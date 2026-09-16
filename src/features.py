"""Извлечение признаков: MSE, SSIM, latent_dist, grad_diff, motion, raw latent."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm.auto import tqdm


# ---------- Загрузка изображений ----------

def load_img(path: str | Path, img_h: int = 256, img_w: int = 256) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((img_w, img_h), Image.BICUBIC)
    return np.asarray(im).astype(np.float32) / 255.0


# ---------- Простые метрики ----------

def mse_full(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mse_masked(a: np.ndarray, b: np.ndarray, mask_2d: np.ndarray) -> float:
    d2 = (a - b) ** 2
    m = d2 * mask_2d[..., None]
    return float(m.sum() / (mask_2d.sum() * 3 + 1e-12))


def ssim_fn(a: np.ndarray, b: np.ndarray) -> float:
    return float(tf.image.ssim(a[None], b[None], max_val=1.0).numpy()[0])


def ms_ssim_fn(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(tf.image.ms_ssim(a[None], b[None], max_val=1.0).numpy()[0])
    except Exception:
        return ssim_fn(a, b)


def grad_diff_fn(a: np.ndarray, b: np.ndarray) -> float:
    ga = np.gradient(a, axis=(0, 1))
    gb = np.gradient(b, axis=(0, 1))
    return float(np.mean(np.abs(ga[0] - gb[0])) + np.mean(np.abs(ga[1] - gb[1])))


def motion_fn(
    prev: np.ndarray, cur: np.ndarray, thr: float = 0.08
) -> Tuple[float, float, float]:
    diff = np.abs(cur - prev)
    energy = float(np.mean(diff))
    area = float(np.mean(diff.mean(axis=2) > thr))
    return energy, area, 0.7 * energy + 0.3 * area


# ---------- Парсинг имени файла (для motion) ----------

def _parse_dt(path: str | Path) -> Optional[datetime]:
    import re
    name = Path(path).name
    m = re.match(r"swap_(\d{2})_(\d{2})_(\d{4})_t_(\d{2})_(\d{2})_(\d{2})", name)
    if not m:
        return None
    d, mo, y, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)


# ---------- Mean latent ----------

def compute_mean_latent(
    paths: List[str],
    encoder,
    mask: np.ndarray,
    bottleneck: int = 64,
    batch_size: int = 16,
    img_h: int = 256,
    img_w: int = 256,
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    batch: List[np.ndarray] = []

    def flush():
        arr = np.stack(batch)
        mu = encoder.predict(arr, verbose=0)[0]
        vecs.append(mu.mean(axis=(1, 2)))
        batch.clear()

    for p in tqdm(paths, desc="mean latent"):
        batch.append(load_img(p, img_h, img_w) * mask)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()
    return np.vstack(vecs).mean(axis=0) if vecs else np.zeros(bottleneck, np.float32)


# ---------- Базовые признаки (5) ----------

BASE_COLS = ["mse_full", "mse_masked", "latent_dist", "ssim", "ms_ssim"]
NEW_COLS = ["grad_diff", "motion_energy", "motion_area", "motion_score"]


def compute_base_features(
    paths: List[str],
    encoder,
    decoder,
    mean_latent: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 8,
    img_h: int = 256,
    img_w: int = 256,
) -> List[Dict]:
    """5 базовых признаков: mse_full, mse_masked, latent_dist, ssim, ms_ssim."""
    feats: List[Dict] = []
    bp, bo, bm = [], [], []

    def flush():
        if not bp:
            return
        arr = np.stack(bm)
        mu, lv, z = encoder.predict(arr, verbose=0)
        recs = decoder.predict(z, verbose=0)
        for i in range(len(bp)):
            orig, recon = bo[i], recs[i]
            lat = mu[i].mean(axis=(0, 1))
            feats.append({
                "path": bp[i],
                "mse_full": mse_full(orig, recon),
                "mse_masked": mse_masked(orig, recon, mask[..., 0]),
                "latent_dist": float(np.linalg.norm(lat - mean_latent)),
                "ssim": ssim_fn(orig, recon),
                "ms_ssim": ms_ssim_fn(orig, recon),
            })
        bp.clear(); bo.clear(); bm.clear()

    for p in tqdm(paths, desc="features"):
        orig = load_img(p, img_h, img_w)
        bp.append(p); bo.append(orig); bm.append(orig * mask)
        if len(bp) >= batch_size:
            flush()
    flush()
    return feats


# ---------- Расширенные признаки (73) ----------

def compute_motion_map(
    paths: List[str],
    max_gap: timedelta = timedelta(hours=24),
    img_h: int = 256,
    img_w: int = 256,
) -> Dict[str, Tuple[float, float, float]]:
    """Считает motion для каждого файла, сравнивая с предыдущим кадром (если разрыв < max_gap)."""
    items = sorted([(dt, p) for p in paths if (dt := _parse_dt(p)) is not None])
    motion_map: Dict[str, Tuple[float, float, float]] = {}
    last_dt, last_img = None, None
    for dt, p in tqdm(items, desc="motion"):
        cur = load_img(p, img_h, img_w)
        if last_img is None or (dt - last_dt) > max_gap:
            motion_map[p] = (0.0, 0.0, 0.0)
        else:
            motion_map[p] = motion_fn(last_img, cur)
        last_dt, last_img = dt, cur
    return motion_map


def compute_extended_features(
    paths: List[str],
    encoder,
    decoder,
    mean_latent: np.ndarray,
    mask: np.ndarray,
    motion_map: Dict[str, Tuple[float, float, float]],
    batch_size: int = 8,
    img_h: int = 256,
    img_w: int = 256,
) -> List[Dict]:
    """5 базовых + grad_diff + 3 motion + 64 raw latent = 73 признака."""
    feats: List[Dict] = []
    bp, bo, bm = [], [], []

    def flush():
        if not bp:
            return
        arr = np.stack(bm)
        mu, lv, z = encoder.predict(arr, verbose=0)
        recs = decoder.predict(z, verbose=0)
        for i in range(len(bp)):
            orig, recon = bo[i], recs[i]
            lat = mu[i].mean(axis=(0, 1))
            me, ma, ms = motion_map.get(bp[i], (0.0, 0.0, 0.0))
            rec = {
                "path": bp[i],
                "mse_full": mse_full(orig, recon),
                "mse_masked": mse_masked(orig, recon, mask[..., 0]),
                "latent_dist": float(np.linalg.norm(lat - mean_latent)),
                "ssim": ssim_fn(orig, recon),
                "ms_ssim": ms_ssim_fn(orig, recon),
                "grad_diff": grad_diff_fn(orig, recon),
                "motion_energy": me,
                "motion_area": ma,
                "motion_score": ms,
            }
            for j in range(len(lat)):
                rec[f"lat{j}"] = float(lat[j])
            feats.append(rec)
        bp.clear(); bo.clear(); bm.clear()

    for p in tqdm(paths, desc="VAE+latent"):
        orig = load_img(p, img_h, img_w)
        bp.append(p); bo.append(orig); bm.append(orig * mask)
        if len(bp) >= batch_size:
            flush()
    flush()
    return feats