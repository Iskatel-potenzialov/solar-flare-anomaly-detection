"""Загрузка OMNI + снимков, парсинг имён файлов, построение датасета с метками P10."""
from __future__ import annotations
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


SPLIT_MAP_DEFAULT = {"train": "TRAIN", "val": "VAL", "test": "TEST"}


def parse_swap_filename(fname: str) -> Optional[datetime]:
    """swap_DD_MM_YYYY_t_HH_MM_SS.png -> datetime(UTC)."""
    name = Path(fname).name
    m = re.match(r"swap_(\d{2})_(\d{2})_(\d{4})_t_(\d{2})_(\d{2})_(\d{2})", name)
    if not m:
        return None
    d, mo, y, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)


def build_image_dataframe(
    image_root: str | Path,
    split_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Собирает df_img: foto, data_time, mesto, path — отсортированный по времени."""
    if split_map is None:
        split_map = SPLIT_MAP_DEFAULT

    image_root = Path(image_root)
    rows: List[dict] = []
    for sub, split in split_map.items():
        for p in (image_root / sub).glob("*.png"):
            dt = parse_swap_filename(p.name)
            if dt is not None:
                rows.append(
                    {"foto": p.name, "data_time": dt, "mesto": split, "path": str(p)}
                )
    return pd.DataFrame(rows).sort_values("data_time").reset_index(drop=True)


def load_omni(path: str | Path) -> pd.DataFrame:
    """Читает OMNI csv, парсит time_tag, сортирует."""
    df = pd.read_csv(path)
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    return df.sort_values("time_tag").reset_index(drop=True)


def collect_p10_window(
    t: pd.Timestamp,
    times: np.ndarray,
    p10_arr: np.ndarray,
    delay_min: int = 30,
    window_hours: float = 4.0,
) -> Tuple[List[float], int]:
    """Возвращает (значения P10 в окне, число непустых точек)."""
    start = t + pd.Timedelta(minutes=delay_min)
    end = start + pd.Timedelta(hours=window_hours)
    i0 = np.searchsorted(times, np.datetime64(start), side="left")
    i1 = np.searchsorted(times, np.datetime64(end), side="right")
    vals = p10_arr[i0:i1]
    return vals.tolist(), int(pd.notna(vals).sum())


def attach_p10_labels(
    df_img: pd.DataFrame,
    df_omni: pd.DataFrame,
    delay_min: int = 30,
    window_hours: float = 4.0,
    p10_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Добавляет колонки p10_6h, dlina, max_P10, sobitie.
    Оставляет только полные окна (dlina == EXPECTED) без NaN.
    """
    expected = int(window_hours * 60 / 5)  # 48 измерений по 5 мин

    df_omni = df_omni.sort_values("time_tag").reset_index(drop=True)
    times = df_omni["time_tag"].values
    p10_arr = df_omni["P10"].values

    res = [
        collect_p10_window(t, times, p10_arr, delay_min, window_hours)
        for t in tqdm(df_img["data_time"], desc="P10 windows")
    ]

    out = df_img.copy()
    out["p10_6h"] = [r[0] for r in res]
    out["dlina"] = [r[1] for r in res]

    out = out[out["dlina"] == expected].copy()
    out = out[out["p10_6h"].apply(lambda v: not any(pd.isna(x) for x in v))].copy()
    out["max_P10"] = out["p10_6h"].apply(np.max)
    out["sobitie"] = (out["max_P10"] > p10_threshold).astype(int)
    return out


def split_manifest(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает (train_bg, val_all, test_all)."""
    train_bg = manifest[(manifest.mesto == "TRAIN") & (manifest.sobitie == 0)]
    val_all = manifest[manifest.mesto == "VAL"].copy()
    test_all = manifest[manifest.mesto == "TEST"].copy()
    return train_bg, val_all, test_all