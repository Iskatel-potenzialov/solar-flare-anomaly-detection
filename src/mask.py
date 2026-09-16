"""Круговая маска со сдвигом — используется и при обучении VAE, и при извлечении признаков."""
from __future__ import annotations
import numpy as np


def make_circular_mask(
    height: int,
    width: int,
    radius_frac: float = 0.6,
    shift_x: float = 45.0,
) -> np.ndarray:
    """
    Возвращает маску формы (H, W, 1) типа float32.

    radius_frac — доля от min(H, W)/2
    shift_x     — сдвиг центра по X в пикселях (winning config: +45)
    """
    cy, cx = height / 2.0, width / 2.0
    radius = radius_frac * min(height, width) / 2.0

    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs - shift_x, indexing="ij")
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask_2d = (dist <= radius).astype(np.float32)
    return mask_2d[..., None]