"""Shared Lighting- and Filament-Invariant Patch Feature Extractor for Chiller FaultGuard v12.0.

Used IDENTICALLY by both `train_vision_ml.py` and `vision_inspector.py` to guarantee
0.00 feature skew between offline training and live iPhone 11 bench inference.

Canonical registered frame geometry (640 x 400 BGR):
- Fill ROI:     X in [68, 572], Y in [52, 290]  -> 6 cols x 4 rows = 24 cells
- Strainer ROI: X in [68, 572], Y in [302, 372] -> 6 cols x 1 row  =  6 cells
Total: 30 cells x 14 invariant features per frame (fully vectorized, ~2 ms CPU).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

FRAME_W = 640
FRAME_H = 400

FILL_ROI = (68, 52, 572, 290)       # x1, y1, x2, y2
STRAINER_ROI = (68, 302, 572, 372)  # x1, y1, x2, y2
GRID_COLS = 6
FILL_ROWS = 4
STRAINER_ROWS = 1
TOTAL_CELLS = GRID_COLS * (FILL_ROWS + STRAINER_ROWS)

# Exact half-resolution (320 x 200) tensor slices for vectorized zone extraction:
# Fill:     x=[34:286] (252 = 6 * 42), y=[26:146] (120 = 4 * 30)
# Strainer: x=[34:286] (252 = 6 * 42), y=[151:186] (35 = 1 * 35)
_HALF_X1, _HALF_X2 = 34, 286
_HALF_FILL_Y1, _HALF_FILL_Y2 = 26, 146
_HALF_STR_Y1, _HALF_STR_Y2 = 151, 186
_CELL_W_HALF = (_HALF_X2 - _HALF_X1) // GRID_COLS  # 42
_FILL_H_HALF = (_HALF_FILL_Y2 - _HALF_FILL_Y1) // FILL_ROWS  # 30
_STR_H_HALF = (_HALF_STR_Y2 - _HALF_STR_Y1) // STRAINER_ROWS  # 35

FEATURE_NAMES: Tuple[str, ...] = (
    "flute_edge_ratio",
    "orient_coherence",
    "fft_periodicity",
    "rel_L_z",
    "rel_a_z",
    "rel_b_z",
    "lap_ratio",
    "dark_gap_frac",
    "entropy",
    "glare_score",
    "contrast_iqr",
    "nbr_L_diff",
    "nbr_edge_drop",
    "nbr_chroma_diff",
)


def build_grid_cells() -> List[Dict[str, Any]]:
    """Return exact pixel bounding boxes for the 6x4 fill grid and 6x1 strainer strip."""
    cells: List[Dict[str, Any]] = []
    fx1, fy1, fx2, fy2 = FILL_ROI
    xs = np.linspace(fx1, fx2, GRID_COLS + 1, dtype=int)
    ys = np.linspace(fy1, fy2, FILL_ROWS + 1, dtype=int)
    idx = 0
    for r in range(FILL_ROWS):
        for c in range(GRID_COLS):
            cells.append(
                {
                    "idx": idx,
                    "zone": "fill",
                    "row": r,
                    "col": c,
                    "bbox_xyxy": [int(xs[c]), int(ys[r]), int(xs[c + 1]), int(ys[r + 1])],
                }
            )
            idx += 1

    sx1, sy1, sx2, sy2 = STRAINER_ROI
    sxs = np.linspace(sx1, sx2, GRID_COLS + 1, dtype=int)
    for c in range(GRID_COLS):
        cells.append(
            {
                "idx": idx,
                "zone": "strainer",
                "row": FILL_ROWS,
                "col": c,
                "bbox_xyxy": [int(sxs[c]), int(sy1), int(sxs[c + 1]), int(sy2)],
            }
        )
        idx += 1
    return cells


GRID_CELLS: List[Dict[str, Any]] = build_grid_cells()


def _build_neighbor_matrix() -> np.ndarray:
    """Precompute row-normalized 8-connected same-zone neighbor averaging matrix (30 x 30)."""
    n = len(GRID_CELLS)
    mat = np.zeros((n, n), dtype=np.float32)
    coord_to_idx = {(c["zone"], c["row"], c["col"]): c["idx"] for c in GRID_CELLS}
    for i, cell in enumerate(GRID_CELLS):
        z, r, c = cell["zone"], cell["row"], cell["col"]
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                ni = coord_to_idx.get((z, r + dr, c + dc))
                if ni is not None:
                    nbrs.append(ni)
        if nbrs:
            mat[i, nbrs] = 1.0 / float(len(nbrs))
    return mat


_NBR_WEIGHTS: np.ndarray = _build_neighbor_matrix()

# Precomputed FFT window and frequency masks for half-res cell width (W = 42)
_FFT_FREQS = np.fft.rfftfreq(_CELL_W_HALF, d=1.0)
_FFT_BAND_MASK = (_FFT_FREQS >= (1.0 / 14.0)) & (_FFT_FREQS <= (1.0 / 3.2))
_FFT_AC_MASK = _FFT_FREQS >= (1.0 / 18.0)
_FFT_WIN = np.hanning(_CELL_W_HALF).astype(np.float32)


def apply_bounded_white_balance(bgr: np.ndarray, max_gain_dev: float = 0.22) -> Tuple[np.ndarray, bool]:
    """Fast Shades-of-Gray white balance using stride-4 p=4 norm and uint8 LUT (< 0.8 ms CPU)."""
    roi = bgr[48:374:4, 60:580:4].astype(np.float32)
    if roi.size == 0:
        return bgr, False
    r_pos = np.maximum(roi, 1.0)
    sq = r_pos * r_pos
    means = np.power(np.mean(sq * sq, axis=(0, 1)), 0.25)
    gray_ref = float(np.mean(means))
    if gray_ref < 18.0:
        return bgr, False
    gains = np.clip(gray_ref / np.maximum(means, 1.0), 1.0 - max_gain_dev, 1.0 + max_gain_dev)
    if float(np.max(np.abs(gains - 1.0))) < 0.015:
        return bgr, False
    ramp = np.arange(256, dtype=np.float32)
    lut = np.stack([np.clip(ramp * gains[c], 0.0, 255.0).astype(np.uint8) for c in range(3)], axis=1)
    balanced = cv2.LUT(bgr, lut.reshape(256, 1, 3))
    return balanced, True


def _slice_tiles_2d(arr: np.ndarray, y1: int, y2: int, rows: int, cell_h: int) -> np.ndarray:
    """Reshape a 2D map slice [y1:y2, _HALF_X1:_HALF_X2] into (rows * GRID_COLS, cell_h, _CELL_W_HALF)."""
    sub = arr[y1:y2, _HALF_X1:_HALF_X2]
    return sub.reshape(rows, cell_h, GRID_COLS, _CELL_W_HALF).transpose(0, 2, 1, 3).reshape(
        rows * GRID_COLS, cell_h, _CELL_W_HALF
    )


def _slice_tiles_3d(arr: np.ndarray, y1: int, y2: int, rows: int, cell_h: int) -> np.ndarray:
    """Reshape a 3D map slice [y1:y2, _HALF_X1:_HALF_X2, C] into (rows * GRID_COLS, cell_h, _CELL_W_HALF, C)."""
    c = arr.shape[2]
    sub = arr[y1:y2, _HALF_X1:_HALF_X2, :]
    return sub.reshape(rows, cell_h, GRID_COLS, _CELL_W_HALF, c).transpose(0, 2, 1, 3, 4).reshape(
        rows * GRID_COLS, cell_h, _CELL_W_HALF, c
    )


def _batch_fft_periodicity(gf_tiles: np.ndarray) -> np.ndarray:
    """Fully vectorized 4-band 1D horizontal FFT periodicity across M cells of shape (M, H, W)."""
    m, h, w = gf_tiles.shape
    h4 = (h // 4) * 4
    profs = gf_tiles[:, :h4, :].reshape(m, 4, h4 // 4, w).mean(axis=2)
    profs = (profs - profs.mean(axis=2, keepdims=True)) * _FFT_WIN[None, None, :]
    spec = np.fft.rfft(profs, axis=2)
    pwr = spec.real * spec.real + spec.imag * spec.imag
    total_ac = pwr[:, :, _FFT_AC_MASK].sum(axis=2)
    band_pwr = pwr[:, :, _FFT_BAND_MASK]
    k = np.argmax(band_pwr, axis=2)
    padded = np.pad(band_pwr, ((0, 0), (0, 0), (1, 1)), mode="constant")
    m_idx = np.arange(m)[:, None]
    b_idx = np.arange(4)[None, :]
    peak_e = padded[m_idx, b_idx, k] + padded[m_idx, b_idx, k + 1] + padded[m_idx, b_idx, k + 2]
    scores = np.where(total_ac < 1e-3, 0.0, np.minimum(1.0, peak_e / (total_ac + 1e-6)))
    return scores.mean(axis=1).astype(np.float32)


def _batch_zone_raw_stats(
    bgr_half: np.ndarray,
    gray_u8: np.ndarray,
    gray_f: np.ndarray,
    lab: np.ndarray,
    abs_gx: np.ndarray,
    abs_gy: np.ndarray,
    flute_mag: np.ndarray,
    lap: np.ndarray,
    chroma_full: np.ndarray,
    y1: int,
    y2: int,
    rows: int,
    cell_h: int,
) -> Dict[str, np.ndarray]:
    """Compute raw per-cell descriptors for an entire zone in single vectorized NumPy passes."""
    m = rows * GRID_COLS
    g_tiles = _slice_tiles_2d(gray_u8, y1, y2, rows, cell_h)
    gf_tiles = _slice_tiles_2d(gray_f, y1, y2, rows, cell_h)
    lab_tiles = _slice_tiles_3d(lab, y1, y2, rows, cell_h)
    bgr_tiles = _slice_tiles_3d(bgr_half, y1, y2, rows, cell_h)
    gx_tiles = _slice_tiles_2d(abs_gx, y1, y2, rows, cell_h)
    gy_tiles = _slice_tiles_2d(abs_gy, y1, y2, rows, cell_h)
    fm_tiles = _slice_tiles_2d(flute_mag, y1, y2, rows, cell_h)
    lap_tiles = _slice_tiles_2d(lap, y1, y2, rows, cell_h)
    c_tiles = _slice_tiles_2d(chroma_full, y1, y2, rows, cell_h)

    l_tiles = lab_tiles[:, :, :, 0]
    raw_edge = fm_tiles.mean(axis=(1, 2), dtype=np.float32)
    mgx = gx_tiles.mean(axis=(1, 2), dtype=np.float32)
    mgy = gy_tiles.mean(axis=(1, 2), dtype=np.float32)
    orient_coh = (mgx - mgy) / (mgx + mgy + 1e-4)
    fft_per = _batch_fft_periodicity(gf_tiles)

    mean_lab = lab_tiles.mean(axis=(1, 2), dtype=np.float32)
    mean_L = mean_lab[:, 0]
    mean_a = mean_lab[:, 1]
    mean_b = mean_lab[:, 2]
    mean_bgr = bgr_tiles.mean(axis=(1, 2), dtype=np.float32)
    raw_lap = lap_tiles.std(axis=(1, 2), dtype=np.float32)

    # Fast exact order-statistic quantiles via single np.partition call per zone
    n_pix = cell_h * _CELL_W_HALF
    l_flat = l_tiles.reshape(m, n_pix)
    k10 = int(round(0.10 * (n_pix - 1)))
    k75 = int(round(0.75 * (n_pix - 1)))
    k90 = int(round(0.90 * (n_pix - 1)))
    part = np.partition(l_flat, (k10, k75, k90), axis=1)
    p10 = part[:, k10]
    p75 = part[:, k75]
    p90 = part[:, k90]
    raw_contrast = (p90 - p10).astype(np.float32)
    groove_thresh = p75 - np.maximum(9.0, 0.42 * raw_contrast)
    dark_gap = (l_flat < groove_thresh[:, None]).mean(axis=1, dtype=np.float32)

    # Vectorized 32-bin Shannon entropy across all M cells in one bincount
    bins = (g_tiles >> 3).reshape(m, n_pix).astype(np.int32)
    offsets = (np.arange(m, dtype=np.int32) * 32)[:, None]
    counts = np.bincount((bins + offsets).ravel(), minlength=m * 32).reshape(m, 32).astype(np.float32)
    probs = counts / float(n_pix)
    ent = -np.sum(np.where(probs > 0.0, probs * np.log2(np.maximum(probs, 1e-12)), 0.0), axis=1).astype(np.float32)

    # Specular PLA glare detector
    bright_low_c = (l_tiles > 236.0) & (c_tiles < 12.0)
    glare_cnt = bright_low_c.sum(axis=(1, 2)).astype(np.float32)
    glare_frac = glare_cnt / float(n_pix)
    smooth_mean = (fm_tiles * bright_low_c).sum(axis=(1, 2), dtype=np.float32) / np.maximum(glare_cnt, 1.0)
    mult = np.where(smooth_mean < 24.0, 3.5, 1.2)
    glare = np.where(glare_frac > 0.01, np.minimum(1.0, glare_frac * mult), 0.0).astype(np.float32)

    return {
        "raw_edge": raw_edge,
        "orient_coh": orient_coh,
        "fft_per": fft_per,
        "mean_L": mean_L,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_bgr": mean_bgr,
        "raw_lap": raw_lap,
        "raw_contrast": raw_contrast,
        "dark_gap": dark_gap,
        "ent": ent,
        "glare": glare,
    }


def extract_patch_features(canonical_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extract (30, 14) float32 lighting- & filament-invariant features from a canonical 640x400 BGR image.

    Computes derivatives and colorimetry on a crisp 320x200 anti-aliased working grid (~2 ms CPU)
    while returning exact 640x400 cell bounding boxes for downstream YOLO-compatible annotation.
    """
    bgr_half = cv2.resize(canonical_bgr, (FRAME_W // 2, FRAME_H // 2), interpolation=cv2.INTER_AREA)

    gray_u8 = cv2.cvtColor(bgr_half, cv2.COLOR_BGR2GRAY)
    gray_f = gray_u8.astype(np.float32)
    lab = cv2.cvtColor(bgr_half, cv2.COLOR_BGR2LAB).astype(np.float32)

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    abs_gx = np.abs(gx)
    abs_gy = np.abs(gy)
    flute_mag = 0.55 * abs_gx + 0.45 * np.maximum(np.abs(0.766 * gx + 0.643 * gy), np.abs(0.766 * gx - 0.643 * gy))
    lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    chroma_full = cv2.magnitude(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)

    fill_s = _batch_zone_raw_stats(
        bgr_half, gray_u8, gray_f, lab, abs_gx, abs_gy, flute_mag, lap, chroma_full,
        _HALF_FILL_Y1, _HALF_FILL_Y2, FILL_ROWS, _FILL_H_HALF,
    )
    str_s = _batch_zone_raw_stats(
        bgr_half, gray_u8, gray_f, lab, abs_gx, abs_gy, flute_mag, lap, chroma_full,
        _HALF_STR_Y1, _HALF_STR_Y2, STRAINER_ROWS, _STR_H_HALF,
    )

    raw_edge = np.concatenate([fill_s["raw_edge"], str_s["raw_edge"]])
    orient_coh = np.concatenate([fill_s["orient_coh"], str_s["orient_coh"]])
    fft_per = np.concatenate([fill_s["fft_per"], str_s["fft_per"]])
    mean_L = np.concatenate([fill_s["mean_L"], str_s["mean_L"]])
    mean_a = np.concatenate([fill_s["mean_a"], str_s["mean_a"]])
    mean_b = np.concatenate([fill_s["mean_b"], str_s["mean_b"]])
    mean_bgr = np.concatenate([fill_s["mean_bgr"], str_s["mean_bgr"]], axis=0)
    raw_lap = np.concatenate([fill_s["raw_lap"], str_s["raw_lap"]])
    raw_contrast = np.concatenate([fill_s["raw_contrast"], str_s["raw_contrast"]])
    dark_gap = np.concatenate([fill_s["dark_gap"], str_s["dark_gap"]])
    ent = np.concatenate([fill_s["ent"], str_s["ent"]])
    glare = np.concatenate([fill_s["glare"], str_s["glare"]])

    n = TOTAL_CELLS
    flute_edge_ratio = np.empty(n, dtype=np.float32)
    rel_L_z = np.empty(n, dtype=np.float32)
    rel_a_z = np.empty(n, dtype=np.float32)
    rel_b_z = np.empty(n, dtype=np.float32)
    lap_ratio = np.empty(n, dtype=np.float32)
    contrast_iqr = np.empty(n, dtype=np.float32)
    zone_chroma_scale = np.empty(n, dtype=np.float32)
    zone_L_scale = np.empty(n, dtype=np.float32)

    # Zone-wise robust normalization (fill 0:24 vs strainer 24:30)
    for sl in (slice(0, 24), slice(24, 30)):
        re_z = raw_edge[sl]
        fp_z = fft_per[sl]
        ml_z = mean_L[sl]
        ma_z = mean_a[sl]
        mb_z = mean_b[sl]
        rl_z = raw_lap[sl]
        rc_z = raw_contrast[sl]

        clean_score = re_z * (0.4 + fp_z)
        top_k = max(3, len(re_z) // 2)
        clean_idx = np.argsort(clean_score)[-top_k:]

        ref_edge = max(float(np.percentile(re_z, 75.0)), 8.0)
        ref_lap = max(float(np.percentile(rl_z, 75.0)), 6.0)
        ref_con = max(float(np.percentile(rc_z, 75.0)), 10.0)

        med_L = float(np.median(ml_z[clean_idx]))
        q25_L, q75_L = np.percentile(ml_z, (25.0, 75.0))
        iqr_L = max(float(q75_L - q25_L), 7.0)

        med_a = float(np.median(ma_z[clean_idx]))
        q25_a, q75_a = np.percentile(ma_z, (25.0, 75.0))
        iqr_a = max(float(q75_a - q25_a), 3.0)

        med_b = float(np.median(mb_z[clean_idx]))
        q25_b, q75_b = np.percentile(mb_z, (25.0, 75.0))
        iqr_b = max(float(q75_b - q25_b), 3.0)

        flute_edge_ratio[sl] = np.minimum(3.0, re_z / ref_edge)
        lap_ratio[sl] = np.minimum(3.0, rl_z / ref_lap)
        contrast_iqr[sl] = np.minimum(3.0, rc_z / ref_con)
        rel_L_z[sl] = np.clip((ml_z - med_L) / iqr_L, -8.0, 8.0)
        rel_a_z[sl] = np.clip((ma_z - med_a) / iqr_a, -8.0, 8.0)
        rel_b_z[sl] = np.clip((mb_z - med_b) / iqr_b, -8.0, 8.0)
        zone_L_scale[sl] = iqr_L
        zone_chroma_scale[sl] = max(3.5, float(np.hypot(iqr_a, iqr_b)))

    # Spatial neighborhood contrast features via precomputed adjacency matrix
    nbr_L = _NBR_WEIGHTS @ mean_L
    nbr_a = _NBR_WEIGHTS @ mean_a
    nbr_b = _NBR_WEIGHTS @ mean_b
    nbr_e = _NBR_WEIGHTS @ flute_edge_ratio

    nbr_L_diff = np.clip((mean_L - nbr_L) / np.maximum(zone_L_scale, 7.0), -8.0, 8.0)
    nbr_edge_drop = np.clip(nbr_e - flute_edge_ratio, -3.0, 3.0)
    nbr_chroma_diff = np.minimum(8.0, np.hypot(mean_a - nbr_a, mean_b - nbr_b) / np.maximum(zone_chroma_scale, 3.5))

    X = np.column_stack(
        [
            flute_edge_ratio,
            orient_coh,
            fft_per,
            rel_L_z,
            rel_a_z,
            rel_b_z,
            lap_ratio,
            dark_gap,
            ent,
            glare,
            contrast_iqr,
            nbr_L_diff,
            nbr_edge_drop,
            nbr_chroma_diff,
        ]
    ).astype(np.float32)

    cell_meta: List[Dict[str, Any]] = []
    for i, cell in enumerate(GRID_CELLS):
        cell_meta.append(
            {
                **cell,
                "mean_L": round(float(mean_L[i]), 2),
                "mean_a": round(float(mean_a[i]), 2),
                "mean_b": round(float(mean_b[i]), 2),
                "mean_bgr": [round(float(v), 1) for v in mean_bgr[i]],
                "flute_edge_ratio": round(float(flute_edge_ratio[i]), 3),
                "fft_periodicity": round(float(fft_per[i]), 3),
                "rel_L_z": round(float(rel_L_z[i]), 3),
                "rel_a_z": round(float(rel_a_z[i]), 3),
                "rel_b_z": round(float(rel_b_z[i]), 3),
                "glare_score": round(float(glare[i]), 3),
                "dark_gap_frac": round(float(dark_gap[i]), 3),
            }
        )
    return X, cell_meta


def compute_local_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Fast scalar SSIM over two matching uint8/float32 tiles."""
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    va = float(np.var(a))
    vb = float(np.var(b))
    cov = float(np.mean((a - mu_a) * (b - mu_b)))
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2)
    return float(np.clip(num / max(den, 1e-6), -1.0, 1.0))
