"""Real-World 3-Stage Vision ML Inspector for Chiller FaultGuard v12.0.

Replaces brittle color-threshold heuristics with a lighting- and filament-invariant
3-stage offline pipeline designed for real iPhone 11 photos of 3D-printed Mini-CT1 rigs:
  Stage 1: ArUco 4-corner homography (DICT_4X4_50 IDs 0..3 -> 640x400 canonical frame)
           with aspect-preserving fallback crop (zero portrait stretch) + bounded WB.
  Stage 2: Frozen HistGradientBoostingClassifier (`models/fouling_cv_model.joblib`)
           scoring 14 invariant features across 6x4 fill + 6x1 strainer cells (< 30 ms CPU).
  Stage 3: 1-Tap Live Bench Calibration Fusion (local SSIM + illumination-matched Lab dE
           + edge/FFT occlusion drop against the user's live Clean Part 07 baseline).

Also maintains rock-solid `spray_flow` (4-nozzle cone symmetry + optical rotameter GPM)
and `basin_water` (white-balance calibrated CIELAB NTU + 5-in-1 CoC strip) pipelines
upgraded with aspect-preserving framing.

All outputs remain strictly advisory_only=True (IEC 62443-3-3; verified_water_savings=None).
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .vision_features import (
    FEATURE_NAMES,
    FILL_ROI,
    FILL_ROWS,
    FRAME_H,
    FRAME_W,
    GRID_CELLS,
    GRID_COLS,
    STRAINER_ROI,
    apply_bounded_white_balance,
    compute_local_ssim,
    extract_patch_features,
)

VALID_VISION_MODES = ("fill_strainer", "spray_flow", "basin_water")
MAX_IMAGE_BYTES = 6 * 1024 * 1024

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
VISION_MODEL_PATH = MODELS_DIR / "fouling_cv_model.joblib"
VISION_META_PATH = MODELS_DIR / "vision_model_meta.json"
CALIB_DIR = MODELS_DIR / "live_calibration"
CALIB_META_PATH = CALIB_DIR / "bench_calibration.json"
CALIB_IMG_PATH = CALIB_DIR / "clean_ref_640x400.png"
CALIB_FEATS_PATH = CALIB_DIR / "clean_ref_features.npy"

# Canonical ArUco DICT_4X4_50 marker center targets on 640x400 frame (enclosing Fill & Strainer ROIs)
ARUCO_DST_PTS = np.array(
    [
        [36.0, 44.0],    # ID 0 (TL)
        [604.0, 44.0],   # ID 1 (TR)
        [604.0, 376.0],  # ID 2 (BR)
        [36.0, 376.0],   # ID 3 (BL)
    ],
    dtype=np.float32,
)

_CACHED_MODEL: Any = None
_CACHED_META: Optional[Dict[str, Any]] = None
_ARUCO_DETECTOR: Any = None
_ACTIVE_CALIB_DIR: Optional[Path] = None
_MEM_CALIB_META: Optional[Dict[str, Any]] = None
_MEM_CALIB_REF: Optional[Tuple[np.ndarray, np.ndarray]] = None
_SYNTHETIC_VISION_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_JPEG_PREVIEW_CACHE: Dict[str, str] = {}


def _get_calib_dir() -> Path:
    """Resolve writable calibration store (local models/live_calibration or /tmp on Vercel read-only FS)."""
    global _ACTIVE_CALIB_DIR
    if _ACTIVE_CALIB_DIR is not None:
        return _ACTIVE_CALIB_DIR
    if os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"):
        _ACTIVE_CALIB_DIR = Path(tempfile.gettempdir()) / "chiller_faultguard_v12_calibration"
        return _ACTIVE_CALIB_DIR
    try:
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        probe = CALIB_DIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        _ACTIVE_CALIB_DIR = CALIB_DIR
    except OSError:
        _ACTIVE_CALIB_DIR = Path(tempfile.gettempdir()) / "chiller_faultguard_v12_calibration"
    return _ACTIVE_CALIB_DIR


def _get_calib_paths() -> Tuple[Path, Path, Path, Path]:
    cdir = _get_calib_dir()
    return (
        cdir,
        cdir / "bench_calibration.json",
        cdir / "clean_ref_640x400.png",
        cdir / "clean_ref_features.npy",
    )


def _get_aruco_detector() -> Any:
    global _ARUCO_DETECTOR
    if _ARUCO_DETECTOR is None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(aruco_dict, params)
    return _ARUCO_DETECTOR


def _load_vision_ml() -> Tuple[Any, Dict[str, Any]]:
    global _CACHED_MODEL, _CACHED_META
    if _CACHED_MODEL is not None and _CACHED_META is not None:
        return _CACHED_MODEL, _CACHED_META
    if VISION_MODEL_PATH.exists() and VISION_META_PATH.exists():
        _CACHED_MODEL = joblib.load(VISION_MODEL_PATH)
        _CACHED_META = json.loads(VISION_META_PATH.read_text(encoding="utf-8"))
        return _CACHED_MODEL, _CACHED_META
    return None, {
        "version": "12.0.0",
        "model_type": "HistGradientBoostingClassifier",
        "model_sha256": None,
        "model_sha256_short": None,
        "decision_threshold": 0.55,
        "loaded": False,
    }


def reload_vision_ml_cache() -> None:
    global _CACHED_MODEL, _CACHED_META
    _CACHED_MODEL = None
    _CACHED_META = None


def get_vision_ml_card() -> Dict[str, Any]:
    """Return metadata card for the frozen v12 Vision ML classifier."""
    model, meta = _load_vision_ml()
    return {
        **meta,
        "loaded": model is not None,
    }


def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)


def _nms_detections(dets: List[Dict[str, Any]], iou_thresh: float = 0.42) -> List[Dict[str, Any]]:
    ordered = sorted(dets, key=lambda d: (d["confidence"], d["area_pct"]), reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in ordered:
        if all(
            (cand["class_name"] != prev["class_name"])
            or (_compute_iou(cand["bbox_xyxy"], prev["bbox_xyxy"]) < iou_thresh)
            for prev in kept
        ):
            kept.append(cand)
    return kept


def decode_base64_image(image_base64: str) -> Tuple[np.ndarray, str]:
    """Decode a base64 JPEG/PNG payload (with or without data URI prefix) to BGR ndarray + SHA-256."""
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ValueError("image_base64 must be a non-empty base64 string.")
    s = image_base64.strip()
    if "," in s and s.lower().startswith("data:image/"):
        s = s.split(",", 1)[1].strip()
    if len(s) > int(MAX_IMAGE_BYTES * 1.4):
        raise ValueError("Uploaded image exceeds 6 MB limit.")
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image encoding.") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Decoded image is empty or exceeds 6 MB limit.")
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0 or bgr.shape[0] < 16 or bgr.shape[1] < 16:
        raise ValueError("Unable to decode valid JPEG/PNG image pixels.")
    frame_sha = hashlib.sha256(raw).hexdigest()
    return bgr, frame_sha


def encode_png_data_uri(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("Failed to encode annotated inspection PNG.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def encode_jpeg_data_uri(bgr: np.ndarray, quality: int = 86) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode annotated inspection JPEG.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def convert_data_uri_to_jpeg(data_uri: str, cache_key: Optional[str] = None, quality: int = 86) -> str:
    """Convert an annotated PNG data URI to crisp lightweight JPEG (~12x smaller for web transfer)."""
    if not data_uri or data_uri.startswith("data:image/jpeg;base64,"):
        return data_uri
    if cache_key and cache_key in _JPEG_PREVIEW_CACHE:
        return _JPEG_PREVIEW_CACHE[cache_key]
    try:
        raw_b64 = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
        raw = base64.b64decode(raw_b64)
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            return data_uri
        jpg_uri = encode_jpeg_data_uri(bgr, quality=quality)
        if cache_key:
            if len(_JPEG_PREVIEW_CACHE) >= 48:
                _JPEG_PREVIEW_CACHE.clear()
            _JPEG_PREVIEW_CACHE[cache_key] = jpg_uri
        return jpg_uri
    except Exception:
        return data_uri


# ---------------------------------------------------------------------------
# Stage 1: ArUco 4-Corner Homography & Aspect-Preserving Framing
# ---------------------------------------------------------------------------

def aspect_preserving_canonical(bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    """Center-crop to 1.60 (640:400) aspect ratio before resizing so portrait photos are never stretched."""
    h, w = bgr.shape[:2]
    if w == FRAME_W and h == FRAME_H:
        return bgr, "canonical_direct"
    target_ar = float(FRAME_W) / float(FRAME_H)
    cur_ar = float(w) / float(max(h, 1))
    if abs(cur_ar - target_ar) <= 0.04:
        return cv2.resize(bgr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA), "aspect_preserving_resize"
    if cur_ar < target_ar:
        # Taller than 1.60 (e.g. 3:4 portrait or 4:3 landscape iPhone photo): crop vertically
        crop_h = max(16, int(round(w / target_ar)))
        y0 = max(0, (h - crop_h) // 2)
        cropped = bgr[y0 : y0 + crop_h, :]
    else:
        # Wider than 1.60 (e.g. 16:9 or ultra-wide): crop horizontally
        crop_w = max(16, int(round(h * target_ar)))
        x0 = max(0, (w - crop_w) // 2)
        cropped = bgr[:, x0 : x0 + crop_w]
    return cv2.resize(cropped, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA), "aspect_preserving_crop"


def register_fill_frame(bgr: np.ndarray) -> Tuple[np.ndarray, bool, List[int], str]:
    """Stage 1: Detect DICT_4X4_50 markers 0..3 -> perspective homography, else aspect-preserving crop."""
    h, w = bgr.shape[:2]
    detector = _get_aruco_detector()

    scale = 1.0
    det_img = bgr
    if max(h, w) > 1400:
        scale = 1280.0 / float(max(h, w))
        det_img = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    found_centers: Dict[int, np.ndarray] = {}
    if ids is not None and len(ids) > 0:
        flat_ids = np.asarray(ids).ravel().tolist()
        for c_pts, mid_val in zip(corners, flat_ids):
            mid = int(mid_val)
            if mid in (0, 1, 2, 3):
                pts = np.asarray(c_pts, dtype=np.float32).reshape(-1, 2)
                center = np.mean(pts, axis=0) / scale
                found_centers[mid] = center.astype(np.float32)

    markers_detected = sorted(found_centers.keys())
    if len(found_centers) == 4:
        src_pts = np.array([found_centers[0], found_centers[1], found_centers[2], found_centers[3]], dtype=np.float32)
        area = float(cv2.contourArea(src_pts))
        if area >= 400.0:
            H = cv2.getPerspectiveTransform(src_pts, ARUCO_DST_PTS)
            warped = cv2.warpPerspective(
                bgr, H, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            return warped, True, markers_detected, "aruco_homography"

    canon, reg_mode = aspect_preserving_canonical(bgr)
    return canon, False, markers_detected, reg_mode


# ---------------------------------------------------------------------------
# Stage 3: 1-Tap Live Bench Calibration Store (models/live_calibration/)
# ---------------------------------------------------------------------------

def get_calibration_status() -> Dict[str, Any]:
    """Return current live bench calibration status."""
    _, meta_path, img_path, feats_path = _get_calib_paths()
    if meta_path.exists() and img_path.exists() and feats_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["calibrated"] = True
            return meta
        except Exception:
            pass
    if _MEM_CALIB_META is not None and _MEM_CALIB_REF is not None:
        return dict(_MEM_CALIB_META)
    return {
        "calibrated": False,
        "calibrated_at": None,
        "label": None,
        "mode": "fill_strainer",
        "aruco_locked": False,
        "markers_detected": [],
        "registration_mode": None,
        "ref_sha256": None,
        "ref_sha256_short": None,
    }


def save_bench_calibration(
    bgr: np.ndarray,
    mode: str = "fill_strainer",
    label: str = "Clean Part 07",
) -> Dict[str, Any]:
    """Register and persist the user's live Clean Part 07 baseline for Stage 3 structural fusion."""
    global _MEM_CALIB_META, _MEM_CALIB_REF
    if mode not in VALID_VISION_MODES:
        raise ValueError(f"mode must be one of {VALID_VISION_MODES}")
    canon, aruco_locked, markers, reg_mode = register_fill_frame(bgr)
    wb_bgr, wb_applied = apply_bounded_white_balance(canon)
    feats, _ = extract_patch_features(wb_bgr)

    ok, png_buf = cv2.imencode(".png", wb_bgr)
    if not ok:
        raise RuntimeError("Failed to encode calibration reference PNG.")
    png_bytes = png_buf.tobytes()

    ref_sha = hashlib.sha256(png_bytes).hexdigest()
    meta: Dict[str, Any] = {
        "calibrated": True,
        "calibrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": (label or "Clean Part 07").strip()[:80],
        "mode": mode,
        "aruco_locked": aruco_locked,
        "markers_detected": markers,
        "registration_mode": reg_mode,
        "wb_normalized": wb_applied,
        "ref_sha256": ref_sha,
        "ref_sha256_short": ref_sha[:16],
        "feature_shape": list(feats.shape),
    }
    _MEM_CALIB_META = dict(meta)
    _MEM_CALIB_REF = (wb_bgr.copy(), feats.astype(np.float32))

    cdir, meta_path, img_path, feats_path = _get_calib_paths()
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(png_bytes)
        np.save(feats_path, feats.astype(np.float32))
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        # Read-only filesystem fallback: keep in-memory calibration active
        pass
    return meta


def clear_bench_calibration() -> Dict[str, Any]:
    """Delete any persisted live bench calibration baseline."""
    global _MEM_CALIB_META, _MEM_CALIB_REF
    _MEM_CALIB_META = None
    _MEM_CALIB_REF = None
    _, meta_path, img_path, feats_path = _get_calib_paths()
    for p in (meta_path, img_path, feats_path):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    return get_calibration_status()


def _load_calibration_ref() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    _, meta_path, img_path, feats_path = _get_calib_paths()
    if meta_path.exists() and img_path.exists() and feats_path.exists():
        ref_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if ref_bgr is not None and ref_bgr.shape[:2] == (FRAME_H, FRAME_W):
            try:
                ref_feats = np.load(feats_path)
                if ref_feats.shape == (len(GRID_CELLS), len(FEATURE_NAMES)):
                    return ref_bgr, ref_feats.astype(np.float32)
            except Exception:
                pass
    if _MEM_CALIB_REF is not None:
        return _MEM_CALIB_REF
    return None


# ---------------------------------------------------------------------------
# Deterministic Photorealistic 640x400 Mini-CT1 Synthetic Frame Generators
# ---------------------------------------------------------------------------

def _base_housing_canvas() -> np.ndarray:
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    y_grad = np.linspace(34, 22, FRAME_H, dtype=np.float32)[:, None]
    img[:, :, 0] = np.clip(y_grad + 10, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(y_grad + 4, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(y_grad, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (0, 28), (18, FRAME_H - 1), (48, 54, 62), -1)
    cv2.rectangle(img, (FRAME_W - 18, 28), (FRAME_W - 1, FRAME_H - 1), (48, 54, 62), -1)
    cv2.line(img, (18, 28), (18, FRAME_H - 1), (74, 84, 96), 1)
    cv2.line(img, (FRAME_W - 18, 28), (FRAME_W - 18, FRAME_H - 1), (74, 84, 96), 1)
    return img


def generate_fill_strainer_frame(condition: str = "fouled_065") -> np.ndarray:
    """Synthesize a crisp 640x400 Station 2 Mini-CT1 Fill Media & Strainer inspection image."""
    rng = np.random.default_rng(20261001 if condition == "fouled_065" else 20261002 if condition == "fouled_mild" else 20261003)
    img = _base_housing_canvas()

    cv2.rectangle(img, (54, 42), (586, 382), (44, 50, 58), -1)
    cv2.rectangle(img, (54, 42), (586, 382), (96, 108, 122), 2)

    fx1, fy1, fx2, fy2 = FILL_ROI
    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (38, 43, 49), -1)
    for offset in range(-260, 560, 14):
        cv2.line(img, (fx1 + offset, fy1), (fx1 + offset + 115, fy2), (58, 65, 73), 2)
        cv2.line(img, (fx1 + offset + 6, fy1), (fx1 + offset + 121, fy2), (28, 32, 37), 1)
    for offset in range(-260, 560, 28):
        cv2.line(img, (fx1 + offset + 115, fy1), (fx1 + offset, fy2), (50, 57, 64), 1)
    cv2.rectangle(img, (56, 44), (67, 294), (44, 50, 58), -1)
    cv2.rectangle(img, (573, 44), (584, 294), (44, 50, 58), -1)
    cv2.rectangle(img, (56, 44), (584, 51), (44, 50, 58), -1)

    sx1, sy1, sx2, sy2 = STRAINER_ROI
    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (62, 70, 78), -1)
    for gx in range(sx1 + 10, sx2 - 6, 16):
        for gy in range(sy1 + 10, sy2 - 6, 14):
            cv2.circle(img, (gx, gy), 4, (28, 32, 38), -1)
            cv2.circle(img, (gx, gy), 4, (86, 95, 106), 1)
    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (104, 114, 126), 1)
    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (96, 108, 122), 1)

    if condition == "fouled_065":
        scale_polys = [
            np.array([[94, 72], [178, 66], [228, 92], [238, 148], [198, 182], [126, 174], [88, 130]], dtype=np.int32),
            np.array([[272, 108], [362, 98], [412, 134], [398, 198], [322, 212], [268, 170]], dtype=np.int32),
        ]
        for poly in scale_polys:
            mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            noise = rng.normal(0, 11, (FRAME_H, FRAME_W, 3))
            crust = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)
            crust[:, :, 0] = 222.0
            crust[:, :, 1] = 229.0
            crust[:, :, 2] = 235.0
            crust = np.clip(crust + noise, 190, 250).astype(np.uint8)
            img[mask > 0] = crust[mask > 0]
            cv2.polylines(img, [poly], True, (182, 190, 198), 2)

        bio_poly = np.array([[424, 152], [534, 144], [556, 212], [540, 272], [446, 276], [412, 218]], dtype=np.int32)
        bmask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        cv2.fillPoly(bmask, [bio_poly], 255)
        bnoise = rng.normal(0, 12, (FRAME_H, FRAME_W, 3))
        bio = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)
        bio[:, :, 0] = 44.0
        bio[:, :, 1] = 132.0
        bio[:, :, 2] = 56.0
        bio = np.clip(bio + bnoise, 20, 185).astype(np.uint8)
        img[bmask > 0] = bio[bmask > 0]
        cv2.polylines(img, [bio_poly], True, (34, 96, 42), 2)

        clog_poly = np.array([[108, 310], [312, 308], [326, 362], [116, 366], [96, 342]], dtype=np.int32)
        cmask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        cv2.fillPoly(cmask, [clog_poly], 255)
        cnoise = rng.normal(0, 10, (FRAME_H, FRAME_W, 3))
        debris = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)
        debris[:, :, 0] = 198.0
        debris[:, :, 1] = 208.0
        debris[:, :, 2] = 212.0
        debris = np.clip(debris + cnoise, 175, 240).astype(np.uint8)
        img[cmask > 0] = debris[cmask > 0]

    elif condition == "fouled_mild":
        poly = np.array([[138, 108], [216, 104], [228, 152], [164, 166], [128, 140]], dtype=np.int32)
        mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        noise = rng.normal(0, 9, (FRAME_H, FRAME_W, 3))
        crust = np.clip(np.array([220.0, 226.0, 232.0]) + noise, 190, 248).astype(np.uint8)
        img[mask > 0] = crust[mask > 0]

    grain = rng.normal(0, 2.2, img.shape)
    return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def generate_spray_flow_frame(condition: str = "healthy") -> np.ndarray:
    """Synthesize a 640x400 Station 1/2 Spray Nozzles (4-cone header) + Left CW Rotameter Float image."""
    rng = np.random.default_rng(20262001 if condition == "healthy" else 20262002)
    img = _base_housing_canvas()

    flow_gpm = 325.0 if condition == "healthy" else 162.0
    clogged_idx = set() if condition == "healthy" else {1, 3}

    cv2.rectangle(img, (28, 44), (134, 376), (40, 47, 56), -1)
    cv2.rectangle(img, (28, 44), (134, 376), (92, 106, 120), 2)
    cv2.rectangle(img, (54, 62), (104, 358), (58, 68, 80), -1)
    cv2.rectangle(img, (58, 62), (65, 358), (82, 96, 112), -1)
    cv2.rectangle(img, (54, 62), (104, 358), (130, 146, 164), 1)

    for gpm in (0, 100, 200, 250, 300, 400):
        yt = int(round(344.0 - (gpm / 400.0) * 268.0))
        col = (60, 165, 245) if gpm == 250 else (165, 178, 192)
        cv2.line(img, (104, yt), (114, yt), col, 2 if gpm == 250 else 1)
        cv2.putText(img, str(gpm), (115, yt + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1, cv2.LINE_AA)

    y_float = int(round(344.0 - (flow_gpm / 400.0) * 268.0))
    cv2.circle(img, (79, y_float), 14, (16, 118, 252), -1)
    cv2.circle(img, (75, y_float - 4), 5, (85, 185, 255), -1)
    cv2.circle(img, (79, y_float), 14, (10, 82, 205), 2)

    cv2.rectangle(img, (152, 44), (616, 376), (34, 40, 48), -1)
    cv2.rectangle(img, (152, 44), (616, 376), (92, 106, 120), 2)
    cv2.rectangle(img, (164, 54), (604, 76), (86, 98, 112), -1)
    cv2.rectangle(img, (164, 54), (604, 60), (118, 132, 148), -1)

    nozzle_xs = (212, 326, 440, 554)
    for idx, nx in enumerate(nozzle_xs):
        cv2.rectangle(img, (nx - 12, 76), (nx + 12, 92), (104, 118, 134), -1)
        if idx not in clogged_idx:
            cone = np.array(
                [[nx - 8, 94], [nx + 8, 94], [nx + 48, 342], [nx - 48, 342]],
                dtype=np.int32,
            )
            cmask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            cv2.fillPoly(cmask, [cone], 255)
            overlay = img.copy()
            overlay[cmask > 0] = (232, 192, 54)
            img = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)
            for dx in (-34, -20, -8, 0, 8, 20, 34):
                cv2.line(img, (nx + dx // 5, 98), (nx + dx, 336), (250, 224, 128), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (nx, 92), 11, (218, 224, 232), -1)
            cv2.circle(img, (nx + 5, 96), 7, (205, 212, 220), -1)
            cv2.line(img, (nx - 3, 98), (nx - 12, 195), (150, 136, 72), 2, cv2.LINE_AA)

    grain = rng.normal(0, 2.0, img.shape)
    return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def generate_basin_water_frame(condition: str = "clear_coc4") -> np.ndarray:
    """Synthesize a 640x400 Station 3 Basin Turbidity Cuvette + Color Cal Card + 5-in-1 CoC Strip image."""
    rng = np.random.default_rng(20263001 if condition == "clear_coc4" else 20263002)
    img = _base_housing_canvas()

    wb_cast = np.array([0.94, 0.99, 1.04], dtype=np.float32)

    cv2.rectangle(img, (190, 42), (450, 86), (48, 54, 62), -1)
    cv2.rectangle(img, (190, 42), (450, 86), (132, 144, 158), 1)
    cv2.rectangle(img, (202, 48), (288, 80), (240, 240, 240), -1)
    cv2.rectangle(img, (296, 48), (344, 80), (132, 132, 132), -1)
    cv2.rectangle(img, (352, 48), (378, 80), (50, 62, 210), -1)
    cv2.rectangle(img, (384, 48), (410, 80), (62, 185, 72), -1)
    cv2.rectangle(img, (416, 48), (442, 80), (210, 120, 45), -1)

    cv2.rectangle(img, (64, 106), (344, 368), (52, 60, 70), -1)
    cv2.rectangle(img, (78, 120), (330, 356), (156, 168, 178), 2)
    wx1, wy1, wx2, wy2 = 82, 124, 326, 352
    if condition == "clear_coc4":
        img[wy1:wy2, wx1:wx2] = (218, 208, 192)
        for bx in range(wx1 + 20, wx2 - 14, 28):
            cv2.rectangle(img, (bx, wy1 + 36), (bx + 10, wy2 - 36), (72, 86, 102), -1)
        cv2.line(img, (wx1, wy1 + 18), (wx2, wy1 + 18), (242, 238, 226), 2)
    else:
        turb = np.zeros((wy2 - wy1, wx2 - wx1, 3), dtype=np.float32)
        turb[:, :, 0] = 68.0
        turb[:, :, 1] = 132.0
        turb[:, :, 2] = 116.0
        turb += rng.normal(0, 9.0, turb.shape)
        img[wy1:wy2, wx1:wx2] = np.clip(turb, 35, 185).astype(np.uint8)
        for bx in range(wx1 + 20, wx2 - 14, 28):
            cv2.rectangle(img, (bx, wy1 + 36), (bx + 10, wy2 - 36), (62, 114, 102), -1)
        for _ in range(42):
            px = int(rng.integers(wx1 + 12, wx2 - 12))
            py = int(rng.integers(wy1 + 24, wy2 - 12))
            cv2.circle(img, (px, py), int(rng.integers(2, 5)), (48, 108, 84), -1)
        cv2.line(img, (wx1, wy1 + 18), (wx2, wy1 + 18), (112, 158, 138), 2)

    cv2.rectangle(img, (396, 106), (576, 368), (44, 52, 62), -1)
    cv2.rectangle(img, (396, 106), (576, 368), (112, 126, 142), 2)
    cv2.rectangle(img, (454, 118), (518, 358), (236, 238, 240), -1)

    if condition == "clear_coc4":
        pad_colors = [(72, 148, 224), (168, 116, 146), (196, 152, 48)]
    else:
        pad_colors = [(118, 52, 224), (138, 48, 198), (74, 206, 184)]

    pad_ys = ((136, 188), (212, 264), (288, 340))
    for (py1, py2), bgr_col in zip(pad_ys, pad_colors):
        cv2.rectangle(img, (464, py1), (508, py2), bgr_col, -1)
        cv2.rectangle(img, (464, py1), (508, py2), (160, 166, 172), 1)

    out = img.astype(np.float32) * wb_cast[None, None, :] + rng.normal(0, 1.8, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 2 + Stage 3: Trained Vision ML + Live Bench Calibration Fusion
# ---------------------------------------------------------------------------

def _infer_cell_subclass(cm: Dict[str, Any]) -> str:
    if cm["zone"] == "strainer":
        return "clogged_strainer"
    b, g, r = cm["mean_bgr"]
    rel_a = cm["rel_a_z"]
    rel_b = cm["rel_b_z"]
    rel_l = cm["rel_L_z"]
    mean_l = cm["mean_L"]
    chroma = math.hypot(cm["mean_a"] - 128.0, cm["mean_b"] - 128.0)

    # Green/olive algal biofilm
    if (rel_a <= -1.05) or ((g - r >= 10.0) and (g - b >= 12.0)) or (rel_a <= -0.65 and rel_b >= 0.75):
        return "algal_biofilm"
    # Bright neutral/chalky calcium carbonate scale crust
    if (rel_l >= 0.85 and mean_l >= 148.0 and chroma <= 24.0) or (mean_l >= 182.0 and chroma <= 18.0 and rel_l >= 0.35):
        return "calcium_scale"
    # Unpainted PLA physical bridge/nodule, dark silt, or foreign debris
    return "channel_occlusion"


def _merge_positive_cells(
    pos_indices: List[int],
    probs: np.ndarray,
    cell_meta: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge 4-connected positive cells of matching (zone, subclass) into clean bounding boxes."""
    if not pos_indices:
        return []
    pos_map: Dict[Tuple[str, int, int], Tuple[int, str]] = {}
    for idx in pos_indices:
        cm = cell_meta[idx]
        subcls = _infer_cell_subclass(cm)
        pos_map[(cm["zone"], cm["row"], cm["col"])] = (idx, subcls)

    visited = set()
    dets: List[Dict[str, Any]] = []
    fill_area = float((FILL_ROI[2] - FILL_ROI[0]) * (FILL_ROI[3] - FILL_ROI[1]))
    str_area = float((STRAINER_ROI[2] - STRAINER_ROI[0]) * (STRAINER_ROI[3] - STRAINER_ROI[1]))

    for key, (idx, subcls) in pos_map.items():
        if key in visited:
            continue
        zone = key[0]
        stack = [key]
        visited.add(key)
        comp: List[int] = []
        while stack:
            z, r, c = stack.pop()
            comp.append(pos_map[(z, r, c)][0])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nk = (z, r + dr, c + dc)
                if nk in pos_map and nk not in visited and pos_map[nk][1] == subcls:
                    visited.add(nk)
                    stack.append(nk)

        xs1 = [cell_meta[i]["bbox_xyxy"][0] for i in comp]
        ys1 = [cell_meta[i]["bbox_xyxy"][1] for i in comp]
        xs2 = [cell_meta[i]["bbox_xyxy"][2] for i in comp]
        ys2 = [cell_meta[i]["bbox_xyxy"][3] for i in comp]
        box = [min(xs1), min(ys1), max(xs2), max(ys2)]
        box_area = float((box[2] - box[0]) * (box[3] - box[1]))
        denom = fill_area if zone == "fill" else str_area
        apct = round(min(100.0, (box_area / max(denom, 1.0)) * 100.0), 1)
        conf = round(float(np.clip(np.max(probs[comp]), 0.55, 0.99)), 2)
        sev = "HIGH" if (len(comp) >= 2 or conf >= 0.85 or apct >= 12.0) else "MEDIUM"
        dets.append(
            {
                "class_name": subcls,
                "confidence": conf,
                "bbox_xyxy": box,
                "area_pct": apct,
                "severity": sev,
            }
        )
    return _nms_detections(dets, iou_thresh=0.55)


def _detect_fill_strainer_v12(
    canonical_bgr: np.ndarray,
    use_calibration: bool = True,
) -> Tuple[str, float, Dict[str, Any], List[Dict[str, Any]], str, Dict[str, Any]]:
    """Run Stage 2 (Trained HGB Vision ML) + Stage 3 (1-Tap Live Bench Calibration Fusion)."""
    wb_bgr, wb_applied = apply_bounded_white_balance(canonical_bgr)
    X, cell_meta = extract_patch_features(wb_bgr)

    model, meta = _load_vision_ml()
    thresh = float(meta.get("decision_threshold", 0.55))
    ml_sha = meta.get("model_sha256")

    if model is not None:
        with threadpool_limits(limits=1, user_api="openmp"):
            ml_probs = np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
    else:
        # Safe fallback if model artifact missing: structural anomaly heuristic
        ml_probs = np.clip(0.8 - X[:, 0] * 0.5 - X[:, 2] * 0.5, 0.0, 1.0)

    fused_probs = ml_probs.copy()
    calibration_used = False
    max_cal_diff = 0.0

    if use_calibration:
        cal_ref = _load_calibration_ref()
        if cal_ref is not None:
            ref_bgr, ref_feats = cal_ref
            cur_gray = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
            cur_lab = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Global median luminance match so booth lamp drift does not cause false Delta-E
            fx1, fy1, fx2, fy2 = FILL_ROI
            dL_global = float(np.median(cur_lab[fy1:fy2, fx1:fx2, 0]) - np.median(ref_lab[fy1:fy2, fx1:fx2, 0]))

            for i, cell in enumerate(GRID_CELLS):
                x1, y1, x2, y2 = cell["bbox_xyxy"]
                ssim_val = compute_local_ssim(cur_gray[y1:y2, x1:x2], ref_gray[y1:y2, x1:x2])
                ssim_diff = max(0.0, 1.0 - ssim_val)

                c_l = cur_lab[y1:y2, x1:x2, 0] - dL_global
                r_l = ref_lab[y1:y2, x1:x2, 0]
                da = cur_lab[y1:y2, x1:x2, 1] - ref_lab[y1:y2, x1:x2, 1]
                db = cur_lab[y1:y2, x1:x2, 2] - ref_lab[y1:y2, x1:x2, 2]
                delta_e = float(np.mean(np.sqrt((c_l - r_l) ** 2 + da ** 2 + db ** 2)))

                edge_drop = max(0.0, float(ref_feats[i, 0] - X[i, 0])) / max(float(ref_feats[i, 0]), 0.30)
                fft_drop = max(0.0, float(ref_feats[i, 2] - X[i, 2])) / max(float(ref_feats[i, 2]), 0.20)

                cal_prob = float(
                    np.clip(
                        1.85 * max(0.0, ssim_diff - 0.16)
                        + 0.052 * max(0.0, delta_e - 8.5)
                        + 0.72 * max(0.0, edge_drop - 0.24)
                        + 0.55 * max(0.0, fft_drop - 0.26),
                        0.0,
                        0.99,
                    )
                )
                max_cal_diff = max(max_cal_diff, cal_prob)

                # Visually matches user's own calibrated Clean Part 07 -> suppress spurious ML noise
                if ssim_diff < 0.11 and delta_e < 7.5 and edge_drop < 0.18:
                    fused_probs[i] = min(float(ml_probs[i]) * 0.30, 0.22)
                elif cal_prob >= 0.48:
                    fused_probs[i] = max(float(ml_probs[i]), cal_prob)
                else:
                    fused_probs[i] = 0.70 * float(ml_probs[i]) + 0.30 * cal_prob
            calibration_used = True

    pos_indices = [i for i in range(len(GRID_CELLS)) if float(fused_probs[i]) >= thresh]
    max_prob = float(np.max(fused_probs)) if len(fused_probs) else 0.0

    # Frame-level gating: require >=2 cells OR a high-confidence single-cell fault (>= max(0.72, thresh+0.14))
    single_cell_floor = max(0.72, min(0.90, thresh + 0.14))
    is_fouled = (len(pos_indices) >= 2) or (len(pos_indices) == 1 and max_prob >= single_cell_floor)
    if not is_fouled:
        pos_indices = []

    fill_pos = [i for i in pos_indices if cell_meta[i]["zone"] == "fill"]
    str_pos = [i for i in pos_indices if cell_meta[i]["zone"] == "strainer"]

    scale_cells = [i for i in fill_pos if _infer_cell_subclass(cell_meta[i]) == "calcium_scale"]
    bio_cells = [i for i in fill_pos if _infer_cell_subclass(cell_meta[i]) == "algal_biofilm"]
    occ_cells = [i for i in fill_pos if _infer_cell_subclass(cell_meta[i]) == "channel_occlusion"]

    scale_area_pct = round((len(scale_cells) / 24.0) * 100.0, 1)
    biofilm_area_pct = round((len(bio_cells) / 24.0) * 100.0, 1)
    channel_occ_pct = round((len(occ_cells) / 24.0) * 100.0, 1)
    strainer_occlusion_pct = round((len(str_pos) / 6.0) * 100.0, 1)
    fouling_coverage_pct = round(min(100.0, (len(fill_pos) / 24.0) * 100.0 + 0.30 * strainer_occlusion_pct), 1)

    if is_fouled:
        dets = _merge_positive_cells(pos_indices, fused_probs, cell_meta)
        verdict = "ISSUE_FOULING"
        confidence = round(min(0.99, max(max_prob, 0.72)), 2)
        summary = (
            f"Fouling: {fouling_coverage_pct:.1f}% coverage "
            f"(scale {scale_area_pct:.1f}%, biofilm {biofilm_area_pct:.1f}%, "
            f"occlusion {channel_occ_pct:.1f}%, strainer {strainer_occlusion_pct:.1f}%)."
        )
    else:
        verdict = "OK"
        confidence = round(min(0.98, max(0.80, 1.0 - max_prob * 0.45)), 2)
        fx1, fy1, fx2, fy2 = FILL_ROI
        dets = [
            {
                "class_name": "clean_fill",
                "confidence": confidence,
                "bbox_xyxy": [fx1 + 4, fy1 + 4, fx2 - 4, fy2 - 4],
                "area_pct": 100.0,
                "severity": "OK",
            }
        ]
        summary = f"Clear: 0.0% fouling coverage (max ML p={max_prob:.2f})."

    metrics = {
        "occlusion_pct": fouling_coverage_pct,
        "fouling_coverage_pct": fouling_coverage_pct,
        "scale_area_pct": scale_area_pct,
        "biofilm_area_pct": biofilm_area_pct,
        "channel_occlusion_pct": channel_occ_pct,
        "strainer_occlusion_pct": strainer_occlusion_pct,
        "scale_patch_count": sum(1 for d in dets if d["class_name"] == "calcium_scale"),
        "biofilm_patch_count": sum(1 for d in dets if d["class_name"] == "algal_biofilm"),
        "positive_cells": len(pos_indices),
        "total_cells": len(GRID_CELLS),
        "ml_max_prob": round(max_prob, 4),
    }
    extra_meta = {
        "wb_normalized": wb_applied,
        "calibration_used": calibration_used,
        "ml_model_sha256": ml_sha,
        "ml_max_prob": round(max_prob, 4),
        "fouling_coverage_pct": fouling_coverage_pct,
    }
    return verdict, confidence, metrics, dets, summary, extra_meta


def _detect_spray_flow(bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any], List[Dict[str, Any]], str]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dets: List[Dict[str, Any]] = []

    rx1, ry1, rx2, ry2 = 40, 54, 120, 364
    r_hsv = hsv[ry1:ry2, rx1:rx2]
    orange_mask = (
        (r_hsv[:, :, 0] >= 4) & (r_hsv[:, :, 0] <= 24) & (r_hsv[:, :, 1] >= 120) & (r_hsv[:, :, 2] >= 110)
    ).astype(np.uint8) * 255
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    ocnts, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rotameter_detected = False
    optical_flow_gpm = 0.0
    if ocnts:
        best = max(ocnts, key=cv2.contourArea)
        if cv2.contourArea(best) >= 60:
            x, y, w, h = cv2.boundingRect(best)
            cy = float(ry1 + y + h / 2.0)
            optical_flow_gpm = round(max(0.0, min(400.0, (344.0 - cy) / 268.0 * 400.0)), 1)
            rotameter_detected = True
            dets.append(
                {
                    "class_name": "rotameter_float",
                    "confidence": 0.95,
                    "bbox_xyxy": [rx1 + x - 3, ry1 + y - 3, rx1 + x + w + 3, ry1 + y + h + 3],
                    "area_pct": round((cv2.contourArea(best) / float((rx2 - rx1) * (ry2 - ry1))) * 100.0, 1),
                    "severity": "HIGH" if optical_flow_gpm < 250.0 else "OK",
                }
            )

    bay_edges = [(158, 268), (272, 382), (386, 496), (500, 610)]
    by1, by2 = 88, 346
    healthy_nozzles = 0
    clogged_nozzles = 0
    sym_scores: List[float] = []

    for bx1, bx2 in bay_edges:
        b_hsv = hsv[by1:by2, bx1:bx2]
        cone_mask = (
            (b_hsv[:, :, 0] >= 84) & (b_hsv[:, :, 0] <= 122) & (b_hsv[:, :, 1] >= 50) & (b_hsv[:, :, 2] >= 85)
        ).astype(np.uint8) * 255
        bay_area = float((bx2 - bx1) * (by2 - by1))
        cov_pct = round((int(np.count_nonzero(cone_mask)) / bay_area) * 100.0, 1)

        mid = (bx2 - bx1) // 2
        left_px = float(np.count_nonzero(cone_mask[:, :mid]))
        right_px = float(np.count_nonzero(cone_mask[:, mid:]))
        sym = (min(left_px, right_px) / max(1.0, max(left_px, right_px))) if (left_px + right_px) > 100 else 0.0
        sym_scores.append(sym * 100.0)

        if cov_pct >= 14.0:
            healthy_nozzles += 1
            dets.append(
                {
                    "class_name": "healthy_cone",
                    "confidence": round(min(0.96, 0.80 + min(0.15, cov_pct / 220.0)), 2),
                    "bbox_xyxy": [bx1 + 6, by1 + 4, bx2 - 6, by2 - 4],
                    "area_pct": cov_pct,
                    "severity": "OK",
                }
            )
        else:
            clogged_nozzles += 1
            dets.append(
                {
                    "class_name": "clogged_nozzle",
                    "confidence": 0.91,
                    "bbox_xyxy": [bx1 + 14, by1 - 10, bx2 - 14, by1 + 44],
                    "area_pct": round(max(1.0, 100.0 - cov_pct), 1),
                    "severity": "HIGH",
                }
            )
            dets.append(
                {
                    "class_name": "dry_zone",
                    "confidence": 0.88,
                    "bbox_xyxy": [bx1 + 10, by1 + 52, bx2 - 10, by2 - 10],
                    "area_pct": round(max(5.0, 65.0 - cov_pct), 1),
                    "severity": "MEDIUM",
                }
            )

    spray_symmetry_pct = round(sum(sym_scores) / max(1, len(sym_scores)), 1)
    flow_below_floor = optical_flow_gpm < 250.0

    if clogged_nozzles > 0 or flow_below_floor:
        verdict = "ISSUE_LOW_FLOW_OR_NOZZLE"
        confidence = 0.92
        summary = f"{healthy_nozzles}/4 cones OK · {optical_flow_gpm:.1f} GPM ({'< 250 floor' if flow_below_floor else '≥ 250'})."
    else:
        verdict = "OK"
        confidence = 0.95
        summary = f"4/4 cones OK · {optical_flow_gpm:.1f} GPM (≥ 250)."

    metrics = {
        "optical_flow_gpm": optical_flow_gpm,
        "flow_floor_gpm": 250.0,
        "flow_below_floor": flow_below_floor,
        "rotameter_detected": rotameter_detected,
        "healthy_nozzles": healthy_nozzles,
        "clogged_nozzles": clogged_nozzles,
        "total_nozzles": 4,
        "spray_symmetry_pct": spray_symmetry_pct,
    }
    return verdict, confidence, metrics, dets, summary


def _detect_basin_water(bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any], List[Dict[str, Any]], str]:
    wb_roi = bgr[50:78, 205:285].astype(np.float32)
    mean_bgr = np.maximum(np.mean(wb_roi, axis=(0, 1)), 40.0)
    gains = np.clip(240.0 / mean_bgr, 0.75, 1.35)
    cal_bgr = np.clip(bgr.astype(np.float32) * gains[None, None, :], 0, 255).astype(np.uint8)
    cal_lab = cv2.cvtColor(cal_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cal_hsv = cv2.cvtColor(cal_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    cuv_lab = cal_lab[146:344, 88:320]
    cuv_gray = cv2.cvtColor(cal_bgr[146:344, 88:320], cv2.COLOR_BGR2GRAY)
    mean_l = float(np.mean(cuv_lab[:, :, 0]))
    mean_a = float(np.mean(cuv_lab[:, :, 1])) - 128.0
    mean_b = float(np.mean(cuv_lab[:, :, 2])) - 128.0
    edge_contrast = float(np.std(cv2.Sobel(cuv_gray, cv2.CV_32F, 1, 0, ksize=3)))

    l_atten = max(0.0, (190.0 - mean_l) * 0.34)
    scatter_b = max(0.0, (mean_b - 2.0) * 0.95)
    blur_term = max(0.0, (95.0 - edge_contrast) * 0.14)
    turbidity_ntu = round(min(90.0, max(1.2, 2.0 + l_atten + scatter_b + blur_term)), 1)
    algae_index = round(min(1.0, max(0.0, (max(0.0, -mean_a) * 0.025) + max(0.0, mean_b) * 0.022)), 2)

    p1_hsv = cal_hsv[142:182, 470:502]
    p2_hsv = cal_hsv[218:258, 470:502]
    p3_hsv = cal_hsv[294:334, 470:502]

    h1 = float(np.mean(p1_hsv[:, :, 0]))
    h3 = float(np.mean(p3_hsv[:, :, 0]))
    s2 = float(np.mean(p2_hsv[:, :, 1]))

    ph_est = round(8.65 if (h1 >= 135.0 or h1 <= 4.0) else max(6.8, min(8.4, 7.2 + (abs(h1 - 12.0) * 0.04))), 2)
    hardness_ppm = int(round(max(80.0, min(520.0, 120.0 + s2 * 1.65))))
    if 80.0 <= h3 <= 125.0:
        estimated_coc = round(min(4.8, max(3.6, 3.8 + (h3 - 85.0) * 0.02)), 2)
    elif 20.0 <= h3 < 80.0:
        estimated_coc = round(max(2.1, min(3.1, 2.2 + (h3 - 25.0) * 0.014)), 2)
    else:
        estimated_coc = 3.5

    is_water_issue = (turbidity_ntu > 15.0) or (estimated_coc < 3.2)

    dets: List[Dict[str, Any]] = [
        {
            "class_name": "wb_cal_card",
            "confidence": 0.97,
            "bbox_xyxy": [190, 42, 450, 86],
            "area_pct": 4.5,
            "severity": "OK",
        },
        {
            "class_name": "turbid_basin" if turbidity_ntu > 15.0 else "clear_basin",
            "confidence": 0.93 if turbidity_ntu > 15.0 else 0.95,
            "bbox_xyxy": [78, 120, 330, 356],
            "area_pct": 23.2,
            "severity": "HIGH" if turbidity_ntu > 15.0 else "OK",
        },
        {
            "class_name": "coc_strip_low" if estimated_coc < 3.2 else "coc_strip_ok",
            "confidence": 0.91 if estimated_coc < 3.2 else 0.94,
            "bbox_xyxy": [450, 118, 522, 356],
            "area_pct": 6.7,
            "severity": "HIGH" if estimated_coc < 3.2 else "OK",
        },
    ]

    if is_water_issue:
        verdict = "ISSUE_WATER_CHEM"
        confidence = 0.92
        summary = f"Water issue: {turbidity_ntu:.1f} NTU, CoC {estimated_coc:.2f}, pH {ph_est:.2f}."
    else:
        verdict = "OK"
        confidence = 0.95
        summary = f"Water OK: {turbidity_ntu:.1f} NTU, CoC {estimated_coc:.2f}, pH {ph_est:.2f}."

    metrics = {
        "turbidity_ntu": turbidity_ntu,
        "algae_index": algae_index,
        "ph_est": ph_est,
        "hardness_ppm": hardness_ppm,
        "estimated_coc": estimated_coc,
        "wb_gain_b": round(float(gains[0]), 3),
        "wb_gain_g": round(float(gains[1]), 3),
        "wb_gain_r": round(float(gains[2]), 3),
        "verified_water_savings": None,
    }
    return verdict, confidence, metrics, dets, summary


# ---------------------------------------------------------------------------
# Crisp YOLO-Style Bounding Box & HUD Annotator
# ---------------------------------------------------------------------------

def _class_color_bgr(class_name: str, severity: str) -> Tuple[int, int, int]:
    if severity == "HIGH":
        return (68, 68, 239)   # Crisp ISA red
    if severity == "MEDIUM":
        return (24, 158, 245)  # Crisp amber-orange
    if class_name in ("healthy_cone", "wb_cal_card"):
        return (212, 182, 22)  # Cyan-teal
    return (129, 185, 16)      # Emerald green


def _annotate_frame(
    bgr: np.ndarray,
    mode: str,
    verdict: str,
    detections: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    sha_short: str,
    latency_ms: float,
    aruco_locked: bool = False,
    calibration_used: bool = False,
) -> np.ndarray:
    vis = bgr.copy()
    overlay = vis.copy()

    for d in detections:
        if d["severity"] in ("HIGH", "MEDIUM"):
            x1, y1, x2, y2 = d["bbox_xyxy"]
            col = _class_color_bgr(d["class_name"], d["severity"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
    vis = cv2.addWeighted(overlay, 0.16, vis, 0.84, 0)

    for d in detections:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        col = _class_color_bgr(d["class_name"], d["severity"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        tk = 9
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(vis, (cx, cy), (cx + dx * tk, cy), (255, 255, 255), 2)
            cv2.line(vis, (cx, cy), (cx, cy + dy * tk), (255, 255, 255), 2)

    # Anti-overlap label pill placement (draw all labels after boxes so borders never strike text)
    placed_labels: List[Tuple[int, int, int, int]] = []

    def _overlaps_any(cand: Tuple[int, int, int, int], pad: int = 2) -> bool:
        ax1, ay1, ax2, ay2 = cand
        for bx1, by1, bx2, by2 in placed_labels:
            if not (ax2 + pad <= bx1 or bx2 + pad <= ax1 or ay2 + pad <= by1 or by2 + pad <= ay1):
                return True
        return False

    for d in detections:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        col = _class_color_bgr(d["class_name"], d["severity"])

        extra = ""
        if d["class_name"] == "rotameter_float":
            extra = f" {metrics.get('optical_flow_gpm', 0.0):.0f}GPM"
        elif d["class_name"] in ("turbid_basin", "clear_basin"):
            extra = f" {metrics.get('turbidity_ntu', 0.0):.1f}NTU"
        elif d["class_name"] in ("coc_strip_low", "coc_strip_ok"):
            extra = f" CoC {metrics.get('estimated_coc', 0.0):.1f}"
        label = f"{d['class_name']}{extra} {int(round(d['confidence'] * 100))}%"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        pw = tw + 8
        ph = th + 6

        def _clamp_rect(cx1: int, cy1: int) -> Tuple[int, int, int, int]:
            rx1 = max(4, min(FRAME_W - pw - 4, int(cx1)))
            ry1 = max(31, min(FRAME_H - ph - 4, int(cy1)))
            return (rx1, ry1, rx1 + pw, ry1 + ph)

        candidates: List[Tuple[int, int, int, int]] = [
            _clamp_rect(x1, y1 - ph - 2),
            _clamp_rect(x1 + 3, y1 + 3),
            _clamp_rect(x2 - pw, y1 - ph - 2),
            _clamp_rect(x1 + 3, y2 - ph - 3),
            _clamp_rect(x1, y2 + 2),
        ]
        # Systematic downward/rightward stagger fallbacks for densely packed multi-box frames
        for step in range(1, 8):
            candidates.append(_clamp_rect(x1, y1 + step * (ph + 3)))
            candidates.append(_clamp_rect(x1 + step * 28, y1 - ph - 2))
            candidates.append(_clamp_rect(x2 - pw, y1 + step * (ph + 3)))

        chosen = candidates[0]
        for cand in candidates:
            if not _overlaps_any(cand):
                chosen = cand
                break
        lx1, ly1, lx2, ly2 = chosen
        placed_labels.append(chosen)
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), col, -1)
        cv2.putText(vis, label, (lx1 + 4, ly2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(vis, (0, 0), (FRAME_W, 27), (18, 23, 30), -1)
    cv2.line(vis, (0, 27), (FRAME_W, 27), (64, 76, 92), 1)
    tags = []
    if aruco_locked:
        tags.append("ARUCO")
    if calibration_used:
        tags.append("CAL")
    tag_str = (" | " + "+".join(tags)) if tags else ""
    hud_left = f"v12 ML-CV | {mode.upper()}{tag_str} | {sha_short[:8]} | {latency_ms:.1f}ms"
    cv2.putText(vis, hud_left, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (218, 228, 240), 1, cv2.LINE_AA)

    v_col = (68, 68, 239) if verdict != "OK" else (129, 185, 16)
    v_txt = verdict
    (vw, _), _ = cv2.getTextSize(v_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(vis, (FRAME_W - vw - 18, 3), (FRAME_W - 6, 23), v_col, -1)
    cv2.putText(vis, v_txt, (FRAME_W - vw - 12, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def inspect_bgr_frame(
    bgr: np.ndarray,
    mode: str,
    source: str = "sim_bench",
    precomputed_sha256: Optional[str] = None,
    use_calibration: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run v12 3-stage pipeline on a BGR image and return YOLO-compatible dict with ML + ArUco metadata."""
    if mode not in VALID_VISION_MODES:
        raise ValueError(f"mode must be one of {VALID_VISION_MODES}")
    t0 = time.perf_counter()

    # Apply live calibration fusion automatically on real uploaded/camera photos, never on sim_bench presets
    apply_cal = (not source.startswith("sim_bench:")) if use_calibration is None else bool(use_calibration)

    _, ml_meta = _load_vision_ml()
    aruco_locked = False
    markers_detected: List[int] = []
    wb_normalized = False
    calibration_used = False
    ml_model_sha256 = ml_meta.get("model_sha256")
    ml_max_prob = 0.0
    fouling_coverage_pct = 0.0

    if mode == "fill_strainer":
        work_bgr, aruco_locked, markers_detected, reg_mode = register_fill_frame(bgr)
        frame_sha = precomputed_sha256 or hashlib.sha256(work_bgr.tobytes()).hexdigest()
        verdict, conf, metrics, dets, summary, extra = _detect_fill_strainer_v12(
            work_bgr, use_calibration=apply_cal
        )
        wb_normalized = bool(extra["wb_normalized"])
        calibration_used = bool(extra["calibration_used"])
        ml_model_sha256 = extra["ml_model_sha256"]
        ml_max_prob = float(extra["ml_max_prob"])
        fouling_coverage_pct = float(extra["fouling_coverage_pct"])
    elif mode == "spray_flow":
        work_bgr, reg_mode = aspect_preserving_canonical(bgr)
        frame_sha = precomputed_sha256 or hashlib.sha256(work_bgr.tobytes()).hexdigest()
        verdict, conf, metrics, dets, summary = _detect_spray_flow(work_bgr)
    else:
        work_bgr, reg_mode = aspect_preserving_canonical(bgr)
        frame_sha = precomputed_sha256 or hashlib.sha256(work_bgr.tobytes()).hexdigest()
        verdict, conf, metrics, dets, summary = _detect_basin_water(work_bgr)
        wb_normalized = True

    numbered_dets = [{"id": f"D{i}", **d} for i, d in enumerate(dets, start=1)]
    detect_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    annotated = _annotate_frame(
        work_bgr,
        mode,
        verdict,
        numbered_dets,
        metrics,
        frame_sha[:16],
        detect_ms,
        aruco_locked=aruco_locked,
        calibration_used=calibration_used,
    )
    data_uri = encode_png_data_uri(annotated)
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "mode": mode,
        "source": source,
        "verdict": verdict,
        "confidence": conf,
        "metrics": metrics,
        "detections": numbered_dets,
        "annotated_image_data_uri": data_uri,
        "frame_sha256": frame_sha,
        "frame_sha256_short": frame_sha[:16],
        "image_width": FRAME_W,
        "image_height": FRAME_H,
        "latency_ms": total_ms,
        "advisory_only": True,
        "summary": summary,
        "aruco_locked": aruco_locked,
        "markers_detected": markers_detected,
        "registration_mode": reg_mode,
        "wb_normalized": wb_normalized,
        "calibration_used": calibration_used,
        "ml_model_sha256": ml_model_sha256,
        "ml_max_prob": ml_max_prob,
        "fouling_coverage_pct": fouling_coverage_pct,
    }


def run_synthetic_vision(mode: str, condition: str, source: Optional[str] = None) -> Dict[str, Any]:
    """Generate a deterministic Mini-CT1 frame for `mode` + `condition` and run real v12 detection (memoized)."""
    if mode not in VALID_VISION_MODES:
        raise ValueError(f"mode must be one of {VALID_VISION_MODES}")
    src = source or f"sim_bench:{mode}:{condition}"
    ckey = (mode, condition, src)
    cached = _SYNTHETIC_VISION_CACHE.get(ckey)
    if cached is not None:
        return copy.deepcopy(cached)
    if mode == "fill_strainer":
        bgr = generate_fill_strainer_frame(condition)
    elif mode == "spray_flow":
        bgr = generate_spray_flow_frame(condition)
    else:
        bgr = generate_basin_water_frame(condition)
    res = inspect_bgr_frame(bgr, mode=mode, source=src, use_calibration=False)
    if len(_SYNTHETIC_VISION_CACHE) < 64:
        _SYNTHETIC_VISION_CACHE[ckey] = copy.deepcopy(res)
    return res


def inspect_uploaded_base64(
    image_base64: str,
    mode: str,
    source: str = "iphone11_upload",
    use_calibration: Optional[bool] = None,
) -> Dict[str, Any]:
    """Decode an uploaded base64 JPEG/PNG frame from iPhone 11 and execute v12 detection."""
    if mode not in VALID_VISION_MODES:
        raise ValueError(f"mode must be one of {VALID_VISION_MODES}")
    bgr, raw_sha = decode_base64_image(image_base64)
    return inspect_bgr_frame(
        bgr,
        mode=mode,
        source=source,
        precomputed_sha256=raw_sha,
        use_calibration=use_calibration,
    )
