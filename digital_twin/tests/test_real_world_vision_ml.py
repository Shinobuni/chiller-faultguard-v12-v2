"""Real-World 3-Stage Vision ML, ArUco Homography, 1-Tap Calibration & Live Audio FFT Suite (v12.0)."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from fastapi.testclient import TestClient

DT_DIR = Path(__file__).resolve().parent.parent
if str(DT_DIR) not in sys.path:
    sys.path.insert(0, str(DT_DIR))

from backend.acoustic_tach_weather import analyze_pcm_fan_audio  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.runtime import ChillerFaultGuardV6Runtime  # noqa: E402
from backend.train_vision_ml import render_domain_randomized_pla_frame  # noqa: E402
from backend.vision_features import FEATURE_NAMES, FRAME_H, FRAME_W, apply_bounded_white_balance, extract_patch_features  # noqa: E402
from backend.vision_inspector import (  # noqa: E402
    ARUCO_DST_PTS,
    VISION_META_PATH,
    VISION_MODEL_PATH,
    clear_bench_calibration,
    generate_basin_water_frame,
    generate_spray_flow_frame,
    inspect_bgr_frame,
)


def _encode_jpg_b64(bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _embed_aruco_and_tilt(canonical_bgr: np.ndarray, tilt_deg: float = 18.0) -> np.ndarray:
    """Place genuine DICT_4X4_50 markers 0..3 at canonical pilaster positions and apply 3D perspective tilt."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas = np.full((520, 800, 3), (42, 46, 52), dtype=np.uint8)
    ox, oy = 80, 60
    canvas[oy : oy + FRAME_H, ox : ox + FRAME_W] = canonical_bgr

    m_size = 44
    pad = 8
    for mid, (cx, cy) in enumerate(ARUCO_DST_PTS):
        mx = int(round(ox + float(cx)))
        my = int(round(oy + float(cy)))
        marker = cv2.aruco.generateImageMarker(aruco_dict, mid, m_size)
        # White quiet-zone border around ArUco tag
        half = (m_size // 2) + pad
        cv2.rectangle(canvas, (mx - half, my - half), (mx + half, my + half), (248, 248, 248), -1)
        m_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        canvas[my - m_size // 2 : my + m_size // 2, mx - m_size // 2 : mx + m_size // 2] = m_bgr

    # Real 3D homography combining yaw/pitch tilt (~tilt_deg) + slight roll
    h, w = canvas.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    s = np.deg2rad(tilt_deg)
    dx = float(np.sin(s) * 68.0)
    dy = float(abs(np.sin(s)) * 42.0)
    if tilt_deg >= 0:
        dst = np.array([[30 + dx, 25 + dy], [w - 30, 15], [w - 40, h - 18], [20 + dx, h - 28 - dy]], dtype=np.float32)
    else:
        dst = np.array([[30, 15], [w - 30 + dx, 25 + dy], [w - 20 + dx, h - 28 - dy], [40, h - 18]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(canvas, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(36, 40, 46))


def test_rw_1_frozen_vision_ml_model_metrics_and_cpu_speed() -> None:
    """Frozen fouling_cv_model.joblib exists, matches SHA256, has CV ROC-AUC >= 0.97, and scores in < 30 ms CPU."""
    assert VISION_MODEL_PATH.exists(), f"Missing trained Vision ML model: {VISION_MODEL_PATH}"
    assert VISION_META_PATH.exists(), f"Missing Vision ML metadata: {VISION_META_PATH}"

    meta = json.loads(VISION_META_PATH.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(VISION_MODEL_PATH.read_bytes()).hexdigest()
    assert meta["model_sha256"] == actual_sha
    assert meta["n_features"] == 14
    assert tuple(meta["feature_names"]) == FEATURE_NAMES
    assert meta["n_patches"] >= 12000
    assert meta["cv_roc_auc"] >= 0.97
    assert meta["inference_ms_cpu"] < 30.0

    # Live CPU timing check (WB + 14-feature extraction + single-thread OpenMP HGB predict_proba, matching production)
    import joblib
    from threadpoolctl import threadpool_limits

    clf = joblib.load(VISION_MODEL_PATH)
    rng = np.random.default_rng(12345)
    sample_bgr, _ = render_domain_randomized_pla_frame(rng, "white_pla", "calcium_scale")
    with threadpool_limits(limits=1, user_api="openmp"):
        # Warm up
        wb, _ = apply_bounded_white_balance(sample_bgr)
        X, _ = extract_patch_features(wb)
        _ = clf.predict_proba(X)

        run_ms = []
        for _ in range(15):
            t0 = time.perf_counter()
            wb, _ = apply_bounded_white_balance(sample_bgr)
            X, _ = extract_patch_features(wb)
            _ = clf.predict_proba(X)[:, 1]
            run_ms.append((time.perf_counter() - t0) * 1000.0)
    med_ms = float(np.median(run_ms))
    assert med_ms < 30.0, f"Live Vision ML CPU inference exceeded 30 ms budget: {med_ms:.2f} ms"


def test_rw_2_all_nine_previously_failing_real_world_pla_photo_cases() -> None:
    """Achieves >= 95% accuracy across all 9+ real-world PLA colors, lighting casts, glare, portrait framing & fouling."""
    clear_bench_calibration()
    rng = np.random.default_rng(20261299)

    cases = [
        # 7 previously failing clean PLA colors/lighting + glare (v11 had 100% false positives on white/light/green!)
        ("white_pla_bright_glare", "white_pla", "clean", True, 1.38, (1.0, 1.0, 1.0), "OK", False),
        ("white_pla_warm_tungsten", "white_pla", "clean", False, 1.15, (0.86, 0.98, 1.14), "OK", False),
        ("light_gray_clean", "light_gray_pla", "clean", True, 1.25, (1.0, 1.0, 1.0), "OK", False),
        ("dark_gray_dim_cool", "dark_gray_pla", "clean", False, 0.50, (1.10, 1.0, 0.88), "OK", False),
        ("industrial_blue_clean", "industrial_blue_pla", "clean", True, 1.10, (1.0, 1.0, 1.0), "OK", False),
        ("teal_green_clean", "teal_green_pla", "clean", False, 1.05, (0.94, 1.05, 0.94), "OK", False),
        ("silver_silk_heavy_glare", "silver_silk_pla", "clean", True, 1.30, (1.0, 1.0, 1.0), "OK", False),
        # Portrait 3:4 phone photo of clean light-gray PLA (must crop aspect-preserving, never stretch 2.1x!)
        ("portrait_3x4_clean_gray", "light_gray_pla", "clean", False, 1.05, (1.0, 1.0, 1.0), "OK", True),
        # Real fouled cartridges across different PLA colors (incl. unpainted single-color relief!)
        ("white_pla_calcium_scale", "white_pla", "calcium_scale", False, 1.10, (1.0, 1.0, 1.0), "ISSUE_FOULING", False),
        ("gray_pla_algal_biofilm", "light_gray_pla", "algal_biofilm", False, 0.95, (0.92, 1.0, 1.08), "ISSUE_FOULING", False),
        ("blue_pla_unpainted_relief", "industrial_blue_pla", "unpainted_relief", False, 1.0, (1.0, 1.0, 1.0), "ISSUE_FOULING", False),
        ("white_pla_strainer_clog", "white_pla", "strainer_clog", False, 1.10, (1.0, 1.0, 1.0), "ISSUE_FOULING", False),
    ]

    passed = 0
    for name, fil, fmode, glare, exp, cast, expected, make_portrait in cases:
        bgr, _ = render_domain_randomized_pla_frame(rng, fil, fouling_mode=fmode, include_glare=glare, exposure=exp, wb_cast=cast)
        if make_portrait:
            # Embed 640x400 cartridge inside a 640x854 (3:4 portrait) phone frame with dark booth background above/below
            port = np.full((854, 640, 3), (38, 42, 48), dtype=np.uint8)
            port[227 : 227 + 400, :] = bgr
            bgr = port

        res = inspect_bgr_frame(bgr, "fill_strainer", source=f"iphone11_test:{name}", use_calibration=False)
        if make_portrait:
            assert res["registration_mode"] == "aspect_preserving_crop"
        assert res["advisory_only"] is True
        assert res["ml_model_sha256"] is not None and len(res["ml_model_sha256"]) == 64

        if res["verdict"] == expected:
            passed += 1
        if expected == "ISSUE_FOULING":
            assert res["fouling_coverage_pct"] > 0.0, f"{name}: expected non-zero fouling_coverage_pct"

    acc = passed / len(cases)
    assert acc >= 0.95, f"Real-world PLA photo accuracy {passed}/{len(cases)} ({acc*100:.1f}%) below 95% floor"


def test_rw_3_aruco_four_corner_homography_under_20deg_tilt() -> None:
    """Stage 1 ArUco 4-corner homography rectifies +/-20 deg tilted photos and accurately separates Clean vs Fouled."""
    clear_bench_calibration()
    rng = np.random.default_rng(20261255)

    clean_white, _ = render_domain_randomized_pla_frame(rng, "white_pla", "clean", include_glare=False, exposure=1.15)
    fouled_blue, _ = render_domain_randomized_pla_frame(rng, "industrial_blue_pla", "mixed_severe", include_glare=False, exposure=1.0)

    for tilt_deg in (20.0, -20.0):
        tilted_clean = _embed_aruco_and_tilt(clean_white, tilt_deg=tilt_deg)
        r_clean = inspect_bgr_frame(tilted_clean, "fill_strainer", source="iphone11_tilted", use_calibration=False)
        assert r_clean["aruco_locked"] is True, f"Failed ArUco lock on clean tilt {tilt_deg}"
        assert r_clean["markers_detected"] == [0, 1, 2, 3]
        assert r_clean["registration_mode"] == "aruco_homography"
        assert r_clean["verdict"] == "OK", f"False alarm on tilted clean white PLA ({tilt_deg} deg): {r_clean['summary']}"

        tilted_fouled = _embed_aruco_and_tilt(fouled_blue, tilt_deg=tilt_deg)
        r_foul = inspect_bgr_frame(tilted_fouled, "fill_strainer", source="iphone11_tilted", use_calibration=False)
        assert r_foul["aruco_locked"] is True
        assert r_foul["markers_detected"] == [0, 1, 2, 3]
        assert r_foul["registration_mode"] == "aruco_homography"
        assert r_foul["verdict"] == "ISSUE_FOULING"
        assert r_foul["fouling_coverage_pct"] > 5.0


def test_rw_4_live_bench_calibration_unpainted_scale_and_foreign_object() -> None:
    """Stage 3 1-Tap Live Bench Calibration achieves 0% FP on Clean Part 07 and catches unpainted scale & foreign objects."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    try:
        c.delete("/api/hardware/calibrate-bench")
        st0 = c.get("/api/hardware/calibration-status").json()
        assert st0["calibration_status"]["calibrated"] is False
        assert st0["vision_ml_card"]["loaded"] is True

        rng = np.random.default_rng(20261288)
        clean_part07, _ = render_domain_randomized_pla_frame(rng, "white_pla", "clean", include_glare=False, exposure=1.10)
        clean_aruco = _embed_aruco_and_tilt(clean_part07, tilt_deg=6.0)

        # 1. Calibrate on user's live Clean White PLA Part 07
        r_cal = c.post(
            "/api/hardware/calibrate-bench",
            json={"image_base64": _encode_jpg_b64(clean_aruco), "mode": "fill_strainer", "label": "Booth Clean White Part 07"},
        )
        assert r_cal.status_code == 200
        cal_body = r_cal.json()
        assert cal_body["calibration_status"]["calibrated"] is True
        assert cal_body["calibration_status"]["aruco_locked"] is True

        # 2. Re-inspect Clean Part 07 under slight booth brightness drift (+10%) -> 0% false positive!
        clean_shifted = np.clip(clean_part07.astype(np.float32) * 1.10, 0, 255).astype(np.uint8)
        r_ok = c.post(
            "/api/hardware/inspect-frame",
            json={"mode": "fill_strainer", "image_base64": _encode_jpg_b64(_embed_aruco_and_tilt(clean_shifted, tilt_deg=5.0)), "source": "iphone11_booth"},
        ).json()["detection"]
        assert r_ok["calibration_used"] is True
        assert r_ok["verdict"] == "OK"
        assert r_ok["fouling_coverage_pct"] == 0.0

        # 3. Unpainted single-color white-on-white PLA relief bridge on Part 08 -> caught!
        unpainted_patch = clean_part07.copy()
        cv2.ellipse(unpainted_patch, (260, 165), (72, 46), 12, 0, 360, (228, 232, 236), -1)
        r_unp = c.post(
            "/api/hardware/inspect-frame",
            json={"mode": "fill_strainer", "image_base64": _encode_jpg_b64(_embed_aruco_and_tilt(unpainted_patch, tilt_deg=6.0)), "source": "iphone11_booth"},
        ).json()["detection"]
        assert r_unp["calibration_used"] is True
        assert r_unp["verdict"] == "ISSUE_FOULING"
        assert r_unp["fouling_coverage_pct"] > 0.0

        # 4. Live foreign-object obstruction (paper/chalk piece stuck onto Clean Part 07 in front of judges) -> caught!
        foreign_obj = clean_part07.copy()
        cv2.rectangle(foreign_obj, (340, 110), (470, 225), (242, 238, 220), -1)
        cv2.rectangle(foreign_obj, (340, 110), (470, 225), (180, 175, 160), 2)
        r_for = c.post(
            "/api/hardware/inspect-frame",
            json={"mode": "fill_strainer", "image_base64": _encode_jpg_b64(_embed_aruco_and_tilt(foreign_obj, tilt_deg=6.0)), "source": "iphone11_booth"},
        ).json()["detection"]
        assert r_for["calibration_used"] is True
        assert r_for["verdict"] == "ISSUE_FOULING"
        assert r_for["fouling_coverage_pct"] > 0.0
    finally:
        # Always reset calibration cleanly so subsequent tests start uncalibrated
        r_del = c.delete("/api/hardware/calibrate-bench")
        assert r_del.status_code == 200
        assert r_del.json()["calibration_status"]["calibrated"] is False


def test_rw_5_live_pcm_fan_audio_fft_endpoint() -> None:
    """POST /api/hardware/inspect-audio runs real 16 kHz Hann FFT for 7-blade BPF RPM and mechanical faults."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    sr = 16000
    t = np.arange(8000, dtype=np.float64) / float(sr)
    rng = np.random.default_rng(4242)

    # 1. Healthy 7-blade 120mm fan @ 1,500 RPM -> BPF = 175.0 Hz
    healthy_pcm = (0.35 * np.sin(2.0 * np.pi * 175.0 * t) + rng.normal(0.0, 0.02, t.size)).tolist()
    r_ok = c.post("/api/hardware/inspect-audio", json={"pcm_samples": healthy_pcm, "sample_rate": sr, "source": "iphone11_mic"})
    assert r_ok.status_code == 200
    a_ok = r_ok.json()["audio_analysis"]
    assert a_ok["verdict"] == "OK"
    assert abs(a_ok["bpf_derived_rpm"] - 1500.0) <= 35.0
    assert len(a_ok["spectrum_80"]) == 80
    assert a_ok["advisory_only"] is True

    # 2. Mechanical bearing defect (290 Hz + 365 Hz harsh harmonics alongside 140 Hz BPF)
    bearing_pcm = (
        0.22 * np.sin(2.0 * np.pi * 140.0 * t)
        + 0.40 * np.sin(2.0 * np.pi * 290.0 * t)
        + 0.34 * np.sin(2.0 * np.pi * 365.0 * t)
        + rng.normal(0.0, 0.025, t.size)
    ).tolist()
    r_brg = c.post("/api/hardware/inspect-audio", json={"pcm_samples": bearing_pcm, "sample_rate": sr, "source": "iphone11_mic"})
    assert r_brg.status_code == 200
    a_brg = r_brg.json()["audio_analysis"]
    assert a_brg["verdict"] == "ISSUE_MECHANICAL"
    assert a_brg["detail_verdict"] == "ISSUE_BEARING"

    # 3. Mechanical belt squeal / fan rub (2.9 kHz + 3.4 kHz)
    squeal_pcm = (
        0.20 * np.sin(2.0 * np.pi * 140.0 * t)
        + 0.38 * np.sin(2.0 * np.pi * 2920.0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 3420.0 * t)
        + rng.normal(0.0, 0.025, t.size)
    ).tolist()
    r_sq = c.post("/api/hardware/inspect-audio", json={"pcm_samples": squeal_pcm, "sample_rate": sr, "source": "iphone11_mic"})
    assert r_sq.status_code == 200
    assert r_sq.json()["audio_analysis"]["verdict"] == "ISSUE_MECHANICAL"
    assert r_sq.json()["audio_analysis"]["detail_verdict"] == "ISSUE_BELT_SLIP"

    # 4. Stalled/silent fan -> ISSUE_MECHANICAL / ISSUE_STALL
    r_st = c.post("/api/hardware/inspect-audio", json={"pcm_samples": [0.0] * 1024, "sample_rate": sr})
    assert r_st.status_code == 200
    assert r_st.json()["audio_analysis"]["verdict"] == "ISSUE_MECHANICAL"
    assert r_st.json()["audio_analysis"]["detail_verdict"] == "ISSUE_STALL"

    # 5. Direct unit check of analyze_pcm_fan_audio invalid inputs
    assert c.post("/api/hardware/inspect-audio", json={"pcm_samples": [0.1] * 10, "sample_rate": sr}).status_code == 422
    assert c.post("/api/hardware/inspect-audio", json={"pcm_samples": [0.1] * 512, "sample_rate": 100}).status_code == 422
    try:
        analyze_pcm_fan_audio([float("nan")] * 512, 16000)
        assert False, "Expected ValueError on NaN PCM samples"
    except ValueError:
        pass


def test_rw_6_v12_hardware_status_portrait_other_modes_and_aruco_sheet() -> None:
    """Verify v12 /api/hardware/status metadata, portrait spray/basin aspect-preserving framing, and /api/hardware/aruco-sheet."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    hw = c.get("/api/hardware/status").json()
    assert hw["version"] == "12.0.0"
    assert "v12.0" in hw["cad_version"]
    assert hw["cad_summary"]["parts_count"] == 15
    assert hw["vision_ml_card"]["loaded"] is True
    assert hw["vision_ml_card"]["cv_roc_auc"] >= 0.97
    assert "calibration_status" in hw

    # Portrait 3:4 spray_flow & basin_water preserve aspect ratio
    for mode, gen_fn in (("spray_flow", lambda: generate_spray_flow_frame("healthy")), ("basin_water", lambda: generate_basin_water_frame("clear_coc4"))):
        base = gen_fn()
        port = np.full((854, 640, 3), (38, 42, 48), dtype=np.uint8)
        port[227 : 227 + 400, :] = base
        res = inspect_bgr_frame(port, mode, source="iphone11_portrait")
        assert res["registration_mode"] == "aspect_preserving_crop"
        assert res["verdict"] == "OK"

    # ArUco sheet endpoint: invalid format -> 422; png/pdf -> 200 (if generated) or clean 404 (never 500)
    assert c.get("/api/hardware/aruco-sheet?format=svg").status_code == 422
    for fmt, magic in (("png", b"\x89PNG"), ("pdf", b"%PDF")):
        resp = c.get(f"/api/hardware/aruco-sheet?format={fmt}")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            assert resp.content[:4] == magic


def test_rw_7_zero_feature_skew_latest_frame_calibration_and_fuzz() -> None:
    """Verify 0.00 feature extraction determinism, latest-frame calibration fallback, and 422 fuzzing."""
    rng = np.random.default_rng(1001)
    bgr, _ = render_domain_randomized_pla_frame(rng, "light_gray_pla", "calcium_scale")
    wb, _ = apply_bounded_white_balance(bgr)
    x1, m1 = extract_patch_features(wb)
    x2, m2 = extract_patch_features(wb)
    assert x1.shape == (30, 14)
    assert np.all(np.isfinite(x1))
    assert np.array_equal(x1, x2)
    assert m1 == m2

    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    try:
        # Calibrate using fallback / latest frame without supplying image_base64
        r_fb = c.post("/api/hardware/calibrate-bench", json={"mode": "fill_strainer", "label": "Latest Frame Ref"})
        assert r_fb.status_code == 200
        assert r_fb.json()["calibration_status"]["calibrated"] is True
        assert r_fb.json()["calibration_status"]["label"] == "Latest Frame Ref"

        # Invalid calibration inputs -> 422
        assert c.post("/api/hardware/calibrate-bench", json={"mode": "bogus_mode"}).status_code == 422
        assert c.post("/api/hardware/calibrate-bench", json={"mode": "fill_strainer", "image_base64": "!!!bad!!!"}).status_code == 422
    finally:
        c.delete("/api/hardware/calibrate-bench")

