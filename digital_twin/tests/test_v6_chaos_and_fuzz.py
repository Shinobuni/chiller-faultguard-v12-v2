"""Adversarial Chaos, Fuzzing, Concurrency & Whole-Repo Red-Line Audit Suite for Chiller FaultGuard v11.0."""

from __future__ import annotations

import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DT_DIR.parent
ASSETS_DIR = (DT_DIR / "assets") if (DT_DIR / "assets").exists() else (REPO_ROOT / "assets")
CAD_DIR = (DT_DIR / "hardware_cad_v12") if (DT_DIR / "hardware_cad_v12").exists() else (REPO_ROOT / "hardware_cad_v12")
if str(DT_DIR) not in sys.path:
    sys.path.insert(0, str(DT_DIR))

from backend.main import create_app  # noqa: E402
from backend.runtime import ChillerFaultGuardV6Runtime  # noqa: E402

EXPECTED_MODEL_SHA256 = "6eeda35fadcb27d9d617a9fb64805763607b5da58de149b09a015b60141e4a0a"


def test_chaos_1_frozen_model_sha256_immutable() -> None:
    """Quan's frozen HistGradientBoostingClassifier artifact MUST remain byte-for-byte identical."""
    model_path = DT_DIR / "backend" / "models_ml" / "quan_hgb_model.joblib"
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert actual == EXPECTED_MODEL_SHA256, f"Frozen model tampered! Got {actual}"


def test_chaos_2_endpoint_fuzzing_never_returns_500() -> None:
    """Fuzzing every endpoint with invalid, boundary, oversized, or hostile inputs must return 4xx, never 500."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    bad_requests = [
        ("POST", "/api/step", {"steps": 0}),
        ("POST", "/api/step", {"steps": -5}),
        ("POST", "/api/step", {"steps": 5000}),
        ("POST", "/api/step", {"steps": "not_an_int"}),
        ("POST", "/api/scenario", {"scenario": "../../etc/passwd"}),
        ("POST", "/api/scenario", {"scenario": ""}),
        ("POST", "/api/jump", {"kind": "drop_table"}),
        ("POST", "/api/seek", {"row_index": -1}),
        ("POST", "/api/seek", {"row_index": 1440}),
        ("POST", "/api/seek", {"row_index": 999999}),
        ("POST", "/api/seek", {"row_index": 100, "scenario": "bogus"}),
        ("POST", "/api/inject-fault", {"fault_type": "invalid_stress"}),
        ("POST", "/api/inject-fault", {"fault_type": ""}),
        ("POST", "/api/illustrative", {"imposed_excess_kw": -10}),
        ("POST", "/api/illustrative", {"imposed_excess_kw": 0}),
        ("POST", "/api/illustrative", {"imposed_excess_kw": 9999}),
        ("POST", "/api/illustrative", {"earlier_pm_day": 1}),
        ("POST", "/api/illustrative", {"existing_pm_day": 10}),
        ("POST", "/api/illustrative", {"restored_fraction": 2.5}),
        ("POST", "/api/acknowledge", {"verdict": "AUTO_CONTROL_PLC"}),
        ("POST", "/api/acknowledge", {"verdict": "INCONCLUSIVE", "engineer_name": "A" * 500}),
        ("POST", "/api/acknowledge", {"verdict": "INCONCLUSIVE", "note": "X" * 5000}),
        ("POST", "/api/pm-optimizer", {"pm_cost_vnd": 0}),
        ("POST", "/api/pm-optimizer", {"pm_cost_vnd": 500_000}),
        ("POST", "/api/pm-optimizer", {"pm_cost_vnd": 999_999_999}),
        ("POST", "/api/pm-optimizer", {"pm_cost_vnd": "not_a_number"}),
        ("POST", "/api/illustrative", {"pm_cost_vnd": 100}),
        ("POST", "/api/rca-agent", {"query": "A" * 500}),
        ("POST", "/api/rca-agent", {"preset": "nonexistent_preset"}),
        ("POST", "/api/rca-agent", {"basis_mode": "invalid_basis"}),
        ("POST", "/api/rca-agent", {"lang": "fr"}),
        ("POST", "/api/hardware/simulate", {"preset": "bogus_preset"}),
        ("POST", "/api/hardware/simulate", {"preset": "all_healthy", "vision_mode": "bogus_mode"}),
        ("POST", "/api/hardware/inspect-frame", {"mode": "invalid_mode", "image_base64": "AAAA"}),
        ("POST", "/api/hardware/inspect-frame", {"mode": "fill_strainer", "image_base64": "!!!not_base64!!!"}),
        ("POST", "/api/hardware/inspect-frame", {"mode": "fill_strainer", "image_base64": ""}),
        ("POST", "/api/hardware/weather", {"preset": "mars_storm"}),
        ("POST", "/api/hardware/weather", {"dry_bulb_c": 150.0}),
        ("POST", "/api/hardware/weather", {"rh_pct": 150.0}),
        ("POST", "/api/hardware/calibrate-bench", {"mode": "invalid_mode"}),
        ("POST", "/api/hardware/calibrate-bench", {"mode": "fill_strainer", "image_base64": "!!!corrupt!!!"}),
        ("POST", "/api/hardware/inspect-audio", {"pcm_samples": [0.1] * 10, "sample_rate": 16000}),
        ("POST", "/api/hardware/inspect-audio", {"pcm_samples": [0.1] * 512, "sample_rate": 500}),
        ("GET", "/api/hardware/aruco-sheet?format=exe", None),
    ]
    for method, url, payload in bad_requests:
        resp = c.request(method, url, json=payload) if payload is not None else c.request(method, url)
        assert resp.status_code in (400, 409, 422), f"{method} {url} {payload} returned {resp.status_code}"


def test_chaos_3_boundary_seeks_row0_and_row1439() -> None:
    """Seeking to exact start (00:00) and end (23:59) handles ABSTAIN, at_end, advance and jump safely."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=False))
    c = TestClient(app)

    # Row 0 (00:00) is off-hours ABSTAIN
    s0 = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 0}).json()
    assert s0["row_index"] == 0
    assert s0["status"] == "ABSTAIN"
    assert s0["at_end"] is False
    assert c.post("/api/acknowledge", json={"verdict": "CONFIRMED", "note": "Blocked at row 0"}).status_code == 409

    # Row 1439 (23:59) is at_end
    s_end = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 1439}).json()
    assert s_end["row_index"] == 1439
    assert s_end["at_end"] is True
    assert c.post("/api/step", json={"steps": 10}).json()["row_index"] == 1439
    assert c.post("/api/jump", json={"kind": "alert"}).json()["row_index"] == 1439


def test_chaos_4_multithread_concurrency_under_lock() -> None:
    """Concurrent multi-thread API calls never trigger race conditions, KeyErrors, or HTTP 500."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    def worker(idx: int) -> list[int]:
        codes = []
        for k in range(20):
            op = (idx + k) % 10
            if op == 0:
                r = c.get("/api/state")
            elif op == 1:
                r = c.post("/api/step", json={"steps": 2})
            elif op == 2:
                r = c.post("/api/illustrative", json={"imposed_excess_kw": 3.0 + (k % 3) * 2.0})
            elif op == 3:
                r = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 640 + (k % 10)})
            elif op == 4:
                r = c.post("/api/inject-fault", json={"fault_type": "domain_ood" if k % 2 == 0 else "none"})
            elif op == 5:
                r = c.post("/api/pm-optimizer", json={"pm_cost_vnd": [4_000_000, 8_000_000, 15_000_000][k % 3]})
            elif op == 6:
                r = c.post("/api/rca-agent", json={"preset": "why_alert" if k % 2 == 0 else "field_ai"})
            elif op == 7:
                r = c.get("/api/hardware/status")
            elif op == 8:
                r = c.post("/api/hardware/simulate", json={"preset": "all_healthy" if k % 2 == 0 else "fill_fouling_065"})
            else:
                r = c.get("/api/health")
            codes.append(r.status_code)
        return codes

    with ThreadPoolExecutor(max_workers=6) as pool:
        all_codes = [code for batch in pool.map(worker, range(6)) for code in batch]

    assert all(code in (200, 409) for code in all_codes), f"Unexpected status codes: {set(all_codes)}"
    # Ensure runtime recovers cleanly afterwards
    final = c.post("/api/reset").json()
    assert final["scenario"] == "normal"
    assert final["read_only"] is True
    assert final["controls"] is False


def test_chaos_5_whole_repo_red_line_and_offline_cdn_sweep() -> None:
    """Every served frontend, backend and SVG file is 100% offline (zero CDN/URLs) and zero red-line strings."""
    targets = [
        DT_DIR / "frontend" / "index.html",
        DT_DIR / "frontend" / "styles.css",
        DT_DIR / "frontend" / "app.js",
        DT_DIR / "frontend" / "charts_svg.js",
        DT_DIR / "backend" / "main.py",
        DT_DIR / "backend" / "runtime.py",
        DT_DIR / "backend" / "quan_engine.py",
        DT_DIR / "backend" / "causal_graph.py",
        DT_DIR / "backend" / "tou_optimizer.py",
        DT_DIR / "backend" / "rca_agent.py",
        DT_DIR / "backend" / "vision_features.py",
        DT_DIR / "backend" / "train_vision_ml.py",
        DT_DIR / "backend" / "vision_inspector.py",
        DT_DIR / "backend" / "acoustic_tach_weather.py",
        ASSETS_DIR / "chiller_plant_pid.svg",
        ASSETS_DIR / "system_pipeline_architecture.svg",
    ]
    optional_targets = [
        DT_DIR / "frontend" / "mobile_cam.html",
        CAD_DIR / "assemblies" / "MiniCT1_Interactive_3D_Viewer.html",
        CAD_DIR / "assembly" / "MiniCT1_Interactive_3D_Viewer.html",
    ]
    for opt in optional_targets:
        if opt.exists():
            targets.append(opt)
    forbidden_terms = ("yield protected", "yield first", "yield critical", "generating excess", "chillguardai")
    for path in targets:
        assert path.exists(), f"Missing target file: {path}"
        txt = path.read_text(encoding="utf-8")
        low = txt.lower()
        for bad in forbidden_terms:
            assert bad not in low, f"Forbidden red-line '{bad}' found in {path.name}"
        if path.suffix in (".html", ".css", ".js", ".svg", ".py"):
            cleaned = txt.replace("http://www.w3.org/2000/svg", "")
            assert "http://" not in cleaned and "https://" not in cleaned, f"External URL found in {path.name}"


def test_chaos_6_stage_launcher_and_packaging_clean() -> None:
    """1-click launcher, helper script, DEV_LOOP_LOG, RESEARCH_INSIGHTS and README_V6 exist and compile cleanly."""
    import py_compile

    bat = REPO_ROOT / "run_v6.bat"
    if not bat.exists():
        pytest.skip("Standalone web deployment package (monorepo stage launcher not bundled)")
    launcher_py = REPO_ROOT / "scripts" / "launch_v6_stage.py"
    capture_py = REPO_ROOT / "scripts" / "capture_v6_previews.py"
    for p in (launcher_py, capture_py, bat, REPO_ROOT / "DEV_LOOP_LOG.md", REPO_ROOT / "RESEARCH_INSIGHTS.md", REPO_ROOT / "README_V6.md"):
        assert p.exists(), f"Missing deliverable file: {p}"
    py_compile.compile(str(launcher_py), doraise=True)
    py_compile.compile(str(capture_py), doraise=True)
    assert "launch_v6_stage.py" in bat.read_text(encoding="utf-8")
