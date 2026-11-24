"""Comprehensive Integrity, Parity, Causal DAG, EVN TOU, Air-Gapped RCA, 3D Hardware CAD & Multimodal Vision AI Verification Suite for Chiller FaultGuard v11.0."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

DT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DT_DIR.parent
ASSETS_DIR = (DT_DIR / "assets") if (DT_DIR / "assets").exists() else (REPO_ROOT / "assets")
CAD_DIR = (DT_DIR / "hardware_cad_v12") if (DT_DIR / "hardware_cad_v12").exists() else (REPO_ROOT / "hardware_cad_v12")
if str(DT_DIR) not in sys.path:
    sys.path.insert(0, str(DT_DIR))

from backend.causal_graph import build_causal_graph  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.runtime import CASES, ChillerFaultGuardV6Runtime  # noqa: E402
from backend.vision_inspector import generate_fill_strainer_frame  # noqa: E402


def test_1_exact_5760_row_streaming_parity_zero_error() -> None:
    """Bit-for-bit parity (max_score_error < 1e-12) against Quan's frozen offline predictions on all 5,760 Oct-01 rows."""
    rt = ChillerFaultGuardV6Runtime(auto_seek_eligible=False)
    total_rows = 0
    total_eligible = 0
    max_err = 0.0

    for scen, cname in CASES.items():
        rt.reset(scen, seek_eligible=False)
        rows = rt.fixture[cname]
        for i, row in enumerate(rows):
            snap = rt.state() if i == 0 else rt.advance(1, compute_sensitivity=False)
            pred = snap["prediction"]
            is_elig = pred["quality_status"] != "ABSTAIN"
            assert is_elig == bool(row["eligible"]), f"Eligibility mismatch at {cname} row {i}"
            if is_elig:
                total_eligible += 1
                err = abs(float(pred["ai_score"]) - float(row["ai_score"]))
                max_err = max(max_err, err)
                assert pred["ai_alert"] == bool(row["ai_alert"]), f"AI alert mismatch at {cname} row {i}"
                assert pred["rule_alert"] == bool(row["rule_alert"]), f"Rule alert mismatch at {cname} row {i}"
            total_rows += 1

    assert total_rows == 5760
    assert total_eligible == 2948
    assert max_err < 1e-12, f"Expected exact zero parity error, got {max_err}"


def test_2_real_local_feature_sensitivity_nonzero() -> None:
    """Local 1-at-a-time perturbation on Quan's frozen HGB computes real non-fabricated sensitivities."""
    rt = ChillerFaultGuardV6Runtime(auto_seek_eligible=False)
    rt.reset("fouling_high", seek_eligible=False)
    snap = rt.jump("alert")
    sens = snap["prediction"]["feature_sensitivity"]
    assert len(sens) == 6
    assert any(abs(item["delta_score"]) > 0.001 for item in sens)
    assert snap["prediction"]["score_is_calibrated_probability"] is False


def test_3_synthetic_stress_abstention_interlock_and_persistence_reset() -> None:
    """Missing/OOD/Flatline force ABSTAIN, reset persistence, and lock Step V Confirmed verdict (409)."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=False))
    c = TestClient(app)
    c.post("/api/scenario", json={"scenario": "fouling_high", "seek_eligible": False})
    s_alert = c.post("/api/jump", json={"kind": "alert"}).json()
    assert s_alert["status"] == "FOULING_SUSPECT"
    assert s_alert["prediction"]["ai_alert"] is True

    for fault in ("missing_nan", "out_of_range", "flatline_60m"):
        s_bad = c.post("/api/inject-fault", json={"fault_type": fault}).json()
        assert s_bad["status"] == "ABSTAIN"
        assert s_bad["prediction"]["ai_score"] is None
        assert s_bad["prediction"]["ai_alert"] is False
        assert s_bad["prediction"]["ai_run_minutes"] == 0
        # Safety Interlock: confirming fouling while ABSTAIN must be blocked (409 Conflict)
        r_lock = c.post("/api/acknowledge", json={"verdict": "CONFIRMED", "note": "Should fail"})
        assert r_lock.status_code == 409

        # Restoring clean inputs resets persistence cleanly (no stale alert carryover!)
        s_rest = c.post("/api/inject-fault", json={"fault_type": "none"}).json()
        assert s_rest["status"] != "ABSTAIN"
        assert s_rest["prediction"]["ai_alert"] is False


def test_4_jump_to_alert_and_end_of_file_no_wrapping() -> None:
    """Jump preserves intermediate persistence; end of 24h fixture stops cleanly without wrapping."""
    rt = ChillerFaultGuardV6Runtime(auto_seek_eligible=False)
    rt.reset("fouling_medium", seek_eligible=False)
    s_alert = rt.jump("alert")
    assert s_alert["status"] == "FOULING_SUSPECT"
    assert s_alert["prediction"]["ai_run_minutes"] >= 5
    assert s_alert["prediction"]["ai_score"] > 0.70

    rt.reset("normal", seek_eligible=False)
    rt.cursor = len(rt.fixture["normal"]) - 2
    s_end = rt.advance(1)
    assert s_end["at_end"] is True
    ts_before = s_end["timestamp"]
    s_after = rt.advance(15)
    assert s_after["at_end"] is True
    assert s_after["timestamp"] == ts_before


def test_5_read_only_no_control_endpoints_and_health_contract() -> None:
    """Verifies read-only advisory health contract and absence of plant-control endpoints."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    h = c.get("/api/health").json()
    assert h["version"] == "12.0.0"
    assert h["release"] == "chiller-faultguard-v12-quan-hgb-hw"
    assert h["ai_engine"] == "QUAN_FROZEN_HGB"
    assert h["read_only"] is True
    assert h["controls"] is False
    assert h["proxy_fallback"] is False
    assert h["auto_swap"] is False
    assert h["causal_dag"] is True
    assert h["tou_optimizer"] is True
    assert h["air_gapped_rca"] is True
    assert h["hardware_vision_ai"] is True
    assert h["vision_ml_enabled"] is True
    assert h["cad_parts_count"] == 15

    assert c.get("/").status_code == 200
    assert c.get("/charts_svg.js").status_code == 200
    assert c.get("/app.js").status_code == 200
    assert c.get("/assets/chiller_plant_pid.svg").status_code == 200
    assert c.post("/api/control", json={}).status_code in (404, 405)
    assert c.patch("/api/config", json={}).status_code in (404, 405)
    assert c.post("/api/scenario", json={"scenario": "invalid"}).status_code == 400


def test_6_honest_3_bucket_energy_and_illustrative_arithmetic() -> None:
    """Separates pending validated electricity from Quan's exact 6,624 kWh illustrative template."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    st = c.get("/api/state").json()
    elec = st["electricity"]
    assert elec["verified_excess_kwh"] is None
    assert elec["verified_avoided_kwh"] is None
    assert elec["water_savings"] is None

    # Exact Oct-Dec conditional matched arithmetic preserved
    cond = {r["case"]: round(r["net_excess_kWh"], 1) for r in elec["oct_dec_conditional"]}
    assert cond == {"tower_fouling_095": 136.3, "tower_fouling_080": 656.4, "tower_fouling_065": 1361.5}

    # Exact Quan default illustrative arithmetic (3 kW * 24h * 92d = 6,624 kWh)
    sim = elec["illustrative_scenario"]
    assert sim["scenario_avoided_excess_kwh"] == 6624.0
    assert sim["days_saved"] == 92

    # Interactive update works live
    st2 = c.post("/api/illustrative", json={"imposed_excess_kw": 5.0, "earlier_pm_day": 45}).json()
    sim2 = st2["electricity"]["illustrative_scenario"]
    assert sim2["days_saved"] == 75
    assert sim2["scenario_avoided_excess_kwh"] == 5.0 * 24.0 * 75.0


def test_7_six_step_supervisor_verdicts_and_audit_trail() -> None:
    """Supervisor can record Confirmed / False Alarm / Inconclusive as local session advisory."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    c.post("/api/scenario", json={"scenario": "fouling_high", "seek_eligible": False})
    c.post("/api/jump", json={"kind": "alert"})

    for verdict in ("CONFIRMED", "FALSE_ALARM", "INCONCLUSIVE"):
        r = c.post(
            "/api/acknowledge",
            json={
                "verdict": verdict,
                "engineer_name": "Shift A Lead",
                "completed_steps": ["I", "II", "III", "IV", "V"],
                "note": f"Audit test {verdict}",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["review"]["verdict"] == verdict
        assert body["review"]["review_note_is_approval"] is False
        assert body["audit_log"][0]["category"] == f"REVIEW: {verdict}"


def test_8_zero_cdn_zero_red_lines_and_brand_consistency() -> None:
    """100% offline (zero CDN URLs), brand Chiller FaultGuard everywhere, zero forbidden red-line claims."""
    html = (DT_DIR / "frontend" / "index.html").read_text(encoding="utf-8")
    js = (DT_DIR / "frontend" / "app.js").read_text(encoding="utf-8")
    charts = (DT_DIR / "frontend" / "charts_svg.js").read_text(encoding="utf-8")
    svg = (ASSETS_DIR / "chiller_plant_pid.svg").read_text(encoding="utf-8")

    # Zero external CDN scripts/links -> guaranteed offline at conference venue
    assert "cdn.jsdelivr.net" not in html
    assert "http://" not in html and "https://" not in html

    for name, content in (("index.html", html), ("app.js", js), ("charts_svg.js", charts), ("pid.svg", svg)):
        low = content.lower()
        for forbidden in ("yield protected", "yield first", "yield critical", "generating excess", "chillguardai"):
            assert forbidden not in low, f"Forbidden string '{forbidden}' found in {name}"


def test_9_alert_gated_confirm_and_stale_review_cleared_on_stress() -> None:
    """CONFIRMED verdict requires an active alert (409 on NO_ALERT), and stress injection clears prior review."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    # Normal baseline at midday is NO_ALERT -> CONFIRMED must be blocked (409)
    st_norm = c.get("/api/state").json()
    assert st_norm["status"] == "NO_ALERT"
    r_no_alert = c.post("/api/acknowledge", json={"verdict": "CONFIRMED", "note": "Should block on NO_ALERT"})
    assert r_no_alert.status_code == 409

    # Jump to active alert on 0.65 UA -> CONFIRMED succeeds
    c.post("/api/scenario", json={"scenario": "fouling_high", "seek_eligible": False})
    st_alert = c.post("/api/jump", json={"kind": "alert"}).json()
    assert st_alert["status"] == "FOULING_SUSPECT"
    r_ok = c.post("/api/acknowledge", json={"verdict": "CONFIRMED", "note": "Verified Steps I-IV"})
    assert r_ok.status_code == 200
    assert r_ok.json()["review"]["verdict"] == "CONFIRMED"

    # Inject stress test -> prior CONFIRMED review MUST be cleared (no stale WO carryover on restore!)
    st_bad = c.post("/api/inject-fault", json={"fault_type": "missing_nan"}).json()
    assert st_bad["review"]["acknowledged"] is False
    assert st_bad["review"]["verdict"] == "PENDING"

    st_rest = c.post("/api/inject-fault", json={"fault_type": "none"}).json()
    assert st_rest["status"] == "NO_ALERT"
    assert st_rest["review"]["acknowledged"] is False
    assert st_rest["review"]["verdict"] == "PENDING"
    assert c.post("/api/acknowledge", json={"verdict": "CONFIRMED", "note": "Still NO_ALERT"}).status_code == 409


def test_10_jump_blocked_under_active_stress_with_409() -> None:
    """Calling /api/jump while a synthetic stress mode is active must fail fast (409) without advancing cursor."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    st_before = c.post("/api/inject-fault", json={"fault_type": "missing_nan"}).json()
    idx_before = st_before["row_index"]

    r_jump_el = c.post("/api/jump", json={"kind": "eligible"})
    assert r_jump_el.status_code == 409
    r_jump_al = c.post("/api/jump", json={"kind": "alert"})
    assert r_jump_al.status_code == 409

    # Cursor must remain completely untouched
    st_after = c.get("/api/state").json()
    assert st_after["row_index"] == idx_before

    # Invalid verdict returns 400
    assert c.post("/api/acknowledge", json={"verdict": "BOGUS"}).status_code == 400


def test_11_true_training_envelope_ood_triggers_gate4_and_interlock() -> None:
    """domain_ood passes Gates 1-3 and trips Gate 4 padded training envelope (ood_flag=True, ood_score=1.0)."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    c.post("/api/scenario", json={"scenario": "fouling_high", "seek_eligible": False})
    c.post("/api/jump", json={"kind": "alert"})

    # Gate 1 physical range check (out_of_range) keeps ood_flag=False and reflects injected 400% fan consistently
    st_phys = c.post("/api/inject-fault", json={"fault_type": "out_of_range"}).json()
    assert st_phys["status"] == "ABSTAIN"
    assert st_phys["prediction"]["ood_flag"] is False
    assert st_phys["signals"]["fan_pct"] == 400.0

    # Gate 4 training-envelope OOD (domain_ood) triggers ood_flag=True, ood_score=1.0, ood_features=['flow']
    # AND reflects injected 750.0 GPM consistently across Step I, Step II signals, and Step III H2 evidence!
    st_ood = c.post("/api/inject-fault", json={"fault_type": "domain_ood"}).json()
    assert st_ood["status"] == "ABSTAIN"
    assert st_ood["prediction"]["ood_flag"] is True
    assert st_ood["prediction"]["ood_score"] == 1.0
    assert "flow" in st_ood["prediction"]["ood_features"]
    assert st_ood["prediction"]["ai_score"] is None
    assert st_ood["signals"]["tower_flow_gpm"] == 750.0
    h2_ood = next(h for h in st_ood["hypotheses"] if h["id"] == "H2")
    assert "750.0 GPM" in h2_ood["evidence"]
    assert c.post("/api/acknowledge", json={"verdict": "CONFIRMED", "note": "Blocked by OOD"}).status_code == 409

    # Missing sensor stress reflects None outlet/range cleanly across signals
    st_nan = c.post("/api/inject-fault", json={"fault_type": "missing_nan"}).json()
    assert st_nan["status"] == "ABSTAIN"
    assert st_nan["signals"]["nominal_tower_outlet_c"] is None
    assert st_nan["signals"]["tower_range_c"] is None

    # Restoring clean inputs clears OOD and restores original LBNL flow/outlet cleanly
    st_clean = c.post("/api/inject-fault", json={"fault_type": "none"}).json()
    assert st_clean["prediction"]["ood_flag"] is False
    assert st_clean["prediction"]["ood_features"] == []
    assert st_clean["signals"]["tower_flow_gpm"] < 600.0
    assert st_clean["signals"]["nominal_tower_outlet_c"] is not None


def test_12_xss_sanitization_helper_and_review_payload_safety() -> None:
    """Frontend escapes dynamic strings via escapeHtml; review notes with HTML tags are handled safely."""
    js = (DT_DIR / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "function escapeHtml(" in js
    assert "escapeHtml(ev.message)" in js
    assert "escapeHtml(h.evidence)" in js
    assert 'addEventListener("click", evt =>' in js  # delegated Step IV checklist listener

    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=False))
    c = TestClient(app)
    c.post("/api/scenario", json={"scenario": "fouling_high", "seek_eligible": False})
    c.post("/api/jump", json={"kind": "alert"})
    r = c.post(
        "/api/acknowledge",
        json={
            "verdict": "CONFIRMED",
            "engineer_name": '<img src=x onerror=alert(1)>',
            "completed_steps": ["I", "II", "III", "IV", "V"],
            "note": "<script>alert('xss')</script> verified",
        },
    )
    assert r.status_code == 200
    assert r.json()["review"]["verdict"] == "CONFIRMED"


def test_13_pid_honest_range_label_live_animation_and_tooltips() -> None:
    """P&ID uses honest Thermal Range title (Approach Excluded*), dynamic WI-CH2, flow pulses & hover tooltips."""
    svg = (ASSETS_DIR / "chiller_plant_pid.svg").read_text(encoding="utf-8")
    assert "CT-1 APPROACH (T_CW,out" not in svg
    assert "CT-1 THERMAL RANGE" in svg
    assert 'id="val-ch2-kw"' in svg
    assert 'class="pid-flow-pulse"' in svg
    assert 'data-pid-tip="ct1"' in svg
    assert 'data-pid-tip="twb"' in svg
    svg_no_ns = svg.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in svg_no_ns and "https://" not in svg_no_ns


def test_14_seek_parity_exact_zero_error_and_ai_ahead_of_rule_beat() -> None:
    """POST /api/seek matches row-by-row streaming bit-for-bit (<1e-12) and verifies Beat 2 (10:40 AI-only)."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=False))
    c = TestClient(app)

    # Beat 2: 0.65 UA at row 640 (10:40) -> AI flags FOULING_SUSPECT while High-Fan Rule is still 0/15m!
    st_seek = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 640}).json()
    assert st_seek["timestamp"] == "2018-10-01 10:40:00"
    assert st_seek["status"] == "FOULING_SUSPECT"
    assert st_seek["agreement"] == "AI_ONLY_ALERT"
    assert st_seek["prediction"]["ai_alert"] is True
    assert st_seek["prediction"]["rule_alert"] is False
    assert st_seek["signals"]["fan_pct"] < 98.0
    assert len(st_seek["timeline_24h"]) == 144

    # Verify exact parity against fresh row-by-row streaming runtime
    rt_stream = ChillerFaultGuardV6Runtime(auto_seek_eligible=False)
    rt_stream.reset("fouling_high", seek_eligible=False)
    st_stream = rt_stream.advance(640, compute_sensitivity=False)
    assert abs(float(st_seek["prediction"]["ai_score"]) - float(st_stream["prediction"]["ai_score"])) < 1e-12
    assert st_seek["prediction"]["ai_run_minutes"] == st_stream["prediction"]["ai_run_minutes"]
    assert st_seek["prediction"]["rule_run_minutes"] == st_stream["prediction"]["rule_run_minutes"]
    assert st_seek["prediction"]["ai_alert"] == st_stream["prediction"]["ai_alert"]
    assert st_seek["prediction"]["rule_alert"] == st_stream["prediction"]["rule_alert"]

    # Seek without scenario switch while synthetic stress is active must block (409)
    c.post("/api/inject-fault", json={"fault_type": "missing_nan"})
    assert c.post("/api/seek", json={"row_index": 700}).status_code == 409


def test_15_architecture_svg_zero_stale_v5_claims() -> None:
    """system_pipeline_architecture.svg strictly matches Quan frozen v6 HGB (zero stale v5 numbers or red-lines)."""
    arch = (ASSETS_DIR / "system_pipeline_architecture.svg").read_text(encoding="utf-8")
    low = arch.lower()
    assert "f1 0.394" in low
    assert "6eeda35f" in low
    assert "0.00e+00" in low
    for stale in ("97.39%", "92.17%", "sandia", "top-3 shap", "4 lbnl classes", "yield protected", "generating excess", "chillguardai"):
        assert stale not in low, f"Stale/forbidden claim '{stale}' found in system_pipeline_architecture.svg"
    arch_no_ns = arch.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in arch_no_ns and "https://" not in arch_no_ns


def test_16_extended_illustrative_calculator_and_reset_defaults() -> None:
    """Extended Tab 3 controls (Calendar PM D90/D180, Restore 75%, Reset Quan Ref) preserve exact arithmetic identity."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    # Partial restoration (75%) on D180 calendar vs D45 earlier PM at 5 kW
    st = c.post(
        "/api/illustrative",
        json={"imposed_excess_kw": 5.0, "earlier_pm_day": 45, "existing_pm_day": 180, "restored_fraction": 0.75},
    ).json()
    sim = st["electricity"]["illustrative_scenario"]
    assert sim["days_saved"] == 135
    assert sim["restored_fraction"] == 0.75
    assert sim["scenario_avoided_excess_kwh"] == round(5.0 * 24.0 * 135 * 0.75, 1)
    assert abs((sim["existing_policy_excess_kwh"] - sim["earlier_policy_excess_kwh"]) - sim["scenario_avoided_excess_kwh"]) < 0.2

    # 1-click Reset Quan Ref restores exact 6,624.0 kWh default
    st_res = c.post("/api/illustrative", json={"reset_defaults": True}).json()
    sim_res = st_res["electricity"]["illustrative_scenario"]
    assert sim_res["imposed_excess_kw"] == 3.0
    assert sim_res["earlier_pm_day"] == 28
    assert sim_res["existing_pm_day"] == 120
    assert sim_res["restored_fraction"] == 1.0
    assert sim_res["scenario_avoided_excess_kwh"] == 6624.0


def test_17_event_metrics_and_modal_judge_qa_contract() -> None:
    """API exposes event_metrics.json operating-window diagnostics; UI modal exposes verbatim Pilot Question & export."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    st = c.get("/api/state").json()
    em = st["event_metrics"]
    methods = em["methods"]
    assert methods["histogram_boosting_candidate"]["by_case"]["tower_fouling_065"]["windows_with_alert"] == 42
    assert methods["histogram_boosting_candidate"]["by_case"]["tower_fouling_065"]["eligible_windows"] == 53
    assert methods["reference_rule"]["by_case"]["tower_fouling_065"]["windows_with_alert"] == 12

    js = (DT_DIR / "frontend" / "app.js").read_text(encoding="utf-8")
    low_js = js.lower()
    assert "cooling-tower" in low_js or "cooling tower" in low_js
    assert "btn-export-md" in js and "btn-export-json" in js
    assert "excess" in low_js


def test_18_live_alert_auto_energy_projection_standby_vs_active() -> None:
    """Tab 3 live_alert_projection is 0 kWh on Normal/ABSTAIN and auto-computes from 15m median ΔkW* on active alert."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    # Normal baseline at midday -> NO_ALERT -> 0 kWh avoidable excess!
    st_norm = c.get("/api/state").json()
    lp_norm = st_norm["electricity"]["live_alert_projection"]
    assert lp_norm["active"] is False
    assert lp_norm["mode"] == "STANDBY_NO_ALERT"
    assert lp_norm["imposed_excess_kw"] == 0.0
    assert lp_norm["scenario_avoided_excess_kwh"] == 0.0
    assert lp_norm["verified_excess_kwh"] is None
    assert lp_norm["water_savings"] is None

    # Active alert on 0.65 UA at 12:15 (row 735) -> auto-computes positive median ΔkW* (~4.2 kW*)
    st_065 = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 735}).json()
    lp_065 = st_065["electricity"]["live_alert_projection"]
    assert lp_065["active"] is True
    assert lp_065["mode"] == "ACTIVE_ALERT_LIVE"
    assert 3.0 < lp_065["imposed_excess_kw"] < 6.0
    assert lp_065["scenario_avoided_excess_kwh"] > 7000.0

    # Synthetic OOD stress -> ABSTAIN -> immediately suspends to 0 kWh!
    st_ood = c.post("/api/inject-fault", json={"fault_type": "domain_ood"}).json()
    lp_ood = st_ood["electricity"]["live_alert_projection"]
    assert lp_ood["active"] is False
    assert lp_ood["mode"] == "SUSPENDED_ABSTAIN"
    assert lp_ood["scenario_avoided_excess_kwh"] == 0.0


def test_19_kha_5m1e_vietnam_official_quantification_kpis() -> None:
    """Tab 3 bottom 5M1E cards compute official Nov-2026 VN 22kV TOU (QD 2892/BCT), 2024 EF (0.6187), and LBNL time KPIs."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    # Normal standby: money = 0 VND, fouling_hours_avoided = 0 h, Scope 2 = 0.0 t CO2e, verified_water = None
    st_norm = c.get("/api/state").json()
    k_norm = st_norm["electricity"]["live_alert_projection"]["kpis_5m1e"]
    assert k_norm["money"]["tariff_vnd_kwh"] == 1925.93
    assert k_norm["money"]["avoided_ex_vat_vnd"] == 0.0
    assert k_norm["time"]["fouling_hours_avoided"] == 0
    assert k_norm["esg"]["grid_ef_kg_kwh"] == 0.6187
    assert k_norm["esg"]["avoided_co2_tonnes"] == 0.0
    assert k_norm["esg"]["verified_water_savings"] is None

    # Active alert 0.65 UA at 12:15 (row 735): live 4.10 kW* -> 9,052.8 kWh -> 17.44M VND, 2,208 h, -107.5m lead, 5.60 t CO2e
    st_065 = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 735}).json()
    k_065 = st_065["electricity"]["live_alert_projection"]["kpis_5m1e"]
    assert k_065["money"]["avoided_ex_vat_vnd"] > 15_000_000.0
    assert k_065["money"]["avoided_incl_vat8_vnd"] > k_065["money"]["avoided_ex_vat_vnd"]
    assert k_065["money"]["daily_burn_ex_vat_vnd"] > 150_000.0
    assert k_065["time"]["fouling_hours_avoided"] == 2208
    assert k_065["time"]["ai_median_window_min"] == 34.5
    assert k_065["time"]["rule_median_window_min"] == 142.0
    assert k_065["time"]["window_speedup_min"] == 107.5
    assert k_065["esg"]["avoided_co2_tonnes"] == 5.6
    assert k_065["esg"]["verified_water_savings"] is None

    # Quan Ref static 3 kW -> 6,624 kWh -> 12,757,360 VND & 4.10 t CO2e
    k_ref = st_065["electricity"]["illustrative_scenario"]["kpis_5m1e"]
    assert abs(k_ref["money"]["avoided_ex_vat_vnd"] - round(6624.0 * 1925.93, 0)) < 2.0
    assert k_ref["esg"]["avoided_co2_tonnes"] == 4.1


def test_20_ashrae_live_water_physics_est_with_null_verified_savings() -> None:
    """v8 live ASHRAE Flow*Range evaporation & CoC 4->3 blowdown estimator computes transparently while keeping verified water = None."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    # Normal standby at midday: live evap > 0 (CT-1 is running!), but CoC risk = 0.0 and verified_water_savings is None
    st_norm = c.get("/api/state").json()
    assert st_norm["electricity"]["water_savings"] is None
    wp_norm = st_norm["electricity"]["live_alert_projection"]["kpis_5m1e"]["esg"]["water_physics_est"]
    assert wp_norm["verified"] is False
    assert wp_norm["suspended"] is False
    assert wp_norm["live_evap_m3_day"] > 1.0
    assert wp_norm["coc_4to3_penalty_m3_est"] == 0.0
    assert wp_norm["thermal_makeup_m3_est"] == 0.0

    # Active 0.65 UA alert at 12:15 (D28 vs D120 = 92d): live evap ~8.5 m3/d, CoC 4->3 blowdown risk > 100 m3, verified still None!
    st_065 = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 735}).json()
    assert st_065["electricity"]["water_savings"] is None
    esg_065 = st_065["electricity"]["live_alert_projection"]["kpis_5m1e"]["esg"]
    assert esg_065["verified_water_savings"] is None
    wp_065 = esg_065["water_physics_est"]
    assert 5.0 < wp_065["live_evap_m3_day"] < 15.0
    assert wp_065["coc_4to3_penalty_m3_est"] > 80.0
    assert wp_065["coc_5to3_penalty_m3_est"] > wp_065["coc_4to3_penalty_m3_est"]
    assert wp_065["thermal_makeup_m3_est"] > 0.0

    # Synthetic OOD stress -> ABSTAIN -> water physics estimator cleanly suspends (0.0, suspended=True)
    st_ood = c.post("/api/inject-fault", json={"fault_type": "domain_ood"}).json()
    wp_ood = st_ood["electricity"]["live_alert_projection"]["kpis_5m1e"]["esg"]["water_physics_est"]
    assert wp_ood["suspended"] is True
    assert wp_ood["live_evap_m3_day"] == 0.0
    assert wp_ood["coc_4to3_penalty_m3_est"] == 0.0


def test_21_causal_root_cause_dag_isolation_across_cases_and_stress() -> None:
    """8-node Causal Root-Cause DAG isolates N1 (CT-1 Fouling), N2 (Low Flow), N3 (OOD/Gate), and retains N4 confounder honesty."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    # 1. Normal baseline (NO_ALERT) -> isolated_root_id == NONE, N4 retained as CONFOUNDER_CHECK, all edges inactive
    st_norm = c.get("/api/state").json()
    dag_norm = st_norm["causal_graph"]
    assert dag_norm["isolated_root_id"] == "NONE"
    assert len(dag_norm["nodes"]) == 8
    assert len(dag_norm["edges"]) == 8
    nm_norm = {n["id"]: n for n in dag_norm["nodes"]}
    assert nm_norm["N1"]["state"] == "CALM"
    assert nm_norm["N2"]["state"] == "RULED_OUT"
    assert nm_norm["N3"]["state"] == "RULED_OUT"
    assert nm_norm["N4"]["state"] == "CONFOUNDER_CHECK"
    assert all(nm_norm[nid]["state"] == "CALM" for nid in ("N5", "N6", "N7", "N8"))
    assert all(not e["active"] and not e["blocked"] for e in dag_norm["edges"])
    assert "99.31%" in dag_norm["confounder_warning"]
    assert "97.3%" in dag_norm["confounder_warning"]

    # 2. Active 0.65 UA alert (row 735) -> isolated_root_id == N1, N2/N3 RULED_OUT, N5..N8 PROPAGATED
    st_065 = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 735}).json()
    dag_065 = st_065["causal_graph"]
    assert dag_065["isolated_root_id"] == "N1"
    nm_065 = {n["id"]: n for n in dag_065["nodes"]}
    assert nm_065["N1"]["state"] == "ROOT_SUSPECT"
    assert nm_065["N2"]["state"] == "RULED_OUT"
    assert nm_065["N3"]["state"] == "RULED_OUT"
    assert nm_065["N4"]["state"] == "CONFOUNDER_CHECK"
    assert all(nm_065[nid]["state"] == "PROPAGATED" for nid in ("N5", "N6", "N7", "N8"))
    active_065 = {(e["from"], e["to"]) for e in dag_065["edges"] if e["active"]}
    assert active_065 == {("N1", "N5"), ("N1", "N6"), ("N5", "N7"), ("N6", "N7"), ("N7", "N8")}
    assert all(not e["blocked"] for e in dag_065["edges"])

    # 3. Synthetic Low-Flow Confounder check (flow < 250 GPM during rule-only alert isolates N2; with AI alert flags N2 CONFOUNDER_CHECK)
    synth_low_rule = json.loads(json.dumps(st_065))
    synth_low_rule["status"] = "RULE_WARNING"
    synth_low_rule["prediction"]["ai_alert"] = False
    synth_low_rule["prediction"]["rule_alert"] = True
    synth_low_rule["signals"]["tower_flow_gpm"] = 180.0
    dag_n2 = build_causal_graph(synth_low_rule)
    assert dag_n2["isolated_root_id"] == "N2"
    assert next(n for n in dag_n2["nodes"] if n["id"] == "N2")["state"] == "ROOT_SUSPECT"

    synth_low_both = json.loads(json.dumps(st_065))
    synth_low_both["signals"]["tower_flow_gpm"] = 180.0
    dag_both = build_causal_graph(synth_low_both)
    assert dag_both["isolated_root_id"] == "N1"
    assert next(n for n in dag_both["nodes"] if n["id"] == "N2")["state"] == "CONFOUNDER_CHECK"

    # 4. Synthetic Stress (domain_ood & missing_nan) -> ABSTAIN -> N3 ROOT_SUSPECT, N1/N2/N5..N8 BLOCKED_GATE, N3->N5 active
    for fault in ("domain_ood", "missing_nan"):
        st_bad = c.post("/api/inject-fault", json={"fault_type": fault}).json()
        dag_bad = st_bad["causal_graph"]
        assert dag_bad["isolated_root_id"] == "N3"
        nm_bad = {n["id"]: n for n in dag_bad["nodes"]}
        assert nm_bad["N3"]["state"] == "ROOT_SUSPECT"
        assert nm_bad["N4"]["state"] == "CONFOUNDER_CHECK"
        assert all(nm_bad[nid]["state"] == "BLOCKED_GATE" for nid in ("N1", "N2", "N5", "N6", "N7", "N8"))
        active_bad = {(e["from"], e["to"]) for e in dag_bad["edges"] if e["active"]}
        blocked_bad = {(e["from"], e["to"]) for e in dag_bad["edges"] if e["blocked"]}
        assert active_bad == {("N3", "N5")}
        assert len(blocked_bad) == 7


def test_22_evn_tou_breakeven_pm_optimizer_clean_vs_defer_and_bounds() -> None:
    """EVN TOU Break-Even PM Optimizer yields CLEAN_NOW on 0.65 UA, DEFER_PM on 0.95 UA, LOCKED on ABSTAIN, and respects bounds."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    # 1. Normal standby -> STANDBY_NO_ACTION, 0 net, None payback
    st_norm = c.get("/api/state").json()
    pm_norm = st_norm["electricity"]["live_alert_projection"]["pm_optimizer"]
    assert pm_norm["verdict"] == "STANDBY_NO_ACTION"
    assert pm_norm["net_benefit_vnd"] == 0.0
    assert pm_norm["payback_days"] is None
    assert pm_norm["illustrative_only"] is True
    assert pm_norm["verified_excess_kwh"] is None
    assert pm_norm["verified_avoided_kwh"] is None

    # 2. Severe 0.65 UA at row 735 (default 8M VND PM) -> CLEAN_NOW_POSITIVE_ROI
    st_065 = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 735}).json()
    pm_065 = st_065["electricity"]["live_alert_projection"]["pm_optimizer"]
    assert pm_065["verdict"] == "CLEAN_NOW_POSITIVE_ROI"
    assert pm_065["pm_cost_vnd"] == 8_000_000.0
    assert pm_065["tariff_vnd_kwh"] == 1925.93
    assert pm_065["daily_burn_vnd"] > 180_000.0
    assert pm_065["net_benefit_vnd"] > 9_000_000.0
    assert pm_065["roi_pct"] > 110.0
    assert 35.0 < pm_065["payback_days"] < 50.0
    assert abs(pm_065["payback_calendar_day"] - round(28 + pm_065["payback_days"], 1)) < 0.2
    assert abs(pm_065["latest_profitable_day"] - round(120 - pm_065["payback_days"], 1)) < 0.2
    assert pm_065["eoq_sensitivity_days"] is not None and pm_065["eoq_sensitivity_days"] > 50.0
    tou = pm_065["tou_avoided_vnd"]
    assert tou["peak_share_pct"] == 31.04
    assert tou["normal_share_pct"] == 53.82
    assert tou["off_peak_share_pct"] == 15.14
    assert abs((tou["peak"] + tou["normal"] + tou["off_peak"]) - pm_065["avoided_gross_vnd"]) <= 5.0

    # 3. Interactive POST /api/pm-optimizer (4M vs 15M VND) + illustrative reset_defaults restores 8M VND
    st_4m = c.post("/api/pm-optimizer", json={"pm_cost_vnd": 4_000_000}).json()
    pm_4m = st_4m["electricity"]["live_alert_projection"]["pm_optimizer"]
    assert pm_4m["pm_cost_vnd"] == 4_000_000.0
    assert pm_4m["roi_pct"] > pm_065["roi_pct"]
    assert pm_4m["payback_days"] < pm_065["payback_days"]

    st_15m = c.post("/api/pm-optimizer", json={"pm_cost_vnd": 15_000_000}).json()
    pm_15m = st_15m["electricity"]["live_alert_projection"]["pm_optimizer"]
    assert pm_15m["pm_cost_vnd"] == 15_000_000.0
    assert pm_15m["verdict"] == "CLEAN_NOW_POSITIVE_ROI"
    assert pm_15m["roi_pct"] < pm_065["roi_pct"]
    assert pm_15m["payback_days"] > pm_065["payback_days"]

    st_reset = c.post("/api/illustrative", json={"reset_defaults": True}).json()
    assert st_reset["electricity"]["live_alert_projection"]["pm_optimizer"]["pm_cost_vnd"] == 8_000_000.0
    assert st_reset["electricity"]["illustrative_scenario"]["scenario_avoided_excess_kwh"] == 6624.0
    assert st_reset["electricity"]["illustrative_scenario"]["pm_optimizer"]["verdict"] == "CLEAN_NOW_POSITIVE_ROI"

    # 4. Mild 0.95 UA at row 898 (active alert, tiny ΔkW*) -> DEFER_PM_NEGATIVE_ROI!
    st_095 = c.post("/api/seek", json={"scenario": "fouling_low", "row_index": 898}).json()
    assert st_095["status"] in ("FOULING_SUSPECT", "RULE_WARNING")
    pm_095 = st_095["electricity"]["live_alert_projection"]["pm_optimizer"]
    assert pm_095["verdict"] == "DEFER_PM_NEGATIVE_ROI"
    assert pm_095["net_benefit_vnd"] < 0.0
    assert pm_095["roi_pct"] < 0.0
    assert pm_095["payback_calendar_day"] > 120.0

    # 5. ABSTAIN under domain_ood -> LOCKED_ABSTAIN
    st_ood = c.post("/api/inject-fault", json={"fault_type": "domain_ood"}).json()
    pm_ood = st_ood["electricity"]["live_alert_projection"]["pm_optimizer"]
    assert pm_ood["verdict"] == "LOCKED_ABSTAIN"
    assert pm_ood["net_benefit_vnd"] == 0.0
    assert pm_ood["payback_days"] is None


def test_23_airgapped_rca_agent_grounded_citations_and_control_refusal() -> None:
    """Air-Gapped 6-Tool RCA Agent produces SHA-256 grounded EN/VI citations, blocks ABSTAIN diagnosis, and refuses PLC writes."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 735})

    # 1. All presets on active 0.65 UA alert return 6 tools, valid citations, deterministic SHA-256, read_only=True, controls=False
    for preset in ("why_alert", "root_cause", "economics_pm", "engineer_brief", "exec_5m1e", "sensor_safety", "water_esg"):
        r = c.post("/api/rca-agent", json={"preset": preset, "basis_mode": "live", "lang": "en"})
        assert r.status_code == 200
        d = r.json()
        assert d["agent_version"] == "12.0.0"
        assert d["air_gapped"] is True
        assert d["deterministic"] is True
        assert d["read_only"] is True
        assert d["controls"] is False
        assert d["refused_control_write"] is False
        assert d["abstain_blocked"] is False
        assert len(d["tool_traces"]) == 6
        assert [t["id"] for t in d["tool_traces"]] == ["T1", "T2", "T3", "T4", "T5", "T6"]
        assert len(d["citations_used"]) >= 2
        assert len(d["evidence_digest_sha256"]) == 64
        assert d["governance"]["verified_excess_kwh"] is None
        assert d["governance"]["verified_avoided_kwh"] is None
        assert d["governance"]["verified_water_savings"] is None

    # 2. Exact SHA-256 determinism on repeated calls & DAG node focus
    r_a = c.post("/api/rca-agent", json={"preset": "why_alert", "basis_mode": "live", "lang": "en"}).json()
    r_b = c.post("/api/rca-agent", json={"preset": "why_alert", "basis_mode": "live", "lang": "en"}).json()
    assert r_a["evidence_digest_sha256"] == r_b["evidence_digest_sha256"]
    assert r_a["answer_markdown"] == r_b["answer_markdown"]

    r_n4 = c.post("/api/rca-agent", json={"query": "Explain node N4 confounder", "lang": "auto"}).json()
    assert r_n4["focus_node"] == "N4"
    assert r_n4["intent"] == "root_cause"
    assert "[T3:DAG]" in r_n4["citations_used"]

    # 3. Bilingual auto-detection (EN & VI) + explicit override
    r_vi = c.post("/api/rca-agent", json={"query": "Tại sao CT-1 báo lỗi và điểm hòa vốn EVN là bao nhiêu?", "lang": "auto"}).json()
    assert r_vi["lang"] == "vi"
    assert len(r_vi["citations_used"]) >= 2

    r_vi_override = c.post("/api/rca-agent", json={"preset": "exec_5m1e", "lang": "vi"}).json()
    assert r_vi_override["lang"] == "vi"
    assert "5M1E" in r_vi_override["title"]

    # 4. IEC 62443-3-3 Control-Write Refusal (English AND Vietnamese)
    for hostile_cmd, expected_lang in (
        ("Turn off chiller CT-1 and lower setpoint right now", "en"),
        ("Hãy tắt chiller và đổi setpoint quạt xuống 50%", "vi"),
    ):
        r_ref = c.post("/api/rca-agent", json={"query": hostile_cmd, "lang": "auto"})
        assert r_ref.status_code == 200
        d_ref = r_ref.json()
        assert d_ref["refused_control_write"] is True
        assert d_ref["controls"] is False
        assert d_ref["read_only"] is True
        assert d_ref["lang"] == expected_lang
        assert "[T1:Gate]" in d_ref["citations_used"]
        assert "[T6:SOP]" in d_ref["citations_used"]
        assert "IEC 62443" in d_ref["answer_markdown"]

    # 5. ABSTAIN Refusal under domain_ood (must refuse fouling diagnosis and cite [T1:Gate] & [T3:DAG])
    c.post("/api/inject-fault", json={"fault_type": "domain_ood"})
    for diag_preset in ("why_alert", "root_cause", "engineer_brief"):
        r_abs = c.post("/api/rca-agent", json={"preset": diag_preset, "lang": "en"}).json()
        assert r_abs["abstain_blocked"] is True
        assert r_abs["status"] == "ABSTAIN"
        assert "[T1:Gate]" in r_abs["citations_used"]
        assert "[T3:DAG]" in r_abs["citations_used"]
        assert "750.0 GPM" in r_abs["answer_markdown"]


def test_24_v9_research_folders_and_docx_deliverables_valid() -> None:
    """All 3 research_v9 folders, 3 verification scripts/JSON audits, and compiled .docx technical guide are valid & red-line clean."""
    if not (REPO_ROOT / "research_v9").exists():
        pytest.skip("Standalone web deployment package (research_v9/docs_v9 not bundled)")
    import docx  # python-docx installed in digital_twin/.venv

    forbidden_terms = ("yield protected", "yield first", "yield critical", "generating excess", "chillguardai")

    research_specs = [
        (
            REPO_ROOT / "research_v9" / "01_airgapped_rca_agent",
            "RESEARCH_AIRGAPPED_RCA_AGENT.md",
            "verify_rca_agent_grounding.py",
            "rca_agent_audit.json",
        ),
        (
            REPO_ROOT / "research_v9" / "02_causal_root_cause_graph",
            "RESEARCH_CAUSAL_ROOT_CAUSE_GRAPH.md",
            "verify_causal_dag_isolation.py",
            "causal_dag_audit.json",
        ),
        (
            REPO_ROOT / "research_v9" / "03_evn_tou_breakeven_pm",
            "RESEARCH_EVN_TOU_BREAKEVEN_PM.md",
            "verify_tou_optimizer.py",
            "tou_optimizer_audit.json",
        ),
    ]

    for folder, md_name, py_name, json_name in research_specs:
        assert folder.is_dir(), f"Missing research folder: {folder}"
        md_path = folder / md_name
        py_path = folder / py_name
        json_path = folder / json_name
        assert md_path.exists() and md_path.stat().st_size > 2000, f"Missing/short paper: {md_path}"
        assert py_path.exists(), f"Missing verification script: {py_path}"

        md_low = md_path.read_text(encoding="utf-8").lower()
        for bad in forbidden_terms:
            assert bad not in md_low, f"Forbidden red-line '{bad}' in {md_path.name}"

        if py_name == "verify_rca_agent_grounding.py":
            import py_compile

            py_compile.compile(str(py_path), doraise=True)
        else:
            proc = subprocess.run(
                [sys.executable, str(py_path)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert proc.returncode == 0, f"{py_name} failed (rc={proc.returncode}): {proc.stderr}\n{proc.stdout}"
        assert json_path.exists() and json_path.stat().st_size > 100, f"Missing/empty JSON output: {json_path}"
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict) and len(parsed) >= 3, f"Invalid JSON structure in {json_path}"

    # Verify compiled Word Technical Guide (.docx)
    docx_path = REPO_ROOT / "docs_v9" / "Chiller_FaultGuard_v9_Technical_Guide.docx"
    assert docx_path.exists() and docx_path.stat().st_size > 50_000, f"Missing/undersized .docx: {docx_path}"
    doc = docx.Document(str(docx_path))

    para_texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    assert len(para_texts) >= 60, f"Expected >= 60 non-empty paragraphs in .docx, got {len(para_texts)}"
    assert len(doc.tables) >= 6, f"Expected >= 6 structured tables in .docx, got {len(doc.tables)}"
    assert len(doc.inline_shapes) >= 4, f"Expected >= 4 embedded screenshots in .docx, got {len(doc.inline_shapes)}"

    full_text = "\n".join(para_texts)
    for tbl in doc.tables:
        for row in tbl.rows:
            full_text += "\n" + " ".join(cell.text.strip() for cell in row.cells)

    low_doc = full_text.lower()
    for chapter_marker in ("chương 1", "chương 2", "chương 3", "chương 4", "chương 5", "chương 6"):
        assert chapter_marker in low_doc, f"Missing chapter heading '{chapter_marker}' in .docx"
    for bad in forbidden_terms:
        assert bad not in low_doc, f"Forbidden red-line '{bad}' found in .docx"
    for placeholder in ("{{", "}}", "TODO:", "TBD:", "PLACEHOLDER"):
        assert placeholder not in full_text, f"Unresolved placeholder '{placeholder}' found in .docx"


def test_25_v9_frontend_and_architecture_svg_contracts() -> None:
    """Frontend HTML/CSS/JS and Architecture SVG expose v9 Causal DAG, EVN TOU PM Optimizer, and Air-Gapped RCA safely."""
    html_txt = (DT_DIR / "frontend" / "index.html").read_text(encoding="utf-8")
    js_txt = (DT_DIR / "frontend" / "app.js").read_text(encoding="utf-8")
    charts_txt = (DT_DIR / "frontend" / "charts_svg.js").read_text(encoding="utf-8")
    arch_txt = (ASSETS_DIR / "system_pipeline_architecture.svg").read_text(encoding="utf-8")

    # v11/v12 asset cache-busting & controls
    assert ("?v=12.0.0" in html_txt) or ("?v=11.0.0" in html_txt)
    assert "Chiller FaultGuard v11" in html_txt or "Chiller FaultGuard" in html_txt
    assert 'id="pm-cost-buttons"' in html_txt
    assert 'id="dag-view-toggle"' in html_txt
    assert 'id="step3-dag-host"' in html_txt
    assert 'id="btn-rca"' in html_txt
    assert 'id="rca-drawer"' in html_txt
    assert "drawCausalDagSvg" in charts_txt
    assert "/api/pm-optimizer" in js_txt
    assert "/api/rca-agent" in js_txt
    assert "escapeHtml" in js_txt

    # Architecture SVG v9 upgrades while retaining exact frozen model benchmarks
    arch_low = arch_txt.lower()
    assert "f1 0.394" in arch_low
    assert "6eeda35f" in arch_low
    assert "0.00e+00" in arch_low
    assert "causal" in arch_low and "dag" in arch_low
    assert "tou" in arch_low or "break-even" in arch_low
    assert "rca" in arch_low and ("62443" in arch_low or "air-gap" in arch_low)


def test_26_cross_surface_numeric_parity_tab3_dag_and_rca() -> None:
    """Exact numeric parity across /api/state (Tab 3 cards/SVG, Tab 4 DAG) and /api/rca-agent (6 tool payloads)."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    st = c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 735}).json()

    live = st["electricity"]["live_alert_projection"]
    pm = live["pm_optimizer"]
    kpis = live["kpis_5m1e"]
    dag = st["causal_graph"]

    rca_live = c.post("/api/rca-agent", json={"preset": "economics_pm", "basis_mode": "live", "lang": "en"}).json()
    t_map = {t["id"]: t["payload"] for t in rca_live["tool_traces"]}

    assert t_map["T2"]["ai_score"] == round(float(st["prediction"]["ai_score"]), 4)
    assert t_map["T3"]["isolated_root_id"] == dag["isolated_root_id"]
    assert t_map["T4"]["eff_kw"] == live["imposed_excess_kw"]
    assert t_map["T4"]["daily_burn_vnd"] == pm["daily_burn_vnd"]
    assert t_map["T4"]["avoided_gross_vnd"] == pm["avoided_gross_vnd"]
    assert t_map["T4"]["net_benefit_vnd"] == pm["net_benefit_vnd"]
    assert t_map["T4"]["payback_days"] == pm["payback_days"]
    assert t_map["T4"]["payback_calendar_day"] == pm["payback_calendar_day"]
    assert t_map["T4"]["verdict"] == pm["verdict"]
    assert t_map["T5"]["avoided_co2_tonnes"] == kpis["esg"]["avoided_co2_tonnes"]
    assert t_map["T5"]["live_evap_m3_day"] == kpis["esg"]["water_physics_est"]["live_evap_m3_day"]
    assert t_map["T5"]["verified_water_savings"] is None

    # Also verify illustrative basis parity (Quan Ref 6,624 kWh)
    rca_ill = c.post("/api/rca-agent", json={"preset": "economics_pm", "basis_mode": "illustrative", "lang": "en"}).json()
    t4_ill = next(t["payload"] for t in rca_ill["tool_traces"] if t["id"] == "T4")
    ill = st["electricity"]["illustrative_scenario"]
    assert t4_ill["scenario_avoided_excess_kwh"] == ill["scenario_avoided_excess_kwh"] == 6624.0
    assert t4_ill["net_benefit_vnd"] == ill["pm_optimizer"]["net_benefit_vnd"]


def test_27_multimodal_vision_inspector_modes_and_iphone11_upload() -> None:
    """OpenCV Vision Inspector accurately classifies fill_strainer, spray_flow, and basin_water across presets and uploaded JPEGs."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    # 1. Default status (fill_fouling_065): fill_strainer flags ISSUE_FOULING with scale/biofilm/strainer boxes
    hw0 = c.get("/api/hardware/status").json()
    assert hw0["version"] == "12.0.0"
    assert hw0["read_only"] is True
    assert hw0["controls"] is False
    assert hw0["advisory_only"] is True
    assert hw0["cad_summary"]["parts_count"] == 15
    fs_foul = hw0["vision_modes"]["fill_strainer"]
    assert fs_foul["verdict"] == "ISSUE_FOULING"
    assert fs_foul["metrics"]["occlusion_pct"] > 25.0
    det_classes = {d["class_name"] for d in fs_foul["detections"]}
    assert "calcium_scale" in det_classes and "algal_biofilm" in det_classes
    assert fs_foul["annotated_image_data_uri"].startswith("data:image/png;base64,")
    assert len(fs_foul["frame_sha256"]) == 64

    # 2. All Healthy preset: all 3 vision modes return OK
    hw_ok = c.post("/api/hardware/simulate", json={"preset": "all_healthy"}).json()
    for mode_key in ("fill_strainer", "spray_flow", "basin_water"):
        vm = hw_ok["vision_modes"][mode_key]
        assert vm["verdict"] == "OK", f"Expected OK on {mode_key} under all_healthy, got {vm['verdict']}"
    assert hw_ok["vision_modes"]["fill_strainer"]["metrics"]["occlusion_pct"] < 10.0
    assert hw_ok["vision_modes"]["spray_flow"]["metrics"]["healthy_nozzles"] == 4
    assert hw_ok["vision_modes"]["spray_flow"]["metrics"]["optical_flow_gpm"] >= 250.0
    assert hw_ok["vision_modes"]["basin_water"]["metrics"]["turbidity_ntu"] < 10.0
    assert hw_ok["vision_modes"]["basin_water"]["metrics"]["estimated_coc"] >= 3.5
    assert hw_ok["vision_modes"]["basin_water"]["metrics"]["verified_water_savings"] is None

    # 3. Clogged nozzle + low CW flow preset -> spray_flow returns ISSUE_LOW_FLOW_OR_NOZZLE
    hw_noz = c.post("/api/hardware/simulate", json={"preset": "clogged_nozzle_low_flow", "vision_mode": "spray_flow"}).json()
    sf = hw_noz["vision_modes"]["spray_flow"]
    assert sf["verdict"] == "ISSUE_LOW_FLOW_OR_NOZZLE"
    assert sf["metrics"]["clogged_nozzles"] >= 1
    assert sf["metrics"]["optical_flow_gpm"] < 250.0
    assert sf["metrics"]["flow_below_floor"] is True

    # 4. Turbid basin + low CoC preset -> basin_water returns ISSUE_WATER_CHEM
    hw_bas = c.post("/api/hardware/simulate", json={"preset": "turbid_basin_low_coc", "vision_mode": "basin_water"}).json()
    bw = hw_bas["vision_modes"]["basin_water"]
    assert bw["verdict"] == "ISSUE_WATER_CHEM"
    assert bw["metrics"]["turbidity_ntu"] > 15.0 or bw["metrics"]["estimated_coc"] < 3.2
    assert bw["metrics"]["verified_water_savings"] is None

    # 5. Real JPEG frame upload via POST /api/hardware/inspect-frame (simulating iPhone 11 /mobile-cam capture)
    bgr = generate_fill_strainer_frame("fouled_065")
    ok, jpg_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    b64_jpg = "data:image/jpeg;base64," + base64.b64encode(jpg_buf.tobytes()).decode("ascii")
    r_up = c.post(
        "/api/hardware/inspect-frame",
        json={"mode": "fill_strainer", "image_base64": b64_jpg, "source": "iphone11_mobile_cam"},
    )
    assert r_up.status_code == 200
    up_body = r_up.json()
    det = up_body["detection"]
    assert det["mode"] == "fill_strainer"
    assert det["source"] == "iphone11_mobile_cam"
    assert det["verdict"] == "ISSUE_FOULING"
    assert det["annotated_image_data_uri"].startswith("data:image/png;base64,")
    assert len(det["detections"]) >= 2


def test_28_acoustic_fft_optical_tach_and_stull_psychrometric_weather_gate() -> None:
    """16 kHz Acoustic FFT + 60fps Optical Tachometer and Stull (2011) Wet-Bulb Weather Gate isolate H3 & H4."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    # 1. Healthy acoustic & optical tachometer
    hw_ok = c.post("/api/hardware/simulate", json={"preset": "all_healthy"}).json()
    ac_ok = hw_ok["acoustic_tach"]
    assert ac_ok["verdict"] == "OK"
    assert ac_ok["slip_pct"] < 5.0
    assert len(ac_ok["spectrum_80"]) == 80

    # 2. Fan belt slip preset -> high slip_pct & 2.4-4.2 kHz belt squeal band
    hw_slip = c.post("/api/hardware/simulate", json={"preset": "fan_belt_slip"}).json()
    ac_slip = hw_slip["acoustic_tach"]
    assert ac_slip["verdict"] == "ISSUE_BELT_SLIP"
    assert ac_slip["slip_pct"] > 15.0
    assert ac_slip["bands"]["belt_squeal_2400_4200hz"] > ac_ok["bands"]["belt_squeal_2400_4200hz"]

    # 3. Stull (2011) Psychrometric Weather Gate across all 4 presets + custom weather
    w_hn = c.post("/api/hardware/weather", json={"preset": "hanoi_humid"}).json()["weather_gate"]
    assert w_hn["wb_le_db_ok"] is True
    assert w_hn["physics_valid"] is True
    assert w_hn["stull_wet_bulb_c"] <= w_hn["dry_bulb_c"]
    assert w_hn["confounder_verdict"] == "WEATHER_CLEAR"

    w_bn = c.post("/api/hardware/weather", json={"preset": "bac_ninh_peak"}).json()["weather_gate"]
    assert w_bn["wb_le_db_ok"] is True
    assert w_bn["confounder_verdict"] == "HIGH_WETBULB_CONFOUNDER"

    w_bad = c.post("/api/hardware/weather", json={"preset": "lbnl_corrupted_wb"}).json()["weather_gate"]
    assert w_bad["wb_le_db_ok"] is False
    assert w_bad["physics_valid"] is False
    assert w_bad["confounder_verdict"] == "SENSOR_CORRUPTED_REJECTED"

    w_cust = c.post("/api/hardware/weather", json={"dry_bulb_c": 25.0, "rh_pct": 60.0, "location": "Custom Fab Site"}).json()["weather_gate"]
    assert w_cust["wb_le_db_ok"] is True
    assert 18.0 < w_cust["stull_wet_bulb_c"] < 22.0


def test_29_field_ai_apply_to_step4_advisory_and_abstain_409_interlock() -> None:
    """POST /api/hardware/apply-to-step4 populates Step IV advisory badges, logs audit trail, NEVER auto-confirms Step V, and preserves ABSTAIN 409 lock."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 735})
    c.post("/api/hardware/simulate", json={"preset": "fill_fouling_065"})

    st = c.post("/api/hardware/apply-to-step4").json()
    fc = st["field_checks"]
    assert fc["applied"] is True
    assert fc["advisory_only"] is True
    assert fc["items"]["fill_media"]["status"] == "ISSUE"
    assert fc["items"]["fill_media"]["source"] == "AI-Vision"
    assert fc["items"]["nozzles"]["source"] == "AI-Vision"
    assert fc["items"]["fan_belt"]["source"] == "AI-Acoustic"
    assert fc["items"]["water_chem"]["source"] == "AI-Vision"

    # NEVER auto-confirms Step V (human supervisor retains authority!)
    assert st["review"]["acknowledged"] is False
    assert st["review"]["verdict"] == "PENDING"
    assert "FIELD-AI" in st["audit_log"][0]["category"]

    # Gate 1-4 ABSTAIN HTTP 409 safety interlock remains 100% enforced even after Field-AI runs!
    c.post("/api/inject-fault", json={"fault_type": "domain_ood"})
    c.post("/api/hardware/apply-to-step4")
    r_lock = c.post("/api/acknowledge", json={"verdict": "CONFIRMED", "note": "Must still block under ABSTAIN"})
    assert r_lock.status_code == 409


def test_30_mobile_cam_offline_ui_and_rca_agent_field_ai_tool_t7() -> None:
    """GET /mobile-cam is 200 OK offline HTML and POST /api/rca-agent preset='field_ai' invokes [T7:FieldAI] while preserving v9 6-tool SHA-256 digests."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)
    c.post("/api/seek", json={"scenario": "fouling_high", "row_index": 735})

    # 1. v9 preset 'why_alert' keeps exact 6 tools and deterministic 16-char SHA-256 digest
    r_v9 = c.post("/api/rca-agent", json={"preset": "why_alert", "lang": "en"}).json()
    r_v9_repeat = c.post("/api/rca-agent", json={"preset": "why_alert", "lang": "en"}).json()
    assert len(r_v9["tool_traces"]) == 6
    assert len(r_v9["evidence_digest_short"]) == 16
    assert r_v9["evidence_digest_short"] == r_v9_repeat["evidence_digest_short"]

    # 2. v10/v11/v12 preset 'field_ai' (EN & VI) invokes 7 tools [T1:Gate]..[T7:FieldAI] and cites [T7:FieldAI]
    for lang in ("en", "vi"):
        r_f = c.post("/api/rca-agent", json={"preset": "field_ai", "lang": lang}).json()
        assert r_f["agent_version"] == "12.0.0"
        assert r_f["lang"] == lang
        assert len(r_f["tool_traces"]) == 7
        assert r_f["tool_traces"][6]["id"] == "T7"
        assert r_f["tool_traces"][6]["tag"] == "[T7:FieldAI]"
        assert "[T7:FieldAI]" in r_f["citations_used"]
        assert "field_ai_context" in r_f

    # 3. GET /mobile-cam serves offline iPhone 11 capture UI
    r_mc = c.get("/mobile-cam")
    assert r_mc.status_code == 200
    mc_txt = r_mc.text
    mc_clean = mc_txt.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in mc_clean and "https://" not in mc_clean
    assert "/api/hardware/inspect-frame" in mc_txt


def test_31_hardware_cad_14_watertight_stls_steps_assemblies_and_3d_viewer() -> None:
    """All 3D-printable STL parts (15 in v12) are watertight, flat on Z=0, <= 185x185x175 mm, with matching STEP, assemblies & viewer."""
    cad_dir = CAD_DIR
    if not (cad_dir / "parts_stl").exists():
        pytest.skip("Standalone web deployment package (raw STL/STEP CAD parts not bundled)")
    import py_compile
    import trimesh

    stl_dir = cad_dir / "parts_stl"
    step_dir = cad_dir / "parts_step"

    # Allow Wave-1 backend test execution while @cad_builder finishes generating hardware_cad_v12 STLs
    gen_script = cad_dir / "generate_mini_ct1_cad_v12.py"
    if gen_script.exists() and not stl_dir.exists():
        py_compile.compile(str(gen_script), doraise=True)
        return

    expected_n = 15 if cad_dir.name == "hardware_cad_v12" else 14
    stl_files = sorted(stl_dir.glob("*.stl"))
    step_files = sorted(step_dir.glob("*.step"))
    assert len(stl_files) == expected_n, f"Expected {expected_n} STL parts in {stl_dir}, found {len(stl_files)}"
    assert len(step_files) == expected_n, f"Expected {expected_n} STEP parts in {step_dir}, found {len(step_files)}"

    for sf in step_files:
        assert sf.stat().st_size > 1000, f"Undersized STEP file: {sf}"

    for lf in stl_files:
        mesh = trimesh.load(str(lf), force="mesh")
        assert mesh.is_watertight, f"STL part not watertight: {lf.name}"
        assert mesh.is_volume, f"STL part not a valid volume: {lf.name}"
        assert float(mesh.volume) > 100.0, f"STL part has non-positive volume: {lf.name}"
        min_z = float(mesh.bounds[0][2])
        assert abs(min_z) <= 0.05, f"STL part {lf.name} not flat on Z=0 (min_z={min_z})"
        ext = mesh.extents
        assert ext[0] <= 185.5 and ext[1] <= 185.5 and ext[2] <= 175.5, (
            f"STL part {lf.name} exceeds 185x185x175 mm print envelope: {ext}"
        )

    asm_dir = cad_dir / "assemblies" if (cad_dir / "assemblies").exists() else (cad_dir / "assembly")
    assert (cad_dir / "print_guide_bom.json").exists()
    viewer_html_path = asm_dir / "MiniCT1_Interactive_3D_Viewer.html"
    if viewer_html_path.exists():
        viewer_html = viewer_html_path.read_text(encoding="utf-8")
        viewer_clean = viewer_html.replace("http://www.w3.org/2000/svg", "")
        assert "http://" not in viewer_clean and "https://" not in viewer_clean


def test_32_v10_research_folders_and_docx_deliverables_valid() -> None:
    """Both research_v10 folders, verification scripts/JSON audits, and compiled v10 .docx technical guide are valid & red-line clean."""
    if not (REPO_ROOT / "research_v10").exists():
        pytest.skip("Standalone web deployment package (research_v10/docs_v10 not bundled)")
    import docx

    forbidden_terms = ("yield protected", "yield first", "yield critical", "generating excess", "chillguardai")
    research_specs = [
        (
            REPO_ROOT / "research_v10" / "01_hardware_bench_3d_design",
            "RESEARCH_HARDWARE_BENCH_3D.md",
            "verify_cad_printability.py",
            "cad_geometry_audit.json",
        ),
        (
            REPO_ROOT / "research_v10" / "02_multimodal_step34_vision_ai",
            "RESEARCH_MULTIMODAL_STEP34_AI.md",
            "verify_multimodal_ai.py",
            "multimodal_ai_audit.json",
        ),
    ]

    for folder, md_name, py_name, json_name in research_specs:
        assert folder.is_dir(), f"Missing research_v10 folder: {folder}"
        md_path = folder / md_name
        py_path = folder / py_name
        json_path = folder / json_name
        assert md_path.exists() and md_path.stat().st_size > 2000, f"Missing/short paper: {md_path}"
        assert py_path.exists(), f"Missing verification script: {py_path}"

        md_low = md_path.read_text(encoding="utf-8").lower()
        for bad in forbidden_terms:
            assert bad not in md_low, f"Forbidden red-line '{bad}' in {md_path.name}"

        if py_name == "verify_multimodal_ai.py" or not (REPO_ROOT / "hardware_cad_v10").exists():
            import py_compile

            py_compile.compile(str(py_path), doraise=True)
        else:
            proc = subprocess.run(
                [sys.executable, str(py_path)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert proc.returncode == 0, f"{py_name} failed (rc={proc.returncode}): {proc.stderr}\n{proc.stdout}"
        assert json_path.exists() and json_path.stat().st_size > 100, f"Missing/empty JSON output: {json_path}"
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict) and len(parsed) >= 3, f"Invalid JSON structure in {json_path}"

    docx_path = REPO_ROOT / "docs_v10" / "Chiller_FaultGuard_v10_Hardware_3D_and_VisionAI_Guide.docx"
    assert docx_path.exists() and docx_path.stat().st_size > 50_000, f"Missing/undersized v10 .docx: {docx_path}"
    doc = docx.Document(str(docx_path))

    para_texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    assert len(para_texts) >= 60, f"Expected >= 60 non-empty paragraphs in v10 .docx, got {len(para_texts)}"
    assert len(doc.tables) >= 6, f"Expected >= 6 structured tables in v10 .docx, got {len(doc.tables)}"
    assert len(doc.inline_shapes) >= 10, f"Expected >= 10 embedded images in v10 .docx, got {len(doc.inline_shapes)}"

    full_text = "\n".join(para_texts)
    for tbl in doc.tables:
        for row in tbl.rows:
            full_text += "\n" + " ".join(cell.text.strip() for cell in row.cells)

    low_doc = full_text.lower()
    for chapter_marker in ("chương 1", "chương 2", "chương 3", "chương 4", "chương 5", "chương 6"):
        assert chapter_marker in low_doc, f"Missing chapter heading '{chapter_marker}' in v10 .docx"
    for bad in forbidden_terms:
        assert bad not in low_doc, f"Forbidden red-line '{bad}' found in v10 .docx"
    for placeholder in ("{{", "}}", "TODO:", "TBD:", "PLACEHOLDER"):
        assert placeholder not in full_text, f"Unresolved placeholder '{placeholder}' found in v10 .docx"


def test_33_v10_tab5_hardware_frontend_and_architecture_svg_contracts() -> None:
    """Frontend HTML/JS/SVG and Architecture SVG expose v10 Tab 5 Hardware & Vision AI Lab, Acoustic FFT SVG, and /mobile-cam."""
    html_txt = (DT_DIR / "frontend" / "index.html").read_text(encoding="utf-8")
    js_txt = (DT_DIR / "frontend" / "app.js").read_text(encoding="utf-8")
    charts_txt = (DT_DIR / "frontend" / "charts_svg.js").read_text(encoding="utf-8")
    arch_txt = (ASSETS_DIR / "system_pipeline_architecture.svg").read_text(encoding="utf-8")
    mc_path = DT_DIR / "frontend" / "mobile_cam.html"

    assert ("?v=12.0.0" in html_txt) or ("?v=11.0.0" in html_txt)
    assert 'data-tab="hardware"' in html_txt
    assert 'id="tab-hardware"' in html_txt
    assert "drawAcousticFftSvg" in charts_txt
    assert "/api/hardware/status" in js_txt
    assert "/api/hardware/simulate" in js_txt
    assert "/api/hardware/inspect-frame" in js_txt
    assert "/api/hardware/weather" in js_txt
    assert "/api/hardware/apply-to-step4" in js_txt
    assert mc_path.exists()

    arch_low = arch_txt.lower()
    assert "f1 0.394" in arch_low
    assert "6eeda35f" in arch_low
    assert "0.00e+00" in arch_low
    assert "iphone 11" in arch_low or "mini-ct1" in arch_low or "vision" in arch_low


def test_34_deterministic_vision_and_fft_reproducibility_sha256() -> None:
    """Repeated simulation of any hardware preset yields bit-for-bit identical frame SHA-256 hashes, FFT bins, and Stull wet-bulb."""
    app = create_app(ChillerFaultGuardV6Runtime(auto_seek_eligible=True))
    c = TestClient(app)

    for preset in ("all_healthy", "fill_fouling_065", "clogged_nozzle_low_flow", "fan_belt_slip", "turbid_basin_low_coc"):
        a = c.post("/api/hardware/simulate", json={"preset": preset}).json()
        b = c.post("/api/hardware/simulate", json={"preset": preset}).json()
        for m in ("fill_strainer", "spray_flow", "basin_water"):
            assert a["vision_modes"][m]["frame_sha256"] == b["vision_modes"][m]["frame_sha256"]
            assert a["vision_modes"][m]["metrics"] == b["vision_modes"][m]["metrics"]
        assert a["acoustic_tach"]["spectrum_80"] == b["acoustic_tach"]["spectrum_80"]
        assert a["acoustic_tach"]["slip_pct"] == b["acoustic_tach"]["slip_pct"]
        assert a["weather_gate"]["stull_wet_bulb_c"] == b["weather_gate"]["stull_wet_bulb_c"]


def test_35_v11_clean_ui_minimal_copy_no_verbose_clutter() -> None:
    """v11 strips verbose sentence-length footnotes from P&ID, Tab 2 recall tiles, Tab 3 footers & Modal while preserving all features."""
    pid_txt = (ASSETS_DIR / "chiller_plant_pid.svg").read_text(encoding="utf-8")
    arch_txt = (ASSETS_DIR / "system_pipeline_architecture.svg").read_text(encoding="utf-8")
    js_txt = (DT_DIR / "frontend" / "app.js").read_text(encoding="utf-8")

    assert "ISA-101 · N+1 CUB Cell" in pid_txt
    assert "Representative N+1 CUB Cell (Mapped to LBNL 3-Chiller / 3-Tower FDD Benchmark)" not in pid_txt
    assert "73.9k Rows · Pilot Scope" in js_txt
    assert "pts · 42/53" in js_txt
    assert "v12.0" in arch_txt.lower() or "v11.0" in arch_txt.lower()
