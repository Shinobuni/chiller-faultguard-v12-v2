"""Deterministic 8-Node Causal Root-Cause Graph (DAG) for Chiller FaultGuard v11.0.

Separates Layer 0 root causes from Layer 1 thermal-hydraulic symptoms and Layer 2
plant energy/EVN/ESG impacts while preserving strict 3-Bucket LBNL honesty:
- Layer 0 (Roots):
    N1: CT-1 Fill Fouling
    N2: CW Pump / Flow
    N3: Sensor / OOD Gate
    N4: Weather / CHW Map*
- Layer 1 (Intermediate Symptoms):
    N5: CT-1 Range & CW Out
    N6: CT-1 Fan Saturation
- Layer 2 (Plant Impact):
    N7: Chiller Lift & 14-Ch ΔkW*
    N8: EVN Cost / CO₂ / Water*

Valid node states:
    ROOT_SUSPECT | PROPAGATED | RULED_OUT | BLOCKED_GATE | CONFOUNDER_CHECK | CALM
"""

from __future__ import annotations

from typing import Any, Dict, List

CONFOUNDER_WARNING = "OA_TEMP_WB > OA_TEMP in 99.31% of rows; CHW map unresolved 97.3%. Excluded from AI."


def _fmt_vnd_compact(val: float) -> str:
    mag = abs(val)
    sign = "-" if val < 0 else ""
    if mag >= 1_000_000.0:
        return f"{sign}{mag / 1_000_000.0:.2f}M ₫"
    if mag >= 1_000.0:
        return f"{sign}{mag / 1_000.0:.1f}k ₫"
    return f"{sign}{mag:.0f} ₫"


def build_causal_graph(snap_core: Dict[str, Any]) -> Dict[str, Any]:
    """Build the 8-node, 8-edge deterministic Causal Root-Cause DAG from a live snapshot."""
    status = str(snap_core.get("status", "NO_ALERT"))
    agreement = str(snap_core.get("agreement", "NO_ALERT"))
    injection = str(snap_core.get("injection", "none"))
    pred = snap_core.get("prediction", {}) or {}
    sig = snap_core.get("signals", {}) or {}
    elec = snap_core.get("electricity", {}) or {}
    live_proj = elec.get("live_alert_projection", {}) or {}
    kpis = live_proj.get("kpis_5m1e", {}) or {}
    money = kpis.get("money", {}) or {}
    esg = kpis.get("esg", {}) or {}
    water_est = esg.get("water_physics_est", {}) or {}
    pm_opt = live_proj.get("pm_optimizer", {}) or {}

    ai_score = pred.get("ai_score")
    ai_alert = bool(pred.get("ai_alert", False))
    rule_alert = bool(pred.get("rule_alert", False))
    ai_run = int(pred.get("ai_run_minutes", 0))
    ai_req = int(pred.get("ai_required_minutes", 5))
    rule_run = int(pred.get("rule_run_minutes", 0))
    rule_req = int(pred.get("rule_required_minutes", 15))
    ood_flag = bool(pred.get("ood_flag", False))
    ood_features = list(pred.get("ood_features", []) or [])
    reasons = list(pred.get("reasons", []) or [])
    reason_text = " ".join(reasons) if reasons else "Operating/quality gate active."

    flow_gpm = float(sig.get("tower_flow_gpm", 0.0))
    fan_pct = float(sig.get("fan_pct", 0.0))
    tw_in_c = sig.get("tower_inlet_c")
    tw_out_c = sig.get("nominal_tower_outlet_c")
    tw_range_c = sig.get("tower_range_c")
    plant_kw = float(sig.get("plant_kw", 0.0))
    delta_kw = float(sig.get("signed_delta_kw", 0.0))
    p_chillers_kw = float(sig.get("p_chillers_kw", 0.0))
    eff_kw = float(live_proj.get("imposed_excess_kw", 0.0))

    daily_burn_vnd = float(pm_opt.get("daily_burn_vnd", money.get("daily_burn_ex_vat_vnd", 0.0)))
    avoided_co2_t = float(esg.get("avoided_co2_tonnes", 0.0))
    live_evap_m3_day = float(water_est.get("live_evap_m3_day", 0.0))
    verdict = str(pm_opt.get("verdict", "STANDBY_NO_ACTION"))

    is_abstain = status == "ABSTAIN"
    is_alert = status in ("FOULING_SUSPECT", "RULE_WARNING")
    flow_low = flow_gpm < 250.0

    # Determine isolated root cause and node states
    if is_abstain:
        isolated_root_id = "N3"
        isolated_root_label = "Sensor / OOD Gate (ABSTAIN)"
        if ood_flag:
            gate_tag = f"OOD ({','.join(ood_features) or 'domain'})"
        elif injection != "none":
            gate_tag = f"Stress: {injection}"
        else:
            gate_tag = "Regime / Quality Gate"
        isolation_summary = (
            f"N3 {gate_tag}: diagnosis blocked."
        )
        states = {
            "N1": "BLOCKED_GATE",
            "N2": "BLOCKED_GATE",
            "N3": "ROOT_SUSPECT",
            "N4": "CONFOUNDER_CHECK",
            "N5": "BLOCKED_GATE",
            "N6": "BLOCKED_GATE",
            "N7": "BLOCKED_GATE",
            "N8": "BLOCKED_GATE",
        }
        active_edges = {("N3", "N5")}
        blocked_edges = {
            ("N1", "N5"),
            ("N1", "N6"),
            ("N2", "N5"),
            ("N4", "N5"),
            ("N5", "N7"),
            ("N6", "N7"),
            ("N7", "N8"),
        }
    elif is_alert and not flow_low:
        isolated_root_id = "N1"
        isolated_root_label = "CT-1 Fill Fouling"
        score_str = f"{float(ai_score):.3f}" if ai_score is not None else "N/A"
        isolation_summary = (
            f"N1 fouling: HGB {score_str} ({ai_run}/{ai_req}m), flow {flow_gpm:.1f} GPM rules out N2."
        )
        states = {
            "N1": "ROOT_SUSPECT",
            "N2": "RULED_OUT",
            "N3": "RULED_OUT",
            "N4": "CONFOUNDER_CHECK",
            "N5": "PROPAGATED",
            "N6": "PROPAGATED",
            "N7": "PROPAGATED",
            "N8": "PROPAGATED",
        }
        active_edges = {
            ("N1", "N5"),
            ("N1", "N6"),
            ("N5", "N7"),
            ("N6", "N7"),
            ("N7", "N8"),
        }
        blocked_edges = set()
    elif is_alert and flow_low:
        if ai_alert:
            isolated_root_id = "N1"
            isolated_root_label = "CT-1 Fill Fouling (with Low CW Flow Check)"
            score_str = f"{float(ai_score):.3f}" if ai_score is not None else "N/A"
            isolation_summary = (
                f"N1 suspect (HGB {score_str}); N2 low flow {flow_gpm:.1f} GPM — check pump first."
            )
            states = {
                "N1": "ROOT_SUSPECT",
                "N2": "CONFOUNDER_CHECK",
                "N3": "RULED_OUT",
                "N4": "CONFOUNDER_CHECK",
                "N5": "PROPAGATED",
                "N6": "PROPAGATED",
                "N7": "PROPAGATED",
                "N8": "PROPAGATED",
            }
            active_edges = {
                ("N1", "N5"),
                ("N1", "N6"),
                ("N2", "N5"),
                ("N5", "N7"),
                ("N6", "N7"),
                ("N7", "N8"),
            }
        else:
            isolated_root_id = "N2"
            isolated_root_label = "CW Pump / Flow Restriction"
            isolation_summary = (
                f"N2 low flow {flow_gpm:.1f} GPM < 250; fan {fan_pct:.1f}%. Check pump & strainers."
            )
            states = {
                "N1": "CONFOUNDER_CHECK",
                "N2": "ROOT_SUSPECT",
                "N3": "RULED_OUT",
                "N4": "CONFOUNDER_CHECK",
                "N5": "PROPAGATED",
                "N6": "PROPAGATED",
                "N7": "PROPAGATED",
                "N8": "PROPAGATED",
            }
            active_edges = {
                ("N2", "N5"),
                ("N5", "N7"),
                ("N6", "N7"),
                ("N7", "N8"),
            }
        blocked_edges = set()
    else:
        isolated_root_id = "NONE"
        isolated_root_label = "No Active Fault Isolated"
        score_str = f"{float(ai_score):.3f}" if ai_score is not None else "N/A"
        isolation_summary = (
            f"Calm: HGB {score_str} < 0.70, fan {fan_pct:.1f}%, flow {flow_gpm:.1f} GPM."
        )
        states = {
            "N1": "CALM",
            "N2": "RULED_OUT" if not flow_low else "CONFOUNDER_CHECK",
            "N3": "RULED_OUT",
            "N4": "CONFOUNDER_CHECK",
            "N5": "CALM",
            "N6": "CALM",
            "N7": "CALM",
            "N8": "CALM",
        }
        active_edges = set()
        blocked_edges = set()

    # Format concise live metrics & detailed engineering tooltips for each of the 8 nodes
    score_val = float(ai_score) if ai_score is not None else 0.0
    n1_metric = (
        "Suspended (Gate)"
        if is_abstain
        else f"HGB {score_val:.3f} ({ai_run}/{ai_req}m)"
    )
    n1_detail = (
        "Suspended (gate)."
        if is_abstain
        else (
            f"HGB {score_val:.3f} vs 0.70 ({ai_run}/{ai_req}m); {agreement}."
        )
    )

    n2_metric = (
        f"{flow_gpm:.1f} GPM (OOD!)"
        if ood_flag and "flow" in ood_features
        else f"{flow_gpm:.1f} GPM ({'<250!' if flow_low else '≥250 OK'})"
    )
    n2_detail = (
        f"{flow_gpm:.1f} GPM > train max 601.1 (OOD)."
        if (ood_flag and "flow" in ood_features)
        else (
            f"{flow_gpm:.1f} GPM vs 250 floor — " + ("below floor." if flow_low else "OK.")
        )
    )

    if is_abstain:
        n3_metric = "OOD GATE TRIP" if ood_flag else (f"STRESS: {injection}" if injection != "none" else "GATE ABSTAIN")
        n3_score = 1.0
    else:
        n3_metric = "Gates 1–4 PASS"
        n3_score = 0.0
    n3_detail = (
        f"Gate active: {reason_text}"
        if is_abstain
        else "Gates 1–4 passed."
    )

    n4_metric = "Excluded (99.3%*)"
    n4_detail = CONFOUNDER_WARNING

    if is_abstain or tw_out_c is None or tw_range_c is None:
        n5_metric = "Blocked (No Out)" if tw_out_c is None else f"ΔT {tw_range_c:.1f}°C · Blocked"
        n5_score = 0.0
    else:
        n5_metric = f"ΔT {tw_range_c:.1f}°C · Out {tw_out_c:.1f}°C*"
        n5_score = round(min(1.0, max(0.0, (float(tw_out_c) - 18.0) / 14.0)), 3)
    n5_detail = (
        "Blocked by gate."
        if is_abstain
        else (
            f"In {tw_in_c:.2f}°C · Out* {tw_out_c:.2f}°C · ΔT {tw_range_c:.2f}°C."
        )
    )

    n6_metric = (
        f"Fan {fan_pct:.1f}% (Gate)"
        if is_abstain
        else f"Fan {fan_pct:.1f}% ({rule_run}/{rule_req}m)"
    )
    n6_score = 0.0 if is_abstain else round(min(1.0, max(0.0, fan_pct / 100.0)), 3)
    n6_detail = (
        "Suspended (gate)."
        if is_abstain
        else (
            f"Fan {fan_pct:.1f}% (rule >98%: {rule_run}/{rule_req}m)."
        )
    )

    if is_abstain:
        n7_metric = f"Plant {plant_kw:.1f} kW · Gate"
        n7_score = 0.0
    else:
        sign_d = "+" if delta_kw >= 0 else ""
        n7_metric = f"ΔkW* {sign_d}{delta_kw:.2f} · 15m {eff_kw:.2f}"
        n7_score = round(min(1.0, max(0.0, eff_kw / 6.0)), 3)
    n7_detail = (
        f"Plant {plant_kw:.1f} kW; ΔkW* suspended."
        if is_abstain
        else (
            f"Plant {plant_kw:.1f} kW (chillers {p_chillers_kw:.1f}); ΔkW* {delta_kw:+.2f}, 15m {eff_kw:.2f}."
        )
    )

    if is_abstain:
        n8_metric = "Locked · 0 ₫/d"
        n8_score = 0.0
    elif is_alert and eff_kw > 0.0:
        n8_metric = f"{_fmt_vnd_compact(daily_burn_vnd)}/d · {avoided_co2_t:.2f}t*"
        n8_score = round(min(1.0, max(0.0, daily_burn_vnd / 250_000.0)), 3)
    else:
        n8_metric = f"0 ₫/d · Evap {live_evap_m3_day:.1f}m³/d*"
        n8_score = 0.0
    n8_detail = (
        "Locked (gate)."
        if is_abstain
        else (
            f"{_fmt_vnd_compact(daily_burn_vnd)}/d · {avoided_co2_t:.2f} t CO₂e · evap {live_evap_m3_day:.2f} m³/d* (illustrative)."
        )
    )

    nodes: List[Dict[str, Any]] = [
        {
            "id": "N1",
            "layer": 0,
            "short": "CT-1 Fill Fouling",
            "metric": n1_metric,
            "state": states["N1"],
            "score": 0.0 if is_abstain else round(score_val, 3),
            "detail": n1_detail,
        },
        {
            "id": "N2",
            "layer": 0,
            "short": "CW Pump / Flow",
            "metric": n2_metric,
            "state": states["N2"],
            "score": 1.0 if (states["N2"] == "ROOT_SUSPECT") else (0.6 if states["N2"] == "CONFOUNDER_CHECK" else 0.0),
            "detail": n2_detail,
        },
        {
            "id": "N3",
            "layer": 0,
            "short": "Sensor / OOD Gate",
            "metric": n3_metric,
            "state": states["N3"],
            "score": n3_score,
            "detail": n3_detail,
        },
        {
            "id": "N4",
            "layer": 0,
            "short": "Weather / CHW Map*",
            "metric": n4_metric,
            "state": states["N4"],
            "score": 0.5,
            "detail": n4_detail,
        },
        {
            "id": "N5",
            "layer": 1,
            "short": "CT-1 Range & CW Out",
            "metric": n5_metric,
            "state": states["N5"],
            "score": n5_score,
            "detail": n5_detail,
        },
        {
            "id": "N6",
            "layer": 1,
            "short": "CT-1 Fan Saturation",
            "metric": n6_metric,
            "state": states["N6"],
            "score": n6_score,
            "detail": n6_detail,
        },
        {
            "id": "N7",
            "layer": 2,
            "short": "Chiller Lift & 14-Ch ΔkW*",
            "metric": n7_metric,
            "state": states["N7"],
            "score": n7_score,
            "detail": n7_detail,
        },
        {
            "id": "N8",
            "layer": 2,
            "short": "EVN Cost / CO₂ / Water*",
            "metric": n8_metric,
            "state": states["N8"],
            "score": n8_score,
            "detail": n8_detail,
        },
    ]

    edge_specs = [
        ("N1", "N5", "UA ↓ → CW Out ↑"),
        ("N1", "N6", "UA ↓ → Fan VFD ↑"),
        ("N2", "N5", "Flow ↓ → ΔT Shift"),
        ("N3", "N5", "Gate / OOD Block"),
        ("N4", "N5", "Twb / CHW Conf.*"),
        ("N5", "N7", "CW Out ↑ → Lift ↑"),
        ("N6", "N7", "Fan + Chiller ΔkW*"),
        ("N7", "N8", "14-Ch ΔkW* → EVN/CO₂"),
    ]
    edges: List[Dict[str, Any]] = [
        {
            "from": src,
            "to": dst,
            "label": lbl,
            "active": (src, dst) in active_edges,
            "blocked": (src, dst) in blocked_edges,
        }
        for src, dst, lbl in edge_specs
    ]

    return {
        "isolated_root_id": isolated_root_id,
        "isolated_root_label": isolated_root_label,
        "isolation_summary": isolation_summary,
        "confounder_warning": CONFOUNDER_WARNING,
        "nodes": nodes,
        "edges": edges,
    }
