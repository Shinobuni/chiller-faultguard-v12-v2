"""Air-Gapped Evidence-Grounded Deterministic RCA Agent for Chiller FaultGuard v12.0.

Architecture:
- Purdue Level 3 Air-Gapped OT Boundary (100% offline, zero external HTTP/API calls)
- IEC 62443-3-3 Read-Only Conduit (controls=False, read_only=True, strict PLC write refusal)
- 6 Core Deterministic Grounding Tools (preserved verbatim from v9):
    [T1:Gate]    tool_check_safety_gates
    [T2:HGB]     tool_score_quan_hgb
    [T3:DAG]     tool_trace_causal_dag
    [T4:EVN-TOU] tool_calc_evn_tou_economics
    [T5:ESG]     tool_calc_scope2_and_water
    [T6:SOP]     tool_retrieve_local_sop
- Plus Field-AI Tool (attached on `field_ai` preset or hardware/vision queries):
    [T7:FieldAI] tool_inspect_field_ai_bench
- Canonical SHA-256 evidence_digest_sha256 over canonical JSON of all executed tool payloads
- Bilingual EN / VI auto-detection with explicit lang override
- Zero-hallucination guarantee: every numeric figure comes verbatim from tool payloads
  and carries inline [T1:Gate]..[T7:FieldAI] citation tags.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .causal_graph import build_causal_graph

VALID_PRESETS = (
    "why_alert",
    "root_cause",
    "economics_pm",
    "engineer_brief",
    "exec_5m1e",
    "sensor_safety",
    "water_esg",
    "control_refusal",
    "field_ai",
)

VALID_BASIS_MODES = ("live", "illustrative")
VALID_LANGS = ("auto", "en", "vi")

TOOL_TAGS = ("[T1:Gate]", "[T2:HGB]", "[T3:DAG]", "[T4:EVN-TOU]", "[T5:ESG]", "[T6:SOP]")
FIELD_AI_TAG = "[T7:FieldAI]"
ALL_TOOL_TAGS = TOOL_TAGS + (FIELD_AI_TAG,)

_VI_DIACRITICS_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)

_VI_KEYWORDS = (
    "tai sao",
    "nguyen nhan",
    "bao loi",
    "canh bao",
    "tat chiller",
    "bat bom",
    "tat bom",
    "doi setpoint",
    "chinh quat",
    "dieu khien",
    "tien dien",
    "hoan von",
    "chi phi",
    "bao tri",
    "ve sinh",
    "ca truc",
    "ban giao",
    "cam bien",
    "cong an toan",
    "nuoc",
    "khi thai",
    "tong quan",
    "mo van",
    "dong van",
    "ghi plc",
    "ngat chiller",
    "tat thap",
)

_CONTROL_WRITE_PATTERNS = (
    # English actuation / PLC write / safety-bypass intents
    "turn off",
    "shut down",
    "shutdown",
    "switch off",
    "trip chiller",
    "trip ct",
    "trip pump",
    "stop chiller",
    "stop tower",
    "stop pump",
    "start pump",
    "start chiller",
    "open valve",
    "close valve",
    "change setpoint",
    "set setpoint",
    "adjust setpoint",
    "raise setpoint",
    "lower setpoint",
    "modify setpoint",
    "write plc",
    "write to plc",
    "write bms",
    "write to bms",
    "modify bms",
    "override plc",
    "override bms",
    "override fan",
    "override valve",
    "actuate",
    "send command",
    "execute command",
    "issue command",
    "control chiller",
    "control fan",
    "control pump",
    "increase fan",
    "decrease fan",
    "reduce fan",
    "set fan speed",
    "force clean",
    "auto dispatch",
    "auto-dispatch",
    "bypass gate",
    "bypass safety",
    "disable interlock",
    "disable gate",
    "trip ",
    # Vietnamese actuation / PLC write / safety-bypass intents
    "tắt chiller",
    "tat chiller",
    "tắt tháp",
    "tat thap",
    "tắt máy",
    "tat may",
    "ngắt chiller",
    "ngat chiller",
    "bật bơm",
    "bat bom",
    "tắt bơm",
    "tat bom",
    "mở van",
    "mo van",
    "đóng van",
    "dong van",
    "đổi setpoint",
    "doi setpoint",
    "thay đổi setpoint",
    "thay doi setpoint",
    "chỉnh quạt",
    "chinh quat",
    "tắt quạt",
    "tat quat",
    "tăng quạt",
    "tang quat",
    "giảm quạt",
    "giam quat",
    "điều khiển chiller",
    "dieu khien chiller",
    "điều khiển bơm",
    "dieu khien bom",
    "điều khiển quạt",
    "dieu khien quat",
    "điều khiển plc",
    "dieu khien plc",
    "ghi plc",
    "ghi xuống plc",
    "ghi xuong plc",
    "ghi bms",
    "bỏ qua cổng",
    "bo qua cong",
    "tắt khóa an toàn",
    "tat khoa an toan",
)

_NODE_ID_RE = re.compile(r"\b(N[1-8])\b", re.IGNORECASE)

# Offline curated SOP corpus (zero external dependencies)
LOCAL_SOP_CORPUS: Dict[str, Dict[str, str]] = {
    "SOP-IEC62443": {
        "id": "SOP-IEC62443",
        "title": "IEC 62443-3-3 Read-Only Conduit & Purdue Level 3 Air-Gap",
        "summary_en": (
            "Advisory telemetry flows one-way only (read_only=True, controls=False). "
            "Zero PLC/BMS write path exists; all operational changes require human supervisor authorization via CMMS."
        ),
        "summary_vi": (
            "Dữ liệu vận hành chỉ đọc một chiều (read_only=True, controls=False). "
            "Không tồn tại đường ghi xuống PLC/BMS; mọi thao tác thiết bị phải do kỹ sư trực ca duyệt qua CMMS."
        ),
    },
    "SOP-ASHRAE12": {
        "id": "SOP-ASHRAE12",
        "title": "ASHRAE Guideline 12 — Cooling Tower N+1 Isolation & Safe Cleaning",
        "summary_en": (
            "Inspect CT-1 basin, spray nozzles, drift eliminators, and CW basket strainers physically. "
            "Isolate via N+1 standby tower cell during off-peak window (Sunday 00:00–06:00); never trip an active chiller blindly."
        ),
        "summary_vi": (
            "Kiểm tra trực quan bể nước CT-1, béc phun, tấm chắn giọt và lưới lọc bơm CW. "
            "Cô lập theo tháp dự phòng N+1 trong khung giờ thấp điểm (Chủ nhật 00:00–06:00); tuyệt đối không ngắt chiller đột ngột."
        ),
    },
    "SOP-SEMIS23": {
        "id": "SOP-SEMIS23",
        "title": "SEMI S23 — Fab Utility Separation (PCW-C vs PCW-T)",
        "summary_en": (
            "Cooling tower CT-1 serves the condenser heat-rejection loop (PCW-C), physically separated from cleanroom "
            "process cooling water (PCW-T). No fab process tool or wafer-level guarantee is claimed from LBNL office data."
        ),
        "summary_vi": (
            "Tháp giải nhiệt CT-1 thuộc vòng giải nhiệt ngưng tụ (PCW-C), tách biệt với vòng nước giải nhiệt công cụ phòng sạch (PCW-T). "
            "Không suy diễn cam kết vận hành công cụ wafer từ dữ liệu tòa nhà LBNL."
        ),
    },
    "SOP-LBNL-AUDIT": {
        "id": "SOP-LBNL-AUDIT",
        "title": "LBNL Physical Audit & 3-Bucket Scientific Honesty",
        "summary_en": (
            "OA_TEMP_WB > OA_TEMP in 99.31% of raw LBNL rows and CHW supply/return mapping is unresolved in 97.3%; "
            "weather/CHW are excluded from AI with zero auto-swap. Keep verified_excess_kwh=None, verified_avoided_kwh=None, "
            "and verified_water_savings=None until calibrated site metering."
        ),
        "summary_vi": (
            "OA_TEMP_WB > OA_TEMP ở 99.31% dòng dữ liệu gốc LBNL và ánh xạ CHW cấp/hồi chưa xác minh ở 97.3%; "
            "thời tiết/CHW bị loại khỏi AI và không tự động hoán đổi cảm biến. Giữ verified_excess_kwh=None, "
            "verified_avoided_kwh=None, verified_water_savings=None đến khi có đo lường Pilot."
        ),
    },
    "SOP-EVN-ESG": {
        "id": "SOP-EVN-ESG",
        "title": "Vietnam Nov-2026 Tariff (QĐ 2892 & 963/QĐ-BCT) & Grid EF (CV 11232/BNNMT)",
        "summary_en": (
            "MOIT Decision 2892/QD-BCT (eff. 01/11/2026) + Decision 963/QD-BCT 22kV 24/7 weighted TOU ex-VAT = 1,925.93 VND/kWh "
            "(Peak 17:30–22:30 Mon–Sat = 31.04% of bill; VAT 8%; FX 25,980 VND/USD). Official Dispatch 11232/BNNMT-BDKH 2024 EF = 0.6187 kg CO2/kWh."
        ),
        "summary_vi": (
            "QĐ 2892/QĐ-BCT (hiệu lực 01/11/2026) + QĐ 963/QĐ-BCT biểu giá 22kV 24/7 chưa VAT = 1,925.93 ₫/kWh "
            "(Giờ cao điểm 17:30–22:30 T2–T7 = 31.04% hóa đơn; VAT 8%; tỷ giá 25,980 ₫/USD). CV 11232/BNNMT-BĐKH EF 2024 = 0.6187 kg CO2/kWh."
        ),
    },
    "SOP-PILOT90": {
        "id": "SOP-PILOT90",
        "title": "90-Day Shadow-Mode Site Pilot Exit Gates",
        "summary_en": (
            "Exit criteria before operational rollout: (1) verify physical sensor mapping & wet-bulb calibration, "
            "(2) install WTP makeup/blowdown sub-meters, (3) confirm independent engineer fouling findings and pre/post-clean kW deltas."
        ),
        "summary_vi": (
            "Cổng thoát Pilot 90 ngày chế độ Shadow: (1) xác minh ánh xạ cảm biến & nhiệt kế bầu ướt tại hiện trường, "
            "(2) lắp đồng hồ đo nước cấp/xả đáy WTP, (3) đối chiếu xác nhận đóng cặn độc lập và chênh lệch kW trước/sau vệ sinh."
        ),
    },
}


def _fmt_vnd(val: float) -> str:
    return f"{int(round(val)):,} ₫"


def _fmt_vnd_m(val: float) -> str:
    sign = "+" if val > 0 else "-" if val < 0 else ""
    mag = abs(val)
    if mag >= 1_000_000.0:
        return f"{sign}{mag / 1_000_000.0:.2f}M ₫"
    if mag >= 1_000.0:
        return f"{sign}{mag / 1_000.0:.0f}k ₫"
    return f"{sign}{mag:.0f} ₫"


def detect_language(query: str, lang: str = "auto") -> str:
    """Detect EN or VI deterministically; explicit 'en'/'vi' overrides auto."""
    if lang in ("en", "vi"):
        return lang
    if not query:
        return "en"
    if _VI_DIACRITICS_RE.search(query):
        return "vi"
    low = query.lower()
    if any(kw in low for kw in _VI_KEYWORDS):
        return "vi"
    return "en"


def detect_control_write_request(query: str, preset: Optional[str]) -> bool:
    """Detect any PLC/BMS write, actuation, setpoint-change, or safety-bypass attempt."""
    if preset == "control_refusal":
        return True
    if not query:
        return False
    low = " " + re.sub(r"\s+", " ", query.strip().lower()) + " "
    for pat in _CONTROL_WRITE_PATTERNS:
        if pat in low:
            return True
    return False


def classify_intent(query: str, preset: Optional[str], refused_control_write: bool) -> Tuple[str, Optional[str]]:
    """Classify deterministic intent and optional focus DAG node (N1..N8)."""
    if refused_control_write:
        return "control_refusal", None

    focus_node: Optional[str] = None
    if query:
        m = _NODE_ID_RE.search(query)
        if m:
            focus_node = m.group(1).upper()

    if preset and preset in VALID_PRESETS:
        return preset, focus_node

    low = query.strip().lower()
    if not low:
        return "why_alert", focus_node

    if focus_node == "N3":
        return "sensor_safety", focus_node
    if focus_node == "N8":
        return "economics_pm", focus_node
    if focus_node in ("N1", "N2", "N4", "N5", "N6", "N7"):
        return "root_cause", focus_node

    if any(
        k in low
        for k in (
            "field ai",
            "vision",
            "yolo",
            "camera",
            "iphone",
            "rotameter",
            "nozzle",
            "spray",
            "fft",
            "belt",
            "bearing",
            "tachometer",
            "stull",
            "turbidity",
            "ntu",
            "cuvette",
            "hardware",
            "mini-ct1",
            "vòi phun",
            "voi phun",
            "dây curoa",
            "day curoa",
            "bạc đạn",
            "bac dan",
            "độ đục",
            "do duc",
            "phần cứng",
            "phan cung",
        )
    ):
        return "field_ai", focus_node
    if any(k in low for k in ("water", "co2", "scope 2", "esg", "blowdown", "coc", "evap", "nước", "nuoc", "khí thải", "khi thai", "môi trường", "moi truong")):
        return "water_esg", focus_node
    if any(k in low for k in ("roi", "evn", "tou", "payback", "break-even", "breakeven", "eoq", "pm cost", "cost", "money", "tariff", "clean now", "defer", "tiền điện", "tien dien", "hoàn vốn", "hoan von", "chi phí", "chi phi", "bảo trì", "bao tri", "vệ sinh", "ve sinh")):
        return "economics_pm", focus_node
    if any(k in low for k in ("root", "dag", "causal", "cause", "confounder", "wet-bulb", "wetbulb", "pump", "flow", "nguyên nhân", "nguyen nhan", "gốc", "goc", "bơm", "bom")):
        return "root_cause", focus_node
    if any(k in low for k in ("brief", "handover", "shift", "checklist", "sop", "triage", "ashrae", "semi", "bàn giao", "ban giao", "ca trực", "ca truc", "quy trình", "quy trinh", "kiểm tra", "kiem tra")):
        return "engineer_brief", focus_node
    if any(k in low for k in ("exec", "5m1e", "cfo", "summary", "judge", "pilot", "tổng quan", "tong quan", "giám đốc", "giam doc", "lãnh đạo", "lanh dao")):
        return "exec_5m1e", focus_node
    if any(k in low for k in ("gate", "ood", "sensor", "flatline", "abstain", "safety", "iec", "air-gap", "airgap", "security", "cảm biến", "cam bien", "cổng an toàn", "cong an toan", "bảo mật", "bao mat")):
        return "sensor_safety", focus_node
    if any(k in low for k in ("why", "alert", "status", "score", "hgb", "fan", "tại sao", "tai sao", "cảnh báo", "canh bao", "báo lỗi", "bao loi", "trạng thái", "trang thai")):
        return "why_alert", focus_node

    return "diagnostic", focus_node


# ---------------------------------------------------------------------------
# 6 Deterministic Grounding Tools
# ---------------------------------------------------------------------------

def tool_check_safety_gates(snapshot: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """[T1:Gate] Inspect Gates 1-4, ABSTAIN state, OOD bounds, and Read-Only Conduit."""
    status = str(snapshot.get("status", "NO_ALERT"))
    pred = snapshot.get("prediction", {}) or {}
    sig = snapshot.get("signals", {}) or {}
    injection = str(snapshot.get("injection", "none"))
    is_abstain = status == "ABSTAIN" or pred.get("quality_status") == "ABSTAIN"
    ood_flag = bool(pred.get("ood_flag", False))
    ood_features = list(pred.get("ood_features", []) or [])
    reasons = list(pred.get("reasons", []) or [])

    payload = {
        "status": status,
        "quality_status": pred.get("quality_status", "ABSTAIN"),
        "is_abstain": is_abstain,
        "injection": injection,
        "ood_flag": ood_flag,
        "ood_score": float(pred.get("ood_score", 0.0)),
        "ood_features": ood_features,
        "reasons": reasons,
        "ct1_on": int(sig.get("ct1_on", 0)),
        "tower_flow_gpm": float(sig.get("tower_flow_gpm", 0.0)),
        "fan_pct": float(sig.get("fan_pct", 0.0)),
        "logged_load_kw": float(sig.get("logged_load_kw", 0.0)),
        "train_flow_max_gpm": 601.1,
        "read_only": bool(snapshot.get("read_only", True)),
        "controls": bool(snapshot.get("controls", False)),
    }
    tool_status = "BLOCKED" if is_abstain else "PASS"
    if is_abstain:
        summary = f"ABSTAIN ({'OOD ' + ','.join(ood_features) if ood_flag else injection or 'Gate'}) · controls=False"
    else:
        summary = f"Gates 1–4 PASS · Flow {payload['tower_flow_gpm']:.1f} GPM · Fan {payload['fan_pct']:.1f}%"
    return tool_status, summary, payload


def tool_score_quan_hgb(snapshot: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """[T2:HGB] Inspect frozen Quan HistGradientBoostingClassifier score, persistence & sensitivity."""
    pred = snapshot.get("prediction", {}) or {}
    metrics = snapshot.get("metrics", {}) or {}
    ai_m = metrics.get("AI", {}) or {}
    rule_m = metrics.get("rule", {}) or {}
    per_case = ai_m.get("per_case", {}) or {}

    sens = list(pred.get("feature_sensitivity", []) or [])[:3]
    top_features = [
        {
            "feature": item.get("feature"),
            "label": item.get("label"),
            "delta_score": item.get("delta_score"),
            "current_value": item.get("current_value"),
        }
        for item in sens
    ]

    ai_score = pred.get("ai_score")
    ai_alert = bool(pred.get("ai_alert", False))
    rule_alert = bool(pred.get("rule_alert", False))
    is_abstain = pred.get("quality_status") == "ABSTAIN"

    payload = {
        "model": "Quan HistGradientBoostingClassifier (frozen)",
        "model_sha256_short": str(pred.get("model_sha256", "6eeda35f"))[:8],
        "agreement": str(snapshot.get("agreement", "NO_ALERT")),
        "ai_score": round(float(ai_score), 4) if ai_score is not None else None,
        "ai_threshold": float(pred.get("ai_threshold", 0.70)),
        "ai_run_minutes": int(pred.get("ai_run_minutes", 0)),
        "ai_required_minutes": int(pred.get("ai_required_minutes", 5)),
        "ai_alert": ai_alert,
        "rule_threshold_pct": float(pred.get("rule_threshold_pct", 98.0)),
        "rule_run_minutes": int(pred.get("rule_run_minutes", 0)),
        "rule_required_minutes": int(pred.get("rule_required_minutes", 15)),
        "rule_alert": rule_alert,
        "score_is_calibrated_probability": False,
        "top_features": top_features,
        "benchmark_ai_f1": round(float(ai_m.get("f1", 0.3941)), 3),
        "benchmark_rule_f1": round(float(rule_m.get("f1", 0.2372)), 3),
        "benchmark_ai_precision_pct": round(float(ai_m.get("precision", 0.9782)) * 100.0, 1),
        "benchmark_recall_065_pct": round(float(per_case.get("tower_fouling_065", {}).get("recall", 0.4653)) * 100.0, 1),
        "benchmark_recall_095_pct": round(float(per_case.get("tower_fouling_095", {}).get("recall", 0.0446)) * 100.0, 1),
        "severity_supported": False,
        "forecast_supported": False,
    }
    if is_abstain:
        tool_status = "ABSTAIN"
        summary = f"Suspended (ABSTAIN) · Frozen HGB {payload['model_sha256_short']}"
    elif ai_alert or rule_alert:
        tool_status = "ALERT"
        sc_txt = f"{payload['ai_score']:.3f}" if payload["ai_score"] is not None else "N/A"
        summary = f"HGB {sc_txt} ({payload['ai_run_minutes']}/{payload['ai_required_minutes']}m) · Rule {payload['rule_run_minutes']}/{payload['rule_required_minutes']}m"
    else:
        tool_status = "NOMINAL"
        sc_txt = f"{payload['ai_score']:.3f}" if payload["ai_score"] is not None else "N/A"
        summary = f"No alert · HGB {sc_txt} vs 0.70 ({payload['ai_run_minutes']}/{payload['ai_required_minutes']}m)"
    return tool_status, summary, payload


def tool_trace_causal_dag(snapshot: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """[T3:DAG] Trace 8-node Causal Root-Cause Graph isolation and active propagation edges."""
    cg = snapshot.get("causal_graph") or build_causal_graph(snapshot)
    nodes = list(cg.get("nodes", []) or [])
    edges = list(cg.get("edges", []) or [])
    active_paths = [f"{e['from']}->{e['to']}" for e in edges if e.get("active")]
    blocked_paths = [f"{e['from']}->{e['to']}" for e in edges if e.get("blocked")]
    node_map = {
        n["id"]: {"short": n["short"], "state": n["state"], "metric": n["metric"], "detail": n["detail"]}
        for n in nodes
    }
    iso_id = str(cg.get("isolated_root_id", "NONE"))
    payload = {
        "isolated_root_id": iso_id,
        "isolated_root_label": str(cg.get("isolated_root_label", "No Active Fault Isolated")),
        "isolation_summary": str(cg.get("isolation_summary", "")),
        "confounder_warning": str(cg.get("confounder_warning", "")),
        "active_edges": active_paths,
        "blocked_edge_count": len(blocked_paths),
        "nodes": node_map,
    }
    if iso_id == "N3":
        tool_status = "BLOCKED"
        summary = f"Isolated N3 (Gate Block) · {len(blocked_paths)} edges blocked"
    elif iso_id in ("N1", "N2"):
        tool_status = "ISOLATED"
        summary = f"Root {iso_id} ({payload['isolated_root_label']}) · {len(active_paths)} active edges"
    else:
        tool_status = "CALM"
        summary = "No active fault isolated (N4 confounder check retained)"
    return tool_status, summary, payload


def tool_calc_evn_tou_economics(snapshot: Dict[str, Any], basis_mode: str) -> Tuple[str, str, Dict[str, Any]]:
    """[T4:EVN-TOU] Extract EVN 22kV TOU Break-Even PM Optimizer economics for live or illustrative basis."""
    elec = snapshot.get("electricity", {}) or {}
    sim = (
        elec.get("live_alert_projection", {})
        if basis_mode == "live"
        else elec.get("illustrative_scenario", {})
    ) or {}
    kpis = sim.get("kpis_5m1e", {}) or {}
    money = kpis.get("money", {}) or {}
    pm = sim.get("pm_optimizer", {}) or {}
    tou = pm.get("tou_avoided_vnd", {}) or {}

    verdict = str(pm.get("verdict", "STANDBY_NO_ACTION"))
    payload = {
        "basis_mode": basis_mode,
        "basis_code": str(money.get("basis_code", "QĐ 2892/BCT · 22kV TOU")),
        "eff_kw": float(sim.get("imposed_excess_kw", 0.0)),
        "instant_delta_kw": float(sim.get("instant_delta_kw", elec.get("signed_delta_kw_diagnostic", 0.0))),
        "days_saved": int(sim.get("days_saved", 92)),
        "earlier_pm_day": int(sim.get("earlier_pm_day", 28)),
        "existing_pm_day": int(sim.get("existing_pm_day", 120)),
        "restored_fraction": float(sim.get("restored_fraction", 1.0)),
        "scenario_avoided_excess_kwh": float(sim.get("scenario_avoided_excess_kwh", 0.0)),
        "tariff_vnd_kwh": float(money.get("tariff_vnd_kwh", 1925.93)),
        "fx_vnd_usd": float(money.get("fx_vnd_usd", 25980.0)),
        "vat_rate": float(money.get("vat_rate", 0.08)),
        "pm_cost_vnd": float(pm.get("pm_cost_vnd", 8_000_000.0)),
        "daily_burn_vnd": float(pm.get("daily_burn_vnd", 0.0)),
        "avoided_gross_vnd": float(pm.get("avoided_gross_vnd", 0.0)),
        "avoided_incl_vat8_vnd": float(money.get("avoided_incl_vat8_vnd", 0.0)),
        "avoided_ex_vat_usd": float(money.get("avoided_ex_vat_usd", 0.0)),
        "net_benefit_vnd": float(pm.get("net_benefit_vnd", 0.0)),
        "roi_pct": float(pm.get("roi_pct", 0.0)),
        "payback_days": pm.get("payback_days"),
        "payback_calendar_day": pm.get("payback_calendar_day"),
        "latest_profitable_day": pm.get("latest_profitable_day"),
        "eoq_sensitivity_days": pm.get("eoq_sensitivity_days"),
        "tou_peak_vnd": float(tou.get("peak", 0.0)),
        "tou_normal_vnd": float(tou.get("normal", 0.0)),
        "tou_off_peak_vnd": float(tou.get("off_peak", 0.0)),
        "peak_bill_share_pct": float(money.get("peak_bill_share_pct", 31.04)),
        "verdict": verdict,
        "verdict_label": str(pm.get("verdict_label", verdict)),
        "recommended_window": str(pm.get("recommended_window", "Sunday / Off-Peak 00:00–06:00 (1,166 ₫/kWh)")),
        "verified_excess_kwh": None,
        "verified_avoided_kwh": None,
    }
    if verdict == "LOCKED_ABSTAIN":
        tool_status = "LOCKED"
        summary = "Locked on ABSTAIN (0 ₫)"
    elif verdict == "CLEAN_NOW_POSITIVE_ROI":
        tool_status = "POSITIVE_ROI"
        summary = f"CLEAN NOW · Net {_fmt_vnd_m(payload['net_benefit_vnd'])} ({payload['roi_pct']:+.1f}% ROI, PB {payload['payback_days']}d)"
    elif verdict == "DEFER_PM_NEGATIVE_ROI":
        tool_status = "DEFER"
        summary = f"DEFER PM · Net {_fmt_vnd_m(payload['net_benefit_vnd'])} (PB {payload['payback_days']}d > D{payload['existing_pm_day']})"
    else:
        tool_status = "STANDBY"
        summary = f"Standby 0 ₫/d (PM cost {_fmt_vnd_m(payload['pm_cost_vnd'])})"
    return tool_status, summary, payload


def tool_calc_scope2_and_water(snapshot: Dict[str, Any], basis_mode: str) -> Tuple[str, str, Dict[str, Any]]:
    """[T5:ESG] Extract Scope 2 CO2e (CV 11232/BNNMT), ASHRAE live water sensitivity, and Time KPIs."""
    elec = snapshot.get("electricity", {}) or {}
    sim = (
        elec.get("live_alert_projection", {})
        if basis_mode == "live"
        else elec.get("illustrative_scenario", {})
    ) or {}
    kpis = sim.get("kpis_5m1e", {}) or {}
    esg = kpis.get("esg", {}) or {}
    wp = esg.get("water_physics_est", {}) or {}
    tm = kpis.get("time", {}) or {}

    suspended = bool(wp.get("suspended", False))
    payload = {
        "basis_mode": basis_mode,
        "ef_basis_code": str(esg.get("ef_basis_code", "CV 11232/BNNMT · 2024")),
        "grid_ef_kg_kwh": float(esg.get("grid_ef_kg_kwh", 0.6187)),
        "avoided_co2_kg": float(esg.get("avoided_co2_kg", 0.0)),
        "avoided_co2_tonnes": float(esg.get("avoided_co2_tonnes", 0.0)),
        "daily_co2_kg": float(esg.get("daily_co2_kg", 0.0)),
        "trees_yr_eq": int(esg.get("trees_yr_eq", 0)),
        "verified_water_savings": None,
        "water_suspended": suspended,
        "live_evap_m3_day": float(wp.get("live_evap_m3_day", 0.0)),
        "live_makeup_coc4_m3_day": float(wp.get("live_makeup_coc4_m3_day", 0.0)),
        "thermal_makeup_m3_est": float(wp.get("thermal_makeup_m3_est", 0.0)),
        "coc_4to3_penalty_m3_est": float(wp.get("coc_4to3_penalty_m3_est", 0.0)),
        "coc_5to3_penalty_m3_est": float(wp.get("coc_5to3_penalty_m3_est", 0.0)),
        "fouling_hours_avoided": int(tm.get("fouling_hours_avoided", 0)),
        "ai_median_window_min": float(tm.get("ai_median_window_min", 0.0)),
        "rule_median_window_min": float(tm.get("rule_median_window_min", 0.0)),
        "window_speedup_min": float(tm.get("window_speedup_min", 0.0)),
        "far_ep_per_hr": float(tm.get("far_ep_per_hr", 0.096)),
    }
    if suspended:
        tool_status = "SUSPENDED"
        summary = "Suspended on ABSTAIN · verified_water_savings=None"
    else:
        tool_status = "ESTIMATED"
        summary = f"CO₂ {payload['avoided_co2_tonnes']:.2f}t · Live Evap {payload['live_evap_m3_day']:.1f} m³/d* (Verified=null)"
    return tool_status, summary, payload


def tool_retrieve_local_sop(
    intent: str,
    refused_control_write: bool,
    abstain_blocked: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    """[T6:SOP] Retrieve deterministic local SOP records matching intent & safety state."""
    if refused_control_write:
        keys = ["SOP-IEC62443", "SOP-ASHRAE12", "SOP-SEMIS23"]
    elif abstain_blocked or intent == "sensor_safety":
        keys = ["SOP-IEC62443", "SOP-LBNL-AUDIT", "SOP-PILOT90"]
    elif intent == "economics_pm":
        keys = ["SOP-EVN-ESG", "SOP-ASHRAE12", "SOP-PILOT90"]
    elif intent == "water_esg":
        keys = ["SOP-EVN-ESG", "SOP-LBNL-AUDIT", "SOP-PILOT90"]
    elif intent == "engineer_brief":
        keys = ["SOP-ASHRAE12", "SOP-SEMIS23", "SOP-IEC62443", "SOP-LBNL-AUDIT"]
    elif intent == "exec_5m1e":
        keys = ["SOP-EVN-ESG", "SOP-IEC62443", "SOP-LBNL-AUDIT", "SOP-PILOT90"]
    else:
        keys = ["SOP-ASHRAE12", "SOP-LBNL-AUDIT", "SOP-SEMIS23", "SOP-IEC62443"]

    matched = [LOCAL_SOP_CORPUS[k] for k in keys]
    payload = {
        "matched_ids": keys,
        "corpus_size": len(LOCAL_SOP_CORPUS),
        "offline_only": True,
        "records": matched,
    }
    return "MATCHED", f"{len(keys)} local SOPs ({', '.join(keys[:3])})", payload


def tool_inspect_field_ai_bench(snapshot: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """[T7:FieldAI] Summarize Benchtop Mini-CT1 Multi-Modal Vision, Acoustic FFT, and Stull Weather Gate."""
    hw = snapshot.get("hardware_summary", {}) or {}
    fc = snapshot.get("field_checks", {}) or {}
    fill_v = str(hw.get("fill_verdict", "OK"))
    spray_v = str(hw.get("spray_verdict", "OK"))
    basin_v = str(hw.get("basin_verdict", "OK"))
    ac_v = str(hw.get("acoustic_verdict", "OK"))
    wx_v = str(hw.get("weather_verdict", "WEATHER_CLEAR"))

    issues = []
    if fill_v != "OK":
        issues.append("Fill/Strainer")
    if spray_v != "OK":
        issues.append("Nozzle/Flow")
    if basin_v != "OK":
        issues.append("Basin/CoC")
    if ac_v != "OK":
        issues.append("Fan/FFT")
    if wx_v != "WEATHER_CLEAR":
        issues.append("Weather")

    payload = {
        "active_preset": str(hw.get("active_preset", "fill_fouling_065")),
        "fill_verdict": fill_v,
        "fill_occlusion_pct": float(hw.get("fill_occlusion_pct", 0.0)),
        "fill_sha_short": str(hw.get("fill_sha_short", "")),
        "spray_verdict": spray_v,
        "optical_flow_gpm": float(hw.get("optical_flow_gpm", 0.0)),
        "healthy_nozzles": int(hw.get("healthy_nozzles", 4)),
        "basin_verdict": basin_v,
        "turbidity_ntu": float(hw.get("turbidity_ntu", 0.0)),
        "estimated_coc": float(hw.get("estimated_coc", 4.0)),
        "acoustic_verdict": ac_v,
        "optical_rpm": float(hw.get("optical_rpm", 0.0)),
        "slip_pct": float(hw.get("slip_pct", 0.0)),
        "weather_verdict": wx_v,
        "stull_wet_bulb_c": float(hw.get("stull_wet_bulb_c", 23.86)),
        "weather_physics_valid": bool(hw.get("weather_physics_valid", True)),
        "step4_applied": bool(fc.get("applied", False)),
        "issues_detected": issues,
        "advisory_only": True,
        "verified_water_savings": None,
    }
    tool_status = "ISSUE" if issues else "OK"
    summary = (
        f"Preset {payload['active_preset']} · Issues: {', '.join(issues)} (advisory)"
        if issues
        else f"Preset {payload['active_preset']} · All 5 Field-AI checks OK"
    )
    return tool_status, summary, payload


def _run_tool(
    tool_id: str,
    tag: str,
    tool_name: str,
    fn: Callable[[], Tuple[str, str, Dict[str, Any]]],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    status, summary, payload = fn()
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "id": tool_id,
        "tag": tag,
        "tool_name": tool_name,
        "status": status,
        "latency_ms": latency_ms,
        "summary": summary,
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Grounded Bilingual Response Composer (Zero Hallucination)
# ---------------------------------------------------------------------------

def _compose_answer(
    lang: str,
    intent: str,
    focus_node: Optional[str],
    refused_control_write: bool,
    abstain_blocked: bool,
    snapshot: Dict[str, Any],
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    p3: Dict[str, Any],
    p4: Dict[str, Any],
    p5: Dict[str, Any],
    p6: Dict[str, Any],
    p7: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    vi = lang == "vi"
    ts = str(snapshot.get("timestamp", ""))
    case_label = str((snapshot.get("case_meta") or {}).get("label", snapshot.get("scenario", "")))
    reasons_str = "; ".join(p1["reasons"]) if p1["reasons"] else "Gate active."
    sc_str = f"{p2['ai_score']:.3f}" if p2["ai_score"] is not None else "None"
    top_feat_str = (
        ", ".join(f"{f['label']} ({f['current_value']}, Δ{f['delta_score']:+.3f})" for f in p2["top_features"])
        if p2["top_features"]
        else ("N/A (ABSTAIN)" if p1["is_abstain"] else "N/A")
    )

    # 1. Control-Write Refusal (Highest Priority — IEC 62443-3-3)
    if refused_control_write:
        if vi:
            title = "TỪ CHỐI LỆNH GHI PLC/BMS — CỔNG CHỈ ĐỌC IEC 62443-3-3"
            md = (
                f"**Từ chối ghi PLC/BMS:** Chặn theo **IEC 62443-3-3 Read-Only** (`read_only={p1['read_only']}`, `controls={p1['controls']}`) `[T1:Gate]` `[T6:SOP]`.\n\n"
                f"- **Trạng thái ({ts} · {case_label}):** `status={p1['status']}`, Flow `{p1['tower_flow_gpm']:.1f} GPM`, Fan `{p1['fan_pct']:.1f}%` `[T1:Gate]`.\n"
                f"- **Quy trình an toàn:** Xác minh Bước I–VI, cô lập N+1 theo **ASHRAE 12** (`{p4['recommended_window']}`), tạo phiếu CMMS thủ công (tách PCW-C/PCW-T **SEMI S23**) `[T4:EVN-TOU]` `[T6:SOP]`."
            )
        else:
            title = "CONTROL WRITE REFUSED — IEC 62443-3-3 READ-ONLY CONDUIT"
            md = (
                f"**Refused:** PLC/BMS actuation blocked by **IEC 62443-3-3 Read-Only Conduit** (`read_only={p1['read_only']}`, `controls={p1['controls']}`) `[T1:Gate]` `[T6:SOP]`.\n\n"
                f"- **Snapshot ({ts} · {case_label}):** `status={p1['status']}`, Flow `{p1['tower_flow_gpm']:.1f} GPM`, Fan `{p1['fan_pct']:.1f}%` `[T1:Gate]`.\n"
                f"- **Manual SOP:** Verify Steps I–VI, isolate N+1 cell (**ASHRAE 12**, `{p4['recommended_window']}`), issue manual CMMS order (**SEMI S23** PCW-C/PCW-T) `[T4:EVN-TOU]` `[T6:SOP]`."
            )
        return title, md

    # 2. ABSTAIN Guard for diagnostic intents
    if abstain_blocked:
        ood_line_en = (
            f"Gate 4 Training-Envelope OOD tripped (`ood_flag={p1['ood_flag']}`, features={p1['ood_features']}, "
            f"`CT_FLOW_1={p1['tower_flow_gpm']:.1f} GPM` vs Jan–Jun max `{p1['train_flow_max_gpm']} GPM`; zero domain remapping)"
            if p1["ood_flag"]
            else f"Gate reason: {reasons_str}"
        )
        ood_line_vi = (
            f"Cổng 4 OOD biên huấn luyện kích hoạt (`ood_flag={p1['ood_flag']}`, đặc trưng={p1['ood_features']}, "
            f"`CT_FLOW_1={p1['tower_flow_gpm']:.1f} GPM` > ngưỡng Jan–Jun `{p1['train_flow_max_gpm']} GPM`; không nội suy ngoài miền)"
            if p1["ood_flag"]
            else f"Lý do cổng an toàn: {reasons_str}"
        )
        if vi:
            title = f"TẠM DỪNG CHẨN ĐOÁN (ABSTAIN) — CỔNG N3 KHÓA AN TOÀN ({ts})"
            md = (
                f"**Tạm dừng chẩn đoán (ABSTAIN):** {ood_line_vi} `[T1:Gate]`.\n\n"
                f"- **DAG:** Chặn tại **{p3['isolated_root_id']} ({p3['isolated_root_label']})** ({p3['blocked_edge_count']} cạnh khóa) `[T3:DAG]`.\n"
                f"- **Kinh tế & Nước:** EVN `{p4['verdict']}` (`0 ₫/ngày`), nước `suspended={p5['water_suspended']}` `[T4:EVN-TOU]` `[T5:ESG]`.\n"
                f"- **SOP:** Kiểm tra cảm biến (`CT_STA_1={p1['ct1_on']}`, Fan `{p1['fan_pct']:.1f}%`, Flow `{p1['tower_flow_gpm']:.1f} GPM`) `[T6:SOP]`."
            )
        else:
            title = f"DIAGNOSIS SUSPENDED (ABSTAIN) — GATE N3 ACTIVE ({ts})"
            md = (
                f"**Diagnosis Suspended (ABSTAIN):** {ood_line_en} `[T1:Gate]`.\n\n"
                f"- **DAG:** Blocked at **{p3['isolated_root_id']} ({p3['isolated_root_label']})** ({p3['blocked_edge_count']} edges locked) `[T3:DAG]`.\n"
                f"- **Economics & Water:** EVN `{p4['verdict']}` (`0 ₫/d`), water `suspended={p5['water_suspended']}` `[T4:EVN-TOU]` `[T5:ESG]`.\n"
                f"- **SOP:** Check transmitter & regime (`CT_STA_1={p1['ct1_on']}`, Fan `{p1['fan_pct']:.1f}%`, Flow `{p1['tower_flow_gpm']:.1f} GPM`) per `{', '.join(p6['matched_ids'])}` `[T6:SOP]`."
            )
        return title, md

    # 3. Root-Cause DAG Breakdown
    if intent == "root_cause":
        focus_block = ""
        if focus_node and focus_node in p3["nodes"]:
            fn = p3["nodes"][focus_node]
            if vi:
                focus_block = f"\n- **Nút được chọn `{focus_node}` ({fn['short']}):** Trạng thái `{fn['state']}` · `{fn['metric']}` — {fn['detail']} `[T3:DAG]`."
            else:
                focus_block = f"\n- **Focused Node `{focus_node}` ({fn['short']}):** State `{fn['state']}` · `{fn['metric']}` — {fn['detail']} `[T3:DAG]`."
        active_chain = ", ".join(p3["active_edges"]) if p3["active_edges"] else ("Không có cạnh lan truyền" if vi else "None (calm)")
        if vi:
            title = f"PHÂN TÍCH NGUYÊN NHÂN GỐC (CAUSAL DAG) — {p3['isolated_root_id']}: {p3['isolated_root_label']}"
            md = (
                f"**Kết quả cô lập DAG ({ts} · {case_label}):** Nguyên nhân gốc cô lập = **{p3['isolated_root_id']} ({p3['isolated_root_label']})**; chuỗi lan truyền hoạt động: `{active_chain}` `[T3:DAG]`.\n\n"
                f"- **Bằng chứng lõi AI đóng băng (`{p2['model_sha256_short']}`):** HGB score=`{sc_str}` vs ngưỡng `{p2['ai_threshold']:.2f}` (`{p2['ai_run_minutes']}/{p2['ai_required_minutes']}m`), High-Fan Rule=`{p1['fan_pct']:.1f}%` (`{p2['rule_run_minutes']}/{p2['rule_required_minutes']}m`), đồng thuận=`{p2['agreement']}`; độ nhạy cục bộ hàng đầu: {top_feat_str} `[T2:HGB]`.\n"
                f"- **Loại trừ nguyên nhân cạnh tranh:** `N2 (CW Flow)` = `{p3['nodes']['N2']['state']}` (`{p1['tower_flow_gpm']:.1f} GPM`), `N3 (Sensor/OOD Gate)` = `{p3['nodes']['N3']['state']}` (`{p1['quality_status']}`) `[T1:Gate]` `[T3:DAG]`.\n"
                f"- **Cảnh báo trung thực N4 (Weather/CHW Confounder*):** {p3['confounder_warning']} `[T3:DAG]` `[T6:SOP]`."
                f"{focus_block}"
            )
        else:
            title = f"CAUSAL ROOT-CAUSE GRAPH (DAG) — {p3['isolated_root_id']}: {p3['isolated_root_label']}"
            md = (
                f"**DAG Isolation ({ts} · {case_label}):** Isolated root = **{p3['isolated_root_id']} ({p3['isolated_root_label']})**; active propagation chain: `{active_chain}` `[T3:DAG]`.\n\n"
                f"- **Frozen Quan HGB Evidence (`{p2['model_sha256_short']}`):** Score=`{sc_str}` vs `{p2['ai_threshold']:.2f}` (`{p2['ai_run_minutes']}/{p2['ai_required_minutes']}m`), High-Fan Rule=`{p1['fan_pct']:.1f}%` (`{p2['rule_run_minutes']}/{p2['rule_required_minutes']}m`), agreement=`{p2['agreement']}`; top local sensitivities: {top_feat_str} `[T2:HGB]`.\n"
                f"- **Competing-Cause Elimination:** `N2 (CW Pump/Flow)` = `{p3['nodes']['N2']['state']}` (`{p1['tower_flow_gpm']:.1f} GPM`), `N3 (Sensor/OOD Gate)` = `{p3['nodes']['N3']['state']}` (`{p1['quality_status']}`) `[T1:Gate]` `[T3:DAG]`.\n"
                f"- **N4 Confounder Honesty:** {p3['confounder_warning']} `[T3:DAG]` `[T6:SOP]`."
                f"{focus_block}"
            )
        return title, md

    # 4. EVN TOU Break-Even PM Optimizer
    if intent == "economics_pm":
        pb_txt = f"{p4['payback_days']}d (Day {p4['payback_calendar_day']})" if p4["payback_days"] is not None else "N/A"
        cut_txt = f"Day {p4['latest_profitable_day']}" if p4["latest_profitable_day"] is not None else "N/A"
        eoq_txt = f"{p4['eoq_sensitivity_days']}d" if p4["eoq_sensitivity_days"] is not None else "N/A"
        if vi:
            title = f"TỐI ƯU ĐIỂM HÒA VỐN EVN TOU — {p4['verdict_label']}"
            md = (
                f"**Quyết định tài chính minh họa (`{p4['basis_mode']}` · {p4['basis_code']}):** **{p4['verdict']}** — đơn giá 22kV TOU 24/7 chưa VAT = `{p4['tariff_vnd_kwh']:.2f} ₫/kWh` (tỷ giá `{p4['fx_vnd_usd']:,.0f} ₫/USD`, VAT `{int(p4['vat_rate']*100)}%`) `[T4:EVN-TOU]` `[T6:SOP]`.\n\n"
                f"- **Công suất & Chi phí điện tránh được (Minh họa):** ΔkW hiệu dụng=`{p4['eff_kw']:.2f} kW*`, tiết kiệm `{p4['days_saved']} ngày` (Ngày {p4['earlier_pm_day']} vs Ngày {p4['existing_pm_day']}, khôi phục `{int(p4['restored_fraction']*100)}%`) → `{p4['scenario_avoided_excess_kwh']:,.1f} kWh`, tổn thất ngày=`{_fmt_vnd(p4['daily_burn_vnd'])}/ngày`, gộp chưa VAT=`{_fmt_vnd(p4['avoided_gross_vnd'])}` (`~${p4['avoided_ex_vat_usd']:,.0f}`) `[T4:EVN-TOU]`.\n"
                f"- **Phân rã TOU 3 khung giờ:** Cao điểm (`{p4['peak_bill_share_pct']}%`) = `{_fmt_vnd(p4['tou_peak_vnd'])}`, Bình thường = `{_fmt_vnd(p4['tou_normal_vnd'])}`, Thấp điểm = `{_fmt_vnd(p4['tou_off_peak_vnd'])}` `[T4:EVN-TOU]`.\n"
                f"- **Hòa vốn & ROI (Chi phí vệ sinh PM = `{_fmt_vnd(p4['pm_cost_vnd'])}`):** Lợi ích ròng = **`{_fmt_vnd_m(p4['net_benefit_vnd'])}`** (`ROI {p4['roi_pct']:+.1f}%`), Hoàn vốn = `{pb_txt}`, Hạn chót có lãi = `{cut_txt}`, Chu kỳ EOQ tuyến tính `t*` = `{eoq_txt}` `[T4:EVN-TOU]`.\n"
                f"- **Khung giờ PM:** `{p4['recommended_window']}` (minh họa; chờ Pilot 90 ngày) `[T1:Gate]` `[T4:EVN-TOU]` `[T6:SOP]`."
            )
        else:
            title = f"EVN TOU BREAK-EVEN PM OPTIMIZER — {p4['verdict_label']}"
            md = (
                f"**Illustrative Financial Verdict (`{p4['basis_mode']}` · {p4['basis_code']}):** **{p4['verdict']}** — 22kV 24/7 weighted TOU ex-VAT = `{p4['tariff_vnd_kwh']:.2f} VND/kWh` (FX `{p4['fx_vnd_usd']:,.0f} VND/USD`, VAT `{int(p4['vat_rate']*100)}%`) `[T4:EVN-TOU]` `[T6:SOP]`.\n\n"
                f"- **Illustrative Avoided Electricity:** Effective excess=`{p4['eff_kw']:.2f} kW*` over `{p4['days_saved']} days` (Day {p4['earlier_pm_day']} vs Day {p4['existing_pm_day']}, `{int(p4['restored_fraction']*100)}%` restoration) → `{p4['scenario_avoided_excess_kwh']:,.1f} kWh`, daily burn=`{_fmt_vnd(p4['daily_burn_vnd'])}/d`, gross ex-VAT=`{_fmt_vnd(p4['avoided_gross_vnd'])}` (`~${p4['avoided_ex_vat_usd']:,.0f}`) `[T4:EVN-TOU]`.\n"
                f"- **3-Tier EVN TOU Breakdown:** Peak (`{p4['peak_bill_share_pct']}%`) = `{_fmt_vnd(p4['tou_peak_vnd'])}`, Normal = `{_fmt_vnd(p4['tou_normal_vnd'])}`, Off-Peak = `{_fmt_vnd(p4['tou_off_peak_vnd'])}` `[T4:EVN-TOU]`.\n"
                f"- **Break-Even & ROI (PM Cost `{_fmt_vnd(p4['pm_cost_vnd'])}`):** Net Benefit = **`{_fmt_vnd_m(p4['net_benefit_vnd'])}`** (`{p4['roi_pct']:+.1f}% ROI`), Payback = `{pb_txt}`, Latest Profitable Cutoff = `{cut_txt}`, EOQ Sensitivity `t*` = `{eoq_txt}` `[T4:EVN-TOU]`.\n"
                f"- **PM Window:** `{p4['recommended_window']}` (illustrative; Pilot metering req.) `[T1:Gate]` `[T4:EVN-TOU]` `[T6:SOP]`."
            )
        return title, md

    # 5. Water & Scope 2 ESG
    if intent == "water_esg":
        if vi:
            title = f"ESG PHẠM VI 2 (CO₂) & ƯỚC LƯỢNG NƯỚC ASHRAE ({p5['ef_basis_code']})"
            md = (
                f"**Phát thải Scope 2 ({p5['ef_basis_code']}):** Hệ số lưới Việt Nam 2024 = `{p5['grid_ef_kg_kwh']:.4f} kg CO₂/kWh` → tránh phát thải minh họa **`{p5['avoided_co2_tonnes']:.2f} tấn CO₂e`** (`{p5['avoided_co2_kg']:,.1f} kg`, tốc độ `{p5['daily_co2_kg']:.1f} kg/ngày`, tương đương `{p5['trees_yr_eq']} cây xanh/năm`) từ `{p4['scenario_avoided_excess_kwh']:,.1f} kWh` `[T4:EVN-TOU]` `[T5:ESG]` `[T6:SOP]`.\n\n"
                f"- **Ước lượng vật lý nước ASHRAE/CTI (Độ nhạy trực tiếp · Chưa kiểm chứng):** `water_suspended={p5['water_suspended']}`, bay hơi trực tiếp = `{p5['live_evap_m3_day']:.2f} m³/ngày`, nước cấp CoC=4 = `{p5['live_makeup_coc4_m3_day']:.2f} m³/ngày`, rủi ro xả đáy CoC 4→3 = `{p5['coc_4to3_penalty_m3_est']:.1f} m³`, CoC 5→3 = `{p5['coc_5to3_penalty_m3_est']:.1f} m³`, bổ sung nhiệt = `{p5['thermal_makeup_m3_est']:.2f} m³` `[T5:ESG]`.\n"
                f"- **Kỷ luật 3-Bucket:** `verified_water_savings=None` vì bộ dữ liệu LBNL không có đồng hồ đo nước cấp/xả đáy WTP; bắt buộc đo thực tế tại Pilot 90 ngày `[T1:Gate]` `[T5:ESG]` `[T6:SOP]`."
            )
        else:
            title = f"SCOPE 2 CO₂ & ASHRAE WATER SENSITIVITY ({p5['ef_basis_code']})"
            md = (
                f"**Scope 2 Carbon ({p5['ef_basis_code']}):** Vietnam 2024 Grid EF = `{p5['grid_ef_kg_kwh']:.4f} kg CO₂/kWh` → illustrative avoided emissions **`{p5['avoided_co2_tonnes']:.2f} t CO₂e`** (`{p5['avoided_co2_kg']:,.1f} kg`, `{p5['daily_co2_kg']:.1f} kg/day`, `~{p5['trees_yr_eq']} trees·yr`) from `{p4['scenario_avoided_excess_kwh']:,.1f} kWh` `[T4:EVN-TOU]` `[T5:ESG]` `[T6:SOP]`.\n\n"
                f"- **ASHRAE/CTI Live Flow×ΔT Water Sensitivity (Unverified):** `water_suspended={p5['water_suspended']}`, live evaporation = `{p5['live_evap_m3_day']:.2f} m³/d`, makeup at CoC=4 = `{p5['live_makeup_coc4_m3_day']:.2f} m³/d`, blowdown penalty CoC 4→3 = `{p5['coc_4to3_penalty_m3_est']:.1f} m³`, CoC 5→3 = `{p5['coc_5to3_penalty_m3_est']:.1f} m³`, thermal makeup est. = `{p5['thermal_makeup_m3_est']:.2f} m³` `[T5:ESG]`.\n"
                f"- **3-Bucket Discipline:** `verified_water_savings=None` because published LBNL CSVs lack WTP makeup/blowdown sub-meters; calibrated site sub-metering is a mandatory 90-day Pilot gate `[T1:Gate]` `[T5:ESG]` `[T6:SOP]`."
            )
        return title, md

    # 6. Sensor Safety & IEC 62443 Audit
    if intent == "sensor_safety":
        if vi:
            title = f"KIỂM TOÁN CỔNG AN TOÀN 1–4 & BẢO MẬT OT ({p1['status']})"
            md = (
                f"**Trạng thái Cổng An Toàn ({ts}):** `status={p1['status']}`, `quality_status={p1['quality_status']}`, `ood_flag={p1['ood_flag']}`, `ood_features={p1['ood_features']}`, `read_only={p1['read_only']}`, `controls={p1['controls']}` `[T1:Gate]`.\n\n"
                f"- **4 Cổng bảo vệ lõi:** (1) Biên vật lý (`Fan={p1['fan_pct']:.1f}%` trong `[0, 101%]`), (2) Đóng băng cảm biến 60 phút, (3) Chế độ hoạt động (`CT_STA_1={p1['ct1_on']}`, `Flow={p1['tower_flow_gpm']:.1f} GPM`, `Load={p1['logged_load_kw']:.1f} kW`), (4) Biên huấn luyện Jan–Jun (`CT_FLOW_1 <= {p1['train_flow_max_gpm']} GPM`) `[T1:Gate]`.\n"
                f"- **Lõi AI & Causal DAG:** Mô hình đóng băng `{p2['model_sha256_short']}` (`score_is_calibrated_probability=False`, `severity_supported=False`, `forecast_supported=False`); nút cổng `N3` = `{p3['nodes']['N3']['state']}` `[T2:HGB]` `[T3:DAG]`.\n"
                f"- **Chuẩn an toàn nhà máy bán dẫn:** Vận hành ngoại tuyến hoàn toàn tại Purdue Level 3, IEC 62443-3-3 Read-Only Conduit, không hoán đổi cảm biến tự động (`OA_TEMP_WB` & CHW loại trừ) `[T6:SOP]`."
            )
        else:
            title = f"SAFETY GATES 1–4 & OT SECURITY AUDIT ({p1['status']})"
            md = (
                f"**Safety Gate Status ({ts}):** `status={p1['status']}`, `quality_status={p1['quality_status']}`, `ood_flag={p1['ood_flag']}`, `ood_features={p1['ood_features']}`, `read_only={p1['read_only']}`, `controls={p1['controls']}` `[T1:Gate]`.\n\n"
                f"- **4-Gate Interlock:** (1) Physical bounds (`Fan={p1['fan_pct']:.1f}%` within `[0, 101%]`), (2) 60-min sensor flatline check, (3) Active operating regime (`CT_STA_1={p1['ct1_on']}`, `Flow={p1['tower_flow_gpm']:.1f} GPM`, `Load={p1['logged_load_kw']:.1f} kW`), (4) Jan–Jun padded training envelope (`CT_FLOW_1 <= {p1['train_flow_max_gpm']} GPM`, zero domain remapping) `[T1:Gate]`.\n"
                f"- **Frozen Model & DAG State:** Frozen HGB `{p2['model_sha256_short']}` (`score_is_calibrated_probability=False`, `severity_supported=False`, `forecast_supported=False`); Gate node `N3` = `{p3['nodes']['N3']['state']}` `[T2:HGB]` `[T3:DAG]`.\n"
                f"- **OT Compliance:** Air-gapped Purdue Level 3 execution, IEC 62443-3-3 Read-Only Conduit, zero auto-swap of unresolved LBNL weather/CHW channels `[T6:SOP]`."
            )
        return title, md

    # 7. Engineer Shift-Supervisor Handover Brief
    if intent == "engineer_brief":
        pb_txt = f"{p4['payback_days']}d (D{p4['payback_calendar_day']})" if p4["payback_days"] is not None else "N/A"
        if vi:
            title = f"PHIẾU BÀN GIAO CA TRỰC KỸ SƯ — {ts} ({case_label})"
            md = (
                f"**1. Tóm tắt hiện trường:** `status={p1['status']}`, `agreement={p2['agreement']}`, Quan HGB=`{sc_str}` (`{p2['ai_run_minutes']}/{p2['ai_required_minutes']}m`), Fan=`{p1['fan_pct']:.1f}%` (`{p2['rule_run_minutes']}/{p2['rule_required_minutes']}m`), Flow=`{p1['tower_flow_gpm']:.1f} GPM`, Load=`{p1['logged_load_kw']:.1f} kW` `[T1:Gate]` `[T2:HGB]`.\n"
                f"**2. Cô lập nguyên nhân (DAG):** Gốc = `{p3['isolated_root_id']} ({p3['isolated_root_label']})`; `N2={p3['nodes']['N2']['state']}`, `N3={p3['nodes']['N3']['state']}`, `N4={p3['nodes']['N4']['state']}` (cần kiểm chứng nhiệt độ bầu ướt & CHW tại hiện trường) `[T3:DAG]`.\n"
                f"**3. Quyết định bảo trì EVN TOU:** `{p4['verdict']}` · Tổn thất minh họa `{_fmt_vnd(p4['daily_burn_vnd'])}/ngày`, Lợi ích ròng `{_fmt_vnd_m(p4['net_benefit_vnd'])}` (`ROI {p4['roi_pct']:+.1f}%`, Hoàn vốn `{pb_txt}`) · Khung giờ đề xuất: `{p4['recommended_window']}` `[T4:EVN-TOU]`.\n"
                f"**4. ESG & Nước:** Scope 2 = `{p5['avoided_co2_tonnes']:.2f} t CO₂e`, bay hơi trực tiếp = `{p5['live_evap_m3_day']:.2f} m³/ngày*` (`verified_water_savings=None`) `[T5:ESG]`.\n"
                f"**5. Hành động Bước I–VI (SOP):** Kiểm tra trực quan tấm tản nhiệt/béc phun/lưới lọc CT-1, cô lập N+1 theo **ASHRAE 12**, giữ tách biệt PCW-C vs PCW-T theo **SEMI S23**, ghi nhận đánh giá nội bộ (không ghi PLC) `[T6:SOP]`."
            )
        else:
            title = f"SHIFT SUPERVISOR HANDOVER BRIEF — {ts} ({case_label})"
            md = (
                f"**1. Live Plant State:** `status={p1['status']}`, `agreement={p2['agreement']}`, Quan HGB=`{sc_str}` (`{p2['ai_run_minutes']}/{p2['ai_required_minutes']}m`), Fan=`{p1['fan_pct']:.1f}%` (`{p2['rule_run_minutes']}/{p2['rule_required_minutes']}m`), Flow=`{p1['tower_flow_gpm']:.1f} GPM`, Load=`{p1['logged_load_kw']:.1f} kW` `[T1:Gate]` `[T2:HGB]`.\n"
                f"**2. Causal DAG Isolation:** Root = `{p3['isolated_root_id']} ({p3['isolated_root_label']})`; `N2={p3['nodes']['N2']['state']}`, `N3={p3['nodes']['N3']['state']}`, `N4={p3['nodes']['N4']['state']}` (verify outdoor wet-bulb & CHW on site) `[T3:DAG]`.\n"
                f"**3. EVN TOU PM Decision:** `{p4['verdict']}` · Illustrative burn `{_fmt_vnd(p4['daily_burn_vnd'])}/d`, Net Benefit `{_fmt_vnd_m(p4['net_benefit_vnd'])}` (`{p4['roi_pct']:+.1f}% ROI`, Payback `{pb_txt}`) · Target window: `{p4['recommended_window']}` `[T4:EVN-TOU]`.\n"
                f"**4. ESG & Water:** Scope 2 = `{p5['avoided_co2_tonnes']:.2f} t CO₂e`, live evap = `{p5['live_evap_m3_day']:.2f} m³/d*` (`verified_water_savings=None`) `[T5:ESG]`.\n"
                f"**5. Steps I–VI Action (SOP):** Inspect CT-1 fill/nozzles/CW strainers, isolate N+1 cell per **ASHRAE 12**, preserve **SEMI S23** PCW-C vs PCW-T separation, and log local supervisor verdict (`controls=False`) `[T6:SOP]`."
            )
        return title, md

    # 8. Executive 5M1E Summary
    if intent == "exec_5m1e":
        pb_txt = f"{p4['payback_days']}d" if p4["payback_days"] is not None else "N/A"
        if vi:
            title = f"BÁO CÁO ĐIỀU HÀNH 5M1E — {case_label} ({ts})"
            md = (
                f"- **MONEY (Tài chính EVN 22kV TOU · `{p4['basis_mode']}`):** `{p4['verdict']}` · Tránh tiêu thụ điện dư minh họa `{p4['scenario_avoided_excess_kwh']:,.1f} kWh` = `{_fmt_vnd(p4['avoided_gross_vnd'])}` (`~${p4['avoided_ex_vat_usd']:,.0f}` chưa VAT), Chi phí PM `{_fmt_vnd(p4['pm_cost_vnd'])}` → Lợi ích ròng **`{_fmt_vnd_m(p4['net_benefit_vnd'])}`** (`ROI {p4['roi_pct']:+.1f}%`, Hoàn vốn `{pb_txt}`, Cao điểm chiếm `{p4['peak_bill_share_pct']}%`) `[T4:EVN-TOU]`.\n"
                f"- **TIME (Thời gian phát hiện & Vận hành):** AI F1=`{p2['benchmark_ai_f1']:.3f}` (vs Rule `{p2['benchmark_rule_f1']:.3f}`), Precision=`{p2['benchmark_ai_precision_pct']:.1f}%`; rút ngắn trung vị cửa sổ cảnh báo `{p5['window_speedup_min']:.1f} phút` (`{p5['ai_median_window_min']:.1f}m` vs `{p5['rule_median_window_min']:.1f}m`), tránh `{p5['fouling_hours_avoided']:,} giờ` đóng cặn minh họa, FAR=`{p5['far_ep_per_hr']:.3f}` ep/h `[T2:HGB]` `[T5:ESG]`.\n"
                f"- **ESG & NƯỚC:** Scope 2 (`{p5['grid_ef_kg_kwh']:.4f} kg/kWh`) = **`{p5['avoided_co2_tonnes']:.2f} t CO₂e`** (`~{p5['trees_yr_eq']} cây/năm`); bay hơi ASHRAE = `{p5['live_evap_m3_day']:.2f} m³/ngày*`, rủi ro xả đáy CoC 4→3 = `{p5['coc_4to3_penalty_m3_est']:.1f} m³*` (`verified_water_savings=None`) `[T5:ESG]`.\n"
                f"- **QUẢN TRỊ & AN TOÀN:** Air-gapped IEC 62443-3-3 (`controls=False`, `read_only=True`), giữ trung thực 3-Bucket (`verified_excess_kwh=None`, `verified_avoided_kwh=None`), lộ trình Pilot 90 ngày kiểm chứng thực địa `[T1:Gate]` `[T6:SOP]`."
            )
        else:
            title = f"EXECUTIVE 5M1E SUMMARY — {case_label} ({ts})"
            md = (
                f"- **MONEY (EVN 22kV TOU · `{p4['basis_mode']}`):** `{p4['verdict']}` · Illustrative avoided excess consumption `{p4['scenario_avoided_excess_kwh']:,.1f} kWh` = `{_fmt_vnd(p4['avoided_gross_vnd'])}` (`~${p4['avoided_ex_vat_usd']:,.0f}` ex-VAT), PM Cost `{_fmt_vnd(p4['pm_cost_vnd'])}` → Net Benefit **`{_fmt_vnd_m(p4['net_benefit_vnd'])}`** (`{p4['roi_pct']:+.1f}% ROI`, Payback `{pb_txt}`, Peak share `{p4['peak_bill_share_pct']}%`) `[T4:EVN-TOU]`.\n"
                f"- **TIME (Detection Lead & Triage):** Frozen HGB F1=`{p2['benchmark_ai_f1']:.3f}` (vs Rule `{p2['benchmark_rule_f1']:.3f}`), Precision=`{p2['benchmark_ai_precision_pct']:.1f}%`; `{p5['window_speedup_min']:.1f} min` median window speedup (`{p5['ai_median_window_min']:.1f}m` vs `{p5['rule_median_window_min']:.1f}m`), `{p5['fouling_hours_avoided']:,} h` illustrative fouling exposure avoided, FAR=`{p5['far_ep_per_hr']:.3f}` ep/h `[T2:HGB]` `[T5:ESG]`.\n"
                f"- **ESG & WATER:** Scope 2 (`{p5['grid_ef_kg_kwh']:.4f} kg/kWh`) = **`{p5['avoided_co2_tonnes']:.2f} t CO₂e`** (`~{p5['trees_yr_eq']} trees·yr`); live ASHRAE evap = `{p5['live_evap_m3_day']:.2f} m³/d*`, CoC 4→3 blowdown risk = `{p5['coc_4to3_penalty_m3_est']:.1f} m³*` (`verified_water_savings=None`) `[T5:ESG]`.\n"
                f"- **GOVERNANCE:** Air-gapped IEC 62443-3-3 (`controls=False`, `read_only=True`), 3-Bucket honesty (`verified_excess_kwh=None`, `verified_avoided_kwh=None`), 90-day shadow-mode Site Pilot exit gates `[T1:Gate]` `[T6:SOP]`."
            )
        return title, md

    # 8b. Multi-Modal Field-AI Inspection Summary (Mini-CT1 + iPhone 11)
    if intent == "field_ai" and p7 is not None:
        if vi:
            title = f"KIỂM ĐỊNH AI ĐA PHƯƠNG THỨC MINI-CT1 ({p7['active_preset']})"
            md = (
                f"**Tổng hợp AI Hiện trường (Chỉ tư vấn · `advisory_only=True`, `controls=False`):** Bộ mô phỏng/ảnh iPhone 11 `{p7['active_preset']}` — trạng thái Digital Twin `status={p1['status']}`, DAG gốc `{p3['isolated_root_id']} ({p3['isolated_root_label']})` `[T1:Gate]` `[T3:DAG]` `[T7:FieldAI]`.\n\n"
                f"- **Thị giác Fill & Strainer (H1 / Bước IV-3):** `{p7['fill_verdict']}` · Độ che lấp=`{p7['fill_occlusion_pct']:.1f}%` (SHA `{p7['fill_sha_short']}`) `[T7:FieldAI]`.\n"
                f"- **Vòi phun & Phao Rotameter quang học (H2 / Bước IV-1):** `{p7['spray_verdict']}` · `{p7['healthy_nozzles']}/4` nón phun đạt, lưu lượng quang học=`{p7['optical_flow_gpm']:.1f} GPM` (sàn `250 GPM`) `[T7:FieldAI]`.\n"
                f"- **Tốc độ quạt quang học & Phổ FFT âm thanh 16 kHz (H3 / Bước IV-2):** `{p7['acoustic_verdict']}` · `{p7['optical_rpm']:.0f} RPM`, độ trượt=`{p7['slip_pct']:.1f}%` `[T7:FieldAI]`.\n"
                f"- **Cổng nhiệt kế bầu ướt Stull 2011 (H4) & Độ đục/CoC bể nước (Bước IV-4):** Thời tiết=`{p7['weather_verdict']}` (`T_wb={p7['stull_wet_bulb_c']:.2f}°C`, hợp lệ vật lý=`{p7['weather_physics_valid']}`); nước bể=`{p7['basin_verdict']}` (`{p7['turbidity_ntu']:.1f} NTU`, `CoC={p7['estimated_coc']:.2f}`, `verified_water_savings=None`) `[T5:ESG]` `[T7:FieldAI]`.\n"
                f"- **Bước IV/V:** Đồng bộ Bước IV=`{p7['step4_applied']}`; kỹ sư duyệt Bước V (**IEC 62443**) `[T1:Gate]` `[T6:SOP]`."
            )
        else:
            title = f"MINI-CT1 MULTI-MODAL FIELD-AI AUDIT ({p7['active_preset']})"
            md = (
                f"**Multi-Modal Field-AI Summary (`advisory_only=True`, `controls=False`):** Bench preset `{p7['active_preset']}` alongside Digital Twin `status={p1['status']}` and DAG root `{p3['isolated_root_id']} ({p3['isolated_root_label']})` `[T1:Gate]` `[T3:DAG]` `[T7:FieldAI]`.\n\n"
                f"- **Fill & Strainer Vision (H1 / Step IV-3):** `{p7['fill_verdict']}` · Effective occlusion=`{p7['fill_occlusion_pct']:.1f}%` (SHA `{p7['fill_sha_short']}`) `[T7:FieldAI]`.\n"
                f"- **Spray Cones & Optical Rotameter (H2 / Step IV-1):** `{p7['spray_verdict']}` · `{p7['healthy_nozzles']}/4` healthy cones, optical CW flow=`{p7['optical_flow_gpm']:.1f} GPM` (`250 GPM` floor) `[T7:FieldAI]`.\n"
                f"- **Optical Tachometer & 16 kHz Acoustic FFT (H3 / Step IV-2):** `{p7['acoustic_verdict']}` · `{p7['optical_rpm']:.0f} RPM`, slip=`{p7['slip_pct']:.1f}%` `[T7:FieldAI]`.\n"
                f"- **Stull (2011) Wet-Bulb Gate (H4) & Basin Turbidity/CoC (Step IV-4):** Weather=`{p7['weather_verdict']}` (`T_wb={p7['stull_wet_bulb_c']:.2f}°C`, physics_valid=`{p7['weather_physics_valid']}`); basin=`{p7['basin_verdict']}` (`{p7['turbidity_ntu']:.1f} NTU`, `CoC={p7['estimated_coc']:.2f}`, `verified_water_savings=None`) `[T5:ESG]` `[T7:FieldAI]`.\n"
                f"- **Step IV/V:** Synced=`{p7['step4_applied']}`; human supervisor signs Step V (**IEC 62443**) `[T1:Gate]` `[T6:SOP]`."
            )
        return title, md

    # 9. Default: why_alert / diagnostic
    pb_txt = f"{p4['payback_days']}d (D{p4['payback_calendar_day']})" if p4["payback_days"] is not None else "N/A"
    if vi:
        title = f"GIẢI TRÌNH TRẠNG THÁI {p1['status']} — {ts} ({case_label})"
        md = (
            f"**Chẩn đoán tức thời:** `status={p1['status']}`, `agreement={p2['agreement']}` — Cổng 1–4 `{p1['quality_status']}` (`Flow={p1['tower_flow_gpm']:.1f} GPM`, `Fan={p1['fan_pct']:.1f}%`, `Load={p1['logged_load_kw']:.1f} kW`) `[T1:Gate]`.\n\n"
            f"- **Lõi AI đóng băng Quan HGB (`{p2['model_sha256_short']}`):** Điểm = `{sc_str}` so với ngưỡng `{p2['ai_threshold']:.2f}` (`{p2['ai_run_minutes']}/{p2['ai_required_minutes']}m`), High-Fan Rule = `{p2['rule_run_minutes']}/{p2['rule_required_minutes']}m` (F1 `{p2['benchmark_ai_f1']:.3f}` vs Rule `{p2['benchmark_rule_f1']:.3f}`; recall 0.65 UA=`{p2['benchmark_recall_065_pct']}%`, 0.95 UA=`{p2['benchmark_recall_095_pct']}%`). Đặc trưng nhạy nhất: {top_feat_str} `[T2:HGB]`.\n"
            f"- **Cô lập nguyên nhân gốc (DAG):** **{p3['isolated_root_id']} ({p3['isolated_root_label']})** — `N2={p3['nodes']['N2']['state']}`, `N3={p3['nodes']['N3']['state']}`, `N4={p3['nodes']['N4']['state']}` `[T3:DAG]`.\n"
            f"- **Kinh tế & ESG (`{p4['basis_mode']}`):** `{p4['verdict']}` · `{p4['eff_kw']:.2f} kW*`, `{_fmt_vnd(p4['daily_burn_vnd'])}/ngày`, ròng `{_fmt_vnd_m(p4['net_benefit_vnd'])}` (PB `{pb_txt}`), CO₂ `{p5['avoided_co2_tonnes']:.2f} t` `[T4:EVN-TOU]` `[T5:ESG]`.\n"
            f"- **Tiếp theo:** Xác minh Bước I–IV trước khi lập phiếu **ASHRAE 12** `[T6:SOP]`."
        )
    else:
        title = f"STATUS EXPLANATION: {p1['status']} — {ts} ({case_label})"
        md = (
            f"**Live Assessment:** `status={p1['status']}`, `agreement={p2['agreement']}` — Gates 1–4 `{p1['quality_status']}` (`Flow={p1['tower_flow_gpm']:.1f} GPM`, `Fan={p1['fan_pct']:.1f}%`, `Load={p1['logged_load_kw']:.1f} kW`) `[T1:Gate]`.\n\n"
            f"- **Frozen Quan HGB (`{p2['model_sha256_short']}`):** Score=`{sc_str}` vs `{p2['ai_threshold']:.2f}` (`{p2['ai_run_minutes']}/{p2['ai_required_minutes']}m`), High-Fan Rule=`{p2['rule_run_minutes']}/{p2['rule_required_minutes']}m` (F1 `{p2['benchmark_ai_f1']:.3f}` vs Rule `{p2['benchmark_rule_f1']:.3f}`; recall 0.65 UA=`{p2['benchmark_recall_065_pct']}%`, 0.95 UA=`{p2['benchmark_recall_095_pct']}%`). Top local drivers: {top_feat_str} `[T2:HGB]`.\n"
            f"- **Causal Root-Cause DAG:** Isolated root = **{p3['isolated_root_id']} ({p3['isolated_root_label']})** — `N2={p3['nodes']['N2']['state']}`, `N3={p3['nodes']['N3']['state']}`, `N4={p3['nodes']['N4']['state']}` `[T3:DAG]`.\n"
            f"- **Economics & ESG (`{p4['basis_mode']}`):** `{p4['verdict']}` · `{p4['eff_kw']:.2f} kW*`, `{_fmt_vnd(p4['daily_burn_vnd'])}/d`, net `{_fmt_vnd_m(p4['net_benefit_vnd'])}` (PB `{pb_txt}`), `{p5['avoided_co2_tonnes']:.2f} t CO₂e` `[T4:EVN-TOU]` `[T5:ESG]`.\n"
            f"- **Next:** Verify Steps I–IV before manual **ASHRAE 12** CMMS order `[T6:SOP]`."
        )
    return title, md


def run_airgapped_rca_agent(
    snapshot: Dict[str, Any],
    query: str = "",
    preset: Optional[str] = None,
    basis_mode: str = "live",
    lang: str = "auto",
) -> Dict[str, Any]:
    """Execute the deterministic 6-tool Air-Gapped RCA Agent on a live runtime snapshot."""
    t_start = time.perf_counter()

    if basis_mode not in VALID_BASIS_MODES:
        raise ValueError(f"basis_mode must be one of {VALID_BASIS_MODES}")
    if lang not in VALID_LANGS:
        raise ValueError(f"lang must be one of {VALID_LANGS}")
    clean_preset = preset.strip() if isinstance(preset, str) and preset.strip() else None
    if clean_preset is not None and clean_preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}")

    clean_query = (query or "").strip()
    detected_lang = detect_language(clean_query, lang=lang)
    refused_control_write = detect_control_write_request(clean_query, clean_preset)
    intent, focus_node = classify_intent(clean_query, clean_preset, refused_control_write)

    snap_status = str(snapshot.get("status", "NO_ALERT"))
    abstain_blocked = (
        (not refused_control_write)
        and (snap_status == "ABSTAIN")
        and (intent in ("diagnostic", "why_alert", "root_cause", "engineer_brief"))
    )

    # Execute all 6 deterministic tools with real wall-clock timing
    t1 = _run_tool("T1", "[T1:Gate]", "tool_check_safety_gates", lambda: tool_check_safety_gates(snapshot))
    t2 = _run_tool("T2", "[T2:HGB]", "tool_score_quan_hgb", lambda: tool_score_quan_hgb(snapshot))
    t3 = _run_tool("T3", "[T3:DAG]", "tool_trace_causal_dag", lambda: tool_trace_causal_dag(snapshot))
    t4 = _run_tool(
        "T4",
        "[T4:EVN-TOU]",
        "tool_calc_evn_tou_economics",
        lambda: tool_calc_evn_tou_economics(snapshot, basis_mode),
    )
    t5 = _run_tool(
        "T5",
        "[T5:ESG]",
        "tool_calc_scope2_and_water",
        lambda: tool_calc_scope2_and_water(snapshot, basis_mode),
    )
    t6 = _run_tool(
        "T6",
        "[T6:SOP]",
        "tool_retrieve_local_sop",
        lambda: tool_retrieve_local_sop(intent, refused_control_write, abstain_blocked),
    )
    tool_traces = [t1, t2, t3, t4, t5, t6]
    p7_payload: Optional[Dict[str, Any]] = None
    if intent == "field_ai":
        t7 = _run_tool("T7", FIELD_AI_TAG, "tool_inspect_field_ai_bench", lambda: tool_inspect_field_ai_bench(snapshot))
        tool_traces.append(t7)
        p7_payload = t7["payload"]

    # Compute deterministic SHA-256 evidence digest over canonical JSON of all executed tool payloads
    canonical_evidence = {t["id"]: t["payload"] for t in tool_traces}
    canonical_json = json.dumps(canonical_evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest_hex = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    answer_title, answer_markdown = _compose_answer(
        lang=detected_lang,
        intent=intent,
        focus_node=focus_node,
        refused_control_write=refused_control_write,
        abstain_blocked=abstain_blocked,
        snapshot=snapshot,
        p1=t1["payload"],
        p2=t2["payload"],
        p3=t3["payload"],
        p4=t4["payload"],
        p5=t5["payload"],
        p6=t6["payload"],
        p7=p7_payload,
    )

    citations_used = [tag for tag in ALL_TOOL_TAGS if tag in answer_markdown]
    total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

    resp: Dict[str, Any] = {
        "agent_version": "12.0.0",
        "engine": "AIRGAPPED_DETERMINISTIC_7TOOL_RCA" if p7_payload is not None else "AIRGAPPED_DETERMINISTIC_6TOOL_RCA",
        "air_gapped": True,
        "deterministic": True,
        "read_only": True,
        "controls": False,
        "lang": detected_lang,
        "intent": intent,
        "preset": clean_preset,
        "focus_node": focus_node,
        "basis_mode": basis_mode,
        "query": clean_query,
        "status": snap_status,
        "timestamp": snapshot.get("timestamp"),
        "scenario": snapshot.get("scenario"),
        "refused_control_write": refused_control_write,
        "abstain_blocked": abstain_blocked,
        "title": answer_title,
        "answer_markdown": answer_markdown,
        "citations_used": citations_used,
        "tool_traces": tool_traces,
        "evidence_digest_sha256": digest_hex,
        "evidence_digest_short": digest_hex[:16],
        "total_latency_ms": total_ms,
        "governance": {
            "standard": "IEC 62443-3-3 Read-Only Conduit · Purdue Level 3 Air-Gap",
            "verified_excess_kwh": None,
            "verified_avoided_kwh": None,
            "verified_water_savings": None,
            "controls": False,
            "read_only": True,
        },
    }
    if p7_payload is not None:
        resp["field_ai_context"] = p7_payload
    return resp
