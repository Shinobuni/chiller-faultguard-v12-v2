"""Published-Data Supervisor Digital Twin Runtime for Chiller FaultGuard v12.0.

Combines:
- Quan's exact frozen HistGradientBoostingClassifier + 5,760-row Oct-01 test fixture (0.00e+00 parity)
- Full 14-channel per-equipment LBNL power breakdown from the 4 original CSVs
- Interactive Illustrative Earlier-PM Calculator & EVN TOU-Aware Break-Even PM Optimizer
- Deterministic 8-Node Causal Root-Cause Graph (DAG) for root vs symptom isolation
- Benchtop Mini-CT1 v12 Real-World Vision ML (ArUco + HGB + 1-Tap Calibration), Live PCM FFT, Stull Weather
- Supervisor 6-Step Alert Triage (I-VI) with Safety Interlock + Simulation-Clock Audit Trail
"""

from __future__ import annotations

import copy
import json
from collections import deque
from pathlib import Path
from statistics import median
from threading import RLock
from typing import Any, Deque, Dict, List, Optional

from .acoustic_tach_weather import analyze_pcm_fan_audio, evaluate_stull_weather, run_acoustic_tach_analysis
from .causal_graph import build_causal_graph
from .quan_engine import MODELS_DIR, QuanFrozenAdapter
from .tou_optimizer import DEFAULT_PM_COST_VND, MAX_PM_COST_VND, MIN_PM_COST_VND, build_pm_optimizer, clamp_pm_cost_vnd
from .vision_inspector import (
    VALID_VISION_MODES,
    clear_bench_calibration,
    convert_data_uri_to_jpeg,
    decode_base64_image,
    generate_fill_strainer_frame,
    get_calibration_status,
    get_vision_ml_card,
    inspect_bgr_frame,
    run_synthetic_vision,
    save_bench_calibration,
)

DATA_DIR = Path(__file__).resolve().parent / "data_lbnl"
DIGITAL_TWIN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DIGITAL_TWIN_DIR.parent
CAD_V12_DIR = (
    (DIGITAL_TWIN_DIR / "hardware_cad_v12")
    if (DIGITAL_TWIN_DIR / "hardware_cad_v12").exists()
    else (REPO_ROOT / "hardware_cad_v12")
)
CAD_VERSION_V12 = "v12.0 (15-Part Zero-Support + Drift Eliminator + Coaxial Landscape Optics)"

CASES: Dict[str, str] = {
    "normal": "normal",
    "fouling_low": "tower_fouling_095",
    "fouling_medium": "tower_fouling_080",
    "fouling_high": "tower_fouling_065",
}

CASE_META: Dict[str, Dict[str, Any]] = {
    "normal": {"label": "Normal (1.00 UA)", "multiplier": 1.00, "csv": "ChillerPlant.csv"},
    "fouling_low": {"label": "0.95 UA (-5%)", "multiplier": 0.95, "csv": "ChillerPlant_coolingtower_fouling_095.csv"},
    "fouling_medium": {"label": "0.80 UA (-20%)", "multiplier": 0.80, "csv": "ChillerPlant_coolingtower_fouling_080.csv"},
    "fouling_high": {"label": "0.65 UA (-35%)", "multiplier": 0.65, "csv": "ChillerPlant_coolingtower_fouling_065.csv"},
}

NATIVE_KEYS = ["CT_RW_TEMP_1", "CT_SW_TEMP_1", "CT_FAN_SPD_1", "CT_FLOW_1", "CWL_SEC_LOAD", "CT_STA_1"]
STRESS_MODES = ("none", "missing_nan", "out_of_range", "domain_ood", "flatline_60m")

HARDWARE_PRESETS: Dict[str, Dict[str, str]] = {
    "all_healthy": {
        "label": "All Healthy",
        "fill": "clean",
        "spray": "healthy",
        "basin": "clear_coc4",
        "acoustic": "healthy",
        "weather": "hanoi_humid",
        "default_mode": "fill_strainer",
    },
    "fill_fouling_065": {
        "label": "Fill Fouling 0.65 UA",
        "fill": "fouled_065",
        "spray": "healthy",
        "basin": "clear_coc4",
        "acoustic": "healthy",
        "weather": "hanoi_humid",
        "default_mode": "fill_strainer",
    },
    "clogged_nozzle_low_flow": {
        "label": "Clogged Nozzle / Low Flow",
        "fill": "clean",
        "spray": "clogged_low_flow",
        "basin": "clear_coc4",
        "acoustic": "healthy",
        "weather": "hanoi_humid",
        "default_mode": "spray_flow",
    },
    "fan_belt_slip": {
        "label": "Fan Belt Slip",
        "fill": "clean",
        "spray": "healthy",
        "basin": "clear_coc4",
        "acoustic": "belt_slip",
        "weather": "hanoi_humid",
        "default_mode": "fill_strainer",
    },
    "high_wetbulb_confounder": {
        "label": "High Wet-Bulb",
        "fill": "clean",
        "spray": "healthy",
        "basin": "clear_coc4",
        "acoustic": "healthy",
        "weather": "bac_ninh_peak",
        "default_mode": "fill_strainer",
    },
    "turbid_basin_low_coc": {
        "label": "Turbid Basin / Low CoC",
        "fill": "clean",
        "spray": "healthy",
        "basin": "turbid_low_coc",
        "acoustic": "healthy",
        "weather": "hanoi_humid",
        "default_mode": "basin_water",
    },
}

CAD_SUMMARY: Dict[str, Any] = {
    "parts_count": 15,
    "total_pla_g_est": 880,
    "bom_vnd_range": "460k-680k ₫",
    "iphone_model": "iPhone 11 Landscape (150.9×75.7×8.3mm)",
    "working_distance_mm": 145,
    "stations": [
        "Station 1 (+18°, Z=123.3mm): Fan Tach & Spray Nozzles",
        "Station 2 (0°, Z=103.0mm): Fill Media & ArUco 0..3",
        "Station 3 (-18°, Z=78.7mm): Basin NTU & CoC Strip",
    ],
    "dual_mode": "0%-Support Dry-Safe Booth / Optional 5V Wet Mode",
    "print_envelope_mm": "185×185×175 mm",
    "cad_version": CAD_VERSION_V12,
}

# Verified Official Vietnam Nov-2026 Quantification Parameters (see research_kha_quantify/KHA_QUANTIFY_RESEARCH_REPORT.md)
# - Industrial Tariff (MOIT Decision 2892/QD-BCT eff. 01/11/2026 + Decision 963/QD-BCT 22/04/2026 TOU 24/7 weighted avg):
#     22 kV MV (ex-VAT) = (96*1814 + 42*1166 + 30*3348)/168 = 1925.93 VND/kWh (Peak 17:30-22:30 = 31.04% of bill)
#     110 kV HV (ex-VAT) = (96*1792 + 42*1123 + 30*3218)/168 = 1879.39 VND/kWh
# - FX: Vietcombank sell rate 25,980 VND/USD (22/11/2026) | 2026 Electricity VAT = 8%
# - Vietnam Grid Emission Factor (2024): Official Dispatch 11232/BNNMT-BDKH (30/09/2026) = 0.6187 kg CO2/kWh
VN_TARIFF_22KV_VND_KWH = 1925.93
VN_TARIFF_110KV_VND_KWH = 1879.39
VN_FX_VND_USD = 25980.0
VN_VAT_2026 = 0.08
VN_PEAK_BILL_SHARE_PCT = 31.04
VN_GRID_EF_2024_KG_KWH = 0.6187
TREE_CO2_KG_PER_YR = 21.0
TRIAGE_MIN_PER_FA = 15.0

# ASHRAE / CTI Cooling Tower Water-Balance Physics Constants (Sensitivity Estimator ONLY; verified_water_savings = None)
ASHRAE_EVAP_COEFF = 0.00085     # E (GPM) = 0.00085 * CT_FLOW_1 (GPM) * Range (degF) (~80% latent rejection)
GPM_TO_M3_PER_HR = 0.2271247    # 1 US GPM = 0.2271247 m3/h
WATER_LATENT_KJ_KG = 2420.0     # Latent heat of vaporization h_fg at ~30-35 C basin
WATER_LATENT_FRAC = 0.80        # ~80% latent evaporation fraction in wet cooling tower
LBNL_065_CHILLER_SHARE = 0.296  # Measured Oct-01 LBNL chiller share of excess power (fallback for static Quan Ref)


class StateConflictError(ValueError):
    """Raised when an action violates the current read-only safety interlock or replay state (HTTP 409)."""


class ChillerFaultGuardV6Runtime:
    """Thread-safe published-data runtime for Chiller FaultGuard v6.0."""

    def __init__(self, auto_seek_eligible: bool = True) -> None:
        self.fixture: Dict[str, List[Dict[str, Any]]] = json.loads(
            (DATA_DIR / "lbnl_oct01_24h_fixture.json").read_text(encoding="utf-8")
        )
        self.metrics: Dict[str, Any] = json.loads(
            (MODELS_DIR / "metrics.json").read_text(encoding="utf-8")
        )
        self.electricity_audit: Dict[str, Any] = json.loads(
            (MODELS_DIR / "electricity.json").read_text(encoding="utf-8")
        )
        self.illustrative_base: Dict[str, Any] = json.loads(
            (MODELS_DIR / "illustrative_energy.json").read_text(encoding="utf-8")
        )
        self.event_metrics: Dict[str, Any] = json.loads(
            (MODELS_DIR / "event_metrics.json").read_text(encoding="utf-8")
        )
        self.adapter = QuanFrozenAdapter()
        self.lock = RLock()
        self._scenario_scores: Dict[str, Dict[int, float]] = {}
        self._cached_cad_summary: Optional[Dict[str, Any]] = None
        self._cached_cad_bom: Optional[Dict[str, Any]] = None
        self._hardware_initialized: bool = False
        self.timeline_bins: Dict[str, List[Dict[str, Any]]] = self._build_timeline_bins()

        self.scenario: str = "normal"
        self.cursor: int = -1
        self.stress: str = "none"
        self.row: Dict[str, Any] = {}
        self.effective_native: Dict[str, Any] = {}
        self.pred: Dict[str, Any] = {}
        self.history: Deque[Dict[str, Any]] = deque(maxlen=180)
        self.audit_log: Deque[Dict[str, Any]] = deque(maxlen=50)

        # Supervisor Step V verdict (explicitly session-local advisory review; never PLC write)
        self.review: Dict[str, Any] = {
            "acknowledged": False,
            "verdict": "PENDING",
            "engineer_name": None,
            "timestamp": None,
            "completed_steps": [],
            "note": None,
            "review_note_is_approval": False,
        }

        # Interactive Illustrative Earlier-PM Parameters (initialized to Quan's exact config!)
        base_a = self.illustrative_base["assumptions"]
        self.sim_params: Dict[str, Any] = {
            "imposed_excess_kw": float(base_a["imposed_excess_kW"]),      # 3.0 kW default
            "earlier_pm_day": int(base_a["earlier_intervention_day"]),    # Day 28 default
            "existing_pm_day": int(base_a["existing_intervention_day"]),  # Day 120 default
            "alert_day": int(base_a["assumed_alert_day"]),                # Day 20 default
            "verify_days": int(base_a["assumed_verification_days"]),      # 2 days default
            "restored_fraction": float(base_a["restored_fraction_of_excess"]),  # 1.0 default
            "hours_per_day": float(base_a["operating_hours_per_day"]),    # 24h default
            "tariff_vnd_kwh": VN_TARIFF_22KV_VND_KWH,                     # QD 2892/BCT 22kV 24/7 TOU ex-VAT
            "tariff_usd_kwh": round(VN_TARIFF_22KV_VND_KWH / VN_FX_VND_USD, 4),
            "pm_cost_vnd": DEFAULT_PM_COST_VND,
            "pm_cost_usd": round(DEFAULT_PM_COST_VND / VN_FX_VND_USD, 2),
        }

        # Benchtop Mini-CT1 Multi-Modal Field AI State (v12; strictly advisory_only, never PLC control)
        self.hardware_preset: str = "fill_fouling_065"
        self.active_vision_mode: str = "fill_strainer"
        self.hw_vision: Dict[str, Dict[str, Any]] = {}
        self.hw_acoustic: Dict[str, Any] = {}
        self.hw_weather: Dict[str, Any] = {}
        self.latest_raw_bgr: Dict[str, Any] = {}
        self.field_checks: Dict[str, Any] = {
            "applied": False,
            "applied_at": None,
            "preset": None,
            "weather_verdict": None,
            "advisory_only": True,
            "items": {},
        }

        self.reset("normal", seek_eligible=auto_seek_eligible)
        self._log(
            "INFO",
            "SYSTEM",
            f"FaultGuard v12.0 ready · frozen HGB {self.adapter.model_sha256[:8]} · Vision ML + ArUco + offline read-only.",
        )

    def _get_scenario_scores(self, scenario: str) -> Dict[int, float]:
        """Lazily batch-score all eligible rows for a static Oct-01 case once (exact 0.00e+00 parity)."""
        cached = self._scenario_scores.get(scenario)
        if cached is not None:
            return cached
        rows = self.fixture[CASES[scenario]]
        cand_indices: List[int] = []
        cand_vecs: List[List[float]] = []
        for idx, r in enumerate(rows):
            if r.get("quality_ok") and r.get("eligible"):
                rng = float(r["CT_RW_TEMP_1"]) - float(r["CT_SW_TEMP_1"])
                fan = float(r["CT_FAN_SPD_1"])
                cand_indices.append(idx)
                cand_vecs.append(
                    [rng, fan, float(r["CT_FLOW_1"]), float(r["CT_SW_TEMP_1"]), float(r["CWL_SEC_LOAD"]) / 1e6, fan * rng]
                )
        score_map: Dict[int, float] = {}
        if cand_vecs:
            batch_scores = self.adapter._score_batch(cand_vecs)
            for idx, sc in zip(cand_indices, batch_scores):
                score_map[idx] = float(sc)
        self._scenario_scores[scenario] = score_map
        return score_map

    def _ensure_hardware_ready(self) -> None:
        if not self._hardware_initialized:
            with self.lock:
                if not self._hardware_initialized:
                    self._run_hardware_preset(self.hardware_preset, vision_mode=self.active_vision_mode, log_event=False)
                    self._hardware_initialized = True

    def _build_timeline_bins(self) -> Dict[str, List[Dict[str, Any]]]:
        """Precompute compact 10-min (144-bin) 24h status strips for every documented Oct-01 case."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for scen, cname in CASES.items():
            rows = self.fixture[cname]
            bins: List[Dict[str, Any]] = []
            step = 10
            for start in range(0, len(rows), step):
                chunk = rows[start : start + step]
                end = start + len(chunk) - 1
                has_both = any(r.get("ai_alert") and r.get("rule_alert") for r in chunk)
                has_ai = any(r.get("ai_alert") for r in chunk)
                has_rule = any(r.get("rule_alert") for r in chunk)
                has_elig = any(r.get("eligible") for r in chunk)
                st = "both" if has_both else "ai_only" if has_ai else "rule_only" if has_rule else "eligible" if has_elig else "off"
                bins.append(
                    {
                        "start_row": start,
                        "end_row": end,
                        "mid_row": start + len(chunk) // 2,
                        "time": chunk[0]["time"][11:16],
                        "state": st,
                    }
                )
            out[scen] = bins
        return out

    def _log(self, level: str, category: str, message: str) -> None:
        ts = self.row.get("time", "2018-10-01 00:00:00")
        self.audit_log.appendleft(
            {"timestamp": ts, "level": level, "category": category, "message": message}
        )

    def _clear_review(self) -> None:
        self.review = {
            "acknowledged": False,
            "verdict": "PENDING",
            "engineer_name": None,
            "timestamp": None,
            "completed_steps": [],
            "note": None,
            "review_note_is_approval": False,
        }

    def reset(self, scenario: str = "normal", seek_eligible: bool = False) -> Dict[str, Any]:
        if scenario not in CASES:
            raise ValueError(f"Unknown documented LBNL case: '{scenario}'")
        with self.lock:
            target = 720 if seek_eligible else 0
            self._replay_to_index(scenario, target)
            self._log(
                "INFO",
                "SCENARIO",
                f"Loaded {CASE_META[scenario]['label']} @ {self.row['time'][11:16]}.",
            )
            return self.state()

    def _replay_to_index(self, scenario: str, target_index: int) -> None:
        """Exact parity-preserving replay from row 0..target_index using batched frozen HGB scoring + sequential gates."""
        rows = self.fixture[CASES[scenario]]
        if not (0 <= target_index < len(rows)):
            raise ValueError(f"row_index must be in [0, {len(rows) - 1}]")
        self.scenario = scenario
        self.cursor = -1
        self.stress = "none"
        self.history.clear()
        self.adapter.reset("v6-stage")
        self._clear_review()

        # Reuse exact frozen HGB batch scores cached per static scenario (0.00e+00 parity, ~100x faster seeks)
        slice_rows = rows[: target_index + 1]
        score_map = self._get_scenario_scores(scenario)

        # Run every single gate & sequential persistence update from row 0..target_index identically to streaming
        for idx, r in enumerate(slice_rows):
            self.cursor = idx
            is_last = idx == target_index
            self._evaluate_row(
                r,
                include_sensitivity=is_last,
                precomputed_score=score_map.get(idx),
            )
            self.history.append(self._trend_point())

    def seek(self, row_index: int, scenario: Optional[str] = None) -> Dict[str, Any]:
        """Jump directly to any minute (0..1439) of the 24h fixture with exact 0.00e+00 streaming parity."""
        with self.lock:
            target_scen = scenario if scenario is not None else self.scenario
            if target_scen not in CASES:
                raise ValueError(f"Unknown documented LBNL case: '{target_scen}'")
            if scenario is None and self.stress != "none":
                raise StateConflictError(
                    f"Stress '{self.stress}' active: clear before seeking."
                )
            self._replay_to_index(target_scen, int(row_index))
            self._log(
                "INFO",
                "SEEK",
                f"Seek {CASE_META[target_scen]['label']} → {self.row['time'][11:16]} (row {self.cursor + 1}).",
            )
            return self.state()

    def _evaluate_row(
        self,
        row: Dict[str, Any],
        include_sensitivity: bool = False,
        precomputed_score: Optional[float] = None,
    ) -> None:
        self.row = row
        native: Dict[str, Any] = {k: row[k] for k in NATIVE_KEYS}
        force_flatline = False
        if self.stress == "missing_nan":
            native.pop("CT_SW_TEMP_1", None)
        elif self.stress == "out_of_range":
            native["CT_FAN_SPD_1"] = 4.0
        elif self.stress == "domain_ood":
            # Physically plausible flow (750 GPM >= 10 GPM min), passes Gates 1-3, breaches Gate 4 Jan-Jun envelope (max 601.1 GPM)
            native["CT_FLOW_1"] = 750.0
            native["CT_STA_1"] = 1.0
            if float(native["CT_FAN_SPD_1"]) < 0.25:
                native["CT_FAN_SPD_1"] = 0.55
            if float(native["CWL_SEC_LOAD"]) <= 0.0:
                native["CWL_SEC_LOAD"] = 180000.0
        elif self.stress == "flatline_60m":
            force_flatline = True

        self.effective_native = dict(native)

        self.pred = self.adapter.predict(
            {
                "timestamp": row["time"],
                "stream_id": "v6-stage",
                "segment_id": CASES[self.scenario],
                "dq_status": "OK" if row["quality_ok"] else "BAD",
                "telemetry": native,
                "force_flatline": force_flatline,
            },
            include_sensitivity=include_sensitivity,
            precomputed_score=precomputed_score if self.stress == "none" else None,
        )
        if self.stress == "missing_nan":
            self.pred["reasons"] = [
                "Stress: CT_SW_TEMP_1 missing. Diagnosis suspended."
            ]
        elif self.stress == "out_of_range":
            self.pred["reasons"] = [
                "Stress: CT_FAN_SPD_1=400% out of range [0, 101%]."
            ]
        elif self.stress == "domain_ood":
            self.pred["reasons"] = [
                "Stress OOD: CT_FLOW_1=750.0 GPM > train max 601.1 GPM."
            ]
        elif self.stress == "flatline_60m":
            self.pred["reasons"] = [
                "Stress: CT_SW_TEMP_1 flatline 60 min."
            ]

    def _trend_point(self) -> Dict[str, Any]:
        return {
            "timestamp": self.row["time"],
            "plant_kw": round(float(self.row["plant_kW"]), 2),
            "normal_plant_kw": round(float(self.row["normal_plant_kW"]), 2),
            "score": self.pred.get("ai_score"),
            "fan_pct": round(float(self.row["CT_FAN_SPD_1"]) * 100.0, 1),
            "ai_alert": bool(self.pred.get("ai_alert", False)),
            "rule_alert": bool(self.pred.get("rule_alert", False)),
            "eligible": self.pred.get("quality_status") != "ABSTAIN",
        }

    def advance(self, steps: int = 1, compute_sensitivity: bool = True) -> Dict[str, Any]:
        with self.lock:
            rows = self.fixture[CASES[self.scenario]]
            score_map = self._get_scenario_scores(self.scenario) if self.stress == "none" else {}
            prev_alert = bool(self.pred.get("ai_alert", False))
            for i in range(steps):
                if self.cursor + 1 >= len(rows):
                    break
                self.cursor += 1
                is_last = (i == steps - 1) or (self.cursor + 1 >= len(rows))
                self._evaluate_row(
                    rows[self.cursor],
                    include_sensitivity=(compute_sensitivity and is_last),
                    precomputed_score=score_map.get(self.cursor),
                )
                self.history.append(self._trend_point())
            if self.pred.get("ai_alert") and not prev_alert:
                self._log(
                    "CRITICAL",
                    "AI ALERT",
                    f"FOULING_SUSPECT @ {self.row['time'][11:16]} (score {self.pred['ai_score']:.3f} > 0.70, 5/5m).",
                )
            # Prevent stale CONFIRMED work order from persisting if replay leaves active alert state
            if self.review.get("verdict") == "CONFIRMED" and not (
                self.pred.get("ai_alert") or self.pred.get("rule_alert")
            ):
                self._clear_review()
            return self.state()

    def jump(self, kind: str = "eligible") -> Dict[str, Any]:
        if kind not in ("eligible", "alert"):
            raise ValueError("Jump kind must be 'eligible' or 'alert'")
        with self.lock:
            if self.stress != "none":
                raise StateConflictError(
                    f"Stress '{self.stress}' active: clear before jumping."
                )
            rows = self.fixture[CASES[self.scenario]]
            score_map = self._get_scenario_scores(self.scenario)
            found = False
            while self.cursor + 1 < len(rows):
                self.cursor += 1
                self._evaluate_row(
                    rows[self.cursor],
                    include_sensitivity=False,
                    precomputed_score=score_map.get(self.cursor),
                )
                self.history.append(self._trend_point())
                matched = (
                    bool(self.pred.get("ai_alert"))
                    if kind == "alert"
                    else self.pred.get("quality_status") != "ABSTAIN"
                )
                if matched:
                    found = True
                    break
            # Compute local feature sensitivity once on the final landed row
            if self.pred.get("quality_status") != "ABSTAIN":
                r = self.row
                vals = [
                    float(r["CT_RW_TEMP_1"]) - float(r["CT_SW_TEMP_1"]),
                    float(r["CT_FAN_SPD_1"]),
                    float(r["CT_FLOW_1"]),
                    float(r["CT_SW_TEMP_1"]),
                    float(r["CWL_SEC_LOAD"]) / 1e6,
                    float(r["CT_FAN_SPD_1"]) * (float(r["CT_RW_TEMP_1"]) - float(r["CT_SW_TEMP_1"])),
                ]
                self.pred["feature_sensitivity"] = self.adapter._compute_local_sensitivity(
                    vals, float(self.pred["ai_score"])
                )
            if self.review.get("verdict") == "CONFIRMED" and not (
                self.pred.get("ai_alert") or self.pred.get("rule_alert")
            ):
                self._clear_review()
            if kind == "alert":
                if found:
                    self._log(
                        "CRITICAL",
                        "AI ALERT",
                        f"Alert @ {self.row['time'][11:16]} (score {self.pred['ai_score']:.3f}, 5/5m).",
                    )
                else:
                    self._log(
                        "INFO",
                        "REPLAY END",
                        f"End of 24h replay ({self.row['time'][11:16]}).",
                    )
            return self.state()

    def inject_fault(self, fault_type: str) -> Dict[str, Any]:
        if fault_type not in STRESS_MODES:
            raise ValueError(f"Supported synthetic stress modes: {', '.join(STRESS_MODES)}")
        with self.lock:
            self.stress = fault_type
            # Reset persistence and clear any prior session verdict (zero stale alert or WO carryover!)
            self.adapter.reset("v6-stage")
            self._clear_review()
            self._evaluate_row(self.row, include_sensitivity=True)
            if self.history:
                self.history[-1] = self._trend_point()
            if fault_type == "none":
                self._log(
                    "INFO",
                    "STRESS CLEARED",
                    "Clean inputs restored; review cleared.",
                )
            else:
                self._log(
                    "VERIFY",
                    "STRESS TEST",
                    f"Stress '{fault_type}' → ABSTAIN; Steps V–VI locked.",
                )
            return self.state()

    def update_illustrative_params(
        self,
        imposed_excess_kw: Optional[float] = None,
        earlier_pm_day: Optional[int] = None,
        existing_pm_day: Optional[int] = None,
        restored_fraction: Optional[float] = None,
        pm_cost_vnd: Optional[float] = None,
        reset_defaults: bool = False,
    ) -> Dict[str, Any]:
        with self.lock:
            if reset_defaults:
                base_a = self.illustrative_base["assumptions"]
                self.sim_params["imposed_excess_kw"] = float(base_a["imposed_excess_kW"])
                self.sim_params["earlier_pm_day"] = int(base_a["earlier_intervention_day"])
                self.sim_params["existing_pm_day"] = int(base_a["existing_intervention_day"])
                self.sim_params["restored_fraction"] = float(base_a["restored_fraction_of_excess"])
                self.sim_params["pm_cost_vnd"] = DEFAULT_PM_COST_VND
                self.sim_params["pm_cost_usd"] = round(DEFAULT_PM_COST_VND / VN_FX_VND_USD, 2)
                return self.state()
            if imposed_excess_kw is not None:
                self.sim_params["imposed_excess_kw"] = max(1.0, min(50.0, float(imposed_excess_kw)))
            if existing_pm_day is not None:
                self.sim_params["existing_pm_day"] = max(60, min(180, int(existing_pm_day)))
                if self.sim_params["earlier_pm_day"] > self.sim_params["existing_pm_day"] - 10:
                    self.sim_params["earlier_pm_day"] = max(22, self.sim_params["existing_pm_day"] - 10)
            if earlier_pm_day is not None:
                max_d = self.sim_params["existing_pm_day"] - 10
                self.sim_params["earlier_pm_day"] = max(22, min(max_d, int(earlier_pm_day)))
            if restored_fraction is not None:
                self.sim_params["restored_fraction"] = max(0.25, min(1.0, float(restored_fraction)))
            if pm_cost_vnd is not None:
                clamped = clamp_pm_cost_vnd(pm_cost_vnd)
                self.sim_params["pm_cost_vnd"] = clamped
                self.sim_params["pm_cost_usd"] = round(clamped / VN_FX_VND_USD, 2)
            return self.state()

    def update_pm_optimizer(self, pm_cost_vnd: float) -> Dict[str, Any]:
        val = float(pm_cost_vnd)
        if not (MIN_PM_COST_VND <= val <= MAX_PM_COST_VND):
            raise ValueError(f"pm_cost_vnd must be within [{MIN_PM_COST_VND:.0f}, {MAX_PM_COST_VND:.0f}] VND")
        with self.lock:
            clamped = clamp_pm_cost_vnd(val)
            self.sim_params["pm_cost_vnd"] = clamped
            self.sim_params["pm_cost_usd"] = round(clamped / VN_FX_VND_USD, 2)
            return self.state()

    # ------------------------------------------------------------------
    # v10 Benchtop Mini-CT1 Hardware & Multi-Modal Step III-IV Field AI
    # ------------------------------------------------------------------

    def _current_cw_outlet_c(self) -> Optional[float]:
        eff = self.effective_native or self.row
        sw = eff.get("CT_SW_TEMP_1")
        if sw is None:
            return None
        return round((float(sw) - 32.0) / 1.8, 2)

    def _current_vfd_pct(self) -> float:
        eff = self.effective_native or self.row
        fan = float(eff.get("CT_FAN_SPD_1", 0.95)) * 100.0
        return round(max(45.0, min(100.0, fan)), 1)

    def _run_hardware_preset(
        self,
        preset: str,
        vision_mode: Optional[str] = None,
        log_event: bool = True,
        img_fmt: str = "png",
    ) -> Dict[str, Any]:
        if preset not in HARDWARE_PRESETS:
            raise ValueError(f"Unknown hardware preset '{preset}'. Allowed: {', '.join(HARDWARE_PRESETS.keys())}")
        if vision_mode is not None and vision_mode not in VALID_VISION_MODES:
            raise ValueError(f"Unknown vision_mode '{vision_mode}'. Allowed: {', '.join(VALID_VISION_MODES)}")
        spec = HARDWARE_PRESETS[preset]
        with self.lock:
            self.hardware_preset = preset
            self.active_vision_mode = vision_mode or spec["default_mode"]
            self.hw_vision = {
                "fill_strainer": run_synthetic_vision("fill_strainer", spec["fill"], source=f"sim_bench:{preset}"),
                "spray_flow": run_synthetic_vision("spray_flow", spec["spray"], source=f"sim_bench:{preset}"),
                "basin_water": run_synthetic_vision("basin_water", spec["basin"], source=f"sim_bench:{preset}"),
            }
            self.hw_acoustic = run_acoustic_tach_analysis(spec["acoustic"], vfd_pct=self._current_vfd_pct())
            self.hw_weather = evaluate_stull_weather(
                preset=spec["weather"],
                cw_outlet_c=self._current_cw_outlet_c(),
            )
            self._hardware_initialized = True
            if log_event:
                self._log(
                    "INFO",
                    "FIELD-AI",
                    f"Bench preset {preset}: fill {self.hw_vision['fill_strainer']['verdict']}, "
                    f"spray {self.hw_vision['spray_flow']['verdict']}, fan {self.hw_acoustic['verdict']}.",
                )
            return self.get_hardware_status(img_fmt=img_fmt)

    def simulate_hardware(
        self,
        preset: str,
        vision_mode: Optional[str] = None,
        img_fmt: str = "png",
    ) -> Dict[str, Any]:
        return self._run_hardware_preset(preset, vision_mode=vision_mode, log_event=True, img_fmt=img_fmt)

    def inspect_hardware_frame(
        self,
        mode: str,
        image_base64: str,
        source: str = "iphone11_upload",
        img_fmt: str = "png",
    ) -> Dict[str, Any]:
        if mode not in VALID_VISION_MODES:
            raise ValueError(f"Unknown vision mode '{mode}'. Allowed: {', '.join(VALID_VISION_MODES)}")
        self._ensure_hardware_ready()
        # Decode and execute real v12 3-stage Vision ML outside lock, then commit under lock
        bgr, raw_sha = decode_base64_image(image_base64)
        result = inspect_bgr_frame(bgr, mode=mode, source=source, precomputed_sha256=raw_sha)
        with self.lock:
            self.latest_raw_bgr[mode] = bgr
            self.hw_vision[mode] = result
            self.active_vision_mode = mode
            self._log(
                "INFO",
                "VISION-AI",
                f"{source} {mode}: {result['verdict']} ({result['confidence']:.2f}, {result['frame_sha256_short'][:8]}).",
            )
            det_out = self._format_vision_entry(result, img_fmt=img_fmt)
            return {
                "detection": det_out,
                "hardware_status": self.get_hardware_status(img_fmt=img_fmt),
            }

    def calibrate_hardware_bench(
        self,
        image_base64: Optional[str] = None,
        mode: str = "fill_strainer",
        label: str = "Clean Part 07",
        img_fmt: str = "png",
    ) -> Dict[str, Any]:
        if mode not in VALID_VISION_MODES:
            raise ValueError(f"Unknown vision mode '{mode}'. Allowed: {', '.join(VALID_VISION_MODES)}")
        self._ensure_hardware_ready()
        if image_base64 is not None and image_base64.strip():
            bgr, _ = decode_base64_image(image_base64)
        else:
            with self.lock:
                bgr = self.latest_raw_bgr.get(mode)
            if bgr is None:
                bgr = generate_fill_strainer_frame("clean")
        cal_meta = save_bench_calibration(bgr, mode=mode, label=label)
        with self.lock:
            self._log(
                "INFO",
                "BENCH-CAL",
                f"Calibrated {mode} clean ref '{cal_meta['label']}' (ArUco={cal_meta['aruco_locked']}, {cal_meta['ref_sha256_short'][:8]}).",
            )
            return {
                "calibration_status": cal_meta,
                "vision_ml_card": get_vision_ml_card(),
                "hardware_status": self.get_hardware_status(img_fmt=img_fmt),
            }

    def reset_hardware_calibration(self, img_fmt: str = "png") -> Dict[str, Any]:
        self._ensure_hardware_ready()
        cal_meta = clear_bench_calibration()
        with self.lock:
            self._log("INFO", "BENCH-CAL", "Cleared live bench calibration reference.")
            return {
                "calibration_status": cal_meta,
                "vision_ml_card": get_vision_ml_card(),
                "hardware_status": self.get_hardware_status(img_fmt=img_fmt),
            }

    def get_hardware_calibration_status(self) -> Dict[str, Any]:
        return {
            "calibration_status": get_calibration_status(),
            "vision_ml_card": get_vision_ml_card(),
            "aruco_sheet_png_available": (CAD_V12_DIR / "aruco_calibration_sheet_A4.png").exists(),
            "aruco_sheet_pdf_available": (CAD_V12_DIR / "aruco_calibration_sheet_A4.pdf").exists(),
            "cad_version": CAD_VERSION_V12,
        }

    def inspect_hardware_audio(
        self,
        pcm_samples: List[float],
        sample_rate: int = 16000,
        source: str = "iphone11_mic",
        img_fmt: str = "png",
    ) -> Dict[str, Any]:
        self._ensure_hardware_ready()
        res = analyze_pcm_fan_audio(
            pcm_samples,
            sample_rate=sample_rate,
            source=source,
            vfd_pct=self._current_vfd_pct(),
        )
        with self.lock:
            self.hw_acoustic = res
            self._log(
                "INFO",
                "AUDIO-FFT",
                f"{source}: {res['verdict']} ({res['bpf_derived_rpm']:.0f} RPM, peak {res['peak_freq_hz']:.0f} Hz).",
            )
            return {
                "audio_analysis": res,
                "hardware_status": self.get_hardware_status(img_fmt=img_fmt),
            }

    def _load_cad_summary(self) -> Dict[str, Any]:
        if self._cached_cad_summary is not None:
            return self._cached_cad_summary
        summary = dict(CAD_SUMMARY)
        bom_path = CAD_V12_DIR / "print_guide_bom.json"
        if bom_path.exists():
            try:
                bom = json.loads(bom_path.read_text(encoding="utf-8"))
                parts = bom.get("parts") or bom.get("stl_parts")
                if isinstance(parts, list) and len(parts) > 0:
                    summary["parts_count"] = len(parts)
                elif isinstance(bom.get("parts_count"), int):
                    summary["parts_count"] = int(bom["parts_count"])
                for key in ("total_pla_g_est", "total_pla_g", "estimated_pla_g"):
                    if isinstance(bom.get(key), (int, float)):
                        summary["total_pla_g_est"] = round(float(bom[key]), 1)
                        break
                if isinstance(bom.get("bom_vnd_range"), str):
                    summary["bom_vnd_range"] = bom["bom_vnd_range"]
                if isinstance(bom.get("plates"), list):
                    summary["build_plates_count"] = len(bom["plates"])
                summary["bom_loaded"] = True
            except Exception:
                summary["bom_loaded"] = False
        else:
            summary["bom_loaded"] = False
        self._cached_cad_summary = summary
        return summary

    def update_hardware_weather(
        self,
        preset: Optional[str] = None,
        dry_bulb_c: Optional[float] = None,
        rh_pct: Optional[float] = None,
        wet_bulb_override_c: Optional[float] = None,
        location: Optional[str] = None,
        img_fmt: str = "png",
        lite: bool = False,
    ) -> Dict[str, Any]:
        self._ensure_hardware_ready()
        with self.lock:
            self.hw_weather = evaluate_stull_weather(
                preset=preset,
                dry_bulb_c=dry_bulb_c,
                rh_pct=rh_pct,
                wet_bulb_override_c=wet_bulb_override_c,
                location=location,
                cw_outlet_c=self._current_cw_outlet_c(),
            )
            self._log(
                "INFO",
                "WEATHER-GATE",
                f"Weather {self.hw_weather['preset']}: T_wb {self.hw_weather['stull_wet_bulb_c']}°C → {self.hw_weather['confounder_verdict']}.",
            )
            return self.get_hardware_status(img_fmt=img_fmt, lite=lite)

    def _build_step4_ai_proposal(self) -> Dict[str, Dict[str, Any]]:
        fill = self.hw_vision["fill_strainer"]
        spray = self.hw_vision["spray_flow"]
        basin = self.hw_vision["basin_water"]
        ac = self.hw_acoustic
        return {
            "nozzles": {
                "id": "nozzles",
                "label": "Spray Nozzles & CW Flow (H2 / Step IV-1)",
                "checked": True,
                "status": "OK" if spray["verdict"] == "OK" else "ISSUE",
                "verdict": spray["verdict"],
                "source": "AI-Vision",
                "confidence": spray["confidence"],
                "metric": f"{spray['metrics']['healthy_nozzles']}/4 cones · {spray['metrics']['optical_flow_gpm']:.0f} GPM",
                "finding": spray["summary"],
            },
            "fan_belt": {
                "id": "fan_belt",
                "label": "Fan Optical RPM & Acoustic FFT (H3 / Step IV-2)",
                "checked": True,
                "status": "OK" if ac["verdict"] == "OK" else "ISSUE",
                "verdict": ac["verdict"],
                "source": "AI-Acoustic",
                "confidence": 0.94,
                "metric": f"{ac['optical_rpm']:.0f} RPM (slip {ac['slip_pct']:.1f}%)",
                "finding": ac["summary"],
            },
            "fill_media": {
                "id": "fill_media",
                "label": "CT-1 Fill Media & Strainer (H1 / Step IV-3)",
                "checked": True,
                "status": "OK" if fill["verdict"] == "OK" else "ISSUE",
                "verdict": fill["verdict"],
                "source": "AI-Vision",
                "confidence": fill["confidence"],
                "metric": f"Occlusion {fill['metrics']['occlusion_pct']:.1f}%",
                "finding": fill["summary"],
            },
            "water_chem": {
                "id": "water_chem",
                "label": "Basin Turbidity & CoC Strip (Step IV-4)",
                "checked": True,
                "status": "OK" if basin["verdict"] == "OK" else "ISSUE",
                "verdict": basin["verdict"],
                "source": "AI-Vision",
                "confidence": basin["confidence"],
                "metric": f"{basin['metrics']['turbidity_ntu']:.1f} NTU · CoC {basin['metrics']['estimated_coc']:.1f}",
                "finding": basin["summary"],
            },
        }

    def _load_cad_bom(self) -> Dict[str, Any]:
        if self._cached_cad_bom is not None:
            return self._cached_cad_bom
        bom_path = CAD_V12_DIR / "print_guide_bom.json"
        if bom_path.exists():
            try:
                bom = json.loads(bom_path.read_text(encoding="utf-8"))
                if isinstance(bom, dict):
                    self._cached_cad_bom = bom
                    return bom
            except Exception:
                pass
        fallback = {
            "version": "12.0.0",
            "parts_count": 15,
            "optics": {
                "phone_orientation": "landscape",
                "stations": [
                    {"id": 1, "name": "Spray Nozzles", "pitch_deg": 18.0, "lens_error_mm": 0.0},
                    {"id": 2, "name": "Fill & Strainer", "pitch_deg": 0.0, "distance_mm": 150.0, "lens_error_mm": 0.0},
                    {"id": 3, "name": "Basin & CoC", "pitch_deg": -18.0, "lens_error_mm": 0.0},
                ],
            },
            "parts": [
                {"part_no": 1, "name": "Basin Sump Base"},
                {"part_no": 2, "name": "Front Frame w/ ArUco 0..3 Pilasters"},
                {"part_no": 3, "name": "Rear Frame Wall"},
                {"part_no": 4, "name": "Left Louver Side Wall"},
                {"part_no": 5, "name": "Right Louver Side Wall"},
                {"part_no": 6, "name": "Fan Shroud & 120mm Guard"},
                {"part_no": 7, "name": "Clean Cross-Fluted Fill Cartridge (Ref P07)"},
                {"part_no": 8, "name": "Fouled Cross-Fluted Fill Cartridge (P08)"},
                {"part_no": 9, "name": "4-Nozzle Spray Header Manifold"},
                {"part_no": 10, "name": "Basin Outlet Strainer Basket"},
                {"part_no": 11, "name": "Optical Rotameter Gauge Tube"},
                {"part_no": 12, "name": "Coaxial Landscape Gantry Base"},
                {"part_no": 13, "name": "Gantry Vertical Mast (+18/0/-18 deg)"},
                {"part_no": 14, "name": "iPhone 11 Landscape Coaxial Cradle"},
                {"part_no": 15, "name": "Chevron Drift Eliminator Pack"},
            ],
            "bom_loaded_from_disk": False,
        }
        self._cached_cad_bom = fallback
        return fallback

    def _format_vision_entry(self, entry: Dict[str, Any], img_fmt: str = "png", strip_image: bool = False) -> Dict[str, Any]:
        if strip_image:
            out = dict(entry)
            out.pop("annotated_image_data_uri", None)
            return out
        if img_fmt == "jpeg" and entry.get("annotated_image_data_uri"):
            out = dict(entry)
            ckey = f"{entry.get('mode')}:{entry.get('frame_sha256_short')}:{entry.get('verdict')}:{entry.get('calibration_used')}"
            out["annotated_image_data_uri"] = convert_data_uri_to_jpeg(
                entry["annotated_image_data_uri"], cache_key=ckey, quality=86
            )
            return out
        return entry

    def get_hardware_status(self, img_fmt: str = "png", lite: bool = False) -> Dict[str, Any]:
        self._ensure_hardware_ready()
        with self.lock:
            if lite or img_fmt == "jpeg":
                vmodes = {
                    k: self._format_vision_entry(v, img_fmt=img_fmt, strip_image=lite)
                    for k, v in self.hw_vision.items()
                }
            else:
                vmodes = self.hw_vision
            return {
                "version": "12.0.0",
                "cad_version": CAD_VERSION_V12,
                "active_preset": self.hardware_preset,
                "active_vision_mode": self.active_vision_mode,
                "presets": {k: v["label"] for k, v in HARDWARE_PRESETS.items()},
                "vision_modes": vmodes,
                "acoustic_tach": self.hw_acoustic,
                "weather_gate": self.hw_weather,
                "step4_ai_proposal": self._build_step4_ai_proposal(),
                "cad_summary": self._load_cad_summary(),
                "cad_bom": self._load_cad_bom(),
                "vision_ml_card": get_vision_ml_card(),
                "calibration_status": get_calibration_status(),
                "field_checks_applied": bool(self.field_checks.get("applied", False)),
                "read_only": True,
                "controls": False,
                "advisory_only": True,
            }

    def get_hardware_summary(self) -> Dict[str, Any]:
        """Compact metadata summary (no PNG data URIs) embedded in /api/state for Tab 4 & RCA Agent."""
        with self.lock:
            if not self.hw_vision or not self.hw_acoustic or not self.hw_weather:
                return {"active_preset": self.hardware_preset, "advisory_only": True}
            fill = self.hw_vision["fill_strainer"]
            spray = self.hw_vision["spray_flow"]
            basin = self.hw_vision["basin_water"]
            return {
                "active_preset": self.hardware_preset,
                "active_vision_mode": self.active_vision_mode,
                "fill_verdict": fill["verdict"],
                "fill_occlusion_pct": fill["metrics"]["occlusion_pct"],
                "fill_sha_short": fill["frame_sha256_short"],
                "spray_verdict": spray["verdict"],
                "optical_flow_gpm": spray["metrics"]["optical_flow_gpm"],
                "healthy_nozzles": spray["metrics"]["healthy_nozzles"],
                "basin_verdict": basin["verdict"],
                "turbidity_ntu": basin["metrics"]["turbidity_ntu"],
                "estimated_coc": basin["metrics"]["estimated_coc"],
                "acoustic_verdict": self.hw_acoustic["verdict"],
                "optical_rpm": self.hw_acoustic["optical_rpm"],
                "slip_pct": self.hw_acoustic["slip_pct"],
                "weather_verdict": self.hw_weather["confounder_verdict"],
                "stull_wet_bulb_c": self.hw_weather["stull_wet_bulb_c"],
                "weather_physics_valid": self.hw_weather["physics_valid"],
                "advisory_only": True,
            }

    def apply_hardware_to_step4(self) -> Dict[str, Any]:
        """Populate Step IV advisory checklist from Multi-Modal Field AI (NEVER auto-confirms Step V!)."""
        self._ensure_hardware_ready()
        with self.lock:
            proposal = self._build_step4_ai_proposal()
            self.field_checks = {
                "applied": True,
                "applied_at": self.row.get("time", "2018-10-01 12:00:00"),
                "preset": self.hardware_preset,
                "weather_verdict": self.hw_weather["confounder_verdict"],
                "advisory_only": True,
                "items": copy.deepcopy(proposal),
            }
            issues = [k for k, v in proposal.items() if v["status"] != "OK"]
            issue_txt = ", ".join(issues) if issues else "none"
            self._log(
                "INFO",
                "FIELD-AI -> STEP IV",
                f"AI checks applied ({self.hardware_preset}); issues: {issue_txt}. Step V stays human.",
            )
            return self.state()

    def record_review(
        self,
        verdict: str,
        engineer_name: str,
        completed_steps: List[str],
        note: str,
    ) -> Dict[str, Any]:
        if verdict not in ("CONFIRMED", "FALSE_ALARM", "INCONCLUSIVE"):
            raise ValueError("Verdict must be CONFIRMED, FALSE_ALARM, or INCONCLUSIVE")
        with self.lock:
            if self.pred.get("quality_status") == "ABSTAIN" and verdict == "CONFIRMED":
                raise StateConflictError("Interlock active: cannot confirm during ABSTAIN.")
            if not (self.pred.get("ai_alert") or self.pred.get("rule_alert")) and verdict == "CONFIRMED":
                raise StateConflictError(
                    "Alert gate: cannot confirm on NO_ALERT."
                )
            self.review = {
                "acknowledged": True,
                "verdict": verdict,
                "engineer_name": engineer_name,
                "timestamp": self.row["time"],
                "completed_steps": list(completed_steps),
                "note": note,
                "review_note_is_approval": False,
            }
            self._log(
                "INFO",
                f"REVIEW: {verdict}",
                f"{engineer_name}: {note}",
            )
            return self.state()

    def _build_5m1e_kpis(
        self,
        eff_kw: float,
        avoided_kwh: float,
        days_saved: int,
        hrs: float,
        active_cut: bool,
        is_abstain: bool = False,
        restored: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute quantitative 5M1E (MONEY, TIME, IMAGE/ESG + ASHRAE Water Est.) from Nov-2026 VN basis & LBNL signals."""
        tariff_vnd = float(self.sim_params.get("tariff_vnd_kwh", VN_TARIFF_22KV_VND_KWH))
        avoided_ex_vnd = round(avoided_kwh * tariff_vnd, 0)
        avoided_vat_vnd = round(avoided_ex_vnd * (1.0 + VN_VAT_2026), 0)
        avoided_usd = round(avoided_ex_vnd / VN_FX_VND_USD, 0)
        daily_burn_vnd = round(eff_kw * hrs * tariff_vnd, 0)
        daily_burn_usd = round(daily_burn_vnd / VN_FX_VND_USD, 2)

        case_key = CASES[self.scenario]
        hgb_ev = self.event_metrics["methods"]["histogram_boosting_candidate"]["by_case"][case_key]
        rule_ev = self.event_metrics["methods"]["reference_rule"]["by_case"][case_key]
        ai_med = float(hgb_ev["first_alert_latency_minutes_detected_windows_only"]["median"])
        rule_med = float(rule_ev["first_alert_latency_minutes_detected_windows_only"]["median"])
        far_ai = float(self.metrics["AI"]["false_alarm_episodes_per_normal_hour"])
        prec_pct = round(float(self.metrics["AI"]["precision"]) * 100.0, 1)

        co2_kg = round(avoided_kwh * VN_GRID_EF_2024_KG_KWH, 1)
        co2_tonnes = round(co2_kg / 1000.0, 2)
        daily_co2_kg = round(eff_kw * hrs * VN_GRID_EF_2024_KG_KWH, 1)
        trees_eq = int(round(co2_kg / TREE_CO2_KG_PER_YR))

        # Live ASHRAE / CTI Cooling Tower Water-Balance Estimator (Physics Sensitivity ONLY; verified_water_savings = None)
        eff = self.effective_native or self.row
        sw_eff = eff.get("CT_SW_TEMP_1")
        if is_abstain or sw_eff is None:
            live_evap_m3_day = 0.0
            live_makeup_coc4_m3_day = 0.0
            coc_4to3_m3 = 0.0
            coc_5to3_m3 = 0.0
            thermal_makeup_m3 = 0.0
        else:
            flow_gpm = max(0.0, float(eff.get("CT_FLOW_1", self.row["CT_FLOW_1"])))
            range_f = max(0.0, float(self.row["CT_RW_TEMP_1"]) - float(sw_eff))
            evap_gpm = ASHRAE_EVAP_COEFF * flow_gpm * range_f
            live_evap_m3_day = round(evap_gpm * GPM_TO_M3_PER_HR * hrs, 2)
            live_makeup_coc4_m3_day = round(live_evap_m3_day * (4.0 / 3.0), 2)
            if active_cut and days_saved > 0:
                coc_4to3_m3 = round(live_evap_m3_day * days_saved * (1.0 / 2.0 - 1.0 / 3.0), 1)
                coc_5to3_m3 = round(live_evap_m3_day * days_saved * (1.0 / 2.0 - 1.0 / 4.0), 1)
                norm_row = self.fixture["normal"][self.cursor]
                live_dch_kw = max(0.0, float(self.row["p_chillers_kw"]) - float(norm_row["p_chillers_kw"]))
                ch_kw_eff = live_dch_kw if live_dch_kw > 0.0 else (eff_kw * LBNL_065_CHILLER_SHARE)
                th_evap_m3 = (ch_kw_eff * hrs * days_saved * restored * 3600.0 * WATER_LATENT_FRAC / WATER_LATENT_KJ_KG) / 1000.0
                thermal_makeup_m3 = round(th_evap_m3 * (4.0 / 3.0), 2)
            else:
                coc_4to3_m3 = 0.0
                coc_5to3_m3 = 0.0
                thermal_makeup_m3 = 0.0

        return {
            "money": {
                "basis_code": "QĐ 2892/BCT · 22kV TOU",
                "tariff_vnd_kwh": tariff_vnd,
                "tariff_110kv_vnd_kwh": VN_TARIFF_110KV_VND_KWH,
                "tariff_usd_kwh": round(tariff_vnd / VN_FX_VND_USD, 4),
                "fx_vnd_usd": VN_FX_VND_USD,
                "vat_rate": VN_VAT_2026,
                "peak_bill_share_pct": VN_PEAK_BILL_SHARE_PCT,
                "avoided_ex_vat_vnd": avoided_ex_vnd,
                "avoided_incl_vat8_vnd": avoided_vat_vnd,
                "avoided_ex_vat_usd": avoided_usd,
                "daily_burn_ex_vat_vnd": daily_burn_vnd,
                "daily_burn_ex_vat_usd": daily_burn_usd,
            },
            "time": {
                "case_key": case_key,
                "fouling_hours_avoided": int(round(days_saved * hrs)) if active_cut else 0,
                "days_earlier": int(days_saved) if active_cut else 0,
                "eligible_windows": int(hgb_ev["eligible_windows"]),
                "ai_windows_alerted": int(hgb_ev["windows_with_alert"]),
                "rule_windows_alerted": int(rule_ev["windows_with_alert"]),
                "ai_median_window_min": ai_med,
                "rule_median_window_min": rule_med,
                "window_speedup_min": round(rule_med - ai_med, 1),
                "precision_pct": prec_pct,
                "far_ep_per_hr": round(far_ai, 3),
                "mean_normal_hrs_per_fa": round(1.0 / far_ai, 1) if far_ai > 0 else 0.0,
                "fa_triage_hrs_per_100h": round(far_ai * (TRIAGE_MIN_PER_FA / 60.0) * 100.0, 2),
                "triage_min_per_ep_assumed": TRIAGE_MIN_PER_FA,
            },
            "esg": {
                "ef_basis_code": "CV 11232/BNNMT · 2024",
                "grid_ef_kg_kwh": VN_GRID_EF_2024_KG_KWH,
                "avoided_co2_kg": co2_kg,
                "avoided_co2_tonnes": co2_tonnes,
                "daily_co2_kg": daily_co2_kg,
                "trees_yr_eq": trees_eq,
                "verified_water_savings": None,
                "water_physics_est": {
                    "verified": False,
                    "suspended": is_abstain or sw_eff is None,
                    "method": "ASHRAE Flow×ΔT (sensitivity)",
                    "live_evap_m3_day": live_evap_m3_day,
                    "live_makeup_coc4_m3_day": live_makeup_coc4_m3_day,
                    "thermal_makeup_m3_est": thermal_makeup_m3,
                    "coc_4to3_penalty_m3_est": coc_4to3_m3,
                    "coc_5to3_penalty_m3_est": coc_5to3_m3,
                },
            },
        }

    def _build_illustrative_scenario(self) -> Dict[str, Any]:
        """Exact transparent earlier-PM arithmetic built on Quan's config/illustrative_energy.json."""
        p = self.sim_params
        ex_kw = float(p["imposed_excess_kw"])
        hrs = float(p["hours_per_day"])
        early_d = int(p["earlier_pm_day"])
        exist_d = int(p["existing_pm_day"])
        rest = float(p["restored_fraction"])
        days_saved = max(0, exist_d - early_d)

        existing_kwh = round(ex_kw * hrs * exist_d, 1)
        pre_clean_kwh = round(ex_kw * hrs * early_d, 1)
        residual_post_clean_kwh = ex_kw * hrs * days_saved * (1.0 - rest)
        earlier_kwh = round(pre_clean_kwh + residual_post_clean_kwh, 1)
        avoided_kwh = round(ex_kw * hrs * days_saved * rest, 1)
        is_abstain = self.pred.get("quality_status") == "ABSTAIN"
        kpis = self._build_5m1e_kpis(ex_kw, avoided_kwh, days_saved, hrs, active_cut=avoided_kwh > 0, is_abstain=is_abstain, restored=rest)
        pm_cost_vnd = float(p.get("pm_cost_vnd", DEFAULT_PM_COST_VND))
        pm_opt = build_pm_optimizer(
            eff_kw=ex_kw,
            early_day=early_d,
            cal_day=exist_d,
            restored=rest,
            hrs=hrs,
            pm_cost_vnd=pm_cost_vnd,
            tariff_vnd_kwh=float(kpis["money"]["tariff_vnd_kwh"]),
            active=avoided_kwh > 0,
            is_abstain=False,
            fx_vnd_usd=VN_FX_VND_USD,
        )

        return {
            "label": "ILLUSTRATIVE · NOT VERIFIED",
            "equation": f"{ex_kw:.1f} kW × {hrs:.0f} h/day × ({exist_d} − {early_d}) days × {int(rest*100)}%",
            "imposed_excess_kw": ex_kw,
            "hours_per_day": hrs,
            "alert_day": int(p["alert_day"]),
            "verify_days": int(p["verify_days"]),
            "earlier_pm_day": early_d,
            "existing_pm_day": exist_d,
            "days_saved": days_saved,
            "restored_fraction": rest,
            "existing_policy_excess_kwh": existing_kwh,
            "earlier_policy_excess_kwh": earlier_kwh,
            "scenario_avoided_excess_kwh": avoided_kwh,
            "illustrative_tariff_vnd_kwh": kpis["money"]["tariff_vnd_kwh"],
            "illustrative_tariff_usd_kwh": kpis["money"]["tariff_usd_kwh"],
            "illustrative_cost_vnd": kpis["money"]["avoided_ex_vat_vnd"],
            "illustrative_cost_usd": kpis["money"]["avoided_ex_vat_usd"],
            "illustrative_co2_kg_vn": round(kpis["esg"]["avoided_co2_kg"], 0),
            "kpis_5m1e": kpis,
            "pm_cost_vnd": pm_cost_vnd,
            "pm_cost_usd": float(p["pm_cost_usd"]),
            "pm_optimizer": pm_opt,
            "verified_excess_kwh": None,
            "verified_avoided_kwh": None,
            "water_savings": None,
            "exclusions": "Excl. water, cleaning energy, yield (Pilot).",
        }

    def _build_live_alert_projection(self, status: str) -> Dict[str, Any]:
        """Auto-triggered projection driven strictly by real telemetry at the active alert window (0 kWh on Normal/ABSTAIN)."""
        p = self.sim_params
        r = self.row
        pred = self.pred
        hrs = float(p["hours_per_day"])
        early_d = int(p["earlier_pm_day"])
        exist_d = int(p["existing_pm_day"])
        rest = float(p["restored_fraction"])
        days_saved = max(0, exist_d - early_d)
        instant_delta = round(float(r["plant_kW"]) - float(r["normal_plant_kW"]), 2)

        active_alert = status in ("FOULING_SUSPECT", "RULE_WARNING")
        if status == "ABSTAIN":
            mode = "SUSPENDED_ABSTAIN"
            eff_kw = 0.0
            win_count = 0
        elif not active_alert:
            mode = "STANDBY_NO_ALERT"
            eff_kw = 0.0
            win_count = 0
        else:
            # Use 15-min rolling median of eligible samples in history to filter 1-min lag-chiller staging spikes
            recent_deltas = [
                float(pt["plant_kw"]) - float(pt["normal_plant_kw"])
                for pt in list(self.history)[-15:]
                if pt.get("eligible")
            ]
            if not recent_deltas:
                recent_deltas = [instant_delta]
            win_count = len(recent_deltas)
            med_kw = round(float(median(recent_deltas)), 2)
            eff_kw = max(0.0, med_kw)
            mode = "ACTIVE_ALERT_LIVE" if eff_kw > 0.0 else "ALERT_ZERO_EXCESS"

        existing_kwh = round(eff_kw * hrs * exist_d, 1)
        pre_clean_kwh = round(eff_kw * hrs * early_d, 1)
        residual_post_clean_kwh = eff_kw * hrs * days_saved * (1.0 - rest)
        earlier_kwh = round(pre_clean_kwh + residual_post_clean_kwh, 1)
        avoided_kwh = round(eff_kw * hrs * days_saved * rest, 1)
        is_active_live = active_alert and eff_kw > 0.0
        kpis = self._build_5m1e_kpis(
            eff_kw, avoided_kwh, days_saved, hrs, active_cut=is_active_live, is_abstain=(status == "ABSTAIN"), restored=rest
        )
        pm_cost_vnd = float(p.get("pm_cost_vnd", DEFAULT_PM_COST_VND))
        pm_opt = build_pm_optimizer(
            eff_kw=eff_kw,
            early_day=early_d,
            cal_day=exist_d,
            restored=rest,
            hrs=hrs,
            pm_cost_vnd=pm_cost_vnd,
            tariff_vnd_kwh=float(kpis["money"]["tariff_vnd_kwh"]),
            active=is_active_live,
            is_abstain=(status == "ABSTAIN"),
            fx_vnd_usd=VN_FX_VND_USD,
        )

        return {
            "is_live_mode": True,
            "active": is_active_live,
            "mode": mode,
            "label": "LIVE ALERT · 15m MEDIAN ΔkW* · ILLUSTRATIVE",
            "equation": f"{eff_kw:.2f} kW* × {hrs:.0f} h/day × ({exist_d} − {early_d}) days × {int(rest*100)}%",
            "imposed_excess_kw": eff_kw,
            "instant_delta_kw": instant_delta,
            "window_samples": win_count,
            "trigger_timestamp": r["time"],
            "trigger_score": pred.get("ai_score"),
            "trigger_fan_pct": round(float(r["CT_FAN_SPD_1"]) * 100.0, 1),
            "hours_per_day": hrs,
            "alert_day": int(p["alert_day"]),
            "verify_days": int(p["verify_days"]),
            "earlier_pm_day": early_d,
            "existing_pm_day": exist_d,
            "days_saved": days_saved,
            "restored_fraction": rest,
            "existing_policy_excess_kwh": existing_kwh,
            "earlier_policy_excess_kwh": earlier_kwh,
            "scenario_avoided_excess_kwh": avoided_kwh,
            "illustrative_tariff_vnd_kwh": kpis["money"]["tariff_vnd_kwh"],
            "illustrative_tariff_usd_kwh": kpis["money"]["tariff_usd_kwh"],
            "illustrative_cost_vnd": kpis["money"]["avoided_ex_vat_vnd"],
            "illustrative_cost_usd": kpis["money"]["avoided_ex_vat_usd"],
            "illustrative_co2_kg_vn": round(kpis["esg"]["avoided_co2_kg"], 0),
            "kpis_5m1e": kpis,
            "pm_cost_vnd": pm_cost_vnd,
            "pm_cost_usd": float(p["pm_cost_usd"]),
            "pm_optimizer": pm_opt,
            "verified_excess_kwh": None,
            "verified_avoided_kwh": None,
            "water_savings": None,
            "exclusions": "Live ΔkW*; excl. water, cleaning energy, yield.",
        }

    def _build_competing_hypotheses(self) -> List[Dict[str, str]]:
        """Honest Step III competing-cause check driven by effective LBNL / stress-injected signals."""
        r = self.row
        eff = self.effective_native or r
        p = self.pred
        abstain = p.get("quality_status") == "ABSTAIN"
        ai_alert = bool(p.get("ai_alert"))
        rule_alert = bool(p.get("rule_alert"))
        fan_pct = float(eff.get("CT_FAN_SPD_1", r["CT_FAN_SPD_1"])) * 100.0
        flow_gpm = float(eff.get("CT_FLOW_1", r["CT_FLOW_1"]))
        sw_val = eff.get("CT_SW_TEMP_1")
        rng_c = (float(r["CT_RW_TEMP_1"]) - float(sw_val)) / 1.8 if sw_val is not None else None

        if abstain:
            if self.stress == "domain_ood":
                h2_ev = f"CT_FLOW_1={flow_gpm:.1f} GPM (OOD > 601.1 GPM)."
            else:
                h2_ev = f"CT_FLOW_1={flow_gpm:.1f} GPM."
            return [
                {
                    "id": "H3",
                    "name": "Quality / Operating Gate (ABSTAIN)",
                    "status": "PRIMARY_SUSPECT",
                    "evidence": " ".join(p.get("reasons", [])),
                },
                {
                    "id": "H1",
                    "name": "CT-1 Fill Fouling",
                    "status": "MONITORING",
                    "evidence": "Suspended while gate holds.",
                },
                {
                    "id": "H2",
                    "name": "CW Flow / Pump Restriction",
                    "status": "MONITORING",
                    "evidence": h2_ev,
                },
                {
                    "id": "H4",
                    "name": "Weather / Channel Mapping",
                    "status": "CHECK_REQUIRED",
                    "evidence": "OA_TEMP_WB > OA_TEMP in 99.3% of rows; excluded.",
                },
            ]

        flow_low = flow_gpm < 250.0
        rng_txt = f"{rng_c:.2f}°C" if rng_c is not None else "MISSING"
        return [
            {
                "id": "H1",
                "name": "CT-1 Heat-Exchanger Fouling",
                "status": "PRIMARY_SUSPECT" if (ai_alert or rule_alert) else "MONITORING",
                "evidence": (
                    f"HGB {p['ai_score']:.3f} ({p['ai_run_minutes']}/{p['ai_required_minutes']}m) · "
                    f"Fan={fan_pct:.1f}% ({p['rule_run_minutes']}/{p['rule_required_minutes']}m)."
                ),
            },
            {
                "id": "H2",
                "name": "CW Flow / Pump Drop",
                "status": "CHECK_REQUIRED" if flow_low else "RULED_OUT",
                "evidence": f"Flow {flow_gpm:.1f} GPM (floor 250) · Range {rng_txt}.",
            },
            {
                "id": "H3",
                "name": "Missing / OOD / Flatline Sensor",
                "status": "RULED_OUT",
                "evidence": "6 inputs finite, in-bounds, live.",
            },
            {
                "id": "H4",
                "name": "Weather / CHW Mapping Unresolved",
                "status": "CHECK_REQUIRED",
                "evidence": "Wet-bulb/CHW excluded until mapped.",
            },
        ]

    def state(self) -> Dict[str, Any]:
        r = self.row
        p = self.pred
        abstain = p.get("quality_status") == "ABSTAIN"
        ai_alert = bool(p.get("ai_alert"))
        rule_alert = bool(p.get("rule_alert"))

        if abstain:
            status = "ABSTAIN"
            agreement = "BLOCKED_BY_GATE"
            rec_title = "ABSTAIN — GATE ACTIVE"
            rec_action = "Diagnosis suspended. Check sensors & tower regime."
        elif ai_alert and rule_alert:
            status = "FOULING_SUSPECT"
            agreement = "BOTH_FLAGGED"
            rec_title = "FOULING SUSPECT — AI + RULE"
            rec_action = f"AI {p['ai_score']:.3f} and Fan {r['CT_FAN_SPD_1']*100:.1f}% both flagged. Verify I–VI."
        elif ai_alert:
            status = "FOULING_SUSPECT"
            agreement = "AI_ONLY_ALERT"
            rec_title = "FOULING SUSPECT — AI AHEAD OF RULE"
            rec_action = f"AI {p['ai_score']:.3f} flagged; rule {p['rule_run_minutes']}/15m. Verify I–IV."
        elif rule_alert:
            status = "RULE_WARNING"
            agreement = "RULE_ONLY_ALERT"
            rec_title = "HIGH-FAN RULE ALERT"
            rec_action = f"Fan >98% 15/15m; AI {p['ai_score']:.3f}. Check belt, strainer, ambient."
        else:
            status = "NO_ALERT"
            agreement = "NO_ALERT"
            rec_title = "NO ALERT — MONITORING"
            rec_action = f"Score {p['ai_score']:.3f} < 0.70 · Fan {r['CT_FAN_SPD_1']*100:.1f}%."

        rows_total = len(self.fixture[CASES[self.scenario]])
        eff = self.effective_native or r
        tw_in_c = (float(r["CT_RW_TEMP_1"]) - 32.0) / 1.8
        sw_eff = eff.get("CT_SW_TEMP_1")
        tw_out_c = (float(sw_eff) - 32.0) / 1.8 if sw_eff is not None else None
        tw_range_c = (tw_in_c - tw_out_c) if tw_out_c is not None else None
        eff_flow_gpm = float(eff.get("CT_FLOW_1", r["CT_FLOW_1"]))
        eff_fan_pct = float(eff.get("CT_FAN_SPD_1", r["CT_FAN_SPD_1"])) * 100.0
        eff_load_kw = float(eff.get("CWL_SEC_LOAD", r["CWL_SEC_LOAD"])) / 1000.0
        chw_s_c = (float(r["CWL_SEC_SW_TEMP"]) - 32.0) / 1.8
        chw_pri_s_c = (float(r["cwl_pri_sw_f"]) - 32.0) / 1.8
        chw_pri_r_c = (float(r["cwl_pri_rw_f"]) - 32.0) / 1.8

        snap: Dict[str, Any] = {
            "release": "chiller-faultguard-v12-quan-hgb-hw",
            "version": "12.0.0",
            "provenance": "Simulated: published LBNL office-building chiller plant (Oct-01 held-out test)",
            "scenario": self.scenario,
            "case": CASES[self.scenario],
            "case_meta": CASE_META[self.scenario],
            "timestamp": r["time"],
            "row_index": self.cursor,
            "total_rows": rows_total,
            "at_end": self.cursor >= rows_total - 1,
            "injection": self.stress,
            "status": status,
            "agreement": agreement,
            "advisory_title": rec_title,
            "advisory_action": rec_action,
            "prediction": p,
            "signals": {
                "tower_inlet_c": round(tw_in_c, 2),
                "nominal_tower_outlet_c": round(tw_out_c, 2) if tw_out_c is not None else None,
                "tower_range_c": round(tw_range_c, 2) if tw_range_c is not None else None,
                "tower_flow_gpm": round(eff_flow_gpm, 1),
                "tower_flow_kgs": round(eff_flow_gpm * 0.0630902, 1),
                "fan_pct": round(eff_fan_pct, 1),
                "logged_load_kw": round(eff_load_kw, 1),
                "nominal_chw_supply_c": round(chw_s_c, 2),
                "pri_chw_sup_c": round(chw_pri_s_c, 2),
                "pri_chw_ret_c": round(chw_pri_r_c, 2),
                "pri_chw_flow_gpm": round(float(r["cwl_pri_flow_gpm"]), 1),
                "ct1_on": int(round(float(r["CT_STA_1"]))),
                "n_chillers_on": int(r["n_chillers_on"]),
                "n_towers_on": int(r["n_towers_on"]),
                # Exact 14-channel per-equipment logged LBNL kW
                "plant_kw": round(float(r["plant_kW"]), 1),
                "normal_plant_kw": round(float(r["normal_plant_kW"]), 1),
                "signed_delta_kw": round(float(r["plant_kW"]) - float(r["normal_plant_kW"]), 2),
                "p_ch1_kw": round(float(r["p_ch1_kw"]), 1),
                "p_ct1_kw": round(float(r["p_ct1_kw"]), 1),
                "p_chillers_kw": round(float(r["p_chillers_kw"]), 1),
                "p_ct_fans_kw": round(float(r["p_ct_fans_kw"]), 1),
                "p_cwp_kw": round(float(r["p_cwp_kw"]), 1),
                "p_chwp_kw": round(float(r["p_chwp_kw"]), 1),
            },
            "hypotheses": self._build_competing_hypotheses(),
            "electricity": {
                "status": "Pending weather & cooling-service validation",
                "boundary": "14 logged channels: 3 chillers + 3 CT fans + 3 CW pumps + 3 pri & 2 sec CHW pumps",
                "logged_plant_kw": round(float(r["plant_kW"]), 1),
                "matched_normal_kw_diagnostic": round(float(r["normal_plant_kW"]), 1),
                "signed_delta_kw_diagnostic": round(float(r["plant_kW"]) - float(r["normal_plant_kW"]), 2),
                "verified_excess_kwh": None,
                "verified_avoided_kwh": None,
                "water_savings": None,
                "oct_dec_conditional": self.electricity_audit["rows"],
                "illustrative_scenario": self._build_illustrative_scenario(),
                "live_alert_projection": self._build_live_alert_projection(status),
            },
            "metrics": self.metrics,
            "event_metrics": self.event_metrics,
            "model_card": self.adapter.model_card,
            "timeline_24h": self.timeline_bins[self.scenario],
            "review": self.review,
            "field_checks": self.field_checks,
            "hardware_summary": self.get_hardware_summary(),
            "history": list(self.history),
            "audit_log": list(self.audit_log),
            "controls": False,
            "read_only": True,
            "forecast_supported": False,
            "severity_supported": False,
        }
        snap["causal_graph"] = build_causal_graph(snap)
        return snap


TwinRuntime = ChillerFaultGuardV6Runtime
