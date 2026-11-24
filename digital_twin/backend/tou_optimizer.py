"""EVN TOU-Aware Break-Even PM Optimizer for Chiller FaultGuard v11.0.

Implements transparent, illustrative financial break-even arithmetic under
Vietnam official industrial electricity tariffs:
- MOIT Decision 2892/QD-BCT (eff. 01/11/2026) & Decision 963/QD-BCT (22/04/2026)
- 22 kV MV 24/7 weighted TOU average (ex-VAT): 1,925.93 VND/kWh
  (Normal 96h @ 1,814 = 53.82%, Off-Peak 42h @ 1,166 = 15.14%, Peak 30h @ 3,348 = 31.04%)
- Illustrative break-even only: verified_avoided_kwh = None until site pilot metering.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

DEFAULT_PM_COST_VND = 8_000_000.0
MIN_PM_COST_VND = 1_000_000.0
MAX_PM_COST_VND = 50_000_000.0

VN_TARIFF_22KV_VND_KWH = 1925.93
VN_FX_VND_USD = 25980.0

TOU_PEAK_SHARE = 0.3104
TOU_NORMAL_SHARE = 0.5382
TOU_OFF_PEAK_SHARE = 0.1514

RECOMMENDED_PM_WINDOW = "Sunday / Off-Peak 00:00–06:00 (1,166 ₫/kWh)"


def clamp_pm_cost_vnd(value: float) -> float:
    """Clamp configurable PM cleaning cost to the documented [1M, 50M] VND envelope."""
    return round(max(MIN_PM_COST_VND, min(MAX_PM_COST_VND, float(value))), 0)


def _fmt_vnd_short(amount_vnd: float) -> str:
    sign = "+" if amount_vnd > 0 else "-" if amount_vnd < 0 else ""
    mag = abs(amount_vnd)
    if mag >= 1_000_000.0:
        return f"{sign}{mag / 1_000_000.0:.2f}M ₫"
    if mag >= 1_000.0:
        return f"{sign}{mag / 1_000.0:.0f}k ₫"
    return f"{sign}{mag:.0f} ₫"


def build_pm_optimizer(
    eff_kw: float,
    early_day: int,
    cal_day: int,
    restored: float,
    hrs: float,
    pm_cost_vnd: float = DEFAULT_PM_COST_VND,
    tariff_vnd_kwh: float = VN_TARIFF_22KV_VND_KWH,
    active: bool = True,
    is_abstain: bool = False,
    fx_vnd_usd: float = VN_FX_VND_USD,
) -> Dict[str, Any]:
    """Compute EVN TOU Break-Even PM economics, payback calendar day, cutoff day, and EOQ sensitivity."""
    eff_kw_f = max(0.0, float(eff_kw))
    early_d = int(early_day)
    cal_d = int(cal_day)
    days_saved = max(0, cal_d - early_d)
    rest_f = max(0.0, min(1.0, float(restored)))
    hrs_f = max(0.0, float(hrs))
    pm_cost_f = clamp_pm_cost_vnd(pm_cost_vnd)
    tariff_f = max(0.0, float(tariff_vnd_kwh))
    fx_f = float(fx_vnd_usd) if float(fx_vnd_usd) > 0 else VN_FX_VND_USD

    effective_active = bool(active) and (not bool(is_abstain)) and (eff_kw_f > 0.0)

    if effective_active:
        daily_burn_vnd = round(eff_kw_f * hrs_f * tariff_f * rest_f, 0)
        avoided_gross_vnd = round(daily_burn_vnd * days_saved, 0)
        net_benefit_vnd = round(avoided_gross_vnd - pm_cost_f, 0)
        roi_pct = round((net_benefit_vnd / pm_cost_f) * 100.0, 1) if pm_cost_f > 0 else 0.0
        payback_days: Optional[float] = round(pm_cost_f / daily_burn_vnd, 1) if daily_burn_vnd > 0 else None
        payback_calendar_day: Optional[float] = round(early_d + payback_days, 1) if payback_days is not None else None
        latest_profitable_day: Optional[float] = round(cal_d - payback_days, 1) if payback_days is not None else None
        # EOQ sensitivity under illustrative linear fouling growth reaching daily_burn_vnd at cal_d
        k_growth = daily_burn_vnd / max(1, cal_d)
        eoq_sensitivity_days: Optional[float] = (
            round(math.sqrt(2.0 * pm_cost_f / k_growth), 1) if k_growth > 0 else None
        )
    else:
        daily_burn_vnd = 0.0
        avoided_gross_vnd = 0.0
        net_benefit_vnd = 0.0
        roi_pct = 0.0
        payback_days = None
        payback_calendar_day = None
        latest_profitable_day = None
        eoq_sensitivity_days = None

    tou_avoided_vnd = {
        "peak": round(avoided_gross_vnd * TOU_PEAK_SHARE, 0),
        "normal": round(avoided_gross_vnd * TOU_NORMAL_SHARE, 0),
        "off_peak": round(avoided_gross_vnd * TOU_OFF_PEAK_SHARE, 0),
        "peak_share_pct": round(TOU_PEAK_SHARE * 100.0, 2),
        "normal_share_pct": round(TOU_NORMAL_SHARE * 100.0, 2),
        "off_peak_share_pct": round(TOU_OFF_PEAK_SHARE * 100.0, 2),
    }

    if is_abstain:
        verdict = "LOCKED_ABSTAIN"
        verdict_label = "LOCKED — GATE"
        rationale = "Gate active; PM economics locked at 0 ₫."
    elif (not active) or eff_kw_f <= 0.0:
        verdict = "STANDBY_NO_ACTION"
        verdict_label = "STANDBY"
        rationale = f"No active alert; routine PM Day {cal_d}."
    elif net_benefit_vnd > 0.0:
        verdict = "CLEAN_NOW_POSITIVE_ROI"
        verdict_label = "CLEAN NOW — +ROI"
        rationale = (
            f"Avoided {_fmt_vnd_short(avoided_gross_vnd)} vs PM {_fmt_vnd_short(pm_cost_f)}: net {_fmt_vnd_short(net_benefit_vnd)} "
            f"(+{roi_pct:.1f}%), payback {payback_days:.1f}d (D{payback_calendar_day:.1f})."
        )
    else:
        verdict = "DEFER_PM_NEGATIVE_ROI"
        verdict_label = "DEFER PM — −ROI"
        pb_txt = f"{payback_days:.1f}d (D{payback_calendar_day:.1f} > D{cal_d})" if payback_days is not None else "N/A"
        rationale = (
            f"Avoided {_fmt_vnd_short(avoided_gross_vnd)} vs PM {_fmt_vnd_short(pm_cost_f)}: net {_fmt_vnd_short(net_benefit_vnd)} "
            f"({roi_pct:.1f}%), payback {pb_txt}."
        )

    return {
        "verdict": verdict,
        "verdict_label": verdict_label,
        "rationale": rationale,
        "recommended_window": RECOMMENDED_PM_WINDOW,
        "pm_cost_vnd": pm_cost_f,
        "pm_cost_usd": round(pm_cost_f / fx_f, 2),
        "tariff_vnd_kwh": tariff_f,
        "fx_vnd_usd": fx_f,
        "eff_kw": round(eff_kw_f, 2),
        "early_day": early_d,
        "cal_day": cal_d,
        "days_saved": days_saved,
        "restored_fraction": rest_f,
        "hours_per_day": hrs_f,
        "daily_burn_vnd": daily_burn_vnd,
        "daily_burn_usd": round(daily_burn_vnd / fx_f, 2),
        "avoided_gross_vnd": avoided_gross_vnd,
        "avoided_gross_usd": round(avoided_gross_vnd / fx_f, 2),
        "net_benefit_vnd": net_benefit_vnd,
        "net_benefit_usd": round(net_benefit_vnd / fx_f, 2),
        "roi_pct": roi_pct,
        "payback_days": payback_days,
        "payback_calendar_day": payback_calendar_day,
        "latest_profitable_day": latest_profitable_day,
        "eoq_sensitivity_days": eoq_sensitivity_days,
        "tou_avoided_vnd": tou_avoided_vnd,
        "illustrative_only": True,
        "verified_excess_kwh": None,
        "verified_avoided_kwh": None,
    }
