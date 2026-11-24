"""Optical Tachometer + 1D Acoustic FFT (H3 / Step IV-2) & Stull (2011) Wet-Bulb Weather Gate (H4).

100% Offline, Air-Gapped (Purdue Level 3 / IEC 62443-3-3; zero outbound HTTP).
- Acoustic FFT runs real scipy.fft.rfft on 16 kHz audio to compute band RMS energies:
    * blade_pass_60_160hz (healthy fan aerodynamics)
    * bearing_180_420hz   (outer-race / ball-pass wear signature)
    * belt_squeal_2400_4200hz (glazed V-belt slip squeal)
- Stull (2011) Psychrometric Wet-Bulb Validator enforces thermodynamic gate T_wb <= T_db,
  directly resolving the raw LBNL OA_TEMP_WB > OA_TEMP (99.31%) sensor-mapping audit defect.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.fft import rfft, rfftfreq

VALID_ACOUSTIC_MODES = ("healthy", "belt_slip", "bearing_wear", "fan_stall")
VALID_WEATHER_PRESETS = ("hanoi_humid", "bac_ninh_peak", "cool_dry_standard", "lbnl_corrupted_wb")

SAMPLE_RATE_HZ = 16000
N_SAMPLES = 16000
MAX_SPEC_HZ = 5000.0
NUM_SPEC_BINS = 80

WEATHER_PRESET_SPECS: Dict[str, Dict[str, Any]] = {
    "hanoi_humid": {
        "location": "Hanoi Industrial Park (Moderate Humid)",
        "dry_bulb_c": 27.5,
        "rh_pct": 74.0,
        "wet_bulb_override_c": None,
    },
    "bac_ninh_peak": {
        "location": "Bac Ninh Fab Summer Peak (High Wet-Bulb)",
        "dry_bulb_c": 36.8,
        "rh_pct": 72.0,
        "wet_bulb_override_c": None,
    },
    "cool_dry_standard": {
        "location": "Cool Dry Baseline (ASHRAE Standard)",
        "dry_bulb_c": 22.0,
        "rh_pct": 50.0,
        "wet_bulb_override_c": None,
    },
    "lbnl_corrupted_wb": {
        "location": "LBNL Raw Sensor Audit (OA_TEMP_WB > OA_TEMP)",
        "dry_bulb_c": 24.0,
        "rh_pct": 62.0,
        "wet_bulb_override_c": 29.4,
    },
}


# ---------------------------------------------------------------------------
# Optical Tachometer + Real 16 kHz 1D Acoustic FFT (H3 / Step IV-2)
# ---------------------------------------------------------------------------

def run_acoustic_tach_analysis(mode: str = "healthy", vfd_pct: float = 95.0) -> Dict[str, Any]:
    """Run real 16 kHz scipy.fft.rfft acoustic spectrum & optical tachometer slip diagnostics."""
    if mode not in VALID_ACOUSTIC_MODES:
        raise ValueError(f"acoustic mode must be one of {VALID_ACOUSTIC_MODES}")

    t0 = time.perf_counter()
    vfd_clamped = max(10.0, min(100.0, float(vfd_pct)))
    expected_rpm = round((vfd_clamped / 100.0) * 1450.0, 1)

    seed_map = {"healthy": 101, "belt_slip": 202, "bearing_wear": 303, "fan_stall": 404}
    rng = np.random.default_rng(seed_map[mode])
    t = np.arange(N_SAMPLES, dtype=np.float64) / float(SAMPLE_RATE_HZ)

    # 5-blade 120mm/cooling-tower fan blade-pass frequency (BPF)
    if mode == "healthy":
        optical_rpm = round(expected_rpm * 0.988, 1)
        bpf = max(65.0, min(155.0, (optical_rpm / 60.0) * 5.0))
        sig = (
            0.34 * np.sin(2.0 * np.pi * bpf * t)
            + 0.09 * np.sin(2.0 * np.pi * (2.0 * bpf) * t)
            + rng.normal(0.0, 0.035, N_SAMPLES)
        )
    elif mode == "belt_slip":
        optical_rpm = round(expected_rpm * 0.715, 1)
        bpf = max(62.0, min(145.0, (optical_rpm / 60.0) * 5.0))
        sig = (
            0.22 * np.sin(2.0 * np.pi * bpf * t)
            + 0.42 * np.sin(2.0 * np.pi * 2760.0 * t)
            + 0.36 * np.sin(2.0 * np.pi * 3240.0 * t)
            + 0.24 * np.sin(2.0 * np.pi * 3680.0 * t)
            + rng.normal(0.0, 0.045, N_SAMPLES)
        )
    elif mode == "bearing_wear":
        optical_rpm = round(expected_rpm * 0.942, 1)
        bpf = max(65.0, min(155.0, (optical_rpm / 60.0) * 5.0))
        sig = (
            0.28 * np.sin(2.0 * np.pi * bpf * t)
            + 0.44 * np.sin(2.0 * np.pi * 245.0 * t)
            + 0.35 * np.sin(2.0 * np.pi * 318.0 * t)
            + 0.26 * np.sin(2.0 * np.pi * 388.0 * t)
            + rng.normal(0.0, 0.04, N_SAMPLES)
        )
    else:  # fan_stall
        optical_rpm = 0.0
        sig = (
            0.14 * np.sin(2.0 * np.pi * 50.0 * t)
            + 0.05 * np.sin(2.0 * np.pi * 100.0 * t)
            + rng.normal(0.0, 0.025, N_SAMPLES)
        )

    slip_pct = round(max(0.0, min(100.0, (1.0 - (optical_rpm / max(1.0, expected_rpm))) * 100.0)), 1)

    # Real Hann-windowed 1D FFT via scipy.fft.rfft
    window = np.hanning(N_SAMPLES)
    spec = rfft(sig * window)
    freqs = rfftfreq(N_SAMPLES, d=1.0 / float(SAMPLE_RATE_HZ))
    mag = (2.0 / float(np.sum(window))) * np.abs(spec)

    def _band_rms(f_lo: float, f_hi: float) -> float:
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(mask):
            return 0.0
        # Parseval-scaled RMS energy in band
        return round(float(np.sqrt(np.sum(mag[mask] ** 2) / 2.0)), 4)

    blade_rms = _band_rms(60.0, 160.0)
    bearing_rms = _band_rms(180.0, 420.0)
    belt_rms = _band_rms(2400.0, 4200.0)

    # Peak-preserving 80-bin spectrum across 0..5000 Hz (62.5 Hz per bin)
    bin_width = MAX_SPEC_HZ / float(NUM_SPEC_BINS)
    spectrum_80: List[Dict[str, float]] = []
    for i in range(NUM_SPEC_BINS):
        f_lo = i * bin_width
        f_hi = (i + 1) * bin_width
        mask = (freqs >= f_lo) & (freqs < f_hi)
        peak_mag = float(np.max(mag[mask])) if np.any(mask) else 1e-6
        db_val = round(max(-60.0, min(0.0, 20.0 * math.log10(max(peak_mag, 1e-6)))), 1)
        spectrum_80.append({"freq_hz": round(f_lo + bin_width * 0.5, 1), "db": db_val})

    valid_mask = (freqs >= 30.0) & (freqs <= MAX_SPEC_HZ)
    dom_idx = int(np.argmax(mag[valid_mask]))
    dominant_freq_hz = round(float(freqs[valid_mask][dom_idx]), 1)

    # Deterministic verdict from actual FFT energies + optical tachometer slip
    if optical_rpm < 30.0:
        verdict = "ISSUE_STALL"
        summary = f"Stall: {optical_rpm:.0f}/{expected_rpm:.0f} RPM (slip {slip_pct:.1f}%)."
    elif belt_rms >= 0.14 or slip_pct >= 15.0:
        verdict = "ISSUE_BELT_SLIP"
        summary = f"Belt slip: {optical_rpm:.0f}/{expected_rpm:.0f} RPM (slip {slip_pct:.1f}%), squeal RMS {belt_rms:.3f}."
    elif bearing_rms >= 0.14:
        verdict = "ISSUE_BEARING"
        summary = f"Bearing wear: 180–420 Hz RMS {bearing_rms:.3f}, slip {slip_pct:.1f}%."
    else:
        verdict = "OK"
        summary = f"Healthy: {optical_rpm:.0f}/{expected_rpm:.0f} RPM (slip {slip_pct:.1f}%)."

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "mode": mode,
        "verdict": verdict,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "vfd_pct": round(vfd_clamped, 1),
        "expected_rpm": expected_rpm,
        "optical_rpm": optical_rpm,
        "slip_pct": slip_pct,
        "bands": {
            "blade_pass_60_160hz": blade_rms,
            "bearing_180_420hz": bearing_rms,
            "belt_squeal_2400_4200hz": belt_rms,
        },
        "dominant_freq_hz": dominant_freq_hz,
        "spectrum_80": spectrum_80,
        "summary": summary,
        "advisory_only": True,
        "latency_ms": latency_ms,
    }


def analyze_pcm_fan_audio(
    pcm_samples: Any,
    sample_rate: int = 16000,
    source: str = "iphone11_mic",
    num_blades: int = 7,
    vfd_pct: float = 95.0,
) -> Dict[str, Any]:
    """Analyze real live PCM microphone audio (Hann-windowed rfft) for BPF RPM & mechanical faults."""
    t0 = time.perf_counter()
    sr = int(sample_rate)
    if not (4000 <= sr <= 96000):
        raise ValueError("sample_rate must be between 4000 and 96000 Hz.")
    if num_blades < 2 or num_blades > 24:
        raise ValueError("num_blades must be between 2 and 24.")

    try:
        sig = np.asarray(pcm_samples, dtype=np.float64).ravel()
    except Exception as exc:
        raise ValueError("pcm_samples must be a numeric 1D sequence.") from exc
    if sig.size < 256 or sig.size > 240000:
        raise ValueError("pcm_samples length must be between 256 and 240000 samples.")
    if not np.all(np.isfinite(sig)):
        raise ValueError("pcm_samples contains NaN or infinite values.")

    # Auto-scale int16-range captures into [-1.0, 1.0]
    peak_abs = float(np.max(np.abs(sig)))
    if peak_abs > 2.0:
        sig = sig / 32768.0

    sig = sig - float(np.mean(sig))
    sig_rms = float(np.sqrt(np.mean(sig ** 2)))
    n = sig.size

    window = np.hanning(n)
    spec = rfft(sig * window)
    freqs = rfftfreq(n, d=1.0 / float(sr))
    mag = (2.0 / float(np.sum(window))) * np.abs(spec)

    def _band_rms(f_lo: float, f_hi: float, exclude_center: Optional[float] = None, exclude_hw: float = 16.0) -> float:
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if exclude_center is not None:
            mask = mask & (np.abs(freqs - exclude_center) > exclude_hw)
        if not np.any(mask):
            return 0.0
        return round(float(np.sqrt(np.sum(mag[mask] ** 2) / 2.0)), 4)

    nyq = min(MAX_SPEC_HZ, float(sr) * 0.5)
    valid_mask = (freqs >= 35.0) & (freqs <= nyq)
    if np.any(valid_mask) and sig_rms >= 1e-4:
        peak_freq_hz = round(float(freqs[valid_mask][int(np.argmax(mag[valid_mask]))]), 1)
    else:
        peak_freq_hz = 0.0

    # Locate 7-blade fan blade-pass frequency (55..260 Hz -> ~470..2230 RPM)
    bpf_mask = (freqs >= 55.0) & (freqs <= min(260.0, nyq))
    if np.any(bpf_mask) and sig_rms >= 1e-4:
        bpf_sub_mag = mag[bpf_mask]
        bpf_sub_freqs = freqs[bpf_mask]
        bpf_idx = int(np.argmax(bpf_sub_mag))
        bpf_peak_amp = float(bpf_sub_mag[bpf_idx])
        bpf_freq_hz = round(float(bpf_sub_freqs[bpf_idx]), 1) if bpf_peak_amp >= 0.008 else 0.0
    else:
        bpf_freq_hz = 0.0

    bpf_derived_rpm = round((bpf_freq_hz / float(num_blades)) * 60.0, 1) if bpf_freq_hz > 0.0 else 0.0
    expected_rpm = round((max(10.0, min(100.0, float(vfd_pct))) / 100.0) * 1450.0, 1)
    slip_pct = round(max(0.0, min(100.0, abs(expected_rpm - bpf_derived_rpm) / max(expected_rpm, 1.0) * 100.0)), 1)

    blade_rms = _band_rms(60.0, 160.0)
    bpf_band_rms = _band_rms(55.0, 260.0)
    bearing_rms = _band_rms(180.0, 420.0)
    bearing_non_bpf_rms = _band_rms(180.0, 420.0, exclude_center=bpf_freq_hz if bpf_freq_hz > 0 else None, exclude_hw=18.0)
    belt_rms = _band_rms(2400.0, 4200.0)

    defect_rms = math.hypot(bearing_non_bpf_rms, belt_rms)
    bearing_harmonic_ratio = round(defect_rms / max(bpf_band_rms, 1e-4), 4) if sig_rms >= 1e-4 else 0.0

    # Peak-preserving 80-bin FFT spectrum (0..5000 Hz) compatible with Tab 5 SVG renderer
    bin_width = MAX_SPEC_HZ / float(NUM_SPEC_BINS)
    spectrum_80: List[Dict[str, float]] = []
    for i in range(NUM_SPEC_BINS):
        f_lo = i * bin_width
        f_hi = (i + 1) * bin_width
        mask = (freqs >= f_lo) & (freqs < f_hi)
        peak_m = float(np.max(mag[mask])) if np.any(mask) else 1e-6
        db_val = round(max(-60.0, min(0.0, 20.0 * math.log10(max(peak_m, 1e-6)))), 1)
        spectrum_80.append({"freq_hz": round(f_lo + bin_width * 0.5, 1), "db": db_val})

    if sig_rms < 1e-4 or bpf_derived_rpm < 30.0:
        verdict = "ISSUE_MECHANICAL"
        detail_verdict = "ISSUE_STALL"
        summary = f"Live mic stall/silence: BPF {bpf_freq_hz:.1f} Hz ({bpf_derived_rpm:.0f} RPM)."
    elif belt_rms >= 0.08 and (belt_rms / max(bpf_band_rms, 1e-4)) >= 0.55:
        verdict = "ISSUE_MECHANICAL"
        detail_verdict = "ISSUE_BELT_SLIP"
        summary = f"Live mic mechanical fault (belt/rub): peak {peak_freq_hz:.0f} Hz, 2.4–4.2 kHz RMS {belt_rms:.3f}."
    elif bearing_non_bpf_rms >= 0.08 and bearing_harmonic_ratio >= 0.55:
        verdict = "ISSUE_MECHANICAL"
        detail_verdict = "ISSUE_BEARING"
        summary = f"Live mic mechanical fault (bearing): 180–420 Hz ratio {bearing_harmonic_ratio:.2f}, peak {peak_freq_hz:.0f} Hz."
    else:
        verdict = "OK"
        detail_verdict = "OK"
        summary = f"Live mic OK: BPF {bpf_freq_hz:.1f} Hz → {bpf_derived_rpm:.0f} RPM ({num_blades} blades), ratio {bearing_harmonic_ratio:.2f}."

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "mode": f"live_pcm:{source}",
        "source": source,
        "verdict": verdict,
        "detail_verdict": detail_verdict,
        "sample_rate_hz": sr,
        "num_samples": int(n),
        "num_blades": int(num_blades),
        "vfd_pct": round(float(vfd_pct), 1),
        "expected_rpm": expected_rpm,
        "optical_rpm": bpf_derived_rpm,
        "bpf_freq_hz": bpf_freq_hz,
        "bpf_derived_rpm": bpf_derived_rpm,
        "peak_freq_hz": peak_freq_hz,
        "dominant_freq_hz": peak_freq_hz,
        "slip_pct": slip_pct,
        "bearing_harmonic_ratio": bearing_harmonic_ratio,
        "bands": {
            "blade_pass_60_160hz": blade_rms,
            "blade_pass_60_260hz": bpf_band_rms,
            "bearing_180_420hz": bearing_rms,
            "belt_squeal_2400_4200hz": belt_rms,
        },
        "spectrum_80": spectrum_80,
        "summary": summary,
        "advisory_only": True,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Stull (2011) Psychrometric Wet-Bulb Weather Validator (H4)
# ---------------------------------------------------------------------------

def stull_wet_bulb_c(dry_bulb_c: float, rh_pct: float) -> float:
    """Exact Stull (2011) empirical wet-bulb temperature equation (degC)."""
    t = float(dry_bulb_c)
    rh = float(rh_pct)
    twb = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return round(twb, 2)


def magnus_dew_point_c(dry_bulb_c: float, rh_pct: float) -> float:
    """Magnus-Tetens dew-point temperature (degC)."""
    t = float(dry_bulb_c)
    rh = max(1.0, min(100.0, float(rh_pct)))
    a, b = 17.625, 243.04
    gamma = math.log(rh / 100.0) + (a * t) / (b + t)
    tdp = (b * gamma) / (a - gamma)
    return round(tdp, 2)


def evaluate_stull_weather(
    preset: Optional[str] = None,
    dry_bulb_c: Optional[float] = None,
    rh_pct: Optional[float] = None,
    wet_bulb_override_c: Optional[float] = None,
    location: Optional[str] = None,
    cw_outlet_c: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate psychrometric wet-bulb physics (T_wb <= T_db) & H4 high-wet-bulb confounder."""
    t0 = time.perf_counter()

    active_preset = "custom"
    if preset is not None:
        if preset not in VALID_WEATHER_PRESETS:
            raise ValueError(f"weather preset must be one of {VALID_WEATHER_PRESETS}")
        spec = WEATHER_PRESET_SPECS[preset]
        active_preset = preset
        t_db = float(spec["dry_bulb_c"]) if dry_bulb_c is None else float(dry_bulb_c)
        rh = float(spec["rh_pct"]) if rh_pct is None else float(rh_pct)
        wb_ovr = spec["wet_bulb_override_c"] if wet_bulb_override_c is None else float(wet_bulb_override_c)
        loc = location or str(spec["location"])
    else:
        if dry_bulb_c is None or rh_pct is None:
            spec = WEATHER_PRESET_SPECS["hanoi_humid"]
            active_preset = "hanoi_humid"
            t_db = float(spec["dry_bulb_c"]) if dry_bulb_c is None else float(dry_bulb_c)
            rh = float(spec["rh_pct"]) if rh_pct is None else float(rh_pct)
            wb_ovr = wet_bulb_override_c
            loc = location or str(spec["location"])
        else:
            t_db = float(dry_bulb_c)
            rh = float(rh_pct)
            wb_ovr = wet_bulb_override_c
            loc = location or "Local Site Weather Input (Air-Gapped)"

    if not (-20.0 <= t_db <= 55.0):
        raise ValueError("dry_bulb_c must be within [-20.0, 55.0] degC.")
    if not (5.0 <= rh <= 100.0):
        raise ValueError("rh_pct must be within [5.0, 100.0] %.")
    if wb_ovr is not None and not (-25.0 <= float(wb_ovr) <= 60.0):
        raise ValueError("wet_bulb_override_c must be within [-25.0, 60.0] degC.")

    stull_twb = stull_wet_bulb_c(t_db, rh)
    dew_pt = magnus_dew_point_c(t_db, rh)
    reported_twb = round(float(wb_ovr), 2) if wb_ovr is not None else stull_twb

    wb_le_db_ok = reported_twb <= (t_db + 0.05)
    dp_le_wb_ok = dew_pt <= (reported_twb + 0.5)
    stull_consistent = abs(reported_twb - stull_twb) <= 2.5
    physics_valid = bool(wb_le_db_ok and dp_le_wb_ok and stull_consistent)

    tower_approach_est_c: Optional[float] = (
        round(float(cw_outlet_c) - stull_twb, 2)
        if (physics_valid and cw_outlet_c is not None)
        else None
    )

    if not physics_valid:
        confounder_verdict = "SENSOR_CORRUPTED_REJECTED"
        summary = f"Rejected: T_wb {reported_twb:.2f}°C > T_db {t_db:.2f}°C (Stull {stull_twb:.2f}°C)."
    elif stull_twb >= 27.5:
        confounder_verdict = "HIGH_WETBULB_CONFOUNDER"
        summary = f"High wet-bulb: Stull T_wb {stull_twb:.2f}°C at {t_db:.1f}°C / {rh:.0f}% RH."
    else:
        confounder_verdict = "WEATHER_CLEAR"
        summary = f"Gate pass: T_wb {stull_twb:.2f}°C ≤ T_db {t_db:.1f}°C, T_dp {dew_pt:.2f}°C."

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "preset": active_preset,
        "location": loc,
        "dry_bulb_c": round(t_db, 2),
        "rh_pct": round(rh, 1),
        "stull_wet_bulb_c": stull_twb,
        "reported_wet_bulb_c": reported_twb,
        "dew_point_c": dew_pt,
        "wb_le_db_ok": wb_le_db_ok,
        "physics_valid": physics_valid,
        "tower_approach_est_c": tower_approach_est_c,
        "confounder_verdict": confounder_verdict,
        "air_gapped": True,
        "advisory_only": True,
        "summary": summary,
        "latency_ms": latency_ms,
    }
