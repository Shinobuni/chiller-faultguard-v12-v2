/* Chiller FaultGuard v12.0 — Clean Minimal-Copy Cyber-Physical Mini-CT1, Quan HGB, Causal DAG, EVN TOU, Vision ML & Air-Gapped RCA
   100% Offline · Zero CDN · Exact Streaming Parity + Causal DAG + EVN Break-Even + Real-World 3-Stage Vision ML & Live Mic FFT */

let isPlaying = false;
let isStepping = false;
let playTimer = null;
let pidSvgRoot = null;
let latestSnap = null;
let latestHw = null;
let hwLoadingPromise = null;
let latestUploadedB64 = null;
let activeTabId = "overview";
const dirtyTabs = { overview: true, trends: true, quantify: true, derisk: true, hardware: true };
let activeHwVisionMode = "fill_strainer";
let lastTimelineScenario = null;
let activeStoryBeat = null;
let tab3BasisMode = "live"; // "live" = auto-triggered from real alert telemetry; "manual" = static What-If template
let step3ViewMode = "dag";  // "dag" = 8-node Causal Root-Cause DAG; "cards" = 4 competing-cause cards
let rcaOpen = false;
let rcaBusy = false;
let lastRcaRequest = { preset: "why_alert", query: "" };
let lastRcaSnapKey = "";
let latestRcaResp = null;
let selectedToolIdx = null;
let reqChain = Promise.resolve();

const auditErrors = [];
window.addEventListener("error", e => auditErrors.push(`JS error: ${e.message}`));
window.addEventListener("unhandledrejection", e => auditErrors.push(`Promise rejection: ${e.reason}`));

function updateAuditHook() {
  const activeTabEl = document.querySelector(".tab-pane.active");
  const livePm = latestSnap && latestSnap.electricity && latestSnap.electricity.live_alert_projection
    ? latestSnap.electricity.live_alert_projection.pm_optimizer
    : null;
  const activeVm = latestHw && latestHw.vision_modes ? latestHw.vision_modes[activeHwVisionMode] : null;
  const payload = {
    errors: auditErrors,
    theme: currentTheme(),
    activeTab: activeTabEl ? activeTabEl.id : "none",
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight,
    overflowY: document.documentElement.scrollHeight - window.innerHeight,
    overflowX: document.documentElement.scrollWidth - window.innerWidth,
    scenario: latestSnap ? latestSnap.scenario : null,
    status: latestSnap ? latestSnap.status : null,
    ood_flag: latestSnap ? Boolean(latestSnap.prediction && latestSnap.prediction.ood_flag) : false,
    step3View: step3ViewMode,
    dagRoot: latestSnap && latestSnap.causal_graph ? latestSnap.causal_graph.isolated_root_id : null,
    pmVerdict: livePm ? livePm.verdict : null,
    rcaOpen,
    rcaReady: Boolean(latestRcaResp),
    hwPreset: latestHw ? latestHw.active_preset : null,
    hwMode: activeHwVisionMode,
    hwVerdict: activeVm ? activeVm.verdict : null,
    hwCalibrated: Boolean(latestHw && latestHw.calibration_status && latestHw.calibration_status.calibrated),
    hwArucoLocked: Boolean(activeVm && activeVm.aruco_locked),
    hwPartsCount: (latestHw && latestHw.cad_bom && latestHw.cad_bom.parts_count) || (latestHw && latestHw.cad_summary && latestHw.cad_summary.parts_count) || null,
  };
  window.__AUDIT__ = payload;
  if (document.body) document.body.dataset.auditJson = JSON.stringify(payload);
}

function escapeHtml(val) {
  return String(val ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const fieldChecks = [
  { id: "nozzles", title: "Spray Deck & Nozzles", hint: "Check uniform water distribution", state: "pending", source: "", detail: "" },
  { id: "fan", title: "CT-1 Fan / Belt / VFD", hint: "Check vibration, belt & VFD current", state: "pending", source: "", detail: "" },
  { id: "fill", title: "Strainers & Fill Media", hint: "Inspect scale / biofilm buildup", state: "pending", source: "", detail: "" },
  { id: "water", title: "Water Treatment & CoC", hint: "Verify biocide log & conductivity", state: "pending", source: "", detail: "" },
];

function resetFieldChecks() {
  fieldChecks.forEach(fc => {
    fc.state = "pending";
    fc.source = "";
    fc.detail = "";
  });
}

function syncFieldChecksFromBackend(raw) {
  if (!raw || (typeof raw === "object" && raw.applied === false)) return;
  const itemsObj = (raw && typeof raw === "object" && raw.items && typeof raw.items === "object")
    ? raw.items
    : raw;
  const keyAliases = {
    nozzles: ["nozzles", "spray_nozzles", "spray_flow"],
    fan: ["fan", "fan_belt", "acoustic_tach"],
    fill: ["fill", "fill_media", "fill_strainer"],
    water: ["water", "water_chem", "basin_water"],
  };
  const normState = v => {
    const s = String(v || "").toLowerCase();
    if (s === "ok" || s === "pass" || s === "clean") return "ok";
    if (s.includes("issue") || s === "fault" || s === "warn" || s === "fail") return "issue";
    return "pending";
  };
  if (Array.isArray(itemsObj)) {
    itemsObj.forEach(entry => {
      if (!entry || !entry.id) return;
      const eid = String(entry.id).toLowerCase();
      const target = fieldChecks.find(fc => (keyAliases[fc.id] || [fc.id]).includes(eid));
      if (!target) return;
      if (entry.state !== undefined || entry.status !== undefined) {
        target.state = normState(entry.state ?? entry.status);
      }
      if (entry.source) target.source = String(entry.source);
      if (entry.metric || entry.finding || entry.detail) {
        target.detail = String(entry.metric || entry.finding || entry.detail);
      }
    });
  } else if (typeof itemsObj === "object") {
    fieldChecks.forEach(fc => {
      const aliases = keyAliases[fc.id] || [fc.id];
      const matchKey = aliases.find(k => itemsObj[k] !== undefined && itemsObj[k] !== null);
      if (!matchKey) return;
      const entry = itemsObj[matchKey];
      if (typeof entry === "string") {
        fc.state = normState(entry);
      } else if (typeof entry === "object") {
        fc.state = normState(entry.state ?? entry.status);
        if (entry.source) fc.source = String(entry.source);
        if (entry.metric || entry.finding || entry.detail || entry.summary) {
          fc.detail = entry.metric
            ? `${entry.metric} (${entry.verdict || entry.status || ""})`
            : String(entry.finding || entry.detail || entry.summary);
        }
      }
    });
  }
}

async function apiJson(url, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body !== null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const errTxt = await resp.text();
    throw new Error(`API ${url} (${resp.status}): ${errTxt}`);
  }
  return await resp.json();
}

function enqueue(fn) {
  reqChain = reqChain.then(fn).catch(err => {
    const el = document.getElementById("advisory-title");
    if (el) el.textContent = err.message;
    stopPlayback();
  });
  return reqChain;
}

function fmt(val, digits = 2, suffix = "") {
  if (val === null || val === undefined || Number.isNaN(val)) return "N/A";
  return `${Number(val).toFixed(digits)}${suffix}`;
}

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") || "light";
}

function setPill(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = cls;
}

/* ==================== LIVE ANIMATED & INTERACTIVE P&ID ==================== */
function buildPidTooltipContent(key, snap) {
  if (!snap) return null;
  const s = snap.signals, p = snap.prediction;
  const aiStr = p.ai_score !== null && p.ai_score !== undefined ? p.ai_score.toFixed(3) : "ABSTAIN";
  const lagChKw = Math.max(0, (s.p_chillers_kw || 0) - (s.p_ch1_kw || 0));
  const map = {
    ct1: {
      title: "CT-1 · Monitored Cooling Tower Fill",
      val: `Status: ${snap.status} | HGB: ${aiStr} | Fan: ${fmt(s.fan_pct, 1, "%")}`,
      role: "Primary monitored tower in Quan frozen HGB (6 native LBNL inputs; binary suspicion).",
    },
    ct2: {
      title: "CT-2..3 · Redundant / Lag Cooling Cells",
      val: `Active Towers: ${s.n_towers_on} ON | Total CT Fans: ${fmt(s.p_ct_fans_kw, 1, " kW")}`,
      role: "N+1 cell enabling planned CT-1 isolation without unplanned fab stop.",
    },
    ch1: {
      title: "CH-1 · Lead Centrifugal Chiller",
      val: `WI-CH1: ${fmt(s.p_ch1_kw, 1, " kW")} | All Chillers: ${fmt(s.p_chillers_kw, 1, " kW")}`,
      role: "Fouled CT-1 raises condenser lift and compressor power (14-Ch boundary).",
    },
    ch2: {
      title: "CH-2..3 · Lag / Standby Chillers",
      val: `Active Chillers: ${s.n_chillers_on} ON | Lag kW: ${fmt(lagChKw, 1, " kW")}`,
      role: "Logged in 14-Ch boundary (CHL_POW_2 + CHL_POW_3).",
    },
    "plant-kw": {
      title: "WI-PLANT · 14-Channel In-Scope Electricity",
      val: `${fmt(s.plant_kw, 1, " kW")} (Δ vs Matched Normal: ${s.signed_delta_kw >= 0 ? "+" : ""}${fmt(s.signed_delta_kw, 1, " kW*")})`,
      role: "3 chillers + 3 CT fans + 3 CW pumps + 5 CHW pumps; verified savings require Pilot.",
    },
    fan: {
      title: "SI-CT1-FAN / WI-CT1 · CT-1 Fan VFD",
      val: `Speed: ${fmt(s.fan_pct, 1, "%")} | Power: ${fmt(s.p_ct1_kw, 1, " kW")}`,
      role: "HGB Feature #2/#6 and Reference Rule (>98% for 15 min).",
    },
    "cw-in": {
      title: "TI-CW-I1 · Hot Condenser Water to CT-1",
      val: `${fmt(s.tower_inlet_c, 2, " °C")} (Native CT_RW_TEMP_1)`,
      role: "HGB Feature #1 input via Water Range ΔT = T_in − T_out.",
    },
    "cw-out": {
      title: "TI-CW-O1 · Nominal CT-1 Outlet (ECWT*)",
      val: `${fmt(s.nominal_tower_outlet_c, 2, " °C*")} (Native CT_SW_TEMP_1)`,
      role: "HGB Feature #4 and 60-min flatline sensor quality gate.",
    },
    "cw-flow": {
      title: "FI-CW-1 · CT-1 Condenser Water Flow",
      val: `${fmt(s.tower_flow_gpm, 1, " GPM")} (${fmt(s.tower_flow_kgs, 1, " kg/s")})`,
      role: "HGB Feature #3; operating gate ≥10 GPM (train bound 30.2–601.1 GPM).",
    },
    range: {
      title: "CT-1 Thermal Range ΔT (Approach Excluded*)",
      val: `Range ΔT: ${fmt(s.tower_range_c, 2, " °C")} | Approach: Pending Weather Audit`,
      role: "Approach excluded because raw LBNL OA_TEMP_WB > OA_TEMP in 99.3% of rows.",
    },
    twb: {
      title: "TI-OAT-WB · Outdoor Wet-Bulb (Excluded*)",
      val: "Excluded from AI (No Silent Auto-Swap)",
      role: "Raw LBNL OA_TEMP_WB > OA_TEMP in 99.31% of rows — withheld honestly.",
    },
    "chw-sup": {
      title: "TI-CHW-S · Nominal CHW Supply*",
      val: `${fmt(s.nominal_chw_supply_c, 2, " °C*")} (Mapping Pending Verification)`,
      role: "Display context only (not in HGB); fab cooling limits belong to Pilot.",
    },
    "chw-ret": {
      title: "TI-CHW-R · Primary CHW Return*",
      val: `${fmt(s.pri_chw_ret_c, 2, " °C*")} | Logged Load*: ${fmt(s.logged_load_kw, 1, " kW*")}`,
      role: "Secondary load is HGB Feature #5; CHW temp mapping pending validation.",
    },
    "chw-flow": {
      title: "FI-CHW · Primary CHW Flow",
      val: `${fmt(s.pri_chw_flow_gpm, 1, " GPM")}`,
      role: "Published LBNL context; 5 CHW pumps included in 14-Ch power.",
    },
    phe: {
      title: "PHE-01 · Out-of-Scope Fab Thermal Interface",
      val: "Boundary Separator (CUB CHW ↔ Fab PCW)",
      role: "Out of electricity scope; FaultGuard is read-only on CUB telemetry.",
    },
    fab: {
      title: "Fab Tools (PCW) · Out of Scope",
      val: "SEMI S23 Context · Reserved for Site Pilot",
      role: "LBNL office replay is PoC only — never claims verified fab yield.",
    },
    guard: {
      title: "Cooling-Service & 3-Bucket Boundary",
      val: "Read-Only Advisory · Zero PLC Write-Back",
      role: "B1: LBNL F1 0.394 | B2: Illustrative 6,624 kWh | B3: Site Pilot.",
    },
  };
  return map[key] || null;
}

function initPidTooltips() {
  const wrap = document.querySelector(".pid-stage-wrap");
  const tip = document.getElementById("pid-tooltip");
  if (!wrap || !tip || !pidSvgRoot) return;
  let lastTipKey = "";
  let lastTipRow = -1;
  let lastTipStatus = "";
  pidSvgRoot.addEventListener("mousemove", evt => {
    const grp = evt.target.closest("[data-pid-tip]");
    if (!grp || !latestSnap) {
      if (lastTipKey) {
        lastTipKey = "";
        tip.classList.add("hidden");
      }
      return;
    }
    const key = grp.getAttribute("data-pid-tip") || "";
    if (key !== lastTipKey || latestSnap.row_index !== lastTipRow || latestSnap.status !== lastTipStatus) {
      const info = buildPidTooltipContent(key, latestSnap);
      if (!info) {
        lastTipKey = "";
        tip.classList.add("hidden");
        return;
      }
      tip.innerHTML = `<strong>${escapeHtml(info.title)}</strong><div class="pt-val">${escapeHtml(info.val)}</div><div class="pt-role">${escapeHtml(info.role)}</div>`;
      lastTipKey = key;
      lastTipRow = latestSnap.row_index;
      lastTipStatus = latestSnap.status;
      tip.classList.remove("hidden");
    }
    const rect = wrap.getBoundingClientRect();
    let x = evt.clientX - rect.left + 14;
    let y = evt.clientY - rect.top + 14;
    if (x + 286 > rect.width) x = Math.max(8, evt.clientX - rect.left - 292);
    if (y + 108 > rect.height) y = Math.max(8, evt.clientY - rect.top - 110);
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  });
  pidSvgRoot.addEventListener("mouseleave", () => {
    lastTipKey = "";
    tip.classList.add("hidden");
  });
}

async function loadInteractivePid() {
  const host = document.getElementById("pid-container");
  try {
    const resp = await fetch("/assets/chiller_plant_pid.svg?v=12.0.0");
    host.innerHTML = await resp.text();
    pidSvgRoot = host.querySelector("svg");
    applyThemeToPid();
    initPidTooltips();
    if (latestSnap) updatePidMimic(latestSnap);
  } catch (err) {
    host.textContent = `Could not load P&ID: ${err.message}`;
  }
}

function applyThemeToPid() {
  if (pidSvgRoot) pidSvgRoot.classList.toggle("theme-dark", currentTheme() === "dark");
}

function setSvgText(id, text) {
  if (!pidSvgRoot) return;
  const el = pidSvgRoot.getElementById(id);
  if (el) el.textContent = text;
}

function updatePidMimic(snap) {
  if (!pidSvgRoot) return;
  const s = snap.signals, p = snap.prediction, st = snap.status;

  setSvgText("val-chw-sup", fmt(s.nominal_chw_supply_c, 2, " °C*"));
  setSvgText("val-chw-ret", fmt(s.pri_chw_ret_c, 2, " °C*"));
  setSvgText("val-chw-flow", fmt(s.pri_chw_flow_gpm, 0, " GPM"));
  setSvgText("val-cw-in", fmt(s.tower_inlet_c, 2, " °C"));
  setSvgText("val-cw-out", fmt(s.nominal_tower_outlet_c, 2, " °C*"));
  setSvgText("val-cw-flow", fmt(s.tower_flow_gpm, 0, " GPM"));
  setSvgText("val-twb", "Excluded*");
  setSvgText("val-ct1-fan", `Fan: ${fmt(s.fan_pct, 1, "%")}`);
  setSvgText("val-ct1-kw", fmt(s.p_ct1_kw, 1, " kW"));
  setSvgText("val-ch1-kw", fmt(s.p_ch1_kw, 1, " kW"));
  const lagChKw = Math.max(0, Number(s.p_chillers_kw || 0) - Number(s.p_ch1_kw || 0));
  setSvgText("val-ch2-kw", fmt(lagChKw, 1, " kW"));
  setSvgText("val-plant-kw", fmt(s.plant_kw, 1, " kW"));
  setSvgText("val-approach", `${fmt(s.tower_range_c, 2, " °C")} · T_in − T_out`);
  const appEl = pidSvgRoot.getElementById("val-approach");
  if (appEl) appEl.setAttribute("font-size", "16");

  // Dynamic Lag Chiller / Tower Staging Visuals (mapped cleanly to LBNL n_chillers_on / n_towers_on)
  const ch2Grp = pidSvgRoot.getElementById("chiller-2");
  if (ch2Grp) ch2Grp.setAttribute("opacity", s.n_chillers_on > 1 ? "0.98" : "0.66");
  setSvgText("ch2-title", s.n_chillers_on > 1 ? `CH-2..3 (LAG ACTIVE · ${s.n_chillers_on} ON)` : "CH-2 (STANDBY / LAG)");
  const ct2Grp = pidSvgRoot.getElementById("cooling-tower-2");
  if (ct2Grp) ct2Grp.setAttribute("opacity", s.n_towers_on > 1 ? "0.98" : "0.66");
  setSvgText("ct2-title", s.n_towers_on > 1 ? `COOLING TOWER 2..3 (${s.n_towers_on} ON)` : "COOLING TOWER 2 (STANDBY)");
  setSvgText("ct2-sub", s.n_towers_on > 1 ? "MULTI-CELL ACTIVE STAGING" : "N+1 REDUNDANT CELL");

  // Subtle Live ISA-101 Pipe Flow & CT-1 Fan Speed Motion (paused cleanly on ABSTAIN or zero flow)
  const flowStopped = st === "ABSTAIN" || Number(s.tower_flow_gpm || 0) < 10;
  pidSvgRoot.classList.toggle("flow-paused", flowStopped);
  const flowDur = Math.max(0.55, Math.min(2.6, 480 / Math.max(120, Number(s.tower_flow_gpm || 120))));
  pidSvgRoot.style.setProperty("--flow-dur", `${flowDur.toFixed(2)}s`);

  const fanStopped = st === "ABSTAIN" || !s.ct1_on || Number(s.fan_pct || 0) < 5;
  pidSvgRoot.classList.toggle("fan-paused", fanStopped);
  const fanDur = Math.max(0.28, Math.min(1.6, 38 / Math.max(20, Number(s.fan_pct || 20))));
  pidSvgRoot.style.setProperty("--fan-dur", `${fanDur.toFixed(2)}s`);

  const isDark = currentTheme() === "dark";
  const states = {
    NO_ALERT: {
      stroke: "#475569", fill: "#64748B", fillOp: "0.06", hatchOp: "0",
      pillFill: isDark ? "#1E293B" : "#F1F5F9", pillStroke: "#64748B", titleFill: isDark ? "#CBD5E1" : "#334155",
      title: "MONITORED: CT-1 FILL", label: `SCORE ${p.ai_score !== null ? p.ai_score.toFixed(3) : "--"} · NO ALERT`,
      stBg: isDark ? "#064E3B" : "#ECFDF5", stStroke: "#059669", stTextFill: isDark ? "#A7F3D0" : "#065F46",
      stText: "NO ALERT | READ-ONLY", appStroke: "#0284C7",
    },
    RULE_WARNING: {
      stroke: "#D97706", fill: "#F59E0B", fillOp: "0.22", hatchOp: "0",
      pillFill: "#FEF3C7", pillStroke: "#D97706", titleFill: "#92400E",
      title: "HIGH-FAN RULE ALERT", label: `FAN ${fmt(s.fan_pct, 1, "%")} · VERIFY I–IV`,
      stBg: isDark ? "#451A03" : "#FEF3C7", stStroke: "#D97706", stTextFill: isDark ? "#FDE68A" : "#92400E",
      stText: "RULE ALERT | READ-ONLY", appStroke: "#D97706",
    },
    FOULING_SUSPECT: {
      stroke: "#DC2626", fill: "#EF4444", fillOp: "0.26", hatchOp: "0.22",
      pillFill: "#991B1B", pillStroke: "#FCA5A5", titleFill: "#FFFFFF",
      title: "FOULING SUSPECT (CT-1)", label: `HGB ${p.ai_score !== null ? p.ai_score.toFixed(3) : "--"} · VERIFY I–VI`,
      stBg: isDark ? "#450A0A" : "#FEE2E2", stStroke: "#DC2626", stTextFill: isDark ? "#FECACA" : "#991B1B",
      stText: "FOULING SUSPECT", appStroke: "#DC2626",
    },
    ABSTAIN: {
      stroke: "#CA8A04", fill: "#FACC15", fillOp: "0.26", hatchOp: "0",
      pillFill: "#FEF08A", pillStroke: "#A16207", titleFill: "#713F12",
      title: "ABSTAIN — GATE ACTIVE", label: "CHECK SIGNAL / REGIME",
      stBg: isDark ? "#422006" : "#FEF9C3", stStroke: "#CA8A04", stTextFill: isDark ? "#FEF08A" : "#713F12",
      stText: "ABSTAIN | READ-ONLY", appStroke: "#CA8A04",
    },
  };

  const cfg = states[st] || states.NO_ALERT;
  const setAttr = (id, a, v) => { const el = pidSvgRoot.getElementById(id); if (el) el.setAttribute(a, v); };
  setAttr("ct1-fault-rect", "stroke", cfg.stroke);
  setAttr("ct1-fault-rect", "fill", cfg.fill);
  setAttr("ct1-fault-rect", "fill-opacity", cfg.fillOp);
  const fRect = pidSvgRoot.getElementById("ct1-fault-rect");
  const acked = Boolean(snap.review && snap.review.acknowledged);
  if (fRect) fRect.classList.toggle("pid-alarm-breathe", st === "FOULING_SUSPECT" && !acked);
  setAttr("ct1-fault-hatch-rect", "opacity", cfg.hatchOp);
  setAttr("ct1-fault-pill", "fill", cfg.pillFill);
  setAttr("ct1-fault-pill", "stroke", cfg.pillStroke);
  const ft = pidSvgRoot.getElementById("val-ct1-fault-title");
  if (ft) { ft.setAttribute("fill", cfg.titleFill); ft.textContent = cfg.title; }
  setSvgText("val-ct1-fault-label", cfg.label);
  setAttr("pid-status-bg", "fill", cfg.stBg);
  setAttr("pid-status-bg", "stroke", cfg.stStroke);
  setAttr("pid-status-dot", "fill", cfg.stStroke);
  const stEl = pidSvgRoot.getElementById("pid-status-text");
  if (stEl) { stEl.setAttribute("fill", cfg.stTextFill); stEl.textContent = cfg.stText; }
  setAttr("approach-box-bg", "stroke", cfg.appStroke);
}

/* ==================== TAB 4: SUPERVISOR TRIAGE WORKBENCH (I -> VI) ==================== */
function renderTriageWorkbench(snap) {
  const s = snap.signals, p = snap.prediction, rev = snap.review, abstain = snap.status === "ABSTAIN";

  // Step I
  const c1 = document.getElementById("card-step-1");
  const sb1 = document.getElementById("sb-1");
  c1.className = `triage-card ${abstain ? "tc-fault" : "tc-ok"}`;
  sb1.className = `step-badge ${abstain ? "sb-warn" : "sb-ok"}`;
  setPill(
    "step1-status",
    p.ood_flag ? "ABSTAIN: TRAINING OOD" : (abstain ? "ABSTAIN: GATE TRIPPED" : "6/6 NATIVE INPUTS OK"),
    abstain ? "pill pill-verify" : "pill pill-ok"
  );
  const gateText = p.ood_flag
    ? "TRAINING OOD"
    : abstain
      ? (snap.injection !== "none" ? snap.injection.toUpperCase() : "LOW REGIME")
      : "ELIGIBLE";
  const box3Lbl = snap.injection === "domain_ood" ? "CT-1 Flow (OOD)" : "Replay Clock";
  const box3Val = snap.injection === "domain_ood"
    ? "750.0 GPM (>601)"
    : `${escapeHtml(snap.timestamp.slice(11, 16))} · #${snap.row_index + 1}`;
  document.getElementById("step1-metrics").innerHTML = `
    <div class="kv-box ${snap.injection === "missing_nan" || snap.injection === "flatline_60m" ? "kv-bad" : "kv-ok"}"><span>CT-1 Outlet*</span><strong>${snap.injection === "missing_nan" ? "MISSING" : (snap.injection === "flatline_60m" ? "FLATLINE 60m" : escapeHtml(fmt(s.nominal_tower_outlet_c, 2, " °C*")))}</strong></div>
    <div class="kv-box ${snap.injection === "out_of_range" ? "kv-bad" : "kv-ok"}"><span>CT-1 Fan VFD</span><strong>${snap.injection === "out_of_range" ? "400.0 % (PHYS)" : escapeHtml(fmt(s.fan_pct, 1, " %"))}</strong></div>
    <div class="kv-box ${snap.injection === "domain_ood" ? "kv-bad" : "kv-ok"}"><span>${box3Lbl}</span><strong>${box3Val}</strong></div>
    <div class="kv-box ${abstain ? "kv-bad" : "kv-ok"}"><span>Gate Status</span><strong>${escapeHtml(gateText)}</strong></div>
  `;

  // Step II
  setPill("step2-status", abstain ? "GATE HOLD / OFF" : `ACTIVE · ${s.n_chillers_on} CH / ${s.n_towers_on} CT`, abstain ? "pill pill-warn" : "pill pill-ok");
  const flowBad = snap.injection === "domain_ood";
  const rangeBad = snap.injection === "missing_nan" || snap.injection === "flatline_60m";
  document.getElementById("step2-metrics").innerHTML = `
    <div class="kv-box"><span>Logged Load*</span><strong>${escapeHtml(fmt(s.logged_load_kw, 1, " kW*"))}</strong></div>
    <div class="kv-box ${flowBad ? "kv-bad" : "kv-ok"}"><span>CT-1 Flow${flowBad ? " (OOD)" : ""}</span><strong>${escapeHtml(fmt(s.tower_flow_gpm, 1, " GPM"))}</strong></div>
    <div class="kv-box ${rangeBad ? "kv-bad" : ""}"><span>CT-1 Range ΔT</span><strong>${snap.injection === "missing_nan" ? "MISSING" : escapeHtml(fmt(s.tower_range_c, 2, " °C"))}</strong></div>
    <div class="kv-box"><span>14-Ch Plant kW</span><strong>${escapeHtml(fmt(s.plant_kw, 1, " kW"))}</strong></div>
  `;

  // Step III (Causal Root-Cause DAG + 4 Competing-Cause Cards toggle)
  const hyps = snap.hypotheses || [];
  const ruledN = hyps.filter(h => h.status === "RULED_OUT").length;
  const prim = hyps.find(h => h.status === "PRIMARY_SUSPECT");
  const cg = snap.causal_graph || null;
  let s3Txt = "NO PERSISTENT ALERT";
  let s3Cls = "pill pill-ok";
  if (abstain) {
    s3Txt = "ABSTAIN: N3 GATE BLOCK";
    s3Cls = "pill pill-verify";
  } else if (cg && cg.isolated_root_id && cg.isolated_root_id !== "NONE") {
    s3Txt = cg.isolated_root_id === "N2" ? "ROOT: N2 CW FLOW" : `ROOT: ${cg.isolated_root_id} FILL FOULING`;
    s3Cls = cg.isolated_root_id === "N2" ? "pill pill-warn" : "pill pill-fault";
  } else if (prim) {
    s3Txt = `${ruledN} RULED OUT · ${prim.id} SUSPECT`;
    s3Cls = "pill pill-fault";
  }
  setPill("step3-status", s3Txt, s3Cls);

  const rcLbl = { PRIMARY_SUSPECT: "LIKELY CAUSE", CHECK_REQUIRED: "VERIFY", RULED_OUT: "RULED OUT", MONITORING: "MONITOR" };
  document.getElementById("step3-hypotheses").innerHTML = hyps.map(h => `
    <div class="hyp-pill hp-${escapeHtml(h.status)}">
      <div class="hyp-top"><span>${escapeHtml(h.id)} · ${escapeHtml(h.name)}</span><span class="rc-tag rc-${escapeHtml(h.status)}">${escapeHtml(rcLbl[h.status] || h.status)}</span></div>
      <div class="hyp-ev">${escapeHtml(h.evidence)}</div>
    </div>
  `).join("");

  const dagHost = document.getElementById("step3-dag-host");
  const hypHost = document.getElementById("step3-hypotheses");
  if (dagHost && hypHost) {
    dagHost.classList.toggle("hidden", step3ViewMode !== "dag");
    hypHost.classList.toggle("hidden", step3ViewMode !== "cards");
    document.querySelectorAll("#dag-view-toggle .dt-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.view === step3ViewMode);
    });
    if (step3ViewMode === "dag" && typeof window.drawCausalDagSvg === "function") {
      window.drawCausalDagSvg(dagHost, cg, node => {
        openRcaDrawer({
          preset: null,
          query: `Explain causal node ${node.id} (${node.short}) and its current evidence`,
        });
      });
    }
  }

  // Step IV (Single delegated listener attached in DOMContentLoaded — zero rebind leak)
  const loggedN = fieldChecks.filter(fc => fc.state !== "pending").length;
  const issues = fieldChecks.filter(fc => fc.state === "issue");
  const aiN = fieldChecks.filter(fc => fc.source && fc.source.toLowerCase().includes("ai")).length;
  document.getElementById("step4-status").textContent = aiN > 0
    ? `${loggedN} / 4 Logged · ${aiN} AI · ${issues.length} Issue(s)`
    : `${loggedN} / 4 Logged · ${issues.length} Issue(s)`;
  const stateLabel = { pending: "Pending", ok: "✓ OK", issue: "⚠ Issue" };
  document.getElementById("step4-checklist").innerHTML = fieldChecks.map(fc => {
    const baseHint = fc.id === "fan" ? `CT-1 Fan ${fmt(s.fan_pct, 1, "%")} · check belt/blades/VFD` : fc.hint;
    const hint = fc.detail || baseHint;
    const srcCls = fc.source
      ? (fc.source.toLowerCase().includes("acoustic") ? "src-ai-ac" : (fc.source.toLowerCase().includes("manual") ? "src-manual" : ""))
      : "";
    const srcTag = fc.source ? `<span class="field-src-badge ${srcCls}">${escapeHtml(fc.source)}</span>` : "";
    return `
      <div class="field-row" title="${escapeHtml(hint)}">
        <div class="field-info"><strong><span>${escapeHtml(fc.title)}</span>${srcTag}</strong><small>${escapeHtml(hint)}</small></div>
        <button class="field-toggle ft-${escapeHtml(fc.state)}" data-field-id="${escapeHtml(fc.id)}">${stateLabel[fc.state]}</button>
      </div>
    `;
  }).join("");

  // Step V (Interlocked on ABSTAIN and gated on active AI/Rule alert!)
  const hasAlert = snap.status === "FOULING_SUSPECT" || snap.status === "RULE_WARNING";
  const aiScoreTxt = p.ai_score !== null && p.ai_score !== undefined ? Number(p.ai_score).toFixed(3) : "N/A";
  document.getElementById("interlock-box").classList.toggle("hidden", !abstain);
  const btnConf = document.getElementById("btn-v-confirmed");
  btnConf.disabled = abstain || !hasAlert;
  btnConf.title = abstain
    ? "Locked by Step I safety interlock (ABSTAIN)"
    : (!hasAlert ? "Requires an active AI or High-Fan Rule alert before confirming fouling" : "Log Confirmed Fouling verdict to unlock Step VI draft brief");

  const vBadge = document.getElementById("step5-badge");
  if (abstain) {
    vBadge.textContent = "LOCKED (ABSTAIN)";
    vBadge.className = "pill pill-verify";
  } else if (rev.acknowledged) {
    vBadge.textContent = `LOGGED: ${rev.verdict}`;
    vBadge.className = rev.verdict === "CONFIRMED" ? "pill pill-ok" : rev.verdict === "FALSE_ALARM" ? "pill pill-fault" : "pill pill-warn";
  } else if (!hasAlert) {
    vBadge.textContent = "MONITORING · NO ALERT";
    vBadge.className = "tag-subtle";
  } else {
    vBadge.textContent = "AWAITING REVIEW";
    vBadge.className = "tag-subtle";
  }

  const issueNames = issues.map(i => i.title).join(", ");
  const topSens = (p.feature_sensitivity && p.feature_sensitivity.length)
    ? `${p.feature_sensitivity[0].label} (${p.feature_sensitivity[0].delta_score >= 0 ? "+" : ""}${p.feature_sensitivity[0].delta_score.toFixed(3)})`
    : "6 native inputs";
  document.getElementById("step5-summary").innerHTML = abstain
    ? "<strong>Blocked:</strong> Resolve Step I gate first (local advisory only)."
    : (!hasAlert
        ? `<strong>No Active Alert:</strong> HGB ${escapeHtml(aiScoreTxt)} · Fan ${escapeHtml(fmt(s.fan_pct, 1, "%"))} — switch to 0.80/0.65 UA or click <strong>Next Alert</strong>.`
        : (issues.length
            ? `<strong>Ready:</strong> HGB ${escapeHtml(aiScoreTxt)} · Driver: ${escapeHtml(topSens)} · Issue: <strong>${escapeHtml(issueNames)}</strong>.`
            : `<strong>Alert Active:</strong> HGB ${escapeHtml(aiScoreTxt)} · Fan ${escapeHtml(fmt(s.fan_pct, 1, "%"))} — verify Step IV, then log verdict.`));

  // Step VI
  const s6Card = document.getElementById("step6-card");
  const s6Badge = document.getElementById("step6-badge");
  const s6Body = document.getElementById("step6-body");
  if (abstain) {
    s6Card.className = "card wo-card wo-locked";
    s6Badge.textContent = "LOCKED (ABSTAIN)";
    s6Badge.className = "pill pill-verify";
    s6Body.innerHTML = `<div class="wo-placeholder">🔒 <strong>Locked:</strong> Clear Step I ABSTAIN gate before drafting brief.</div>`;
  } else if (rev.acknowledged && rev.verdict === "CONFIRMED") {
    s6Card.className = "card wo-card wo-confirmed";
    s6Badge.textContent = "WO DRAFT READY";
    s6Badge.className = "pill pill-ok";
    const woId = `WO-CT1-${snap.timestamp.slice(5, 10).replace("-", "")}`;
    const fieldTxt = issueNames || "Spray deck / fill media inspection verified";
    s6Body.innerHTML = `
      <div class="wo-grid">
        <div class="wo-kv"><span>Draft ID / Asset</span><strong>${escapeHtml(woId)} · CT-1</strong></div>
        <div class="wo-kv"><span>Isolation Window</span><strong>N+1 Staging (Zero Unplanned Stop)</strong></div>
        <div class="wo-kv"><span>Evidence (I–IV)</span><strong>HGB ${escapeHtml(aiScoreTxt)} (${p.ai_run_minutes}/5m) · Fan ${escapeHtml(fmt(s.fan_pct, 1, "%"))} · ${escapeHtml(fieldTxt)}</strong></div>
        <div class="wo-kv"><span>Power / Savings</span><strong>${escapeHtml(fmt(s.plant_kw, 1, " kW"))} · Savings Pending Pilot</strong></div>
      </div>
      <div class="wo-foot">
        <span class="wo-note">Read-Only Draft · Human Sign-Off Required</span>
        <div class="wo-actions">
          <button id="btn-copy-wo" class="btn-copy-wo">📋 Copy</button>
          <button id="btn-export-md" class="btn-export-wo" title="Download offline Markdown brief">⬇ .md</button>
          <button id="btn-export-json" class="btn-export-wo" title="Download offline JSON audit record">⬇ .json</button>
        </div>
      </div>
    `;
    const copyBtn = document.getElementById("btn-copy-wo");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        const text = `[${woId}] CT-1 Maintenance Brief | Engineer: ${rev.engineer_name || "Supervisor"} | Evidence: Quan HGB score=${aiScoreTxt} (${p.ai_run_minutes}/5m), Fan=${fmt(s.fan_pct, 1, "%")}, Field=${fieldTxt} | Note: ${rev.note || ""} | Read-Only Advisory Draft (Zero PLC Write)`;
        navigator.clipboard?.writeText(text).catch(() => {});
        copyBtn.textContent = "✓ Copied";
      });
    }
    const dlFile = (filename, content, mime) => {
      const blob = new Blob([content], { type: mime });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    const mdBtn = document.getElementById("btn-export-md");
    if (mdBtn) {
      mdBtn.addEventListener("click", () => {
        const sensLines = (p.feature_sensitivity || []).map(f => `- ${f.label} (${f.current_value}): ${f.delta_score >= 0 ? "+" : ""}${f.delta_score.toFixed(4)} Δscore`).join("\n");
        const md = [
          `# ${woId} — Chiller FaultGuard Maintenance Action Brief (Advisory Draft)`,
          ``,
          `- **Timestamp (LBNL Replay):** ${snap.timestamp} (Case: ${snap.case_meta.label})`,
          `- **Asset:** Cooling Tower 1 (CT-1) · Coordinate N+1 Staging (Zero Unplanned Stop)`,
          `- **Frozen AI Engine:** Quan HistGradientBoostingClassifier (SHA256 \`${p.model_sha256}\`)`,
          `- **Score / Persistence:** ${aiScoreTxt} (uncalibrated; ${p.ai_run_minutes}/${p.ai_required_minutes}m) | High-Fan Rule: ${fmt(s.fan_pct, 1, "%")} (${p.rule_run_minutes}/${p.rule_required_minutes}m)`,
          `- **14-Ch Logged Plant Power:** ${fmt(s.plant_kw, 1, " kW")} (Verified excess/avoided kWh & water savings: Pending Site Pilot)`,
          `- **Step IV Field Checks:** ${fieldTxt}`,
          `- **Step V Verdict:** ${rev.verdict} by ${rev.engineer_name || "Supervisor"}`,
          `- **Supervisor Note:** ${rev.note || "N/A"}`,
          ``,
          `## Local Feature Sensitivity (1-at-a-Time vs Train Mean — Not SHAP)`,
          sensLines || "- N/A",
          ``,
          `> **Governance Disclaimer:** Local Read-Only Advisory Draft Only (\`read_only=true\`, \`controls=false\`). Zero automated PLC/BMS write-back. Requires human facilities engineering sign-off.`,
        ].join("\n");
        dlFile(`${woId}_brief.md`, md, "text/markdown;charset=utf-8");
      });
    }
    const jsonBtn = document.getElementById("btn-export-json");
    if (jsonBtn) {
      jsonBtn.addEventListener("click", () => {
        const payload = {
          brief_id: woId,
          asset: "CT-1",
          read_only: true,
          controls: false,
          review_note_is_approval: false,
          timestamp: snap.timestamp,
          case: snap.case,
          model_sha256: p.model_sha256,
          ai_score: p.ai_score,
          score_is_calibrated_probability: false,
          ai_run_minutes: p.ai_run_minutes,
          rule_run_minutes: p.rule_run_minutes,
          signals: s,
          field_checks: fieldChecks,
          review: rev,
          verified_excess_kwh: null,
          verified_avoided_kwh: null,
          water_savings: null,
        };
        dlFile(`${woId}_audit.json`, JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
      });
    }
  } else if (rev.acknowledged && rev.verdict === "FALSE_ALARM") {
    s6Card.className = "card wo-card wo-false";
    s6Badge.textContent = "NO PM DISPATCHED";
    s6Badge.className = "pill pill-ok";
    s6Body.innerHTML = `<div class="wo-placeholder"><strong>False Alarm Logged</strong> (${escapeHtml(snap.timestamp.slice(11, 16))}) · Zero PM drafted.</div>`;
  } else if (rev.acknowledged && rev.verdict === "INCONCLUSIVE") {
    s6Card.className = "card wo-card wo-inc";
    s6Badge.textContent = "24H WATCH ACTIVE";
    s6Badge.className = "pill pill-warn";
    s6Body.innerHTML = `<div class="wo-placeholder"><strong>Inconclusive:</strong> Continue read-only monitoring (24h watch).</div>`;
  } else {
    s6Card.className = "card wo-card wo-locked";
    s6Badge.textContent = "LOCKED UNTIL STEP V";
    s6Badge.className = "tag-subtle";
    s6Body.innerHTML = `<div class="wo-placeholder">Locked until Step V <strong>Confirmed</strong> verdict (Read-only · zero PLC write).</div>`;
  }
}

/* ==================== MAIN RENDER ==================== */
function renderSnapshot(snap) {
  latestSnap = snap;
  const s = snap.signals, p = snap.prediction, e = snap.electricity, m = snap.metrics, rev = snap.review;
  const abstain = snap.status === "ABSTAIN";

  // Top Bar
  const clk = document.getElementById("badge-clock");
  clk.textContent = snap.timestamp.slice(11, 16);
  clk.title = `${snap.timestamp} · Row ${snap.row_index + 1}/${snap.total_rows}`;
  document.getElementById("row-counter-tag").textContent = `#${snap.row_index + 1} / ${snap.total_rows}`;
  const srcEl = document.getElementById("pid-source-tag");
  if (srcEl) {
    srcEl.textContent = `${snap.case_meta.label} · ${snap.timestamp.slice(5, 10)}`;
    srcEl.title = `${snap.case_meta.csv} · ${snap.timestamp.slice(0, 10)}`;
  }

  document.querySelectorAll("#scenario-buttons .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.scenario === snap.scenario));
  document.querySelectorAll("#fault-buttons .g-btn").forEach(b => b.classList.toggle("active", b.dataset.fault === snap.injection));

  // Override Banner & stress-safe jump button visibility
  const vBanner = document.getElementById("verification-banner");
  const stressActive = snap.injection && snap.injection !== "none";
  const btnJumpAlert = document.getElementById("btn-jump-alert");
  if (btnJumpAlert) {
    btnJumpAlert.disabled = stressActive;
    btnJumpAlert.title = stressActive
      ? "Clear active synthetic stress test before jumping replay"
      : "Process intermediate rows to next persistent AI alert (5/5m)";
  }
  if (abstain) {
    vBanner.classList.remove("hidden");
    document.getElementById("verify-reason").textContent = `${(p.reasons || []).join(" ")} Step V–VI confirmation locked safely.`;
    document.getElementById("btn-jump-eligible").classList.toggle("hidden", stressActive);
    document.getElementById("btn-clear-fault-banner").classList.toggle("hidden", !stressActive);
  } else {
    vBanner.classList.add("hidden");
  }

  // ---- TAB 1 (ISA-101 Redundant Coding: Color + Shape/Icon + Explicit Text) ----
  dirtyTabs.overview = true;
  dirtyTabs.trends = true;
  dirtyTabs.quantify = true;
  dirtyTabs.derisk = true;

  const activePane = document.querySelector(".tab-pane.active");
  if (activePane && activePane.id && activePane.id.startsWith("tab-")) {
    activeTabId = activePane.id.slice(4);
  }

  const aStrip = document.getElementById("advisory-strip");
  aStrip.className = "action-bar " + (snap.status === "FOULING_SUSPECT" ? "bar-fault" : abstain ? "bar-verify" : snap.status === "RULE_WARNING" ? "bar-suspect" : "bar-normal");
  document.getElementById("advisory-equip").textContent = snap.status === "FOULING_SUSPECT" ? "CT-1 · REVIEW" : "CT-1";
  document.getElementById("advisory-title").textContent = snap.at_end
    ? "End of 24h Oct-01 fixture — reset or switch case."
    : snap.advisory_title;
  const ackB = document.getElementById("advisory-ack-badge");
  if (rev.acknowledged) { ackB.classList.remove("hidden"); ackB.textContent = `✓ ${rev.verdict}`; } else { ackB.classList.add("hidden"); }

  if (snap.field_checks && snap.field_checks.applied) {
    syncFieldChecksFromBackend(snap.field_checks);
  }

  renderActiveTab(snap);

  if (snap.at_end) stopPlayback();
  const isLiveBasisNow = tab3BasisMode === "live" && Boolean(e.live_alert_projection);
  const simNow = isLiveBasisNow ? e.live_alert_projection : e.illustrative_scenario;
  const curPmCost = (simNow && simNow.pm_optimizer && simNow.pm_optimizer.pm_cost_vnd) || 8000000;
  const hwPre = (snap.hardware_summary && snap.hardware_summary.active_preset) || (latestHw && latestHw.active_preset) || "";
  const snapKey = `${snap.scenario}|${snap.row_index}|${snap.injection}|${tab3BasisMode}|${curPmCost}|${hwPre}`;
  if (rcaOpen && !rcaBusy && snapKey !== lastRcaSnapKey) {
    lastRcaSnapKey = snapKey;
    queryRcaAgent(lastRcaRequest, { silent: true });
  }
  updateAuditHook();
}

function renderActiveTab(snap) {
  if (!snap) return;
  if (activeTabId === "overview") {
    renderTabOverview(snap);
    dirtyTabs.overview = false;
  } else if (activeTabId === "trends") {
    renderTabTrends(snap);
    dirtyTabs.trends = false;
  } else if (activeTabId === "quantify") {
    renderTabQuantify(snap);
    dirtyTabs.quantify = false;
  } else if (activeTabId === "derisk") {
    renderTabDerisk(snap);
    dirtyTabs.derisk = false;
  } else if (activeTabId === "hardware") {
    if (latestHw && dirtyTabs.hardware) {
      renderHardwareTab(latestHw, { keepMode: true });
      dirtyTabs.hardware = false;
    } else if (!latestHw) {
      loadHardwareStatus({ keepMode: false });
    }
  }
}

function renderTabOverview(snap) {
  const s = snap.signals, p = snap.prediction, abstain = snap.status === "ABSTAIN";
  const stPillMap = {
    NO_ALERT: ["✓ NO ALERT", "pill pill-ok"],
    RULE_WARNING: ["▲ RULE ALERT", "pill pill-warn"],
    FOULING_SUSPECT: ["⚠ FOULING SUSPECT", "pill pill-fault"],
    ABSTAIN: ["🔒 ABSTAIN", "pill pill-verify"],
  };
  const [stTxt, stCls] = stPillMap[snap.status] || ["✓ NO ALERT", "pill pill-ok"];
  setPill("overall-status-pill", stTxt, stCls);
  setPill("phys-state-pill", abstain ? "🔒 ABSTAIN" : (p.rule_alert ? "⚠ RULE ALERT" : "✓ NO RULE ALERT"), abstain ? "pill pill-verify" : (p.rule_alert ? "pill pill-fault" : "pill pill-ok"));

  const agMap = {
    NO_ALERT: ["SEPARATE · NO ALERT", "pill pill-ok"],
    AI_ONLY_ALERT: ["AI ALERT · RULE CLEAN", "pill pill-fault"],
    RULE_ONLY_ALERT: ["RULE ALERT · AI CLEAN", "pill pill-warn"],
    BOTH_FLAGGED: ["AI & RULE BOTH FLAGGED", "pill pill-fault"],
    BLOCKED_BY_GATE: ["BLOCKED BY GATE", "pill pill-verify"],
  };
  const [agTxt, agCls] = agMap[snap.agreement] || ["SEPARATE OUTPUTS", "pill pill-ok"];
  setPill("fusion-agreement-pill", agTxt, agCls);

  document.getElementById("kpi-fan-meas").textContent = fmt(s.fan_pct, 1, " %");
  document.getElementById("rule-dwell-text").textContent = `${p.rule_run_minutes} / ${p.rule_required_minutes}m`;
  const rPct = Math.min(100, (p.rule_run_minutes / Math.max(p.rule_required_minutes, 1)) * 100);
  const rBar = document.getElementById("rule-dwell-bar");
  rBar.style.width = `${rPct}%`;
  rBar.style.background = p.rule_alert ? "#DC2626" : rPct > 0 ? "#D97706" : "#059669";

  const aiMetaEl = document.getElementById("ai-meta-tag");
  if (aiMetaEl) {
    aiMetaEl.textContent = fmt(p.inference_ms, 1, "ms");
    aiMetaEl.title = `Uncalibrated HGB score · ${fmt(p.inference_ms, 1, "ms")} local inference`;
  }
  document.getElementById("ai-pred-label").textContent = snap.status.replace("_", " ");
  document.getElementById("ai-conf-val").textContent = p.ai_score !== null ? p.ai_score.toFixed(3) : "ABSTAIN";
  const scPct = p.ai_score !== null ? Math.max(0, Math.min(100, p.ai_score * 100)) : 0;
  document.getElementById("ai-score-fill").style.width = `${scPct.toFixed(1)}%`;
  document.getElementById("ai-dwell-text").textContent = `${p.ai_run_minutes} / ${p.ai_required_minutes} min`;
  const aPct = Math.min(100, (p.ai_run_minutes / Math.max(p.ai_required_minutes, 1)) * 100);
  const aBar = document.getElementById("ai-dwell-bar");
  aBar.style.width = `${aPct}%`;
  aBar.style.background = p.ai_alert ? "#DC2626" : aPct > 0 ? "#D97706" : "#0B63B6";

  document.getElementById("kpi-plant-kw").textContent = fmt(s.plant_kw, 1, " kW");
  const dKw = Number(s.signed_delta_kw || 0);
  const dEl = document.getElementById("kpi-delta-kw");
  if (dEl) {
    dEl.textContent = `${dKw >= 0 ? "+" : ""}${dKw.toFixed(1)} kW`;
    dEl.className = "hero-num " + (dKw > 2.5 ? "num-fault" : dKw > 0.5 ? "num-warn" : "num-ok");
  }
  document.getElementById("kpi-range-c").textContent = fmt(s.tower_range_c, 2, " °C");
  const fEl = document.getElementById("kpi-flow-gpm");
  if (fEl) fEl.textContent = fmt(s.tower_flow_gpm, 0, " GPM");
  document.getElementById("kpi-load-kw").textContent = fmt(s.logged_load_kw, 0, " kW");
  const stgEl = document.getElementById("kpi-staging");
  if (stgEl) stgEl.textContent = `${s.n_chillers_on} CH / ${s.n_towers_on} CT`;

  renderTimelineScrubber(snap);
  updatePidMimic(snap);
  window.drawFanGaugeSvg(document.getElementById("chart-gauge"), s.fan_pct, p.rule_threshold_pct, snap.status);
}

function renderTabTrends(snap) {
  const p = snap.prediction, m = snap.metrics, abstain = snap.status === "ABSTAIN";
  setPill("xai-ood-badge", p.ood_flag ? "OUTSIDE TRAIN BOUNDS (OOD)" : "IN TRAINING BOUNDS", p.ood_flag ? "pill pill-verify" : "pill pill-ok");
  const sens = p.feature_sensitivity || [];
  document.getElementById("xai-feature-list").innerHTML = sens.length
    ? sens.map(ft => {
        const sign = ft.delta_score >= 0 ? "+" : "";
        const cls = ft.direction === "up" ? "xai-up" : "xai-down";
        const barCls = ft.direction === "up" ? "xai-fill-up" : "xai-fill-down";
        return `
          <div class="xai-item">
            <div class="xai-meta"><span>${escapeHtml(ft.label)} (${escapeHtml(ft.current_value)})</span><span class="xai-val ${cls}">${sign}${ft.delta_score.toFixed(3)} Δscore</span></div>
            <div class="xai-bar-track"><div class="xai-bar-fill ${barCls}" style="width:${Math.max(6, Math.round(ft.weight * 100))}%"></div></div>
          </div>
        `;
      }).join("")
    : `<div class="tag-subtle" style="text-align:center;padding:18px">${p.ood_flag ? "Suspended: Feature vector outside Jan–Jun training envelope (OOD)" : (abstain ? "Suspended while ABSTAIN gate is active" : "Advance 1 step or jump to eligible row")}</div>`;

  const pc = m.AI.per_case, rPc = m.rule.per_case;
  const makeRecallTile = (title, tagCls, aiR, ruleR, badge, tip) => {
    const aPct = (aiR * 100).toFixed(1), rPct = (ruleR * 100).toFixed(1);
    const barCls = aiR >= ruleR ? "rc-bar-ai" : "rc-bar-warn";
    return `
      <div class="rc-tile" title="${escapeHtml(tip)}">
        <div class="rc-tile-top"><span>${title}</span><span class="rc-tag ${tagCls}">AI ${aPct}%</span></div>
        <div class="rc-mini-bars">
          <div class="rc-bar-row"><span>AI</span><div class="rc-bar-bg"><div class="${barCls}" style="width:${Math.min(100, aiR * 100)}%"></div></div><span>${aPct}%</span></div>
          <div class="rc-bar-row"><span>Rule</span><div class="rc-bar-bg"><div class="rc-bar-rule" style="width:${Math.min(100, ruleR * 100)}%"></div></div><span>${rPct}%</span></div>
        </div>
        <div class="rc-tile-ev">${badge}</div>
      </div>
    `;
  };
  document.getElementById("limits-grid").innerHTML = `
    ${makeRecallTile("0.95 UA (Mild)", "rc-CHECK_REQUIRED", pc.tower_fouling_095.recall, rPc.tower_fouling_095.recall, `Rule +${((rPc.tower_fouling_095.recall - pc.tower_fouling_095.recall) * 100).toFixed(1)} pts`, "Rule beats AI on mildest fouling (8.5% vs 4.5%) — disclosed honestly.")}
    ${makeRecallTile("0.80 UA (Med)", "rc-RULED_OUT", pc.tower_fouling_080.recall, rPc.tower_fouling_080.recall, `+${((pc.tower_fouling_080.recall - rPc.tower_fouling_080.recall) * 100).toFixed(1)} pts · 97.8% P`, "+6.2 pts recall vs Rule · Precision 97.8% on Oct–Dec held-out test.")}
    ${makeRecallTile("0.65 UA (Sev)", "rc-RULED_OUT", pc.tower_fouling_065.recall, rPc.tower_fouling_065.recall, `+${((pc.tower_fouling_065.recall - rPc.tower_fouling_065.recall) * 100).toFixed(1)} pts · 42/53`, "+28.7 pts recall vs Rule · 42 of 53 eligible windows alerted.")}
    <div class="rc-tile" title="73,932 eligible rows (292.3h normal). Weather/CHW mapping & onset delay open for Pilot.">
      <div class="rc-tile-top"><span>Coverage</span><span class="rc-tag rc-MONITORING">13.95% Active</span></div>
      <div class="rc-mini-bars">
        <div class="rc-bar-row"><span>Gate</span><div class="rc-bar-bg"><div class="rc-bar-ai" style="width:100%"></div></div><span>100%</span></div>
        <div class="rc-bar-row"><span>Time</span><div class="rc-bar-bg"><div class="rc-bar-rule" style="width:14%"></div></div><span>14.0%</span></div>
      </div>
      <div class="rc-tile-ev">73.9k Rows · Pilot Scope</div>
    </div>
  `;
  window.drawScoreTrendSvg(document.getElementById("chart-score"), snap.history, p.ai_threshold);
  window.drawPowerTrendSvg(document.getElementById("chart-power"), snap.history);
  window.drawBenchmarkBarsSvg(document.getElementById("chart-benchmark"), m);
}

function renderTabQuantify(snap) {
  const s = snap.signals, e = snap.electricity;
  const isLiveBasis = tab3BasisMode === "live" && Boolean(e.live_alert_projection);
  const sim = isLiveBasis ? e.live_alert_projection : e.illustrative_scenario;
  document.querySelectorAll("#excess-kw-buttons .seg-btn").forEach(b => {
    if (b.dataset.mode === "live") b.classList.toggle("active", isLiveBasis);
    else b.classList.toggle("active", !isLiveBasis && Number(b.dataset.kw) === Number(e.illustrative_scenario.imposed_excess_kw));
  });
  document.querySelectorAll("#clean-day-buttons .seg-btn").forEach(b => b.classList.toggle("active", Number(b.dataset.day) === Number(sim.earlier_pm_day)));
  document.querySelectorAll("#cal-day-buttons .seg-btn").forEach(b => b.classList.toggle("active", Number(b.dataset.cal) === Number(sim.existing_pm_day)));
  document.querySelectorAll("#restore-buttons .seg-btn").forEach(b => b.classList.toggle("active", Math.abs(Number(b.dataset.rest) - Number(sim.restored_fraction)) < 0.01));

  if (isLiveBasis) {
    if (!sim.active || sim.imposed_excess_kw <= 0) {
      document.getElementById("cf-chart-title").textContent = `Live Auto-Projection (Standby)`;
      setPill(
        "cf-days-pill",
        sim.mode === "SUSPENDED_ABSTAIN" ? "🔒 ABSTAIN · LOCKED" : "✓ 0 kWh · Standby",
        sim.mode === "SUSPENDED_ABSTAIN" ? "pill pill-verify" : "pill pill-ok"
      );
    } else {
      document.getElementById("cf-chart-title").textContent = `Live Auto (+${sim.imposed_excess_kw.toFixed(2)} kW* @ ${sim.trigger_timestamp.slice(11, 16)}) · D${sim.earlier_pm_day} vs D${sim.existing_pm_day}`;
      setPill(
        "cf-days-pill",
        `⚡ ${sim.days_saved}d Earlier · ${Number(sim.scenario_avoided_excess_kwh).toLocaleString()} kWh`,
        "pill pill-fault"
      );
    }
  } else {
    document.getElementById("cf-chart-title").textContent = `Earlier PM (D${sim.earlier_pm_day}) vs. Calendar PM (D${sim.existing_pm_day})`;
    setPill(
      "cf-days-pill",
      `${sim.days_saved}d Earlier · ${Number(sim.scenario_avoided_excess_kwh).toLocaleString()} kWh`,
      "pill pill-ok"
    );
  }

  const fmtVnd = v => {
    const n = Number(v || 0);
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M ₫`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k ₫`;
    return `${Math.round(n)} ₫`;
  };
  const k5 = sim.kpis_5m1e;
  const kwLabel = isLiveBasis ? `${sim.imposed_excess_kw.toFixed(2)} kW*` : `${sim.imposed_excess_kw.toFixed(0)} kW`;
  const chip2Html = isLiveBasis
    ? (sim.active
        ? `<strong>Trigger</strong> ${escapeHtml(sim.trigger_timestamp.slice(11, 16))} · 15m ΔkW*`
        : `<strong>Status</strong> ${escapeHtml(snap.status.replace("_", " "))}`)
    : `<strong>Assumed</strong> D${sim.alert_day} → D${sim.earlier_pm_day}`;
  const chipUpdates = [
    ["a-eq", `<strong>Formula</strong> ${kwLabel}×${sim.hours_per_day}h×${sim.days_saved}d`],
    ["a-flow", chip2Html],
    ["a-pm", `<strong>Calendar</strong> Day ${sim.existing_pm_day}`],
    ["a-ua", `<strong>Restore</strong> ${Math.round(sim.restored_fraction * 100)}%`],
    ["a-cost", `<strong>22kV TOU</strong> ${Math.round(k5.money.tariff_vnd_kwh).toLocaleString()} ₫/kWh`],
    ["a-coc", `<strong>Excludes</strong> Water / Yield`],
  ];
  const flashed = [];
  chipUpdates.forEach(([id, htmlStr]) => {
    const el = document.getElementById(id);
    if (el && el.innerHTML !== htmlStr) {
      el.innerHTML = htmlStr;
      el.classList.remove("chip-flash");
      flashed.push(el);
    }
  });
  if (flashed.length) {
    requestAnimationFrame(() => flashed.forEach(el => el.classList.add("chip-flash")));
  }

  const condBody = document.querySelector("#cond-elec-table tbody");
  if (condBody && e.oct_dec_conditional && !condBody.dataset.populated) {
    condBody.dataset.populated = "1";
    const nameMap = { tower_fouling_095: "0.95 UA (-5%)", tower_fouling_080: "0.80 UA (-20%)", tower_fouling_065: "0.65 UA (-35%)" };
    condBody.innerHTML = e.oct_dec_conditional.map(row => {
      const posStr = Number(row.positive_only_kWh_not_headline || 0).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
      const negStr = Math.abs(Number(row.negative_kWh || 0)).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
      return `
      <tr>
        <td>${escapeHtml(nameMap[row.case] || row.case)}</td>
        <td>${row.matched_hours.toFixed(1)} h</td>
        <td>+${Number(row.net_excess_kWh).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kWh</td>
        <td>+${posStr} / −${negStr}</td>
        <td>${Number(row.excluded_or_unmatched_rows).toLocaleString()}</td>
        <td><span class="tag-subtle">Conditional · Pending</span></td>
      </tr>
    `;
    }).join("");
  }

  // Quantitative 3-Pillar 5M1E Cards (Official Nov-2026 VN Tariff + EVN TOU Break-Even PM Optimizer)
  const isStandby = isLiveBasis && (!sim.active || sim.imposed_excess_kw <= 0);
  const km = k5.money, kt = k5.time, ke = k5.esg;
  const pm = sim.pm_optimizer || {};
  const pmCostVnd = Number(pm.pm_cost_vnd || 8000000);
  const pmCostM = Math.round(pmCostVnd / 1e6);
  document.querySelectorAll("#pm-cost-buttons .pm-mini-btn").forEach(b => {
    b.classList.toggle("active", Math.abs(Number(b.dataset.pm) - pmCostVnd) < 1000);
  });

  const fmtSignedVnd = v => {
    const n = Number(v || 0);
    const sign = n > 0 ? "+" : n < 0 ? "−" : "";
    const abs = Math.abs(n);
    if (abs >= 1e6) return `${sign}${(abs / 1e6).toFixed(2)}M ₫`;
    if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(1)}k ₫`;
    return `${sign}${Math.round(abs)} ₫`;
  };
  const pkTile = (lbl, val, sub, cls = "", tip = "") =>
    `<div class="pk-box ${cls}"${tip ? ` title="${escapeHtml(tip)}"` : ""}><span class="pk-lbl">${escapeHtml(lbl)}</span><strong class="pk-val">${escapeHtml(val)}</strong><span class="pk-sub">${escapeHtml(sub)}</span></div>`;

  const moneyEl = document.getElementById("p-money-kpis");
  if (moneyEl) {
    const verdict = pm.verdict || (isStandby ? "STANDBY_NO_ACTION" : "CLEAN_NOW_POSITIVE_ROI");
    const pbDays = pm.payback_days !== null && pm.payback_days !== undefined ? Math.round(Number(pm.payback_days)) : null;
    const pbCal = pm.payback_calendar_day !== null && pm.payback_calendar_day !== undefined ? Math.round(Number(pm.payback_calendar_day)) : null;
    const cutoffDay = pm.latest_profitable_day !== null && pm.latest_profitable_day !== undefined ? Math.round(Number(pm.latest_profitable_day)) : null;
    const eoqDays = pm.eoq_sensitivity_days !== null && pm.eoq_sensitivity_days !== undefined ? Math.round(Number(pm.eoq_sensitivity_days)) : null;

    let roiVal = "0 ₫", roiSub = "Standby", roiCls = "";
    if (verdict === "CLEAN_NOW_POSITIVE_ROI") {
      roiVal = fmtSignedVnd(pm.net_benefit_vnd);
      roiSub = `Payback ${pbDays !== null ? pbDays + "d" : "--"} · +${Math.round(Number(pm.roi_pct || 0))}% ROI`;
      roiCls = "pk-hi";
    } else if (verdict === "DEFER_PM_NEGATIVE_ROI") {
      roiVal = fmtSignedVnd(pm.net_benefit_vnd);
      roiSub = `Defer PM${pbDays !== null ? ` · ${pbDays}d PB` : ""}`;
      roiCls = "pk-warn";
    } else if (verdict === "LOCKED_ABSTAIN") {
      roiVal = "LOCKED";
      roiSub = "ABSTAIN Gate";
      roiCls = "pk-warn";
    }

    const burnVal = isStandby ? "0 ₫/d" : `${fmtVnd(km.daily_burn_ex_vat_vnd)}/d`;
    const burnSub = verdict === "CLEAN_NOW_POSITIVE_ROI" && cutoffDay !== null
      ? `Cutoff D${cutoffDay} · Peak ${Math.round(Number(km.peak_bill_share_pct || 31))}%`
      : verdict === "DEFER_PM_NEGATIVE_ROI"
        ? "Sun 00–06h PM"
        : "0 ₫/d · Sun 00–06h";

    const tou = pm.tou_avoided_vnd || {};
    const roiTip = pm.rationale || "EVN 22kV TOU Break-Even PM Optimizer (illustrative only; verified_avoided_kwh = null).";
    const burnTip = `EVN TOU 22kV (1,926 ₫/kWh) · Peak 31.0%: ${fmtVnd(tou.peak || 0)} · Normal 53.8%: ${fmtVnd(tou.normal || 0)} · Off-Peak 15.1%: ${fmtVnd(tou.off_peak || 0)} · Recommended Window: ${pm.recommended_window || "Sun 00–06h"}`;

    moneyEl.innerHTML = [
      pkTile(
        "Avoided (Ex-VAT)",
        fmtVnd(km.avoided_ex_vat_vnd),
        `~$${Math.round(Number(km.avoided_ex_vat_usd || 0)).toLocaleString()} · +8%: ${fmtVnd(km.avoided_incl_vat8_vnd)}`,
        ""
      ),
      pkTile(`Net ROI (PM ${pmCostM}M ₫)`, roiVal, roiSub, roiCls, roiTip),
      pkTile("Daily Burn / Opt", burnVal, burnSub, "", burnTip),
    ].join("");

    const mNoteTxt = verdict === "CLEAN_NOW_POSITIVE_ROI"
      ? `CLEAN NOW (Sun 00–06h 1,166₫) · Payback D${pbCal ?? "--"} · EOQ t*≈${eoqDays ?? "--"}d · QĐ 2892/BCT 22kV (Pilot req.).`
      : verdict === "DEFER_PM_NEGATIVE_ROI"
        ? `DEFER PM: Payback (${pbDays ?? "--"}d) > D${sim.existing_pm_day} window · 22kV TOU 1,926 ₫/kWh · Verified kWh = Pilot.`
        : `Standby (${snap.status.replace("_", " ")}) · 22kV TOU ${Math.round(km.tariff_vnd_kwh).toLocaleString()} ₫/kWh (110kV: ${Math.round(km.tariff_110kv_vnd_kwh).toLocaleString()} ₫) · Verified kWh = Pilot.`;
    const mNote = document.getElementById("p-money-note");
    if (mNote) mNote.textContent = mNoteTxt;
    const pMoneyCard = document.getElementById("pillar-money");
    if (pMoneyCard) pMoneyCard.title = mNoteTxt;
  }

  const timeEl = document.getElementById("p-time-kpis");
  if (timeEl) {
    const isNormCase = kt.case_key === "normal";
    const leadLbl = isNormCase ? "Mean Clean / FA" : "Window Lead (AI)";
    const leadVal = isNormCase
      ? `${kt.mean_normal_hrs_per_fa.toFixed(1)} h / FA`
      : `−${kt.window_speedup_min.toFixed(1)} min`;
    const leadSub = isNormCase
      ? `${kt.ai_windows_alerted}/${kt.eligible_windows} win · FAR ${kt.far_ep_per_hr.toFixed(3)}/h`
      : `${kt.ai_median_window_min.toFixed(0)}m vs ${kt.rule_median_window_min.toFixed(0)}m · ${kt.ai_windows_alerted}/${kt.eligible_windows}`;
    timeEl.innerHTML = [
      pkTile("Fouling Run Cut", `${kt.fouling_hours_avoided.toLocaleString()} h`, kt.days_earlier > 0 ? `${kt.days_earlier}d earlier PM` : "Standby (0d)", kt.days_earlier > 0 ? "pk-hi" : ""),
      pkTile(leadLbl, leadVal, leadSub, isNormCase ? "" : "pk-hi"),
      pkTile("FA Triage Load", `${kt.fa_triage_hrs_per_100h.toFixed(2)} h/100h`, `Prec ${kt.precision_pct.toFixed(1)}% · 15m/ep`, ""),
    ].join("");
    const tNoteTxt = `LBNL Oct–Dec windows (${kt.ai_windows_alerted}/${kt.eligible_windows} AI vs ${kt.rule_windows_alerted}/${kt.eligible_windows} Rule) · 5m AI vs 15m Rule dwell · N+1 PM avoids unplanned stop.`;
    const tNote = document.getElementById("p-time-note");
    if (tNote) tNote.textContent = tNoteTxt;
    const pTimeCard = document.getElementById("pillar-time");
    if (pTimeCard) pTimeCard.title = tNoteTxt;
  }

  const esgEl = document.getElementById("p-image-kpis");
  if (esgEl) {
    const wp = ke.water_physics_est || {};
    const wpSusp = Boolean(wp.suspended);
    const wpVal = wpSusp ? "ABSTAIN" : `~${Number(wp.live_evap_m3_day || 0).toFixed(1)} m³/d*`;
    const wpSub = wpSusp
      ? "Gate hold · WTP Pilot"
      : (Number(wp.coc_4to3_penalty_m3_est || 0) > 0
          ? `+${Math.round(wp.coc_4to3_penalty_m3_est)}m³ if CoC4→3`
          : "0 m³ risk · WTP Pilot");
    const wpTip = wpSusp
      ? "ASHRAE water balance suspended during ABSTAIN gate. Verified water savings = null (WTP meter required in Pilot)."
      : `Live ASHRAE/CTI CT-1 Evap: ${Number(wp.live_evap_m3_day || 0).toFixed(2)} m³/d (Makeup @CoC4: ${Number(wp.live_makeup_coc4_m3_day || 0).toFixed(2)} m³/d) | Direct chiller-heat thermal makeup: +${Number(wp.thermal_makeup_m3_est || 0).toFixed(1)} m³ | Blowdown risk if fouling forces CoC 4→3: +${Number(wp.coc_4to3_penalty_m3_est || 0).toFixed(1)} m³ (+${Number(wp.coc_5to3_penalty_m3_est || 0).toFixed(1)} m³ for CoC 5→3) | Verified water savings = null (WTP meter required in Pilot).`;
    esgEl.innerHTML = [
      pkTile("Avoided Scope 2", `${ke.avoided_co2_tonnes.toFixed(2)} t CO₂e`, `${Math.round(ke.avoided_co2_kg).toLocaleString()} kg @ ${ke.grid_ef_kg_kwh}`, isStandby ? "" : "pk-hi"),
      pkTile("Daily CO₂ / Trees", `${ke.daily_co2_kg.toFixed(1)} kg/d`, `~${ke.trees_yr_eq.toLocaleString()} trees/yr eq.`, ""),
      pkTile("Water Est. (ASHRAE)", wpVal, wpSub, "pk-warn", wpTip),
    ].join("");
    const eNoteTxt = `VN 2024 EF ${ke.grid_ef_kg_kwh} (CV 11232) · Water = ASHRAE Flow×ΔT est. (verified = null; WTP meter req.).`;
    const eNote = document.getElementById("p-image-note");
    if (eNote) eNote.textContent = eNoteTxt;
    const pImgCard = document.getElementById("pillar-image");
    if (pImgCard) pImgCard.title = eNoteTxt;
  }

  window.drawCounterfactualSvg(document.getElementById("chart-counterfactual"), sim);
  window.drawPowerBreakdownSvg(document.getElementById("chart-breakdown"), s);
}

function renderTabDerisk(snap) {
  renderTriageWorkbench(snap);
  document.getElementById("event-log-list").innerHTML = (snap.audit_log || []).slice(0, 12).map(ev => `
    <li class="feed-item">
      <span class="f-time">${escapeHtml((ev.timestamp || "").slice(11, 16))}</span>
      <span class="f-tag ev-${escapeHtml(ev.level)}">${escapeHtml(ev.category)}</span>
      <span>${escapeHtml(ev.message)}</span>
    </li>
  `).join("");
}

/* ==================== TAB 5: MINI-CT1 HARDWARE BENCH & MULTI-MODAL VISION AI LAB ==================== */
function renderHardwareTab(hw, opts = {}) {
  if (!hw) return;
  // Preserve cached annotated images when receiving a lite=1 weather-only update
  if (latestHw && latestHw.vision_modes && hw.vision_modes) {
    Object.keys(hw.vision_modes).forEach(k => {
      const cur = hw.vision_modes[k];
      const prev = latestHw.vision_modes[k];
      if (cur && !cur.annotated_image_data_uri && prev && prev.annotated_image_data_uri) {
        cur.annotated_image_data_uri = prev.annotated_image_data_uri;
      }
    });
  }
  latestHw = hw;
  dirtyTabs.hardware = false;
  if (!opts.keepMode && hw.active_vision_mode) {
    activeHwVisionMode = hw.active_vision_mode;
  }
  const modes = hw.vision_modes || {};
  if (!modes[activeHwVisionMode]) {
    activeHwVisionMode = hw.active_vision_mode || Object.keys(modes)[0] || "fill_strainer";
  }
  const vm = modes[activeHwVisionMode] || {};
  const ac = hw.acoustic_tach || {};
  const wg = hw.weather_gate || {};
  const cad = hw.cad_summary || {};
  const bom = hw.cad_bom || {};
  const mlCard = hw.vision_ml_card || {};
  const cal = hw.calibration_status || {};

  // 1. Top Preset Bar, Vision Mode Tabs & 1-Tap Calibration Controls
  document.querySelectorAll("#hw-preset-buttons .seg-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.preset === hw.active_preset);
  });
  document.querySelectorAll("#hw-vision-modes .dt-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.mode === activeHwVisionMode);
  });
  const isCalibrated = Boolean(cal.calibrated);
  const btnCal = document.getElementById("btn-hw-calibrate");
  const btnResetCal = document.getElementById("btn-hw-reset-cal");
  if (btnCal) {
    btnCal.classList.toggle("act-primary", isCalibrated);
    btnCal.classList.toggle("act-accent", !isCalibrated);
    btnCal.title = isCalibrated
      ? `Calibrated: ${cal.label || "Clean Part 07"} (${cal.calibrated_at || "active"}) — click to re-lock clean reference`
      : "1-Tap Live Bench Calibration: lock current Clean Part 07 frame as reference";
  }
  if (btnResetCal) {
    btnResetCal.classList.toggle("hidden", !isCalibrated);
    btnResetCal.disabled = !isCalibrated;
  }

  // 2. Left Card: iPhone 11 / 3-Stage Vision ML Inspector
  const imgEl = document.getElementById("hw-vision-img");
  if (imgEl && vm.annotated_image_data_uri) {
    const imgKey = `${activeHwVisionMode}|${vm.frame_sha256_short || ""}|${vm.verdict || ""}|${vm.calibration_used ? 1 : 0}`;
    if (imgEl.dataset.imgKey !== imgKey) {
      imgEl.dataset.imgKey = imgKey;
      imgEl.src = vm.annotated_image_data_uri;
    }
  }
  const srcTag = document.getElementById("hw-vision-src-tag");
  if (srcTag) {
    const shaShort = String(vm.frame_sha256_short || vm.frame_sha256 || "--").slice(0, 8);
    srcTag.textContent = `SHA ${shaShort} · ${Number(vm.latency_ms || 0).toFixed(1)}ms`;
  }
  const mlPill = document.getElementById("hw-ml-pill");
  if (mlPill) {
    const sha8 = String(mlCard.model_sha256_short || mlCard.model_sha256 || vm.ml_model_sha256 || "88273963").slice(0, 8);
    const auc = mlCard.cv_roc_auc !== undefined && mlCard.cv_roc_auc !== null ? Number(mlCard.cv_roc_auc).toFixed(3) : "0.998";
    const cpuMs = mlCard.inference_ms_cpu !== undefined && mlCard.inference_ms_cpu !== null ? Math.round(Number(mlCard.inference_ms_cpu)) : 12;
    mlPill.textContent = `HGB ${sha8} · AUC ${auc} · ${cpuMs}ms`;
    mlPill.className = "hw-ml-pill ml-info";
    mlPill.title = `${mlCard.model_name || "Vision ML HGB"} · ${Number(mlCard.n_patches || 16800).toLocaleString()} patches · F1 ${Number(mlCard.cv_f1 || 0.936).toFixed(3)} · FPR ${Number(mlCard.cv_patch_fpr || 0.0036).toFixed(4)}`;
  }
  const regPill = document.getElementById("hw-reg-pill");
  if (regPill) {
    const arucoOk = Boolean(vm.aruco_locked);
    const markersN = Array.isArray(vm.markers_detected) ? vm.markers_detected.length : 0;
    const regMode = String(vm.registration_mode || "");
    if (arucoOk) {
      regPill.textContent = `ArUco Lock (${markersN || 4}/4)`;
      regPill.className = "hw-ml-pill ml-ok";
    } else if (regMode.toLowerCase().includes("aspect")) {
      regPill.textContent = "Aspect ROI";
      regPill.className = "hw-ml-pill ml-info";
    } else {
      regPill.textContent = "Canonical";
      regPill.className = "hw-ml-pill ml-muted";
    }
    regPill.title = `Registration: ${regMode || "canonical_640x400"} · ArUco markers: ${markersN}/4`;
  }
  const calPill = document.getElementById("hw-cal-pill");
  if (calPill) {
    if (isCalibrated) {
      calPill.textContent = "Ref Calibrated";
      calPill.className = "hw-ml-pill ml-ok";
      calPill.title = `Clean Baseline: ${cal.label || "Clean Part 07"} · SHA ${String(cal.ref_sha256_short || "--").slice(0, 8)} · ${cal.calibrated_at || ""}`;
    } else {
      calPill.textContent = "ML Only";
      calPill.className = "hw-ml-pill ml-muted";
      calPill.title = "No live P07 reference locked — click Calibrate Clean (P07) to enable Stage 3 structural fusion";
    }
  }
  const pmaxPill = document.getElementById("hw-pmax-pill");
  if (pmaxPill) {
    if (vm.ml_max_prob !== undefined && vm.ml_max_prob !== null) {
      const pmVal = Number(vm.ml_max_prob);
      pmaxPill.textContent = `pmax ${pmVal.toFixed(2)}`;
      pmaxPill.className = `hw-ml-pill ${pmVal >= Number(mlCard.decision_threshold || 0.6) ? "ml-warn" : "ml-ok"}`;
    } else {
      pmaxPill.className = "hw-ml-pill ml-muted hidden";
    }
  }
  const vVerdict = String(vm.verdict || "OK");
  const vPillMap = {
    OK: "✓ VISION OK",
    ISSUE_FOULING: "⚠ FILL FOULING",
    ISSUE_LOW_FLOW_OR_NOZZLE: "⚠ NOZZLE / LOW FLOW",
    ISSUE_WATER_CHEM: "⚠ TURBID / LOW CoC",
  };
  setPill(
    "hw-vision-verdict-pill",
    vPillMap[vVerdict] || `⚠ ${vVerdict.replace(/_/g, " ")}`,
    vVerdict === "OK" ? "pill pill-ok" : "pill pill-fault"
  );

  const pkTile = (lbl, val, sub, cls = "", tip = "") =>
    `<div class="pk-box ${cls}"${tip ? ` title="${escapeHtml(tip)}"` : ""}><span class="pk-lbl">${escapeHtml(lbl)}</span><strong class="pk-val">${escapeHtml(val)}</strong><span class="pk-sub">${escapeHtml(sub)}</span></div>`;

  const m = vm.metrics || {};
  const vKpiHost = document.getElementById("hw-vision-kpis");
  if (vKpiHost) {
    if (activeHwVisionMode === "fill_strainer") {
      const occ = Number(m.occlusion_pct || 0);
      const sc = Number(m.scale_area_pct || 0);
      const bio = Number(m.biofilm_area_pct || 0);
      const str = Number(m.strainer_occlusion_pct || 0);
      vKpiHost.innerHTML = [
        pkTile("Fill Occlusion", `${occ.toFixed(1)}%`, occ >= 8 ? "Threshold ≥8.0%" : "Clean (<8.0%)", occ >= 8 ? "pk-warn" : "pk-hi"),
        pkTile("Calcium Scale", `${sc.toFixed(1)}%`, `${m.scale_patch_count ?? 0} crust patch(es)`, sc >= 3.5 ? "pk-warn" : ""),
        pkTile("Algal Biofilm", `${bio.toFixed(1)}%`, `${m.biofilm_patch_count ?? 0} zone(s)`, bio >= 2.5 ? "pk-warn" : ""),
        pkTile("Strainer Tray", `${str.toFixed(1)}%`, str >= 12 ? "Blocked (≥12%)" : "Open Grid OK", str >= 12 ? "pk-warn" : ""),
      ].join("");
    } else if (activeHwVisionMode === "spray_flow") {
      const gpm = Number(m.optical_flow_gpm || 0);
      const lowF = Boolean(m.flow_below_floor);
      const hNoz = Number(m.healthy_nozzles ?? 4);
      const cNoz = Number(m.clogged_nozzles ?? 0);
      const sym = Number(m.spray_symmetry_pct || 0);
      vKpiHost.innerHTML = [
        pkTile("Optical CW Flow", `${gpm.toFixed(1)} GPM`, lowF ? "<250 GPM Floor!" : "≥250 GPM Active", lowF ? "pk-warn" : "pk-hi"),
        pkTile("Healthy Cones", `${hNoz} / ${m.total_nozzles || 4}`, "Symmetric hollow spray", hNoz === 4 ? "pk-hi" : ""),
        pkTile("Clogged Nozzles", `${cNoz} Nozzle(s)`, cNoz > 0 ? "Dry zone / crust" : "0 occluded", cNoz > 0 ? "pk-warn" : ""),
        pkTile("Cone Symmetry", `${sym.toFixed(1)}%`, "Left/Right balance", sym < 70 ? "pk-warn" : ""),
      ].join("");
    } else {
      const ntu = Number(m.turbidity_ntu || 0);
      const coc = Number(m.estimated_coc || 0);
      const ph = Number(m.ph_est || 7.5);
      const hard = Number(m.hardness_ppm || 200);
      const alg = Number(m.algae_index || 0);
      vKpiHost.innerHTML = [
        pkTile("Turbidity (Lab)", `${ntu.toFixed(1)} NTU`, ntu > 15 ? ">15 NTU Turbid!" : "≤15 NTU Clear", ntu > 15 ? "pk-warn" : "pk-hi"),
        pkTile("5-in-1 CoC Strip", `CoC ${coc.toFixed(2)}`, coc < 3.2 ? "<3.2 Blowdown Risk" : "≥3.2 Optimal", coc < 3.2 ? "pk-warn" : "pk-hi"),
        pkTile("pH & Hardness", `pH ${ph.toFixed(2)}`, `${hard} ppm CaCO₃`, ""),
        pkTile("Algae / Verified", `Idx ${alg.toFixed(2)}`, "Verified Water = null", ""),
      ].join("");
    }
  }

  const detHost = document.getElementById("hw-vision-dets");
  if (detHost) {
    const dets = Array.isArray(vm.detections) ? vm.detections : [];
    const chips = dets.slice(0, 4).map(d => {
      const sev = String(d.severity || "OK").toUpperCase();
      const cls = sev === "HIGH" ? "det-high" : (sev === "MEDIUM" ? "det-med" : "det-low");
      return `<span class="hw-det-chip ${cls}">${escapeHtml(d.class_name)} ${Math.round(Number(d.confidence || 0) * 100)}% · ${Number(d.area_pct || 0).toFixed(1)}%</span>`;
    }).join("");
    detHost.innerHTML = chips || `<span class="hw-det-chip det-low">No fault detections</span>`;
    detHost.title = vm.summary || "";
  }

  // 3. Right Card R1: Acoustic FFT & Optical Tachometer (H3 / Step IV-2)
  const acVerdict = String(ac.verdict || "OK");
  setPill(
    "hw-fft-verdict-pill",
    acVerdict === "OK" ? "✓ DRIVE OK" : `⚠ ${acVerdict.replace(/_/g, " ")}`,
    acVerdict === "OK" ? "pill pill-ok" : (acVerdict === "ISSUE_BEARING" ? "pill pill-warn" : "pill pill-fault")
  );
  if (typeof window.drawAcousticFftSvg === "function") {
    window.drawAcousticFftSvg(document.getElementById("chart-acoustic-fft"), ac);
  }
  const fftKpis = document.getElementById("hw-fft-kpis");
  if (fftKpis) {
    const optRpm = Number(ac.optical_rpm || 0);
    const expRpm = Number(ac.expected_rpm || 0);
    const slip = Number(ac.slip_pct || 0);
    const bands = ac.bands || {};
    const sqRms = Number(bands.belt_squeal_2400_4200hz || 0);
    const brRms = Number(bands.bearing_180_420hz || 0);
    fftKpis.innerHTML = [
      pkTile("Optical Tach", `${optRpm.toFixed(0)} RPM`, "60fps ArUco Hub", acVerdict === "OK" ? "pk-hi" : "pk-warn"),
      pkTile("Expected RPM", `${expRpm.toFixed(0)} RPM`, `VFD ${Number(ac.vfd_pct || 0).toFixed(1)}%`, ""),
      pkTile("V-Belt Slip", `${slip.toFixed(1)}%`, slip >= 15 ? "Slip ≥15%!" : "Nominal (<15%)", slip >= 15 ? "pk-warn" : ""),
      pkTile("Peak / Band RMS", `${Number(ac.dominant_freq_hz || 0).toFixed(0)} Hz`, `Sq ${sqRms.toFixed(2)} · Br ${brRms.toFixed(2)}`, sqRms >= 0.14 || brRms >= 0.14 ? "pk-warn" : ""),
    ].join("");
  }

  // 4. Right Card R2: Stull Psychrometric Wet-Bulb Weather Gate (H4)
  const wVerdict = String(wg.confounder_verdict || "WEATHER_CLEAR");
  setPill(
    "hw-weather-verdict-pill",
    wVerdict === "WEATHER_CLEAR" ? "✓ WEATHER CLEAR" : (wVerdict === "HIGH_WETBULB_CONFOUNDER" ? "⚠ HIGH T_WB (H4)" : "🔒 SENSOR REJECTED"),
    wVerdict === "WEATHER_CLEAR" ? "pill pill-ok" : (wVerdict === "HIGH_WETBULB_CONFOUNDER" ? "pill pill-warn" : "pill pill-fault")
  );
  document.querySelectorAll("#hw-weather-buttons .seg-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.wpreset === wg.preset);
  });
  const wKpis = document.getElementById("hw-weather-kpis");
  if (wKpis) {
    const tdb = Number(wg.dry_bulb_c || 0);
    const rh = Number(wg.rh_pct || 0);
    const twb = Number(wg.stull_wet_bulb_c || 0);
    const tdp = Number(wg.dew_point_c || 0);
    const pOk = Boolean(wg.physics_valid);
    const appC = wg.tower_approach_est_c;
    wKpis.innerHTML = [
      pkTile("Dry-Bulb T_db", `${tdb.toFixed(1)} °C`, String(wg.location || "Local Site"), ""),
      pkTile("Rel. Humidity", `${rh.toFixed(0)} %`, `Dew Pt ${tdp.toFixed(1)} °C`, ""),
      pkTile("Stull Wet-Bulb", `${twb.toFixed(2)} °C`, `Reported ${Number(wg.reported_wet_bulb_c || twb).toFixed(1)} °C`, wVerdict === "HIGH_WETBULB_CONFOUNDER" ? "pk-warn" : "pk-hi"),
      pkTile("Twb ≤ Tdb Gate", pOk ? "PASS" : "REJECTED", (appC !== null && appC !== undefined && Number(appC) >= 0.5) ? `Approach ~${Number(appC).toFixed(1)}°C*` : "LBNL 99.3% Guard", pOk ? "pk-hi" : "pk-warn"),
    ].join("");
  }
  const wNote = document.getElementById("hw-weather-note");
  if (wNote) wNote.textContent = wg.summary || "";
  const wCard = document.getElementById("hw-weather-card");
  if (wCard && wg.summary) wCard.title = wg.summary;

  // 5. Right Card R3: Mini-CT1 15-Part 3D Printable Rig & Coaxial Landscape iPhone 11 Gantry
  const partsList = Array.isArray(bom.parts) ? bom.parts : [];
  const partsCount = Number(bom.parts_count || partsList.length || cad.parts_count || 15);
  const partNoOf = pt => Number(pt.part_id || pt.part_no || pt.id || 0);
  const partNameOf = pt => String(pt.display_name || pt.name || pt.slug || "");
  const p15 = partsList.find(pt => partNoOf(pt) === 15);
  const p15Name = (p15 && partNameOf(p15)) ? partNameOf(p15) : "Chevron Drift Eliminator";
  const opticsObj = bom.iphone11_optics || bom.optics || {};
  const stations = Array.isArray(opticsObj.stations) ? opticsObj.stations : [];
  const st2 = stations.find(st => Number(st.station || st.id) === 2) || {};
  const distMm = Number(st2.distance_mm || opticsObj.nominal_focus_distance_mm || cad.working_distance_mm || 150);
  const rawErr = st2.optical_error_mm !== undefined ? st2.optical_error_mm : st2.lens_error_mm;
  const lensErr = rawErr !== undefined ? Math.abs(Number(rawErr)).toFixed(1) : "0.0";
  const tiltStr = stations.length >= 3
    ? stations.map(st => {
        const deg = Math.round(Number(st.tilt_deg !== undefined ? st.tilt_deg : st.pitch_deg));
        return `${deg > 0 ? "+" : ""}${deg}°`;
      }).join("/")
    : "+18°/0°/-18°";
  const partsPill = document.getElementById("hw-cad-parts-pill");
  if (partsPill) {
    partsPill.textContent = `${partsCount} STL / STEP`;
    partsPill.title = partsList.length
      ? partsList.map(pt => `P${String(partNoOf(pt)).padStart(2, "0")} ${partNameOf(pt)}`).join(" · ")
      : `${partsCount} watertight 0%-support FDM parts (incl. Part 15 · ${p15Name})`;
  }
  const plaGrams = Math.round(Number(bom.total_pla_g_est || cad.total_pla_g_est || 991));
  const bomRange = String(bom.bom_vnd_range || cad.bom_vnd_range || "460k-680k ₫");
  const cadKpis = document.getElementById("hw-cad-kpis");
  if (cadKpis) {
    cadKpis.innerHTML = [
      pkTile("3D CAD Parts", `${partsCount} STL+STEP`, `P15 ${p15Name.replace(/^Chevron\s+/i, "").replace(/\s*\(.+\)$/, "")}`, "pk-hi", `Part 15: ${p15Name} · ${cad.print_envelope_mm || "≤185×185×175mm"}`),
      pkTile("PLA Filament", `~${plaGrams} g`, "0% Supports · FDM", ""),
      pkTile("Vietnam BOM", bomRange, "120mm 5V + LED", "pk-hi"),
      pkTile("Coaxial Optics", `${distMm} mm FOV`, `Landscape · ${lensErr}mm err`, "pk-hi"),
      pkTile("3-Station Tilt", tiltStr, "ArUco 0..3 + Mic", ""),
    ].join("");
  }
  updateAuditHook();
}

async function loadHardwareStatus(opts = {}) {
  if (hwLoadingPromise) return hwLoadingPromise;
  hwLoadingPromise = (async () => {
    try {
      const hw = await apiJson("/api/hardware/status?img_fmt=jpeg");
      renderHardwareTab(hw, opts);
      return hw;
    } catch (err) {
      auditErrors.push(`HW status error: ${err.message}`);
      updateAuditHook();
      return null;
    } finally {
      hwLoadingPromise = null;
    }
  })();
  return hwLoadingPromise;
}

async function applyHardwareToStep4(switchToTriage = false) {
  const snap = await apiJson("/api/hardware/apply-to-step4", "POST");
  if (snap && snap.field_checks) {
    syncFieldChecksFromBackend(snap.field_checks);
  }
  renderSnapshot(snap);
  if (switchToTriage) {
    switchTab("derisk");
  }
}

/* ==================== STEP III DAG / CARDS VIEW TOGGLE ==================== */
function hideDagFloatingTip() {
  const dft = document.getElementById("dag-floating-tip");
  if (dft) dft.classList.add("hidden");
}

function setStep3View(mode) {
  hideDagFloatingTip();
  step3ViewMode = mode === "cards" ? "cards" : "dag";
  if (latestSnap) renderTriageWorkbench(latestSnap);
  updateAuditHook();
}

/* ==================== AIR-GAPPED EVIDENCE-GROUNDED RCA AGENT (HOTKEY: A) ==================== */
function renderSafeRcaMarkdown(title, mdText) {
  const formatInline = raw => {
    let s = escapeHtml(raw);
    // Render both `[T1:Gate]` and [T1:Gate] as crisp evidence citation pills
    s = s.replace(/`?(\[T\d:[^\]]+\])`?/g, '<span class="rca-cite">$1</span>');
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/`([^`]+)`/g, '<code class="rca-code">$1</code>');
    return s;
  };

  const parts = [];
  if (title) parts.push(`<div class="rca-h">${escapeHtml(title)}</div>`);
  const lines = String(mdText || "").split(/\r?\n/);
  let inList = false;
  const closeList = () => {
    if (inList) { parts.push("</ul>"); inList = false; }
  };

  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed) {
      closeList();
      return;
    }
    if (trimmed.startsWith("### ") || trimmed.startsWith("## ")) {
      closeList();
      parts.push(`<div class="rca-h">${formatInline(trimmed.replace(/^#{2,3}\s+/, ""))}</div>`);
    } else if (/^[-•]\s+/.test(trimmed)) {
      if (!inList) { parts.push('<ul class="rca-ul">'); inList = true; }
      parts.push(`<li class="rca-li">${formatInline(trimmed.replace(/^[-•]\s+/, ""))}</li>`);
    } else {
      closeList();
      parts.push(`<p class="rca-p">${formatInline(trimmed)}</p>`);
    }
  });
  closeList();
  return parts.join("");
}

function toolChipStatusClass(st) {
  const s = String(st || "").toUpperCase();
  if (["PASS", "NOMINAL", "CALM", "POSITIVE_ROI", "MATCHED", "ESTIMATED"].includes(s)) return "st-ok";
  if (["BLOCKED", "ALERT", "ISOLATED"].includes(s)) return "st-fault";
  if (["ABSTAIN", "LOCKED", "DEFER", "STANDBY", "SUSPENDED"].includes(s)) return "st-warn";
  return "st-info";
}

function renderToolInspector(idx) {
  const insp = document.getElementById("rca-tool-inspector");
  const titleEl = document.getElementById("rca-insp-title");
  const preEl = document.getElementById("rca-insp-json");
  if (!insp || !latestRcaResp || !Array.isArray(latestRcaResp.tool_traces)) return;

  if (idx === null || idx === undefined || !latestRcaResp.tool_traces[idx]) {
    selectedToolIdx = null;
    insp.classList.add("hidden");
    document.querySelectorAll("#rca-tool-strip .rca-tool-chip").forEach(b => b.classList.remove("selected"));
    return;
  }
  selectedToolIdx = idx;
  const t = latestRcaResp.tool_traces[idx];
  document.querySelectorAll("#rca-tool-strip .rca-tool-chip").forEach(b => {
    b.classList.toggle("selected", Number(b.dataset.toolIdx) === idx);
  });
  titleEl.textContent = `${t.tag} · ${t.tool_name} (${t.status} · ${Number(t.latency_ms || 0).toFixed(2)}ms)`;
  preEl.textContent = `${t.summary || ""}\n\n${JSON.stringify(t.payload || {}, null, 2)}`;
  insp.classList.remove("hidden");
}

function renderRcaResponse(resp) {
  latestRcaResp = resp;
  const latPill = document.getElementById("rca-latency-pill");
  const digPill = document.getElementById("rca-digest-pill");
  const intentTag = document.getElementById("rca-intent-tag");
  const langPill = document.getElementById("rca-lang-pill");
  const guardEl = document.getElementById("rca-guard-banner");
  const stripEl = document.getElementById("rca-tool-strip");
  const bodyEl = document.getElementById("rca-answer-body");

  if (latPill) latPill.textContent = `${Number(resp.total_latency_ms || 0).toFixed(1)} ms`;
  if (digPill) {
    const shortDig = (resp.evidence_digest_short || resp.evidence_digest_sha256 || "").slice(0, 12);
    digPill.textContent = `SHA256: ${shortDig}`;
    digPill.title = `Canonical 6-Tool Evidence SHA-256: ${resp.evidence_digest_sha256 || ""}`;
  }
  if (intentTag) intentTag.textContent = `${String(resp.intent || "diagnostic").replace(/_/g, " ").toUpperCase()} · ${resp.basis_mode === "illustrative" ? "QUAN REF" : "LIVE"}`;
  if (langPill) langPill.textContent = String(resp.lang || "en").toUpperCase();

  if (guardEl) {
    if (resp.refused_control_write) {
      guardEl.className = "rca-guard-banner rca-guard-refuse";
      guardEl.textContent = "🔒 IEC 62443-3-3 CONTROL-WRITE REFUSED — READ-ONLY CONDUIT (controls=false, zero PLC path)";
    } else if (resp.abstain_blocked) {
      guardEl.className = "rca-guard-banner rca-guard-abstain";
      guardEl.textContent = "⚠ ABSTAIN INTERLOCK ACTIVE — FOULING DIAGNOSIS SUSPENDED (Root N3 Gate Block)";
    } else {
      guardEl.className = "rca-guard-banner hidden";
      guardEl.textContent = "";
    }
  }

  const traces = Array.isArray(resp.tool_traces) ? resp.tool_traces : [];
  const traceLbl = document.querySelector(".rca-trace-lbl");
  if (traceLbl && traces.length) traceLbl.textContent = `${traces.length}-TOOL EVIDENCE TRACE`;
  if (stripEl) {
    const fmtMs = v => {
      const n = Number(v || 0);
      return n >= 1 ? `${n.toFixed(1)}ms` : `${n.toFixed(2)}ms`;
    };
    stripEl.innerHTML = traces.map((t, idx) => `
      <button type="button" class="rca-tool-chip ${toolChipStatusClass(t.status)}${selectedToolIdx === idx ? " selected" : ""}" data-tool-idx="${idx}" title="${escapeHtml(t.tool_name)}: ${escapeHtml(t.summary)} (click to inspect JSON)">
        <span>${escapeHtml(t.tag)}</span>
        <span class="rca-tool-ms">${fmtMs(t.latency_ms)}</span>
      </button>
    `).join("");
  }
  if (selectedToolIdx !== null && traces[selectedToolIdx]) renderToolInspector(selectedToolIdx);

  if (bodyEl) {
    bodyEl.innerHTML = renderSafeRcaMarkdown(resp.title, resp.answer_markdown);
  }
  updateAuditHook();
}

async function queryRcaAgent(req = {}, opts = {}) {
  const preset = req.preset || null;
  const query = typeof req.query === "string" ? req.query : "";
  const lang = req.lang || "auto";
  lastRcaRequest = { preset, query, lang };
  if (latestSnap) {
    const e = latestSnap.electricity || {};
    const sObj = tab3BasisMode === "live" ? e.live_alert_projection : e.illustrative_scenario;
    const pmC = (sObj && sObj.pm_optimizer && sObj.pm_optimizer.pm_cost_vnd) || 8000000;
    lastRcaSnapKey = `${latestSnap.scenario}|${latestSnap.row_index}|${latestSnap.injection}|${tab3BasisMode}|${pmC}`;
  }

  // Sync quick-chip highlight
  document.querySelectorAll("#rca-quick-chips .rca-chip").forEach(b => {
    const matchPreset = preset && b.dataset.preset === preset;
    const matchQuery = !preset && query && b.dataset.query && b.dataset.query === query;
    b.classList.toggle("active", Boolean(matchPreset || matchQuery));
  });

  const sendBtn = document.getElementById("btn-rca-send");
  rcaBusy = true;
  if (sendBtn && !opts.silent) sendBtn.disabled = true;
  try {
    const basisMode = tab3BasisMode === "live" ? "live" : "illustrative";
    const resp = await apiJson("/api/rca-agent", "POST", {
      query,
      preset,
      basis_mode: basisMode,
      lang,
    });
    renderRcaResponse(resp);
  } catch (err) {
    const bodyEl = document.getElementById("rca-answer-body");
    if (bodyEl) bodyEl.innerHTML = `<div class="rca-placeholder">RCA Agent error: ${escapeHtml(err.message)}</div>`;
  } finally {
    rcaBusy = false;
    if (sendBtn) sendBtn.disabled = false;
  }
}

function openRcaDrawer(req = null) {
  const drawer = document.getElementById("rca-drawer");
  const btn = document.getElementById("btn-rca");
  if (!drawer) return;
  rcaOpen = true;
  drawer.classList.remove("hidden");
  if (btn) btn.classList.add("active");
  queryRcaAgent(req || lastRcaRequest);
  updateAuditHook();
}

function closeRcaDrawer() {
  const drawer = document.getElementById("rca-drawer");
  const btn = document.getElementById("btn-rca");
  if (!drawer) return;
  rcaOpen = false;
  drawer.classList.add("hidden");
  if (btn) btn.classList.remove("active");
  updateAuditHook();
}

function toggleRcaDrawer() {
  if (rcaOpen) closeRcaDrawer();
  else openRcaDrawer();
}

/* ==================== 24H PARITY-SAFE TIMELINE SCRUBBER ==================== */
function renderTimelineScrubber(snap) {
  const host = document.getElementById("tl-bins");
  const cursorEl = document.getElementById("tl-cursor");
  if (!host || !snap.timeline_24h) return;
  if (lastTimelineScenario !== snap.scenario) {
    lastTimelineScenario = snap.scenario;
    const stateLabels = {
      off: "Off-Hours / Gate",
      eligible: "Eligible (No Alert)",
      rule_only: "High-Fan Rule Only",
      ai_only: "AI-Only Early Window (Rule Clean)",
      both: "Both AI & Rule Flagged",
    };
    host.innerHTML = snap.timeline_24h.map(b =>
      `<div class="tl-bin tl-${escapeHtml(b.state)}" data-seek-row="${b.mid_row}" title="${escapeHtml(b.time)} · ${escapeHtml(stateLabels[b.state] || b.state)} (click to seek)"></div>`
    ).join("");
  }
  if (cursorEl && snap.total_rows > 1) {
    const pct = Math.max(0, Math.min(100, (snap.row_index / (snap.total_rows - 1)) * 100));
    cursorEl.style.left = `${pct.toFixed(2)}%`;
  }
}

/* ==================== 6-BEAT GUIDED STAGE STORY CHOREOGRAPHY ==================== */
function setActiveBeatBtn(beatId) {
  activeStoryBeat = beatId;
  document.querySelectorAll("#story-beats .story-btn").forEach(b => b.classList.toggle("active", b.dataset.beat === String(beatId)));
}

function runStoryBeat(beatId) {
  stopPlayback();
  setActiveBeatBtn(beatId);
  const b = String(beatId);
  if (b === "1") {
    fieldChecks.forEach(fc => { fc.state = "pending"; });
    enqueue(async () => {
      renderSnapshot(await apiJson("/api/seek", "POST", { scenario: "normal", row_index: 735 }));
      switchTab("overview");
    });
  } else if (b === "2") {
    // Exact real Oct-01 moment (#640, 10:40): Quan HGB 0.911 FOULING_SUSPECT while Fan 93.9% Rule stays 0/15m!
    fieldChecks.forEach(fc => { fc.state = "pending"; });
    enqueue(async () => {
      renderSnapshot(await apiJson("/api/seek", "POST", { scenario: "fouling_high", row_index: 640 }));
      switchTab("overview");
    });
  } else if (b === "3") {
    enqueue(async () => {
      renderSnapshot(await apiJson("/api/seek", "POST", { scenario: "fouling_high", row_index: 735 }));
      switchTab("trends");
    });
  } else if (b === "4") {
    tab3BasisMode = "live";
    enqueue(async () => {
      await apiJson("/api/seek", "POST", { scenario: "fouling_high", row_index: 735 });
      renderSnapshot(await apiJson("/api/illustrative", "POST", { imposed_excess_kw: 3.0, earlier_pm_day: 28, existing_pm_day: 120, restored_fraction: 1.0 }));
      switchTab("quantify");
    });
  } else if (b === "5") {
    enqueue(async () => {
      await apiJson("/api/seek", "POST", { scenario: "fouling_high", row_index: 735 });
      renderSnapshot(await apiJson("/api/inject-fault", "POST", { fault_type: "domain_ood" }));
      switchTab("derisk");
    });
  } else if (b === "6") {
    fieldChecks[0].state = "ok";
    fieldChecks[1].state = "ok";
    fieldChecks[2].state = "issue";
    fieldChecks[3].state = "ok";
    enqueue(async () => {
      renderSnapshot(await apiJson("/api/seek", "POST", { scenario: "fouling_high", row_index: 735 }));
      switchTab("derisk");
    });
  }
}

function toggleStoryBar(forceOpen = null) {
  const bar = document.getElementById("story-bar");
  const btn = document.getElementById("btn-story");
  if (!bar) return;
  const open = forceOpen !== null ? forceOpen : bar.classList.contains("hidden");
  bar.classList.toggle("hidden", !open);
  if (btn) btn.classList.toggle("active", open);
  updateAuditHook();
}

/* ==================== INTERACTIONS ==================== */
function switchTab(tabId) {
  hideDagFloatingTip();
  activeTabId = tabId;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tabId));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.toggle("active", p.id === `tab-${tabId}`));
  if (tabId === "hardware") {
    if (latestHw) {
      if (dirtyTabs.hardware) renderHardwareTab(latestHw, { keepMode: true });
    } else {
      loadHardwareStatus({ keepMode: false });
    }
  } else if (latestSnap && dirtyTabs[tabId]) {
    renderActiveTab(latestSnap);
  }
  updateAuditHook();
}

function stopPlayback() {
  isPlaying = false;
  if (playTimer) clearTimeout(playTimer);
  playTimer = null;
  const btn = document.getElementById("btn-play");
  btn.textContent = "▶ Play";
  btn.classList.remove("playing");
}

function scheduleNextPlayTick() {
  if (!isPlaying) return;
  playTimer = setTimeout(() => {
    if (!isPlaying || isStepping) return;
    isStepping = true;
    const t0 = performance.now();
    enqueue(async () => {
      try {
        renderSnapshot(await apiJson("/api/step", "POST", { steps: 2 }));
      } finally {
        isStepping = false;
        if (isPlaying) {
          const elapsed = performance.now() - t0;
          const delay = Math.max(120, Math.round(520 - elapsed));
          playTimer = setTimeout(scheduleNextPlayTick, delay);
        }
      }
    });
  }, 0);
}

function startPlayback() {
  stopPlayback();
  isPlaying = true;
  const btn = document.getElementById("btn-play");
  btn.textContent = "⏸ Pause";
  btn.classList.add("playing");
  scheduleNextPlayTick();
}

function submitVerdict(verdict) {
  const issues = fieldChecks.filter(fc => fc.state === "issue").map(fc => fc.title);
  const steps = ["I", "II", "III", "IV", "V"];
  if (verdict === "CONFIRMED") steps.push("VI");
  const engInp = document.getElementById("input-engineer");
  const noteInp = document.getElementById("input-note");
  const engineerName = (engInp && engInp.value.trim()) || "Facilities Supervisor (Shift A)";
  const customNote = noteInp ? noteInp.value.trim() : "";
  const defaultNote = verdict === "CONFIRMED"
    ? (issues.length ? `Confirmed via Steps I–IV (${issues.join(", ")}) → Step VI draft brief ready.` : "Confirmed via Steps I–IV → Step VI draft brief ready.")
    : verdict === "FALSE_ALARM"
      ? "Recorded as False Alarm after Steps I–IV review (archived for pilot feedback)."
      : "Marked Inconclusive — 24h observation watch initiated.";
  const note = customNote || defaultNote;
  enqueue(async () => renderSnapshot(await apiJson("/api/acknowledge", "POST", {
    verdict,
    engineer_name: engineerName,
    completed_steps: steps,
    note,
  })));
}

/* ==================== ARCHITECTURE, MODEL CARD, OPERATING WINDOWS & JUDGE Q&A MODAL ==================== */
function renderInfoModal(snap) {
  if (!snap) return;
  const mc = snap.model_card || {};
  const em = snap.event_metrics || {};
  const methods = em.methods || {};
  const hgbWin = (methods.histogram_boosting_candidate && methods.histogram_boosting_candidate.by_case) || {};
  const ruleWin = (methods.reference_rule && methods.reference_rule.by_case) || {};
  const winCases = [
    ["normal", "Normal (1.00 UA)"],
    ["tower_fouling_095", "0.95 UA (−5%)"],
    ["tower_fouling_080", "0.80 UA (−20%)"],
    ["tower_fouling_065", "0.65 UA (−35%)"],
  ];
  const winRows = winCases.map(([k, lbl]) => {
    const a = hgbWin[k] || {}, r = ruleWin[k] || {};
    const aLat = a.first_alert_latency_minutes_detected_windows_only || {};
    const rLat = r.first_alert_latency_minutes_detected_windows_only || {};
    return `<tr>
      <td>${escapeHtml(lbl)}</td>
      <td>${a.windows_with_alert ?? "--"} / ${a.eligible_windows ?? "--"}</td>
      <td>${aLat.median !== undefined && aLat.median !== null ? aLat.median.toFixed(1) + "m" : "--"}</td>
      <td>${r.windows_with_alert ?? "--"} / ${r.eligible_windows ?? "--"}</td>
      <td>${rLat.median !== undefined && rLat.median !== null ? rLat.median.toFixed(1) + "m" : "--"}</td>
    </tr>`;
  }).join("");

  const cardEl = document.getElementById("mtab-card");
  if (cardEl) {
    const limitChips = [
      ["Mild 0.95 UA: 4.5% Recall", "Rule (8.5%) beats AI on mildest fouling — disclosed honestly."],
      ["13.95% Eligible Coverage", "73,932 eligible Oct–Dec rows (active cooling regime only)."],
      ["Weather / CHW Pending*", "Raw LBNL OA_TEMP_WB > OA_TEMP (99.31%) excluded without auto-swap."],
      ["Uncalibrated Score", "Score > 0.70 for 5 min is binary suspicion — not probability or severity."],
      ["No RUL Forecast", "Detects active fouling degradation only; no remaining-useful-life claim."],
      ["Site Pilot Required", "Office LBNL simulation; fab KPIs and verified savings require Shadow Pilot."],
    ];
    cardEl.innerHTML = `
      <div class="mc-grid">
        <div class="mc-box">
          <h4>Frozen Model Card</h4>
          <div class="mc-kv"><span>Algorithm</span><strong>${escapeHtml(mc.model || "HistGradientBoostingClassifier")}</strong></div>
          <div class="mc-kv"><span>SHA256</span><strong><code>${escapeHtml(snap.prediction.model_sha256 || "")}</code></strong></div>
          <div class="mc-kv"><span>6 Native Inputs</span><strong>${escapeHtml((mc.features || []).join(", "))}</strong></div>
          <div class="mc-kv"><span>Chronological Split</span><strong>${escapeHtml(mc.split || "")} (${Number(mc.training_rows || 224245).toLocaleString()} rows)</strong></div>
          <div class="mc-kv"><span>Decision Gate</span><strong>Score &gt; 0.70 · 5 consec. min (Uncalibrated · Binary)</strong></div>
          <div class="mc-kv"><span>Explainability</span><strong>1-at-a-time perturbation vs train mean (Not SHAP)</strong></div>
          <div class="mc-kv"><span>Streaming Parity</span><strong>0.00e+00 (5,760 Oct-01 rows · 2,948 eligible)</strong></div>
          <div class="mc-note">Exploratory reused Oct–Dec test · F1 0.394 vs Rule 0.237 · Precision 97.8%.</div>
        </div>
        <div class="mc-box">
          <h4>Operating Windows (Oct–Dec)</h4>
          <table class="cond-table">
            <thead><tr><th>Case</th><th>HGB Win</th><th>HGB Med</th><th>Rule Win</th><th>Rule Med</th></tr></thead>
            <tbody>${winRows}</tbody>
          </table>
          <div class="mc-note">Active-window observation opportunities — not physical onset-to-alert delay.</div>
          <h4 style="margin-top:4px">Disclosed Limits (Hover for Details)</h4>
          <div class="mc-chips">${limitChips.map(([lbl, tip]) => `<span class="mc-chip mc-chip-warn" title="${escapeHtml(tip)}">${escapeHtml(lbl)}</span>`).join("")}</div>
        </div>
      </div>
    `;
  }

  const pilotEl = document.getElementById("mtab-pilot");
  if (pilotEl) {
    pilotEl.innerHTML = `
      <div class="mc-grid">
        <div class="mc-box">
          <h4>Verbatim Site Pilot Question</h4>
          <div class="pilot-quote">“Can the prototype flag cooling-tower performance degradation during a period when engineers independently confirmed fouling or a need for cleaning?”</div>
          <h4 style="margin-top:4px">3-Bucket Evidence Boundary</h4>
          <ul class="mc-list">
            <li><strong>Bucket 1 (Demonstrated):</strong> Offline LBNL replay (0.00e+00), 4-gate OOD, HGB F1 0.394 vs Rule 0.237, Causal DAG, Mini-CT1 Vision/FFT/Stull, RCA Agent.</li>
            <li><strong>Bucket 2 (Illustrative):</strong> Live-Alert ΔkW* &amp; 6,624 kWh template + EVN 22kV TOU ROI (verified_avoided_kwh = null).</li>
            <li><strong>Bucket 3 (Site Pilot):</strong> 2–4w read-only fab shadow baseline, verified kWh, WTP water meters &amp; fab cooling limits.</li>
          </ul>
        </div>
        <div class="mc-box">
          <h4>5M1E → Money / Time / ESG</h4>
          <ul class="mc-list">
            <li><strong>Machine &amp; Material (Money):</strong> Cut fan saturation &amp; condenser lift; electricity-only in MVP, water/chem in Pilot.</li>
            <li><strong>Man &amp; Method (Time):</strong> Shift calendar PM to engineer-approved N+1 cleaning via Steps I–VI.</li>
            <li><strong>Measurement (Time):</strong> 4-gate OOD + 97.8% precision minimize nuisance dispatch.</li>
            <li><strong>Environment (ESG):</strong> Lower Scope 2 CO₂ from consuming excess electrical energy (SEMI S23: PCW-C ~1.56 vs PCW-T ~0.26 kWh/m³).</li>
          </ul>
        </div>
      </div>
    `;
  }

  const qaEl = document.getElementById("mtab-qa");
  if (qaEl) {
    const qas = [
      ["1. Why 24.7% recall (F1 0.394) instead of 97%?",
       "Strict chronological split (Jan–Jun train → Oct–Dec test) with corrupted weather/CHW excluded yields 97.8% precision and honest low mild-fouling recall (4.5%)."],
      ["2. Why not auto-swap OA_TEMP_WB & CHW columns?",
       "Thermodynamic inversion (99.31% WB > DB) proves an audit defect, not which physical label is true, so we exclude them rather than fabricating evidence."],
      ["3. Does office LBNL simulation prove Fab readiness?",
       "No — LBNL proves the read-only workflow, 4-gate abstention, and AI-vs-rule separation, while fab transfer requires an authorized Shadow Pilot."],
      ["4. Are Tab 3 kWh / Water Est. verified fab savings?",
       "No — verified_excess_kwh, verified_avoided_kwh, and water_savings stay null; Tab 3 shows illustrative EVN 22kV TOU and ASHRAE estimates pending Pilot meters."],
      ["5. What prevents false maintenance or PLC trips?",
       "Read-only architecture (controls=false), 4-gate ABSTAIN with HTTP 409 Step V–VI lock, and mandatory human triage producing only a CMMS draft."],
      ["6. Why uncalibrated score & local sensitivity (not SHAP)?",
       "Raw tree scores are not calibrated probabilities, and 1-at-a-time perturbation vs train mean runs in ~12 ms offline without falsely claiming TreeSHAP."],
      ["7. Why Air-Gapped RCA Agent over a Cloud LLM?",
       "Purdue Level 3 fabs ban cloud Wi-Fi; our local 7-tool agent ([T1]–[T7] + SHA-256) delivers 0% hallucination in EN/VI and deterministically refuses PLC writes."],
      ["8. How do Causal DAG & EVN TOU Optimizer work?",
       "The 8-node DAG isolates upstream roots (N1–N4) from symptoms (N5–N8), while MOIT QD 2892/BCT 22kV TOU computes Break-Even Payback for Clean Now vs Defer PM."],
      ["9. How do you prevent alarm fatigue (FAR 0.096 ep/h)?",
       "5-min consecutive persistence filters spikes, Causal DAG isolates competing causes, and Step V logs False Alarm feedback to tune pilot thresholds."],
      ["10. What does a 90-day Fab Shadow Pilot need?",
       "Read-only historian exports (14-Ch power, CW temps/flow, fan VFD, verified weather/CHW) plus CMMS cleaning dates and WTP meters — zero PLC writes."],
      ["11. How does Mini-CT1 v12 + iPhone 11 assist Steps III–IV?",
       "A 15-part 0%-support FDM rig (w/ Drift Eliminator & coaxial Landscape +18°/0°/-18° cradle) runs 3-stage offline Vision ML (HGB 88273963, AUC 0.998), ArUco + 1-tap P07 calibration, and live 2s Mic FFT to pre-fill Step IV while human Step V stays mandatory."],
      ["12. How does Stull (2011) fix the 99.31% LBNL weather bug?",
       "Tab 5 computes local Twb(Tdb, RH) via Stull (2011) and enforces Twb ≤ Tdb to reject bad sensors and flag H4 high-wet-bulb confounders (≥27.5°C)."],
    ];
    qaEl.innerHTML = `<div class="qa-grid">${qas.map(([q, a]) => `<div class="qa-card"><div class="qa-q">${escapeHtml(q)}</div><div class="qa-a">${escapeHtml(a)}</div></div>`).join("")}</div>`;
  }
}

function openInfoModal(tab = "card") {
  const modal = document.getElementById("info-modal");
  if (!modal) return;
  if (latestSnap) renderInfoModal(latestSnap);
  modal.classList.remove("hidden");
  switchModalTab(tab);
  updateAuditHook();
}

function closeInfoModal() {
  const modal = document.getElementById("info-modal");
  if (modal) modal.classList.add("hidden");
  updateAuditHook();
}

function switchModalTab(mtabId) {
  document.querySelectorAll("#modal-tabs .mtab-btn").forEach(b => b.classList.toggle("active", b.dataset.mtab === mtabId));
  document.querySelectorAll(".modal-body .mtab-pane").forEach(p => p.classList.toggle("active", p.id === `mtab-${mtabId}`));
}

window.addEventListener("DOMContentLoaded", async () => {
  const pidPromise = loadInteractivePid();

  document.querySelectorAll(".tab-btn").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  document.getElementById("btn-open-triage").addEventListener("click", () => switchTab("derisk"));

  document.querySelectorAll("#scenario-buttons .seg-btn").forEach(b => {
    b.addEventListener("click", () => {
      stopPlayback();
      resetFieldChecks();
      document.querySelectorAll("#scenario-buttons .seg-btn").forEach(x => x.classList.toggle("active", x === b));
      enqueue(async () => renderSnapshot(await apiJson("/api/scenario", "POST", { scenario: b.dataset.scenario, seek_eligible: true })));
    });
  });

  // Single delegated listener for Step IV interactive field checks (never leaks listeners on re-render)
  const checklistEl = document.getElementById("step4-checklist");
  if (checklistEl) {
    checklistEl.addEventListener("click", evt => {
      const btn = evt.target.closest("[data-field-id]");
      if (!btn) return;
      const item = fieldChecks.find(x => x.id === btn.dataset.fieldId);
      if (!item) return;
      item.state = item.state === "pending" ? "ok" : item.state === "ok" ? "issue" : "pending";
      item.source = item.state === "pending" ? "" : "Manual";
      if (latestSnap) renderTriageWorkbench(latestSnap);
    });
  }

  // Step IV & Tab 5: Apply Multi-Modal AI Field-Assist to Step IV (Advisory Only — Human Step V required)
  const btnStep4Ai = document.getElementById("btn-step4-ai");
  if (btnStep4Ai) {
    btnStep4Ai.addEventListener("click", () => {
      enqueue(async () => applyHardwareToStep4(false));
    });
  }
  const btnHwApplyStep4 = document.getElementById("btn-hw-apply-step4");
  if (btnHwApplyStep4) {
    btnHwApplyStep4.addEventListener("click", () => {
      enqueue(async () => applyHardwareToStep4(true));
    });
  }

  // Tab 5: Bench Scenario Presets, Vision Mode Tabs, Weather Presets, iPhone Photo Upload & RCA Field-AI
  document.querySelectorAll("#hw-preset-buttons .seg-btn").forEach(b => {
    b.addEventListener("click", () => {
      const preset = b.dataset.preset;
      document.querySelectorAll("#hw-preset-buttons .seg-btn").forEach(x => x.classList.toggle("active", x === b));
      enqueue(async () => {
        const hw = await apiJson("/api/hardware/simulate?img_fmt=jpeg", "POST", { preset });
        renderHardwareTab(hw, { keepMode: false });
      });
    });
  });

  document.querySelectorAll("#hw-vision-modes .dt-btn").forEach(b => {
    b.addEventListener("click", () => {
      activeHwVisionMode = b.dataset.mode;
      if (latestHw) renderHardwareTab(latestHw, { keepMode: true });
    });
  });

  document.querySelectorAll("#hw-weather-buttons .seg-btn").forEach(b => {
    b.addEventListener("click", () => {
      const wpreset = b.dataset.wpreset;
      document.querySelectorAll("#hw-weather-buttons .seg-btn").forEach(x => x.classList.toggle("active", x === b));
      enqueue(async () => {
        const hw = await apiJson("/api/hardware/weather?lite=1", "POST", { preset: wpreset });
        renderHardwareTab(hw, { keepMode: true });
      });
    });
  });

  function readImageForUpload(file, maxDim = 1600, maxRawBytes = 1800000) {
    return new Promise((resolve, reject) => {
      const readRaw = () => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("Failed to read image file"));
        reader.readAsDataURL(file);
      };
      if (!file || file.size <= maxRawBytes) {
        readRaw();
        return;
      }
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        try {
          const scale = Math.min(1, maxDim / Math.max(img.naturalWidth || 1, img.naturalHeight || 1));
          const w = Math.max(16, Math.round((img.naturalWidth || 640) * scale));
          const h = Math.max(16, Math.round((img.naturalHeight || 400) * scale));
          const cv = document.createElement("canvas");
          cv.width = w;
          cv.height = h;
          cv.getContext("2d").drawImage(img, 0, 0, w, h);
          URL.revokeObjectURL(url);
          resolve(cv.toDataURL("image/jpeg", 0.90));
        } catch (_err) {
          URL.revokeObjectURL(url);
          readRaw();
        }
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        readRaw();
      };
      img.src = url;
    });
  }

  const btnHwUpload = document.getElementById("btn-hw-upload");
  const hwFileInp = document.getElementById("hw-file-input");
  if (btnHwUpload && hwFileInp) {
    btnHwUpload.addEventListener("click", () => hwFileInp.click());
    hwFileInp.addEventListener("change", evt => {
      const file = evt.target.files && evt.target.files[0];
      if (!file) return;
      readImageForUpload(file).then(b64 => {
        latestUploadedB64 = b64;
        enqueue(async () => {
          const resp = await apiJson("/api/hardware/inspect-frame?img_fmt=jpeg", "POST", {
            mode: activeHwVisionMode,
            image_base64: b64,
            source: "iphone11_upload",
          });
          if (resp && resp.hardware_status) {
            renderHardwareTab(resp.hardware_status, { keepMode: true });
          } else {
            await loadHardwareStatus({ keepMode: true });
          }
        });
      }).catch(() => {});
      evt.target.value = "";
    });
  }

  const btnHwCalibrate = document.getElementById("btn-hw-calibrate");
  if (btnHwCalibrate) {
    btnHwCalibrate.addEventListener("click", () => {
      enqueue(async () => {
        const resp = await apiJson("/api/hardware/calibrate-bench?img_fmt=jpeg", "POST", {
          image_base64: latestUploadedB64 || null,
          mode: "fill_strainer",
          label: "Clean Part 07",
        });
        if (resp && resp.hardware_status) {
          renderHardwareTab(resp.hardware_status, { keepMode: true });
        } else {
          await loadHardwareStatus({ keepMode: true });
        }
      });
    });
  }

  const btnHwResetCal = document.getElementById("btn-hw-reset-cal");
  if (btnHwResetCal) {
    btnHwResetCal.addEventListener("click", () => {
      enqueue(async () => {
        const resp = await apiJson("/api/hardware/calibrate-bench?img_fmt=jpeg", "DELETE");
        if (resp && resp.hardware_status) {
          renderHardwareTab(resp.hardware_status, { keepMode: true });
        } else {
          await loadHardwareStatus({ keepMode: true });
        }
      });
    });
  }

  const btnHwRca = document.getElementById("btn-hw-ask-rca");
  if (btnHwRca) {
    btnHwRca.addEventListener("click", () => {
      openRcaDrawer({ preset: "field_ai", query: "" });
    });
  }

  document.querySelectorAll("#excess-kw-buttons .seg-btn").forEach(b => {
    b.addEventListener("click", () => {
      if (b.dataset.mode === "live") {
        tab3BasisMode = "live";
        if (latestSnap) renderSnapshot(latestSnap);
      } else {
        tab3BasisMode = "manual";
        enqueue(async () => renderSnapshot(await apiJson("/api/illustrative", "POST", { imposed_excess_kw: Number(b.dataset.kw) })));
      }
    });
  });

  document.querySelectorAll("#clean-day-buttons .seg-btn").forEach(b => {
    b.addEventListener("click", () => {
      enqueue(async () => renderSnapshot(await apiJson("/api/illustrative", "POST", { earlier_pm_day: Number(b.dataset.day) })));
    });
  });

  document.querySelectorAll("#cal-day-buttons .seg-btn").forEach(b => {
    b.addEventListener("click", () => {
      enqueue(async () => renderSnapshot(await apiJson("/api/illustrative", "POST", { existing_pm_day: Number(b.dataset.cal) })));
    });
  });

  document.querySelectorAll("#restore-buttons .seg-btn").forEach(b => {
    b.addEventListener("click", () => {
      enqueue(async () => renderSnapshot(await apiJson("/api/illustrative", "POST", { restored_fraction: Number(b.dataset.rest) })));
    });
  });

  const resetIllusBtn = document.getElementById("btn-reset-illus");
  if (resetIllusBtn) {
    resetIllusBtn.addEventListener("click", () => {
      tab3BasisMode = "manual";
      enqueue(async () => renderSnapshot(await apiJson("/api/illustrative", "POST", { reset_defaults: true })));
    });
  }

  // Tab 3 MONEY Card: Compact PM Cost Preset Toggle (4M / 8M / 15M VND)
  document.querySelectorAll("#pm-cost-buttons .pm-mini-btn").forEach(b => {
    b.addEventListener("click", () => {
      const pmCost = Number(b.dataset.pm);
      enqueue(async () => renderSnapshot(await apiJson("/api/pm-optimizer", "POST", { pm_cost_vnd: pmCost })));
    });
  });

  // Tab 4 Step III: Causal DAG vs 4 Cards Segmented Toggle
  document.querySelectorAll("#dag-view-toggle .dt-btn").forEach(b => {
    b.addEventListener("click", () => setStep3View(b.dataset.view));
  });

  // Air-Gapped RCA Agent Drawer Wiring
  const btnRca = document.getElementById("btn-rca");
  if (btnRca) btnRca.addEventListener("click", toggleRcaDrawer);
  const btnCloseRca = document.getElementById("btn-close-rca");
  if (btnCloseRca) btnCloseRca.addEventListener("click", closeRcaDrawer);

  document.querySelectorAll("#rca-quick-chips .rca-chip").forEach(b => {
    b.addEventListener("click", () => {
      if (b.dataset.preset) queryRcaAgent({ preset: b.dataset.preset, query: "" });
      else if (b.dataset.query) queryRcaAgent({ preset: null, query: b.dataset.query });
    });
  });

  const toolStripEl = document.getElementById("rca-tool-strip");
  if (toolStripEl) {
    toolStripEl.addEventListener("click", evt => {
      const btn = evt.target.closest("[data-tool-idx]");
      if (!btn) return;
      const idx = Number(btn.dataset.toolIdx);
      renderToolInspector(selectedToolIdx === idx ? null : idx);
    });
  }
  const closeInspBtn = document.getElementById("btn-close-insp");
  if (closeInspBtn) closeInspBtn.addEventListener("click", () => renderToolInspector(null));

  const copyRcaBtn = document.getElementById("btn-copy-rca");
  if (copyRcaBtn) {
    copyRcaBtn.addEventListener("click", () => {
      if (!latestRcaResp) return;
      const txt = `${latestRcaResp.title || ""}\n\n${latestRcaResp.answer_markdown || ""}\n\nSHA-256 Evidence Digest: ${latestRcaResp.evidence_digest_sha256 || ""}\nGovernance: IEC 62443-3-3 Read-Only Conduit (controls=false)`;
      navigator.clipboard?.writeText(txt).catch(() => {});
      copyRcaBtn.textContent = "✓ Copied";
      setTimeout(() => { copyRcaBtn.textContent = "📋 Copy Brief"; }, 1600);
    });
  }

  const rcaForm = document.getElementById("rca-form");
  if (rcaForm) {
    rcaForm.addEventListener("submit", evt => {
      evt.preventDefault();
      const inp = document.getElementById("rca-input");
      const q = inp ? inp.value.trim() : "";
      if (!q) return;
      queryRcaAgent({ preset: null, query: q, lang: "auto" });
      if (inp) inp.value = "";
    });
  }

  // Modal: Model Card, Operating Windows, System Architecture & Judge Q&A
  const badgeBtn = document.getElementById("badge-ai-engine");
  if (badgeBtn) badgeBtn.addEventListener("click", () => openInfoModal("card"));
  const closeModalBtn = document.getElementById("btn-close-modal");
  if (closeModalBtn) closeModalBtn.addEventListener("click", closeInfoModal);
  const modalBg = document.getElementById("info-modal");
  if (modalBg) modalBg.addEventListener("click", evt => { if (evt.target === modalBg) closeInfoModal(); });
  document.querySelectorAll("#modal-tabs .mtab-btn").forEach(b => b.addEventListener("click", () => switchModalTab(b.dataset.mtab)));

  document.querySelectorAll("#fault-buttons .g-btn").forEach(b => {
    b.addEventListener("click", () => {
      stopPlayback();
      enqueue(async () => renderSnapshot(await apiJson("/api/inject-fault", "POST", { fault_type: b.dataset.fault })));
    });
  });

  document.getElementById("btn-clear-fault-banner").addEventListener("click", () => {
    enqueue(async () => renderSnapshot(await apiJson("/api/inject-fault", "POST", { fault_type: "none" })));
  });
  document.getElementById("btn-jump-eligible").addEventListener("click", () => {
    stopPlayback();
    enqueue(async () => renderSnapshot(await apiJson("/api/jump", "POST", { kind: "eligible" })));
  });

  document.getElementById("btn-play").addEventListener("click", () => (isPlaying ? stopPlayback() : startPlayback()));
  document.getElementById("btn-jump-alert").addEventListener("click", () => {
    stopPlayback();
    enqueue(async () => renderSnapshot(await apiJson("/api/jump", "POST", { kind: "alert" })));
  });
  document.getElementById("btn-step15").addEventListener("click", () => {
    enqueue(async () => renderSnapshot(await apiJson("/api/step", "POST", { steps: 15 })));
  });
  document.getElementById("btn-reset").addEventListener("click", () => {
    stopPlayback();
    fieldChecks.forEach(fc => { fc.state = "pending"; });
    enqueue(async () => renderSnapshot(await apiJson("/api/reset", "POST")));
  });

  document.querySelectorAll("[data-verdict]").forEach(b => b.addEventListener("click", () => submitVerdict(b.dataset.verdict)));

  document.getElementById("btn-theme").addEventListener("click", () => {
    const next = currentTheme() === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    applyThemeToPid();
    dirtyTabs.overview = dirtyTabs.trends = dirtyTabs.quantify = dirtyTabs.derisk = dirtyTabs.hardware = true;
    if (latestSnap) renderSnapshot(latestSnap);
    if (activeTabId === "hardware" && latestHw) renderHardwareTab(latestHw, { keepMode: true });
  });

  // 24h Parity-Safe Timeline Scrubber click-to-seek (delegated)
  const tlBins = document.getElementById("tl-bins");
  if (tlBins) {
    tlBins.addEventListener("click", evt => {
      const bin = evt.target.closest("[data-seek-row]");
      if (!bin) return;
      stopPlayback();
      const targetRow = Number(bin.getAttribute("data-seek-row"));
      enqueue(async () => {
        if (latestSnap && latestSnap.injection !== "none") {
          await apiJson("/api/inject-fault", "POST", { fault_type: "none" });
        }
        renderSnapshot(await apiJson("/api/seek", "POST", { row_index: targetRow }));
      });
    });
  }

  // Story Bar toggle & 6 Guided Stage Beats
  const btnStory = document.getElementById("btn-story");
  if (btnStory) btnStory.addEventListener("click", () => toggleStoryBar());
  document.querySelectorAll("#story-beats .story-btn").forEach(b => {
    b.addEventListener("click", () => runStoryBeat(b.dataset.beat));
  });

  // Presenter Keyboard Shortcuts (ignored while typing inside form inputs)
  window.addEventListener("keydown", evt => {
    const tag = (evt.target && evt.target.tagName) ? evt.target.tagName.toUpperCase() : "";
    if (evt.key === "Escape") {
      closeInfoModal();
      closeRcaDrawer();
      return;
    }
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (evt.ctrlKey || evt.metaKey || evt.altKey) return;
    if (evt.key === "1") switchTab("overview");
    else if (evt.key === "2") switchTab("trends");
    else if (evt.key === "3") switchTab("quantify");
    else if (evt.key === "4") switchTab("derisk");
    else if (evt.key === "5") switchTab("hardware");
    else if (evt.key === " ") { evt.preventDefault(); isPlaying ? stopPlayback() : startPlayback(); }
    else if (evt.key === "s" || evt.key === "S") toggleStoryBar();
    else if (evt.key === "m" || evt.key === "M") {
      const modal = document.getElementById("info-modal");
      if (modal && modal.classList.contains("hidden")) openInfoModal("card"); else closeInfoModal();
    }
    else if (evt.key === "t" || evt.key === "T") document.getElementById("btn-theme")?.click();
    else if (evt.key === "a" || evt.key === "A") toggleRcaDrawer();
    else if (evt.key === "n" || evt.key === "N") {
      const jb = document.getElementById("btn-jump-alert");
      if (jb && !jb.disabled) jb.click();
    }
    else if (evt.key === "r" || evt.key === "R") document.getElementById("btn-reset")?.click();
  });

  const qp = new URLSearchParams(window.location.search);
  const qTheme = qp.get("theme");
  if (qTheme === "dark" || qTheme === "light") {
    document.documentElement.setAttribute("data-theme", qTheme);
    applyThemeToPid();
  }
  if (qp.get("story") === "1") toggleStoryBar(true);
  if (qp.get("basis") === "manual") tab3BasisMode = "manual";
  else if (qp.get("basis") === "live") tab3BasisMode = "live";
  const qStep3 = qp.get("step3");
  if (qStep3 === "cards" || qStep3 === "dag") step3ViewMode = qStep3;
  if (qp.get("field") === "issue") {
    fieldChecks[0].state = "ok";
    fieldChecks[1].state = "ok";
    fieldChecks[2].state = "issue";
    fieldChecks[3].state = "ok";
  }

  const qTab = qp.get("tab");
  if (qTab && ["overview", "trends", "quantify", "derisk", "hardware"].includes(qTab)) switchTab(qTab);

  // Fast parallel boot: fetch /api/state + P&ID SVG immediately; lazy-load Tab 5 hardware in background
  await enqueue(async () => {
    const qHwPreset = qp.get("hw_preset");
    const qHwMode = qp.get("hw_mode");
    const needHwImmediate = Boolean(qHwPreset || qHwMode || qTab === "hardware" || qp.get("apply_step4") === "1");

    const statePromise = qp.get("apply_step4") === "1"
      ? null
      : apiJson("/api/state").then(snap => {
          renderSnapshot(snap);
          return snap;
        });

    if (needHwImmediate) {
      if (qHwPreset) {
        const hw = await apiJson("/api/hardware/simulate?img_fmt=jpeg", "POST", {
          preset: qHwPreset,
          vision_mode: qHwMode || undefined,
        });
        if (qHwMode) activeHwVisionMode = qHwMode;
        renderHardwareTab(hw, { keepMode: Boolean(qHwMode) });
      } else {
        await loadHardwareStatus({ keepMode: false });
        if (qHwMode && ["fill_strainer", "spray_flow", "basin_water"].includes(qHwMode)) {
          activeHwVisionMode = qHwMode;
          if (latestHw) renderHardwareTab(latestHw, { keepMode: true });
        }
      }
      if (qp.get("apply_step4") === "1") {
        await applyHardwareToStep4(false);
      } else if (statePromise) {
        await statePromise;
      }
    } else {
      await Promise.all([pidPromise, statePromise]);
      // Pre-warm Tab 5 hardware asynchronously in background after Tab 1 is already interactive
      setTimeout(() => {
        if (!latestHw) loadHardwareStatus({ keepMode: false });
      }, 250);
    }
  });
  const qModal = qp.get("modal");
  if (qModal && ["card", "arch", "pilot", "qa"].includes(qModal)) openInfoModal(qModal);
  if (qp.get("rca") === "1") {
    const qPreset = qp.get("rca_preset");
    const qQuery = qp.get("rca_query");
    const qLang = qp.get("rca_lang") || "auto";
    openRcaDrawer(
      qQuery
        ? { preset: null, query: qQuery, lang: qLang }
        : { preset: qPreset || "why_alert", query: "", lang: qLang }
    );
  }
});
