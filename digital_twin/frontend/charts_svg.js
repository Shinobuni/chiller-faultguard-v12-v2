/* Chiller FaultGuard v11.0 — High-DPI Offline Vector SVG Charts, Causal DAG & Acoustic FFT (Zero CDN Required!) */

function escSvg(val) {
  return String(val ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function clipTxt(str, maxLen) {
  const chars = Array.from(String(str ?? ""));
  return chars.length > maxLen ? `${chars.slice(0, maxLen - 1).join("")}…` : chars.join("");
}

function fmtNetShort(v) {
  const n = Number(v || 0);
  const sign = n > 0 ? "+" : n < 0 ? "−" : "";
  const abs = Math.abs(n);
  if (abs >= 1e6) return `${sign}${(abs / 1e6).toFixed(1)}M ₫`;
  if (abs >= 1e3) return `${sign}${Math.round(abs / 1e3)}k ₫`;
  return `${sign}${Math.round(abs)} ₫`;
}

function svgEl(tag, attrs = {}, inner = "") {
  const a = Object.entries(attrs).map(([k, v]) => `${k}="${v}"`).join(" ");
  return `<${tag} ${a}>${inner}</${tag}>`;
}

function chartPalette() {
  const dark = (document.documentElement.getAttribute("data-theme") || "light") === "dark";
  return {
    dark,
    txt: dark ? "#E2E8F0" : "#1E293B",
    muted: dark ? "#94A3B8" : "#64748B",
    grid: dark ? "#243352" : "#E2E8F0",
    track: dark ? "#1E293B" : "#E2E8F0",
    bg: dark ? "#131C31" : "#FFFFFF",
    blue: dark ? "#38BDF8" : "#0B63B6",
    ok: dark ? "#34D399" : "#059669",
    warn: dark ? "#FBBF24" : "#D97706",
    fault: dark ? "#F87171" : "#DC2626",
    purple: dark ? "#A78BFA" : "#6D28D9",
  };
}

/* 1. Radial High-Fan Reference Rule Gauge (0 -> 100%, Threshold 98%, Zero Overlap Geometry) */
window.drawFanGaugeSvg = function drawFanGaugeSvg(host, fanPct, thresholdPct, status) {
  if (!host) return;
  const c = chartPalette();
  const isHold = status === "ABSTAIN" || fanPct === null || fanPct === undefined;
  const val = isHold ? 0 : Math.max(0, Math.min(100, Number(fanPct)));
  const cx = 160, cy = 112, r = 74;
  const ang = deg => (deg * Math.PI) / 180;
  const pt = (deg, rad = r) => [cx + rad * Math.cos(ang(deg)), cy + rad * Math.sin(ang(deg))];

  function arcPath(a1, a2) {
    const [x1, y1] = pt(a1), [x2, y2] = pt(a2);
    const large = Math.abs(a2 - a1) > 180 ? 1 : 0;
    return `M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`;
  }

  // Span 165 deg -> 375 deg (210 deg total sweep)
  const aStart = 165, aSpan = 210;
  const valToDeg = v => aStart + (v / 100) * aSpan;
  const needleDeg = valToDeg(val);
  const [nx, ny] = pt(needleDeg, r - 14);
  const [tx, ty] = pt(valToDeg(thresholdPct), r + 9);
  const [tx2, ty2] = pt(valToDeg(thresholdPct), r - 11);
  const valCol = isHold ? c.warn : val > thresholdPct ? c.fault : val >= 85 ? c.warn : c.ok;

  // Outer tick labels cleanly outside arc so needle NEVER overlaps 98% or any other tick
  const tickMeta = [
    { v: 0, dx: -16, dy: 4, col: c.muted, fw: 700 },
    { v: 25, dx: -15, dy: -4, col: c.muted, fw: 700 },
    { v: 50, dx: 0, dy: -16, col: c.muted, fw: 700 },
    { v: 75, dx: 16, dy: -4, col: c.muted, fw: 700 },
    { v: 98, dx: 19, dy: 2, col: c.fault, fw: 900 },
  ];
  const ticks = tickMeta.map(t => {
    const [bx, by] = pt(valToDeg(t.v), r + 8);
    return svgEl(
      "text",
      {
        x: (bx + t.dx).toFixed(1),
        y: (by + t.dy).toFixed(1),
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        fill: t.col,
        "font-size": t.v === 98 ? 10.5 : 10,
        "font-weight": t.fw,
      },
      `${t.v}%`
    );
  }).join("");

  const needleCol = c.dark ? "#F8FAFC" : "#0F172A";
  host.innerHTML = `<svg viewBox="0 0 320 162" style="width:100%;height:100%;display:block">
    <path d="${arcPath(valToDeg(0), valToDeg(100))}" fill="none" stroke="${c.track}" stroke-width="15" stroke-linecap="butt" opacity="0.45"/>
    <path d="${arcPath(valToDeg(0), valToDeg(80))}" fill="none" stroke="${c.ok}" stroke-width="13" stroke-linecap="butt"/>
    <path d="${arcPath(valToDeg(80), valToDeg(98))}" fill="none" stroke="${c.warn}" stroke-width="13" stroke-linecap="butt"/>
    <path d="${arcPath(valToDeg(98), valToDeg(100))}" fill="none" stroke="${c.fault}" stroke-width="13" stroke-linecap="butt"/>
    <line x1="${tx.toFixed(1)}" y1="${ty.toFixed(1)}" x2="${tx2.toFixed(1)}" y2="${ty2.toFixed(1)}" stroke="${c.fault}" stroke-width="2.6"/>
    ${ticks}
    <line x1="${cx}" y1="${cy}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}" stroke="${needleCol}" stroke-width="4.2" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="6.5" fill="${needleCol}"/>
    <circle cx="${cx}" cy="${cy}" r="2.8" fill="${c.bg}"/>
    <text x="${cx}" y="${cy + 34}" text-anchor="middle" fill="${valCol}" font-size="22" font-weight="900" font-family="Consolas,monospace">${isHold ? "ABSTAIN" : val.toFixed(1) + " %"}</text>
  </svg>`;
};

/* 2. Quan HGB Score vs 0.70 Threshold Time-Series Chart */
window.drawScoreTrendSvg = function drawScoreTrendSvg(host, history, threshold = 0.7) {
  if (!host) return;
  const c = chartPalette();
  const W = 760, H = 335, L = 46, R = 22, T = 22, B = 32;
  const pw = W - L - R, ph = H - T - B;
  const rows = history || [];
  const yMap = v => T + (1 - Math.max(0, Math.min(1, v))) * ph;
  const xMap = i => L + (rows.length > 1 ? (i / (rows.length - 1)) * pw : pw * 0.5);

  let grid = "";
  [0, 0.25, 0.5, 0.7, 1.0].forEach(v => {
    const yy = yMap(v);
    grid += `<line x1="${L}" y1="${yy.toFixed(1)}" x2="${W - R}" y2="${yy.toFixed(1)}" stroke="${v === threshold ? c.fault : c.grid}" stroke-width="${v === threshold ? 1.8 : 1}" ${v === threshold ? 'stroke-dasharray="6,4"' : ""}/>`;
    grid += `<text x="${L - 7}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="${v === threshold ? c.fault : c.muted}" font-size="11" font-weight="${v === threshold ? 800 : 600}">${v.toFixed(2)}</text>`;
  });

  const yTh = yMap(threshold);
  const alertZone = `<rect x="${L}" y="${T}" width="${pw}" height="${(yTh - T).toFixed(1)}" fill="${c.dark ? "rgba(248,113,113,0.12)" : "rgba(220,38,38,0.08)"}"/>
    <text x="${W - R - 8}" y="${(yTh - 7).toFixed(1)}" text-anchor="end" fill="${c.fault}" font-size="11" font-weight="800">Alert Zone (&gt; 0.70 for 5m)</text>`;

  let path = "", gap = true, lastPt = null;
  let alertDots = "";
  rows.forEach((r, i) => {
    if (r.score === null || r.score === undefined) { gap = true; return; }
    const x = xMap(i), y = yMap(r.score);
    path += `${gap ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)} `;
    gap = false;
    lastPt = { x, y, v: r.score, alert: r.ai_alert };
    if (r.ai_alert && i % 2 === 0) {
      alertDots += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.8" fill="${c.fault}" opacity="0.85"/>`;
    }
  });

  let xTicks = "";
  if (rows.length > 0) {
    const nTicks = Math.min(6, rows.length);
    for (let k = 0; k < nTicks; k++) {
      const idx = nTicks === 1 ? 0 : Math.round((k / (nTicks - 1)) * (rows.length - 1));
      const x = xMap(idx);
      const lbl = (rows[idx].timestamp || "").slice(11, 16);
      xTicks += `<text x="${x.toFixed(1)}" y="${H - 10}" text-anchor="middle" fill="${c.muted}" font-size="11" font-family="Consolas,monospace">${lbl}</text>`;
    }
  }

  let endMarker = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${c.muted}" font-size="13.5" font-weight="700">No eligible rows in window — click Next Alert</text>`;
  if (lastPt) {
    const pillX = Math.max(L + 4, Math.min(W - R - 58, lastPt.x - 52));
    const pillY = Math.max(T + 4, lastPt.y - 26);
    endMarker = `
      <circle cx="${lastPt.x.toFixed(1)}" cy="${lastPt.y.toFixed(1)}" r="5.5" fill="${lastPt.alert ? c.fault : c.blue}" stroke="${c.bg}" stroke-width="2.2"/>
      <rect x="${pillX.toFixed(1)}" y="${pillY.toFixed(1)}" width="54" height="19" rx="4" fill="${c.bg}" stroke="${lastPt.alert ? c.fault : c.blue}" stroke-width="1.3"/>
      <text x="${(pillX + 27).toFixed(1)}" y="${(pillY + 13).toFixed(1)}" text-anchor="middle" fill="${lastPt.alert ? c.fault : c.blue}" font-size="11" font-weight="900" font-family="Consolas,monospace">${lastPt.v.toFixed(3)}</text>
    `;
  }

  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%;display:block">
    ${alertZone}${grid}${xTicks}
    ${path ? `<path d="${path}" fill="none" stroke="${c.blue}" stroke-width="2.8" stroke-linejoin="round"/>` : ""}
    ${alertDots}${endMarker}
    <line class="ch-crosshair" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="${c.muted}" stroke-width="1.2" stroke-dasharray="3,3" visibility="hidden"/>
    <circle class="ch-dot" cx="${L}" cy="${T}" r="4.5" fill="${c.blue}" stroke="${c.bg}" stroke-width="2" visibility="hidden"/>
  </svg><div class="chart-tip hidden"></div>`;

  attachSeriesTooltip(host, W, L, R, rows, (r, x) => {
    const sc = r.score !== null && r.score !== undefined ? Number(r.score) : null;
    const dotY = sc !== null ? yMap(sc) : null;
    const badge = r.ai_alert ? "FOULING SUSPECT (5/5m)" : (!r.eligible ? "ABSTAIN" : (sc > threshold ? "ABOVE 0.70 (DWELLING)" : "NO ALERT"));
    return {
      dotX: x,
      dotY,
      dotCol: r.ai_alert ? c.fault : c.blue,
      html: `<strong>${(r.timestamp || "").slice(11, 16)} · Oct-01</strong>Score: <b>${sc !== null ? sc.toFixed(3) : "ABSTAIN"}</b> (Gate ${threshold.toFixed(2)})<br/>Fan: ${Number(r.fan_pct || 0).toFixed(1)}% · ${badge}`,
    };
  });
};

function attachSeriesTooltip(host, W, L, R, rows, buildFn) {
  if (!host || !rows || !rows.length) return;
  const svg = host.querySelector("svg");
  const tip = host.querySelector(".chart-tip");
  const line = host.querySelector(".ch-crosshair");
  const dot = host.querySelector(".ch-dot");
  if (!svg || !tip) return;
  const pw = W - L - R;
  svg.addEventListener("mousemove", evt => {
    const rect = svg.getBoundingClientRect();
    if (rect.width <= 0) return;
    const relX = ((evt.clientX - rect.left) / rect.width) * W;
    const frac = Math.max(0, Math.min(1, (relX - L) / pw));
    const idx = Math.round(frac * (rows.length - 1));
    const snapX = L + (rows.length > 1 ? (idx / (rows.length - 1)) * pw : pw * 0.5);
    const info = buildFn(rows[idx], snapX, idx);
    if (!info) return;
    if (line) { line.setAttribute("x1", snapX.toFixed(1)); line.setAttribute("x2", snapX.toFixed(1)); line.setAttribute("visibility", "visible"); }
    if (dot && info.dotY !== null && info.dotY !== undefined) {
      dot.setAttribute("cx", snapX.toFixed(1)); dot.setAttribute("cy", info.dotY.toFixed(1));
      if (info.dotCol) dot.setAttribute("fill", info.dotCol);
      dot.setAttribute("visibility", "visible");
    } else if (dot) {
      dot.setAttribute("visibility", "hidden");
    }
    tip.innerHTML = info.html;
    tip.classList.remove("hidden");
    const hr = host.getBoundingClientRect();
    let tx = evt.clientX - hr.left + 12;
    let ty = evt.clientY - hr.top - 48;
    if (tx + 195 > hr.width) tx = Math.max(6, evt.clientX - hr.left - 205);
    if (ty < 4) ty = evt.clientY - hr.top + 14;
    tip.style.left = `${tx}px`;
    tip.style.top = `${ty}px`;
  });
  svg.addEventListener("mouseleave", () => {
    tip.classList.add("hidden");
    if (line) line.setAttribute("visibility", "hidden");
    if (dot) dot.setAttribute("visibility", "hidden");
  });
}

/* 3. Logged 14-Channel Plant Power vs Matched Normal Timestamp (Diagnostic) */
window.drawPowerTrendSvg = function drawPowerTrendSvg(host, history) {
  if (!host) return;
  const c = chartPalette();
  const W = 760, H = 335, L = 52, R = 22, T = 30, B = 32;
  const pw = W - L - R, ph = H - T - B;
  const rows = history || [];
  const vals = rows.flatMap(r => [r.plant_kw, r.normal_plant_kw]).filter(v => typeof v === "number" && Number.isFinite(v));
  const lo = vals.length ? Math.max(0, Math.floor((Math.min(...vals) - 12) / 10) * 10) : 0;
  const hi = vals.length ? Math.ceil((Math.max(...vals) + 12) / 10) * 10 : 300;
  const span = Math.max(10, hi - lo);
  const yMap = v => T + (1 - (v - lo) / span) * ph;
  const xMap = i => L + (rows.length > 1 ? (i / (rows.length - 1)) * pw : pw * 0.5);

  let grid = "";
  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * span;
    const yy = yMap(v);
    grid += `<line x1="${L}" y1="${yy.toFixed(1)}" x2="${W - R}" y2="${yy.toFixed(1)}" stroke="${c.grid}" stroke-width="1"/>`;
    grid += `<text x="${L - 7}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="${c.muted}" font-size="11">${v.toFixed(0)}</text>`;
  }

  let pAct = "", pNorm = "", areaPts = [];
  rows.forEach((r, i) => {
    const x = xMap(i);
    const ya = yMap(r.plant_kw), yn = yMap(r.normal_plant_kw);
    pAct += `${i === 0 ? "M" : "L"}${x.toFixed(1)},${ya.toFixed(1)} `;
    pNorm += `${i === 0 ? "M" : "L"}${x.toFixed(1)},${yn.toFixed(1)} `;
    areaPts.push([x, ya, yn]);
  });

  let shade = "";
  if (areaPts.length > 1) {
    const top = areaPts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
    const bot = [...areaPts].reverse().map(p => `${p[0].toFixed(1)},${p[2].toFixed(1)}`).join(" ");
    shade = `<polygon points="${top} ${bot}" fill="${c.dark ? "rgba(251,146,60,0.22)" : "rgba(234,88,12,0.16)"}"/>`;
  }

  let xTicks = "";
  if (rows.length > 0) {
    const nTicks = Math.min(6, rows.length);
    for (let k = 0; k < nTicks; k++) {
      const idx = nTicks === 1 ? 0 : Math.round((k / (nTicks - 1)) * (rows.length - 1));
      xTicks += `<text x="${xMap(idx).toFixed(1)}" y="${H - 10}" text-anchor="middle" fill="${c.muted}" font-size="11" font-family="Consolas,monospace">${(rows[idx].timestamp || "").slice(11, 16)}</text>`;
    }
  }

  let deltaTag = "";
  if (rows.length > 0) {
    const last = rows[rows.length - 1];
    const diff = Number(last.plant_kw) - Number(last.normal_plant_kw);
    const sign = diff >= 0 ? "+" : "";
    deltaTag = `<text x="${W - R}" y="17" text-anchor="end" fill="${diff > 0.2 ? "#EA580C" : c.ok}" font-size="11.5" font-weight="800" font-family="Consolas,monospace">Latest Δ: ${sign}${diff.toFixed(1)} kW*</text>`;
  }

  const legend = `
    <line x1="${L + 4}" y1="13" x2="${L + 22}" y2="13" stroke="#EA580C" stroke-width="2.8"/>
    <text x="${L + 28}" y="17" fill="${c.txt}" font-size="11.5" font-weight="700">Active 14-Ch (kW)</text>
    <line x1="${L + 160}" y1="13" x2="${L + 178}" y2="13" stroke="${c.blue}" stroke-width="2.2" stroke-dasharray="5,3"/>
    <text x="${L + 184}" y="17" fill="${c.txt}" font-size="11.5" font-weight="700">Normal Matched (Diagnostic)</text>
    ${deltaTag}
  `;

  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%;display:block">
    ${legend}${grid}${xTicks}${shade}
    ${pNorm ? `<path d="${pNorm}" fill="none" stroke="${c.blue}" stroke-width="2.2" stroke-dasharray="5,3"/>` : ""}
    ${pAct ? `<path d="${pAct}" fill="none" stroke="#EA580C" stroke-width="2.8" stroke-linejoin="round"/>` : ""}
    <line class="ch-crosshair" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="${c.muted}" stroke-width="1.2" stroke-dasharray="3,3" visibility="hidden"/>
    <circle class="ch-dot" cx="${L}" cy="${T}" r="4.5" fill="#EA580C" stroke="${c.bg}" stroke-width="2" visibility="hidden"/>
  </svg><div class="chart-tip hidden"></div>`;

  attachSeriesTooltip(host, W, L, R, rows, (r, x) => {
    const act = Number(r.plant_kw || 0), norm = Number(r.normal_plant_kw || 0), diff = act - norm;
    return {
      dotX: x,
      dotY: yMap(act),
      dotCol: "#EA580C",
      html: `<strong>${(r.timestamp || "").slice(11, 16)} · 14-Ch Logged</strong>Active: <b>${act.toFixed(1)} kW</b> · Normal: ${norm.toFixed(1)} kW<br/>Signed Δ*: <b>${diff >= 0 ? "+" : ""}${diff.toFixed(2)} kW</b>`,
    };
  });
};

/* 4. Oct-Dec Exploratory Held-Out Benchmark Horizontal Bars (Clean Right F1 Column — Zero Track Overlap) */
window.drawBenchmarkBarsSvg = function drawBenchmarkBarsSvg(host, metrics) {
  if (!host || !metrics) return;
  const c = chartPalette();
  const items = [
    { name: "Quan HGB (Selected)", f1: metrics.AI.f1, prec: metrics.AI.precision, rec: metrics.AI.recall, far: metrics.AI.false_alarm_episodes_per_normal_hour, col: c.ok },
    { name: "Logistic Baseline", f1: metrics.logistic_reference.f1, prec: metrics.logistic_reference.precision, rec: metrics.logistic_reference.recall, far: metrics.logistic_reference.false_alarm_episodes_per_normal_hour, col: c.blue },
    { name: "High-Fan Rule (>98%)", f1: metrics.rule.f1, prec: metrics.rule.precision, rec: metrics.rule.recall, far: metrics.rule.false_alarm_episodes_per_normal_hour, col: c.muted },
  ];
  const W = 520, H = 228, trackX = 8, trackW = 396;
  let body = "";
  items.forEach((it, idx) => {
    const y = 16 + idx * 70;
    const barW = Math.max(8, Math.round((it.f1 / 0.50) * trackW));
    body += `
      <text x="${trackX}" y="${y + 13}" fill="${c.txt}" font-size="12.5" font-weight="800">${it.name}</text>
      <text x="${W - 8}" y="${y + 13}" text-anchor="end" fill="${c.muted}" font-size="10.8" font-family="Consolas,monospace">Prec ${(it.prec * 100).toFixed(1)}% · Rec ${(it.rec * 100).toFixed(1)}% · FAR ${it.far.toFixed(3)}/h</text>
      <rect x="${trackX}" y="${y + 23}" width="${trackW}" height="20" rx="6" fill="${c.track}" opacity="0.72"/>
      <rect x="${trackX}" y="${y + 23}" width="${Math.min(trackW, barW)}" height="20" rx="6" fill="${it.col}"/>
      <text x="${W - 8}" y="${y + 38}" text-anchor="end" fill="${it.col}" font-size="14" font-weight="900" font-family="Consolas,monospace">F1 ${it.f1.toFixed(3)}</text>
    `;
  });
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%;display:block">${body}</svg>`;
};

/* 5. Live-Alert Auto OR Illustrative Earlier-PM Counterfactual Area Chart */
window.drawCounterfactualSvg = function drawCounterfactualSvg(host, sim) {
  if (!host || !sim) return;
  const c = chartPalette();
  const W = 760, H = 390, L = 58, R = 24, T = 36, B = 32;
  const pw = W - L - R, ph = H - T - B;
  const existD = sim.existing_pm_day || 120;
  const earlyD = sim.earlier_pm_day || 28;
  const exKw = Number(sim.imposed_excess_kw || 0);
  const hrs = sim.hours_per_day || 24.0;

  // Live Standby / ABSTAIN / Zero-Penalty state: flat 0 kWh baseline instead of fake red triangle!
  if (sim.is_live_mode && (!sim.active || exKw <= 0)) {
    const xMap = d => L + (d / existD) * pw;
    const y0 = T + ph;
    const stepD = existD <= 120 ? 20 : 30;
    let grid = "", xTicks = "";
    for (let k = 0; k <= 4; k++) {
      const yy = T + (1 - k / 4) * ph;
      grid += `<line x1="${L}" y1="${yy.toFixed(1)}" x2="${W - R}" y2="${yy.toFixed(1)}" stroke="${c.grid}" stroke-width="1"/>`;
      grid += `<text x="${L - 8}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="${c.muted}" font-size="11">${(k * 2).toFixed(1)}k</text>`;
    }
    for (let d = 0; d <= existD; d += stepD) {
      xTicks += `<text x="${xMap(d).toFixed(1)}" y="${H - 10}" text-anchor="middle" fill="${c.muted}" font-size="11">D${d}</text>`;
    }
    const isAbstain = sim.mode === "SUSPENDED_ABSTAIN";
    const isZeroAlert = sim.mode === "ALERT_ZERO_EXCESS";
    const accent = isAbstain ? c.warn : (isZeroAlert ? c.warn : c.ok);
    const title = isAbstain
      ? "🔒 SUSPENDED (ABSTAIN) — NO ENERGY PROJECTION"
      : isZeroAlert
        ? "⚠ ALERT ACTIVE · 0.00 kW MEASURED ΔkW* (POSSIBLE FALSE ALARM)"
        : "✓ NO ACTIVE ALERT — 0 kWh AVOIDABLE EXCESS";
    const sub1 = isAbstain
      ? "Step I gate active — restore clean telemetry first."
      : isZeroAlert
        ? `0.00 kW positive penalty @ ${(sim.trigger_timestamp || "").slice(11, 16)} (ΔkW*: ${Number(sim.instant_delta_kw || 0).toFixed(2)} kW).`
        : `Normal @ ${(sim.trigger_timestamp || "").slice(11, 16)} (ΔkW*: ${Number(sim.instant_delta_kw || 0).toFixed(2)} kW) — no earlier PM needed.`;
    const sub2 = "Switch to 0.80 / 0.65 UA or click 3 kW (Quan Ref).";

    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%;display:block">
      <line x1="${L + 4}" y1="15" x2="${L + 24}" y2="15" stroke="${accent}" stroke-width="3"/>
      <text x="${L + 30}" y="19" fill="${c.txt}" font-size="12" font-weight="700">Live Auto-Projection: 0 kWh (Standby)</text>
      ${grid}${xTicks}
      <line x1="${L}" y1="${y0.toFixed(1)}" x2="${W - R}" y2="${y0.toFixed(1)}" stroke="${accent}" stroke-width="3.5"/>
      <rect x="${L + 34}" y="${T + 86}" width="${pw - 68}" height="122" rx="12" fill="${c.bg}" stroke="${accent}" stroke-width="2" stroke-dasharray="6,4"/>
      <text x="${L + pw / 2}" y="${T + 122}" text-anchor="middle" fill="${accent}" font-size="15" font-weight="900">${title}</text>
      <text x="${L + pw / 2}" y="${T + 150}" text-anchor="middle" fill="${c.txt}" font-size="11.8" font-weight="700">${sub1}</text>
      <text x="${L + pw / 2}" y="${T + 178}" text-anchor="middle" fill="${c.muted}" font-size="11" font-weight="600">${sub2}</text>
    </svg><div class="chart-tip hidden"></div>`;
    return;
  }

  const maxKwh = Math.max(1000, Math.ceil((sim.existing_policy_excess_kwh * 1.14) / 1000) * 1000);

  const xMap = d => L + (d / existD) * pw;
  const yMap = v => T + (1 - v / maxKwh) * ph;

  let grid = "";
  for (let k = 0; k <= 4; k++) {
    const v = (k / 4) * maxKwh;
    const yy = yMap(v);
    grid += `<line x1="${L}" y1="${yy.toFixed(1)}" x2="${W - R}" y2="${yy.toFixed(1)}" stroke="${c.grid}" stroke-width="1"/>`;
    grid += `<text x="${L - 8}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="${c.muted}" font-size="11">${(v / 1000).toFixed(1)}k</text>`;
  }

  let xTicks = "";
  const stepD = existD <= 120 ? 20 : 30;
  for (let d = 0; d <= existD; d += stepD) {
    xTicks += `<text x="${xMap(d).toFixed(1)}" y="${H - 10}" text-anchor="middle" fill="${c.muted}" font-size="11">D${d}</text>`;
  }

  const x0 = xMap(0), y0 = yMap(0);
  const preCleanKwh = exKw * hrs * earlyD;
  const xEarly = xMap(earlyD), yEarly = yMap(preCleanKwh);
  const xExist = xMap(existD), yExist = yMap(sim.existing_policy_excess_kwh);
  const yEarlyEnd = yMap(sim.earlier_policy_excess_kwh);

  const avoidedPoly = `${xEarly.toFixed(1)},${yEarly.toFixed(1)} ${xExist.toFixed(1)},${yExist.toFixed(1)} ${xExist.toFixed(1)},${yEarlyEnd.toFixed(1)}`;
  const earlyPoly = `${x0.toFixed(1)},${y0.toFixed(1)} ${xEarly.toFixed(1)},${yEarly.toFixed(1)} ${xExist.toFixed(1)},${yEarlyEnd.toFixed(1)} ${xExist.toFixed(1)},${y0.toFixed(1)}`;

  // Smart flip callout box to the LEFT when earlyD >= 42% of calendar horizon so it never collides with red diagonal
  const boxW = sim.is_live_mode ? 248 : 172;
  const flipLeft = earlyD >= existD * 0.42;
  const boxX = flipLeft ? Math.max(L + 6, xEarly - boxW - 10) : Math.min(W - R - boxW - 6, xEarly + 10);
  const boxY = T + 12;
  const boxTxt = sim.is_live_mode
    ? `⚡ Live @ ${(sim.trigger_timestamp || "").slice(11, 16)} (+${exKw.toFixed(2)} kW*) → Clean D${earlyD}`
    : `Alert D${sim.alert_day} → Clean D${earlyD}`;

  // Smart center avoided kWh callout cleanly inside the red triangle (zero collision with red diagonal at any setting)
  const fracX = 0.64;
  const lblX = Math.min(W - R - 110, Math.max(xEarly + 85, xEarly + (xExist - xEarly) * fracX));
  const tAlong = (lblX - xEarly) / Math.max(1, xExist - xEarly);
  const yTopAtLbl = yEarly + (yExist - yEarly) * tAlong;
  const yBotAtLbl = yEarly + (yEarlyEnd - yEarly) * tAlong;
  const lblY = Math.max(yTopAtLbl + 22, Math.min(yBotAtLbl - 26, (yTopAtLbl + yBotAtLbl) * 0.52));

  const legend = `
    <line x1="${L + 4}" y1="15" x2="${L + 24}" y2="15" stroke="${c.fault}" stroke-width="2.8" stroke-dasharray="6,4"/>
    <text x="${L + 30}" y="19" fill="${c.txt}" font-size="12" font-weight="700">Calendar PM Day ${existD} (${Number(sim.existing_policy_excess_kwh).toLocaleString()} kWh)</text>
    <line x1="${L + 340}" y1="15" x2="${L + 360}" y2="15" stroke="${c.ok}" stroke-width="3.2"/>
    <text x="${L + 366}" y="19" fill="${c.txt}" font-size="12" font-weight="700">Earlier PM Day ${earlyD} (${Number(sim.earlier_policy_excess_kwh).toLocaleString()} kWh)</text>
  `;

  // Exact red-wedge blended halo + smart gap break so dotted payback line NEVER touches -kWh headline or subtitle
  const pm = sim.pm_optimizer || null;
  const rest = Number(sim.restored_fraction || 1.0);
  const wedgeHalo = c.dark ? "#3C2B3D" : "#FAE1E1";
  const avoidedSubTxt = `(${sim.is_live_mode ? "Live-Alert Auto" : "Illustrative"}: ${exKw.toFixed(sim.is_live_mode ? 2 : 0)} kW${sim.is_live_mode ? "*" : ""} × ${hrs}h × ${sim.days_saved}d × ${Math.round(rest * 100)}%)`;
  const textHalfW = Math.max(85, Math.round(avoidedSubTxt.length * 3.15 + 10));
  const gapTop = lblY - 19;
  const gapBot = lblY + 24;

  // EVN TOU Break-Even Payback / Defer PM marker (placed cleanly below the Earlier-PM green line so ZERO overlap with Clean box or red -kWh label!)
  let paybackLineSvg = "";
  let paybackPillSvg = "";
  if (pm) {
    const pbDay = pm.payback_calendar_day !== null && pm.payback_calendar_day !== undefined
      ? Number(pm.payback_calendar_day)
      : NaN;
    if (pm.verdict === "DEFER_PM_NEGATIVE_ROI") {
      const deferTxt = `⚠ Defer PM (Payback > D${existD})`;
      const pillW = 204, pillH = 22;
      const pillX = Math.max(L + 8, Math.min(W - R - pillW - 4, (xEarly + xExist) / 2 - pillW / 2));
      const pillY = y0 - 27;
      const pillBg = c.dark ? "rgba(245,158,11,0.18)" : "#FFFBEB";
      paybackPillSvg = `
        <rect x="${pillX.toFixed(1)}" y="${pillY.toFixed(1)}" width="${pillW}" height="${pillH}" rx="6" fill="${pillBg}" stroke="${c.warn}" stroke-width="1.5"/>
        <text x="${(pillX + pillW / 2).toFixed(1)}" y="${(pillY + 14.5).toFixed(1)}" text-anchor="middle" fill="${c.warn}" font-size="10.5" font-weight="900">${escSvg(deferTxt)}</text>
      `;
    } else if (Number.isFinite(pbDay) && pbDay > earlyD && pbDay <= existD) {
      const xPb = xMap(pbDay);
      const calAtPb = exKw * hrs * pbDay;
      const earAtPb = exKw * hrs * earlyD + exKw * hrs * (pbDay - earlyD) * (1 - rest);
      const yCalAtPb = yMap(calAtPb);
      const yEarAtPb = yMap(earAtPb);
      const pbCol = pm.verdict === "CLEAN_NOW_POSITIVE_ROI" ? c.ok : c.warn;
      const pbBg = pm.verdict === "CLEAN_NOW_POSITIVE_ROI"
        ? (c.dark ? "rgba(16,185,129,0.18)" : "#ECFDF5")
        : (c.dark ? "rgba(245,158,11,0.18)" : "#FFFBEB");
      const pbTxt = `★ Payback D${Math.round(pbDay)} (${fmtNetShort(pm.net_benefit_vnd)} Net)`;
      const pillW = Math.max(184, Math.min(226, Math.round(pbTxt.length * 6.4 + 20)));
      const pillH = 22;
      const pillX = Math.max(L + 8, Math.min(W - R - pillW - 4, xPb - pillW / 2));
      const pillY = y0 - 27;

      const hitsTextX = Math.abs(xPb - lblX) <= textHalfW;
      let upperSegs = "";
      if (!hitsTextX) {
        upperSegs = `<line x1="${xPb.toFixed(1)}" y1="${yCalAtPb.toFixed(1)}" x2="${xPb.toFixed(1)}" y2="${yEarAtPb.toFixed(1)}" stroke="${pbCol}" stroke-width="1.8" stroke-dasharray="4,3" stroke-opacity="0.82"/>`;
      } else {
        if (yCalAtPb < gapTop - 4) {
          upperSegs += `<line x1="${xPb.toFixed(1)}" y1="${yCalAtPb.toFixed(1)}" x2="${xPb.toFixed(1)}" y2="${Math.min(yEarAtPb, gapTop).toFixed(1)}" stroke="${pbCol}" stroke-width="1.8" stroke-dasharray="4,3" stroke-opacity="0.82"/>`;
        }
        if (yEarAtPb > gapBot + 4) {
          upperSegs += `<line x1="${xPb.toFixed(1)}" y1="${Math.max(yCalAtPb, gapBot).toFixed(1)}" x2="${xPb.toFixed(1)}" y2="${yEarAtPb.toFixed(1)}" stroke="${pbCol}" stroke-width="1.8" stroke-dasharray="4,3" stroke-opacity="0.82"/>`;
        }
      }

      paybackLineSvg = `
        ${upperSegs}
        <line x1="${xPb.toFixed(1)}" y1="${yEarAtPb.toFixed(1)}" x2="${xPb.toFixed(1)}" y2="${y0.toFixed(1)}" stroke="${pbCol}" stroke-width="2" stroke-dasharray="4,3"/>
        <circle cx="${xPb.toFixed(1)}" cy="${yEarAtPb.toFixed(1)}" r="4.8" fill="${pbCol}" stroke="${c.bg}" stroke-width="2"/>
      `;
      paybackPillSvg = `
        <rect x="${pillX.toFixed(1)}" y="${pillY.toFixed(1)}" width="${pillW}" height="${pillH}" rx="6" fill="${pbBg}" stroke="${pbCol}" stroke-width="1.5"/>
        <text x="${(pillX + pillW / 2).toFixed(1)}" y="${(pillY + 14.5).toFixed(1)}" text-anchor="middle" fill="${pbCol}" font-size="10.5" font-weight="900">${escSvg(pbTxt)}</text>
      `;
    }
  }

  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%;display:block">
    ${legend}${grid}${xTicks}
    <polygon points="${earlyPoly}" fill="${c.dark ? "rgba(52,211,153,0.18)" : "rgba(5,150,105,0.16)"}"/>
    <polygon points="${avoidedPoly}" fill="${c.dark ? "rgba(248,113,113,0.18)" : "rgba(220,38,38,0.14)"}"/>
    <line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${xExist.toFixed(1)}" y2="${yExist.toFixed(1)}" stroke="${c.fault}" stroke-width="3" stroke-dasharray="7,5"/>
    <polyline points="${x0.toFixed(1)},${y0.toFixed(1)} ${xEarly.toFixed(1)},${yEarly.toFixed(1)} ${xExist.toFixed(1)},${yEarlyEnd.toFixed(1)}" fill="none" stroke="${c.ok}" stroke-width="3.4"/>
    ${paybackLineSvg}
    <circle cx="${xEarly.toFixed(1)}" cy="${yEarly.toFixed(1)}" r="4.8" fill="${c.blue}" stroke="${c.bg}" stroke-width="2"/>
    <line x1="${xEarly.toFixed(1)}" y1="${T}" x2="${xEarly.toFixed(1)}" y2="${y0.toFixed(1)}" stroke="${c.blue}" stroke-width="2" stroke-dasharray="3,3"/>
    <rect x="${boxX.toFixed(1)}" y="${boxY.toFixed(1)}" width="${boxW}" height="26" rx="6" fill="${c.bg}" stroke="${c.blue}" stroke-width="1.4"/>
    <text x="${(boxX + boxW / 2).toFixed(1)}" y="${(boxY + 17).toFixed(1)}" text-anchor="middle" fill="${c.blue}" font-size="11" font-weight="800">${escSvg(boxTxt)}</text>
    <text x="${lblX.toFixed(1)}" y="${lblY.toFixed(1)}" text-anchor="middle" fill="${c.fault}" font-size="16" font-weight="900" paint-order="stroke" stroke="${wedgeHalo}" stroke-width="5" stroke-linejoin="round">−${Number(sim.scenario_avoided_excess_kwh).toLocaleString()} kWh</text>
    <text x="${lblX.toFixed(1)}" y="${(lblY + 18).toFixed(1)}" text-anchor="middle" fill="${c.muted}" font-size="11" font-weight="700" paint-order="stroke" stroke="${wedgeHalo}" stroke-width="4.5" stroke-linejoin="round">${escSvg(avoidedSubTxt)}</text>
    ${paybackPillSvg}
    <line class="ch-crosshair" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="${c.muted}" stroke-width="1.2" stroke-dasharray="3,3" visibility="hidden"/>
    <circle class="ch-dot" cx="${L}" cy="${T}" r="4.5" fill="${c.fault}" stroke="${c.bg}" stroke-width="2" visibility="hidden"/>
  </svg><div class="chart-tip hidden"></div>`;

  const daysArr = Array.from({ length: existD + 1 }, (_, d) => d);
  const tariff = (sim.kpis_5m1e && sim.kpis_5m1e.money && sim.kpis_5m1e.money.tariff_vnd_kwh) || 1925.93;
  const pmCost = pm && pm.pm_cost_vnd ? Number(pm.pm_cost_vnd) : null;
  attachSeriesTooltip(host, W, L, R, daysArr, (d, x) => {
    const calKwh = exKw * hrs * d;
    const earKwh = d <= earlyD ? calKwh : exKw * hrs * earlyD + exKw * hrs * (d - earlyD) * (1 - rest);
    const saved = Math.max(0, calKwh - earKwh);
    const savedVnd = saved * tariff;
    const pmLine = pmCost !== null
      ? `<br/>EVN Saved: <b>${fmtNetShort(savedVnd)}</b> · Net vs PM: <b>${fmtNetShort(savedVnd - pmCost)}</b>`
      : "";
    return {
      dotX: x,
      dotY: yMap(calKwh),
      dotCol: c.fault,
      html: `<strong>Day ${d} (${sim.is_live_mode ? "Auto from Live Alert" : "Illustrative Template"})</strong>Calendar PM: ${Math.round(calKwh).toLocaleString()} kWh · Earlier: ${Math.round(earKwh).toLocaleString()} kWh<br/>Avoided: <b>${Math.round(saved).toLocaleString()} kWh</b>${pmLine}`,
    };
  });
};

/* 6. Live 14-Channel Logged LBNL Power Breakdown Vertical Bars */
window.drawPowerBreakdownSvg = function drawPowerBreakdownSvg(host, sig) {
  if (!host || !sig) return;
  const c = chartPalette();
  const W = 740, H = 285, L = 48, R = 18, T = 26, B = 34;
  const pw = W - L - R, ph = H - T - B;
  const bars = [
    { lbl: "Chillers (3)", val: sig.p_chillers_kw || 0, col: c.blue, desc: "CHL_POW_1 + CHL_POW_2 + CHL_POW_3" },
    { lbl: "CHW Pumps (5)", val: sig.p_chwp_kw || 0, col: "#0EA5E9", desc: "3 Primary + 2 Secondary CHW Pumps" },
    { lbl: "CW Pumps (3)", val: sig.p_cwp_kw || 0, col: c.ok, desc: "CDWL_PM_POW_1..3 Condenser Pumps" },
    { lbl: "CT Fans (3)", val: sig.p_ct_fans_kw || 0, col: c.warn, desc: "CT_POW_1..3 Cooling Tower Fans" },
    { lbl: "TOTAL 14-CH", val: sig.plant_kw || 0, col: "#EA580C", desc: "In-scope sum (AHU/Fab tools excluded)" },
  ];
  const maxV = Math.max(50, Math.ceil((Math.max(...bars.map(b => b.val)) * 1.18) / 25) * 25);
  const yMap = v => T + (1 - v / maxV) * ph;

  let grid = "";
  for (let k = 0; k <= 4; k++) {
    const v = (k / 4) * maxV;
    const yy = yMap(v);
    grid += `<line x1="${L}" y1="${yy.toFixed(1)}" x2="${W - R}" y2="${yy.toFixed(1)}" stroke="${c.grid}" stroke-width="1"/>`;
    grid += `<text x="${L - 7}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="${c.muted}" font-size="10.5">${v.toFixed(0)}</text>`;
  }

  const slot = pw / bars.length;
  const bw = Math.min(76, slot * 0.56);
  let rects = "";
  const totalKw = Math.max(0.1, Number(sig.plant_kw || 0.1));
  bars.forEach((b, i) => {
    const cx = L + slot * (i + 0.5);
    const x = cx - bw / 2;
    const y = yMap(b.val);
    const h = Math.max(3, yMap(0) - y);
    rects += `<rect data-bar-idx="${i}" style="cursor:pointer" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="6" fill="${b.col}"/>`;
    rects += `<text x="${cx.toFixed(1)}" y="${(y - 7).toFixed(1)}" text-anchor="middle" fill="${c.txt}" font-size="12" font-weight="800" font-family="Consolas,monospace">${b.val.toFixed(1)} kW</text>`;
    rects += `<text x="${cx.toFixed(1)}" y="${H - 11}" text-anchor="middle" fill="${c.txt}" font-size="11.5" font-weight="700">${b.lbl}</text>`;
  });

  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%;display:block">${grid}${rects}</svg><div class="chart-tip hidden"></div>`;
  const svg = host.querySelector("svg");
  const tip = host.querySelector(".chart-tip");
  if (svg && tip) {
    svg.addEventListener("mousemove", evt => {
      const bar = evt.target.closest("[data-bar-idx]");
      if (!bar) { tip.classList.add("hidden"); return; }
      const b = bars[Number(bar.getAttribute("data-bar-idx"))];
      const share = ((b.val / totalKw) * 100).toFixed(1);
      tip.innerHTML = `<strong>${b.lbl}: ${b.val.toFixed(1)} kW (${share}%)</strong>${b.desc}`;
      tip.classList.remove("hidden");
      const hr = host.getBoundingClientRect();
      let tx = evt.clientX - hr.left + 12;
      let ty = evt.clientY - hr.top - 42;
      if (tx + 210 > hr.width) tx = Math.max(6, evt.clientX - hr.left - 215);
      if (ty < 4) ty = evt.clientY - hr.top + 14;
      tip.style.left = `${tx}px`;
      tip.style.top = `${ty}px`;
    });
    svg.addEventListener("mouseleave", () => tip.classList.add("hidden"));
  }
};

/* 7. Interactive 8-Node Causal Root-Cause Graph (DAG) for Tab 4 Step III */
window.drawCausalDagSvg = function drawCausalDagSvg(host, causalGraph, onNodeClick) {
  if (!host) return;
  const c = chartPalette();
  const W = 740, H = 186;

  // Reuse a single viewport-fixed floating tooltip on document.body so it NEVER clips inside
  // .dag-host (overflow:hidden) and NEVER covers adjacent DAG columns.
  let tip = document.getElementById("dag-floating-tip");
  if (!tip) {
    tip = document.createElement("div");
    tip.id = "dag-floating-tip";
    tip.className = "dag-floating-tip hidden";
    document.body.appendChild(tip);
  }
  const hideTip = () => tip.classList.add("hidden");
  hideTip();

  if (!causalGraph || !Array.isArray(causalGraph.nodes) || !causalGraph.nodes.length) {
    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;display:block">
      <text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${c.muted}" font-size="12" font-weight="700">Causal Root-Cause DAG awaiting snapshot…</text>
    </svg>`;
    return;
  }

  // Generous horizontal routing gutters (56px & 56px) + full non-truncated titles on all 8 blocks
  const NODE_POS = {
    N1: { col: 0, x: 6,   y: 16,  w: 196, h: 37 },
    N2: { col: 0, x: 6,   y: 58,  w: 196, h: 37 },
    N3: { col: 0, x: 6,   y: 100, w: 196, h: 37 },
    N4: { col: 0, x: 6,   y: 142, w: 196, h: 37 },
    N5: { col: 1, x: 258, y: 28,  w: 192, h: 58 },
    N6: { col: 1, x: 258, y: 108, w: 192, h: 58 },
    N7: { col: 2, x: 506, y: 28,  w: 228, h: 58 },
    N8: { col: 2, x: 506, y: 108, w: 228, h: 58 },
  };

  function nodeStyle(state, id) {
    if (state === "ROOT_SUSPECT" && id === "N3") {
      return {
        icon: "🔒", tag: "GATE ROOT", shortTag: "GATE",
        stroke: c.dark ? "#FACC15" : "#CA8A04",
        fill: c.dark ? "rgba(250,204,21,0.20)" : "#FEF9C3",
        accent: c.dark ? "#FACC15" : "#CA8A04",
        titleCol: c.dark ? "#FEF08A" : "#713F12",
        pillBg: c.dark ? "rgba(250,204,21,0.28)" : "#FEF08A",
        dash: "", sw: 1.9,
      };
    }
    const styles = {
      ROOT_SUSPECT: {
        icon: "⚠", tag: "ROOT SUSPECT", shortTag: "ROOT",
        stroke: c.fault, fill: c.dark ? "rgba(239,68,68,0.20)" : "#FEF2F2",
        accent: c.fault, titleCol: c.fault,
        pillBg: c.dark ? "rgba(239,68,68,0.28)" : "#FEE2E2",
        dash: "", sw: 1.9,
      },
      PROPAGATED: {
        icon: "→", tag: "PROPAGATED", shortTag: "PROP",
        stroke: c.dark ? "#FB923C" : "#EA580C",
        fill: c.dark ? "rgba(234,88,12,0.18)" : "#FFF7ED",
        accent: c.dark ? "#FB923C" : "#EA580C",
        titleCol: c.dark ? "#FDBA74" : "#C2410C",
        pillBg: c.dark ? "rgba(234,88,12,0.26)" : "#FFEDD5",
        dash: "", sw: 1.7,
      },
      RULED_OUT: {
        icon: "✓", tag: "RULED OUT", shortTag: "OK",
        stroke: c.ok, fill: c.dark ? "rgba(16,185,129,0.14)" : "#ECFDF5",
        accent: c.ok, titleCol: c.ok,
        pillBg: c.dark ? "rgba(16,185,129,0.22)" : "#D1FAE5",
        dash: "", sw: 1.35,
      },
      BLOCKED_GATE: {
        icon: "🔒", tag: "BLOCKED (ABSTAIN)", shortTag: "HOLD",
        stroke: c.dark ? "#FACC15" : "#CA8A04",
        fill: c.dark ? "rgba(250,204,21,0.13)" : "#FEFCE8",
        accent: c.dark ? "#FACC15" : "#CA8A04",
        titleCol: c.dark ? "#FEF08A" : "#854D0E",
        pillBg: c.dark ? "rgba(250,204,21,0.22)" : "#FEF08A",
        dash: "4,2.5", sw: 1.4,
      },
      CONFOUNDER_CHECK: {
        icon: "?", tag: "VERIFY PENDING*", shortTag: "CHECK",
        stroke: c.warn, fill: c.dark ? "rgba(245,158,11,0.14)" : "#FFFBEB",
        accent: c.warn, titleCol: c.warn,
        pillBg: c.dark ? "rgba(245,158,11,0.22)" : "#FEF3C7",
        dash: "3.5,2.5", sw: 1.35,
      },
      CALM: {
        icon: "•", tag: "CALM", shortTag: "CALM",
        stroke: c.grid, fill: c.bg,
        accent: c.muted, titleCol: c.txt,
        pillBg: c.dark ? "rgba(148,163,184,0.16)" : "#F1F5F9",
        dash: "", sw: 1.2,
      },
    };
    return styles[state] || styles.CALM;
  }

  const colActive = c.dark ? "#FB923C" : "#EA580C";
  const colPulse = c.dark ? "#FFF7ED" : "#FEF08A";
  const colGate = c.dark ? "#FACC15" : "#CA8A04";
  const colBlocked = c.warn;
  const colMuted = c.dark ? "#475569" : "#94A3B8";

  const defs = `
    <defs>
      <marker id="dag-arr-active" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="5.8" markerHeight="5.8" orient="auto">
        <path d="M 0 1.4 L 9.2 5 L 0 8.6 z" fill="${colActive}"/>
      </marker>
      <marker id="dag-arr-gate" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="5.8" markerHeight="5.8" orient="auto">
        <path d="M 0 1.4 L 9.2 5 L 0 8.6 z" fill="${colGate}"/>
      </marker>
      <marker id="dag-arr-blocked" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="5.2" markerHeight="5.2" orient="auto">
        <path d="M 0 1.5 L 9.2 5 L 0 8.5 z" fill="${colBlocked}"/>
      </marker>
      <marker id="dag-arr-muted" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="4.8" markerHeight="4.8" orient="auto">
        <path d="M 0 1.6 L 9.2 5 L 0 8.4 z" fill="${colMuted}"/>
      </marker>
    </defs>
  `;

  const headers = `
    <text x="104" y="10.5" text-anchor="middle" fill="${c.muted}" font-size="8.2" font-weight="800" letter-spacing="0.4">LAYER 0 · CANDIDATE ROOTS</text>
    <text x="354" y="10.5" text-anchor="middle" fill="${c.muted}" font-size="8.2" font-weight="800" letter-spacing="0.4">LAYER 1 · CT-1 THERMAL / FAN</text>
    <text x="620" y="10.5" text-anchor="middle" fill="${c.muted}" font-size="8.2" font-weight="800" letter-spacing="0.4">LAYER 2 · PLANT &amp; EVN IMPACT</text>
  `;

  // Spread BOTH outgoing and incoming ports so edges never stack at exits or collide at arrowheads
  const edges = causalGraph.edges || [];
  const inMap = {};
  const outMap = {};
  edges.forEach((e, idx) => {
    const src = NODE_POS[e.from], dst = NODE_POS[e.to];
    if (!src || !dst || src.col === dst.col) return;
    if (!inMap[e.to]) inMap[e.to] = [];
    inMap[e.to].push({ idx, srcCy: src.y + src.h / 2 });
    if (!outMap[e.from]) outMap[e.from] = [];
    outMap[e.from].push({ idx, dstCy: dst.y + dst.h / 2 });
  });
  const portInY = {};
  Object.entries(inMap).forEach(([toId, arr]) => {
    const dst = NODE_POS[toId];
    arr.sort((a, b) => a.srcCy - b.srcCy);
    const n = arr.length;
    arr.forEach((item, k) => {
      const frac = n === 1 ? 0.5 : (0.20 + (0.60 * k) / (n - 1));
      portInY[item.idx] = dst.y + dst.h * frac;
    });
  });
  const portOutY = {};
  Object.entries(outMap).forEach(([fromId, arr]) => {
    const src = NODE_POS[fromId];
    arr.sort((a, b) => a.dstCy - b.dstCy);
    const n = arr.length;
    arr.forEach((item, k) => {
      const frac = n === 1 ? 0.5 : (0.34 + (0.38 * k) / (n - 1));
      portOutY[item.idx] = src.y + src.h * frac;
    });
  });

  // Render inactive edges first, then blocked, then active on top
  const sortedEdges = edges.map((e, idx) => ({ e, idx })).sort((a, b) => {
    const rank = x => (x.e.active ? 2 : x.e.blocked ? 1 : 0);
    return rank(a) - rank(b);
  });

  let edgesSvg = "";
  sortedEdges.forEach(({ e, idx }) => {
    const src = NODE_POS[e.from], dst = NODE_POS[e.to];
    if (!src || !dst) return;
    let dPath = "";
    if (src.col === dst.col) {
      const cx = src.x + src.w / 2;
      const y1 = src.y + src.h + 1;
      const y2 = dst.y - 4.5;
      dPath = `M ${cx.toFixed(1)} ${y1.toFixed(1)} L ${cx.toFixed(1)} ${y2.toFixed(1)}`;
    } else {
      const x1 = src.x + src.w + 1;
      const y1 = portOutY[idx] !== undefined ? portOutY[idx] : (src.y + src.h / 2);
      const x2 = dst.x - 4.5;
      const y2 = portInY[idx] !== undefined ? portInY[idx] : (dst.y + dst.h / 2);
      // 4px horizontal lead-out & lead-in guarantees every arrowhead enters level at 0 deg
      const lead = 5;
      const sx = x1 + lead;
      const ex = x2 - lead;
      const midX = (sx + ex) / 2;
      dPath = `M ${x1.toFixed(1)} ${y1.toFixed(1)} L ${sx.toFixed(1)} ${y1.toFixed(1)} C ${midX.toFixed(1)} ${y1.toFixed(1)}, ${midX.toFixed(1)} ${y2.toFixed(1)}, ${ex.toFixed(1)} ${y2.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)}`;
    }

    if (e.active) {
      const isGateEdge = e.from === "N3";
      const stCol = isGateEdge ? colGate : colActive;
      const mk = isGateEdge ? "url(#dag-arr-gate)" : "url(#dag-arr-active)";
      edgesSvg += `<path d="${dPath}" fill="none" stroke="${stCol}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.94" marker-end="${mk}"/>`;
      edgesSvg += `<path class="dag-edge-flow" d="${dPath}" fill="none" stroke="${colPulse}" stroke-width="1.35" stroke-linecap="round" stroke-dasharray="2.5,19.5" stroke-opacity="0.72"/>`;
    } else if (e.blocked) {
      edgesSvg += `<path d="${dPath}" fill="none" stroke="${colBlocked}" stroke-width="1.5" stroke-dasharray="4,3" stroke-opacity="0.78" marker-end="url(#dag-arr-blocked)"/>`;
    } else {
      edgesSvg += `<path d="${dPath}" fill="none" stroke="${colMuted}" stroke-width="1.15" stroke-opacity="0.42" marker-end="url(#dag-arr-muted)"/>`;
    }
  });

  const nodeMap = {};
  let nodesSvg = "";
  causalGraph.nodes.forEach(node => {
    const pos = NODE_POS[node.id];
    if (!pos) return;
    nodeMap[node.id] = node;
    const st = nodeStyle(node.state, node.id);
    const dashAttr = st.dash ? `stroke-dasharray="${st.dash}"` : "";
    const isCol0 = pos.col === 0;
    const maxTitle = isCol0 ? 30 : (pos.col === 2 ? 34 : 28);
    const maxMetric = isCol0 ? 22 : (pos.col === 2 ? 34 : 28);
    const titleStr = clipTxt(`${st.icon} ${node.id} · ${node.short}`, maxTitle);
    const metricStr = clipTxt(node.metric || "", maxMetric);

    // Compact bottom-right status pill on Layer 0 so full titles NEVER truncate
    const pillW = Math.max(24, st.shortTag.length * 4.8 + 8);
    const pillX = pos.x + pos.w - pillW - 6;

    const innerTxt = isCol0
      ? `
        <text x="${pos.x + 10.5}" y="${pos.y + 14.5}" fill="${st.titleCol}" font-size="8.9" font-weight="800">${escSvg(titleStr)}</text>
        <text x="${pos.x + 10.5}" y="${pos.y + 29.2}" fill="${c.txt}" font-size="8.1" font-weight="700" font-family="Consolas,monospace">${escSvg(metricStr)}</text>
        <rect x="${pillX.toFixed(1)}" y="${pos.y + 20.0}" width="${pillW.toFixed(1)}" height="12.2" rx="4" fill="${st.pillBg}"/>
        <text x="${(pillX + pillW / 2).toFixed(1)}" y="${pos.y + 28.6}" text-anchor="middle" fill="${st.titleCol}" font-size="6.7" font-weight="900" letter-spacing="0.2">${escSvg(st.shortTag)}</text>
      `
      : `
        <text x="${pos.x + 11}" y="${pos.y + 16.5}" fill="${st.titleCol}" font-size="9.2" font-weight="800">${escSvg(titleStr)}</text>
        <text x="${pos.x + 11}" y="${pos.y + 33}" fill="${c.txt}" font-size="8.9" font-weight="800" font-family="Consolas,monospace">${escSvg(metricStr)}</text>
        <text x="${pos.x + 11}" y="${pos.y + 48.5}" fill="${st.titleCol}" font-size="7.9" font-weight="800" letter-spacing="0.25">${escSvg(st.tag)}</text>
        <text x="${pos.x + pos.w - 8}" y="${pos.y + 48.5}" text-anchor="end" fill="${c.muted}" font-size="7.7" font-weight="700">Ask RCA ↗</text>
      `;

    // Inset left accent pill cleanly inside the rounded box (zero corner collision!)
    nodesSvg += `
      <g class="dag-node" data-node-id="${escSvg(node.id)}" role="button" tabindex="0">
        <rect class="dag-box" x="${pos.x}" y="${pos.y}" width="${pos.w}" height="${pos.h}" rx="7" fill="${st.fill}" stroke="${st.stroke}" stroke-width="${st.sw}" ${dashAttr}/>
        <rect x="${pos.x + 2.8}" y="${pos.y + 5.5}" width="3.0" height="${pos.h - 11}" rx="1.5" fill="${st.accent}"/>
        ${innerTxt}
      </g>
    `;
  });

  // Compact bottom-right isolation footer cleanly aligned below Col 1 & Col 2 (y=180, never touches Col 0)
  const isoRoot = causalGraph.isolated_root_id && causalGraph.isolated_root_id !== "NONE"
    ? `Isolated Root: ${causalGraph.isolated_root_id} (${causalGraph.isolated_root_label || ""})`
    : "Isolated Root: None (Calm Baseline)";
  const isoCol = causalGraph.isolated_root_id === "N3"
    ? colGate
    : (causalGraph.isolated_root_id && causalGraph.isolated_root_id !== "NONE" ? c.fault : c.ok);
  const footerLine = `
    <text x="258" y="180" fill="${isoCol}" font-size="8.4" font-weight="800">${escSvg(clipTxt(isoRoot, 56))}</text>
    <text x="734" y="180" text-anchor="end" fill="${c.muted}" font-size="8.1" font-weight="700">Click node → RCA</text>
  `;

  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;display:block">
    ${defs}${headers}${edgesSvg}${nodesSvg}${footerLine}
  </svg>`;

  const svg = host.querySelector("svg");
  if (svg) {
    let activeHoverId = null;
    const showNodeTip = (grp) => {
      const nid = grp.getAttribute("data-node-id");
      const n = nodeMap[nid];
      if (!n) { hideTip(); return; }
      if (activeHoverId !== nid) {
        activeHoverId = nid;
        const detailTxt = clipTxt(n.detail || "", 155);
        tip.innerHTML = `<strong>${escSvg(n.id)} · ${escSvg(n.short)} [${escSvg(n.state)}]</strong><div class="dft-metric">${escSvg(n.metric || "")}</div><div class="dft-detail">${escSvg(detailTxt)}</div><div class="dft-cta">Click node → investigate in Air-Gapped RCA Agent</div>`;
      }
      tip.classList.remove("hidden");
      const box = grp.getBoundingClientRect();
      const tw = tip.offsetWidth || 275;
      const th = tip.offsetHeight || 92;
      // Anchor cleanly ABOVE the hovered block (flips BELOW only if near top edge) -> NEVER covers neighbor column!
      let tx = box.left + (box.width - tw) / 2;
      tx = Math.max(10, Math.min(window.innerWidth - tw - 10, tx));
      let ty = box.top - th - 10;
      if (ty < 10) ty = Math.min(window.innerHeight - th - 10, box.bottom + 10);
      tip.style.left = `${Math.round(tx)}px`;
      tip.style.top = `${Math.round(ty)}px`;
    };

    svg.addEventListener("mousemove", evt => {
      const grp = evt.target.closest("[data-node-id]");
      if (!grp) { activeHoverId = null; hideTip(); return; }
      showNodeTip(grp);
    });
    svg.addEventListener("mouseleave", () => { activeHoverId = null; hideTip(); });
    svg.addEventListener("click", evt => {
      const grp = evt.target.closest("[data-node-id]");
      if (!grp) return;
      hideTip();
      if (typeof onNodeClick !== "function") return;
      const n = nodeMap[grp.getAttribute("data-node-id")];
      if (n) onNodeClick(n);
    });
    svg.addEventListener("keydown", evt => {
      if (evt.key !== "Enter" && evt.key !== " ") return;
      const grp = evt.target.closest("[data-node-id]");
      if (!grp || typeof onNodeClick !== "function") return;
      evt.preventDefault();
      hideTip();
      const n = nodeMap[grp.getAttribute("data-node-id")];
      if (n) onNodeClick(n);
    });
  }
};

/* 8. 1D Acoustic FFT Spectrum Chart (0–5 kHz, Bearing 180–420 Hz & Belt Squeal 2.4–4.2 kHz Zones) */
window.drawAcousticFftSvg = function drawAcousticFftSvg(host, acoustic) {
  if (!host) return;
  const c = chartPalette();
  const W = 600, H = 162, L = 40, R = 14, T = 20, B = 24;
  const pw = W - L - R, ph = H - T - B;

  const bins = (acoustic && (Array.isArray(acoustic.spectrum_80) ? acoustic.spectrum_80 : (Array.isArray(acoustic.spectrum) ? acoustic.spectrum : []))) || [];
  if (!bins.length) {
    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;display:block">
      <text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${c.muted}" font-size="11.5" font-weight="700">Acoustic FFT spectrum awaiting bench audio…</text>
    </svg><div class="chart-tip hidden"></div>`;
    return;
  }

  const maxFreq = Math.max(5000, ...bins.map(b => Number(b.freq_hz || 0)));
  const dbs = bins.map(b => Number(b.db ?? -60)).filter(Number.isFinite);
  const minDb = Math.min(-65, Math.floor((Math.min(...dbs) - 4) / 10) * 10);
  const maxDb = Math.max(0, Math.ceil((Math.max(...dbs) + 6) / 10) * 10);
  const dbSpan = Math.max(20, maxDb - minDb);

  const xMap = f => L + (Math.max(0, Math.min(maxFreq, f)) / maxFreq) * pw;
  const yMap = db => T + (1 - (Math.max(minDb, Math.min(maxDb, db)) - minDb) / dbSpan) * ph;

  // Shaded diagnostic frequency bands: Bearing (180–420 Hz) & Belt Squeal (2400–4200 Hz)
  const bX1 = xMap(180), bX2 = xMap(420);
  const sX1 = xMap(2400), sX2 = xMap(4200);
  const bands = `
    <rect x="${bX1.toFixed(1)}" y="${T}" width="${Math.max(6, bX2 - bX1).toFixed(1)}" height="${ph}" fill="${c.dark ? "rgba(251,191,36,0.16)" : "rgba(217,119,6,0.13)"}"/>
    <text x="${((bX1 + bX2) / 2 + 24).toFixed(1)}" y="${T - 6}" text-anchor="middle" fill="${c.warn}" font-size="8.8" font-weight="800">Bearing 180–420Hz</text>
    <rect x="${sX1.toFixed(1)}" y="${T}" width="${(sX2 - sX1).toFixed(1)}" height="${ph}" fill="${c.dark ? "rgba(248,113,113,0.15)" : "rgba(220,38,38,0.10)"}"/>
    <text x="${((sX1 + sX2) / 2).toFixed(1)}" y="${T - 6}" text-anchor="middle" fill="${c.fault}" font-size="8.8" font-weight="800">Belt Squeal 2.4–4.2 kHz</text>
  `;

  let grid = "";
  for (let k = 0; k <= 3; k++) {
    const db = minDb + (k / 3) * dbSpan;
    const yy = yMap(db);
    grid += `<line x1="${L}" y1="${yy.toFixed(1)}" x2="${W - R}" y2="${yy.toFixed(1)}" stroke="${c.grid}" stroke-width="0.9"/>`;
    grid += `<text x="${L - 5}" y="${(yy + 3.5).toFixed(1)}" text-anchor="end" fill="${c.muted}" font-size="9" font-family="Consolas,monospace">${Math.round(db)}</text>`;
  }
  [0, 1000, 2000, 3000, 4000, 5000].forEach(f => {
    const xx = xMap(f);
    grid += `<text x="${xx.toFixed(1)}" y="${H - 7}" text-anchor="middle" fill="${c.muted}" font-size="9" font-family="Consolas,monospace">${f === 0 ? "0" : (f / 1000) + "kHz"}</text>`;
  });

  const verdict = String(acoustic.verdict || "OK");
  const lineCol = verdict === "OK" ? c.ok : (verdict === "ISSUE_BEARING" ? c.warn : c.fault);
  const fillCol = verdict === "OK"
    ? (c.dark ? "rgba(52,211,153,0.22)" : "rgba(5,150,105,0.16)")
    : verdict === "ISSUE_BEARING"
      ? (c.dark ? "rgba(251,191,36,0.22)" : "rgba(217,119,6,0.16)")
      : (c.dark ? "rgba(248,113,113,0.24)" : "rgba(220,38,38,0.16)");

  let linePath = "";
  let peakBin = bins[0];
  bins.forEach((b, i) => {
    const x = xMap(Number(b.freq_hz || 0));
    const y = yMap(Number(b.db ?? minDb));
    linePath += `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)} `;
    if (Number(b.db ?? -999) > Number(peakBin.db ?? -999)) peakBin = b;
  });
  const firstX = xMap(Number(bins[0].freq_hz || 0));
  const lastX = xMap(Number(bins[bins.length - 1].freq_hz || maxFreq));
  const baseY = yMap(minDb);
  const areaPath = `${linePath} L${lastX.toFixed(1)},${baseY.toFixed(1)} L${firstX.toFixed(1)},${baseY.toFixed(1)} Z`;

  const pkX = xMap(Number(peakBin.freq_hz || 0));
  const pkY = yMap(Number(peakBin.db ?? minDb));
  const pkLbl = `Peak ${Math.round(Number(peakBin.freq_hz || 0))} Hz (${Number(peakBin.db || 0).toFixed(1)} dB)`;
  const pkW = Math.min(148, Math.max(106, pkLbl.length * 5.6 + 12));
  const pkBoxX = Math.max(L + 4, Math.min(W - R - pkW - 4, pkX > W * 0.68 ? pkX - pkW - 8 : pkX + 8));
  const pkBoxY = Math.max(T + 3, Math.min(T + ph - 22, pkY - 10));

  const peakSvg = `
    <circle cx="${pkX.toFixed(1)}" cy="${pkY.toFixed(1)}" r="4" fill="${lineCol}" stroke="${c.bg}" stroke-width="1.8"/>
    <rect x="${pkBoxX.toFixed(1)}" y="${pkBoxY.toFixed(1)}" width="${pkW}" height="17" rx="4" fill="${c.bg}" stroke="${lineCol}" stroke-width="1.1"/>
    <text x="${(pkBoxX + pkW / 2).toFixed(1)}" y="${(pkBoxY + 11.5).toFixed(1)}" text-anchor="middle" fill="${lineCol}" font-size="8.6" font-weight="800" font-family="Consolas,monospace">${escSvg(pkLbl)}</text>
  `;

  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;display:block">
    ${bands}${grid}
    <path d="${areaPath}" fill="${fillCol}"/>
    <path d="${linePath}" fill="none" stroke="${lineCol}" stroke-width="2" stroke-linejoin="round"/>
    ${peakSvg}
    <line class="ch-crosshair" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="${c.muted}" stroke-width="1.1" stroke-dasharray="3,3" visibility="hidden"/>
    <circle class="ch-dot" cx="${L}" cy="${T}" r="3.8" fill="${lineCol}" stroke="${c.bg}" stroke-width="1.8" visibility="hidden"/>
  </svg><div class="chart-tip hidden"></div>`;

  attachSeriesTooltip(host, W, L, R, bins, (b, x) => {
    const f = Number(b.freq_hz || 0);
    const db = Number(b.db ?? minDb);
    const zone = (f >= 2400 && f <= 4200) ? "Belt Squeal Band" : (f >= 180 && f <= 420) ? "Bearing Band" : (f >= 60 && f <= 160) ? "Blade Pass Band" : "Broadband";
    return {
      dotX: x,
      dotY: yMap(db),
      dotCol: lineCol,
      html: `<strong>${Math.round(f)} Hz · ${db.toFixed(1)} dB</strong>${zone} · Optical RPM: ${Math.round(Number(acoustic.optical_rpm || 0))}`,
    };
  });
};
