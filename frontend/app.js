/* SLR-GAT delay forecaster — single-page client, no build step, no CDN. */

const $ = (id) => document.getElementById(id);
let MODE = "manual";
let SEED_LEN = 3;

async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `${r.status} ${r.statusText}`);
  return j;
}

function msg(text, kind = "err") {
  $("msg").innerHTML = text ? `<div class="msg ${kind}">${text}</div>` : "";
}

/* ---------------------------------------------------------------- chart */
function chart(stops) {
  const W = 900, H = 320, L = 48, R = 16, T = 16, B = 86;
  const fc = stops.filter((s) => s.forecast !== null);
  if (!fc.length) { $("chart").innerHTML = ""; return; }

  const haveActual = stops.some((s) => s.actual !== null) && MODE === "replay";
  const xs = stops.map((s) => s.stop);
  const vals = [];
  fc.forEach((s) => { vals.push(s.forecast + (s.band || 0), Math.max(0, s.forecast - (s.band || 0))); });
  stops.forEach((s) => { if (s.actual !== null) vals.push(s.actual); });
  const yMax = Math.max(5, Math.ceil(Math.max(...vals) / 5) * 5);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);

  const px = (v) => L + ((v - xMin) / Math.max(1, xMax - xMin)) * (W - L - R);
  const py = (v) => T + (1 - v / yMax) * (H - T - B);

  const areaTop = fc.map((s) => `${px(s.stop)},${py(s.forecast + (s.band || 0))}`);
  const areaBot = fc.slice().reverse()
    .map((s) => `${px(s.stop)},${py(Math.max(0, s.forecast - (s.band || 0)))}`);
  const line = fc.map((s, i) => `${i ? "L" : "M"}${px(s.stop)},${py(s.forecast)}`).join("");

  let grid = "", ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = (yMax / ticks) * i, y = py(v);
    grid += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="#e8ebef"/>`
          + `<text x="${L - 8}" y="${y + 4}" text-anchor="end" font-size="11"
                   fill="#656d78">${v.toFixed(0)}</text>`;
  }

  const every = Math.max(1, Math.ceil(stops.length / 16));
  let xlab = "";
  stops.forEach((s, i) => {
    if (i % every && i !== stops.length - 1) return;
    const lx = px(s.stop), ly = H - B + 14;
    xlab += `<text x="${lx}" y="${ly}" font-size="10" fill="#656d78"
              text-anchor="end" transform="rotate(-42 ${lx} ${ly})">${s.station}</text>`
          + `<line x1="${lx}" y1="${H - B}" x2="${lx}" y2="${H - B + 4}" stroke="#aeb6c0"/>`;
  });

  const seeds = stops.filter((s) => s.is_seed && s.actual !== null)
    .map((s) => `<rect x="${px(s.stop) - 4}" y="${py(s.actual) - 4}" width="8" height="8"
                   fill="#15181d"/>`).join("");

  let actual = "";
  if (haveActual) {
    const pts = stops.filter((s) => s.actual !== null);
    actual = `<path d="${pts.map((s, i) => `${i ? "L" : "M"}${px(s.stop)},${py(s.actual)}`).join("")}"
                fill="none" stroke="var(--actual)" stroke-width="1.6"
                stroke-dasharray="5 3"/>`
           + pts.map((s) => `<circle cx="${px(s.stop)}" cy="${py(s.actual)}" r="2.6"
                fill="var(--actual)"/>`).join("");
  }

  const lastSeed = Math.max(...stops.filter((s) => s.is_seed).map((s) => s.stop));

  $("chart").innerHTML = `
  <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="forecast delay by stop">
    ${grid}
    <line x1="${px(lastSeed)}" y1="${T}" x2="${px(lastSeed)}" y2="${H - B}"
          stroke="#aeb6c0" stroke-dasharray="3 3"/>
    <polygon points="${areaTop.concat(areaBot).join(" ")}" fill="var(--accent)"
             opacity="0.14"/>
    <path d="${line}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>
    ${fc.map((s) => `<circle cx="${px(s.stop)}" cy="${py(s.forecast)}" r="2.8"
        fill="var(--accent)"><title>${s.station}: ${s.forecast} min</title></circle>`).join("")}
    ${actual}${seeds}
    ${xlab}
    <text x="${L}" y="${H - 6}" font-size="11" fill="#656d78">stop along route &rarr;</text>
  </svg>`;

  $("legend").innerHTML =
    `<span><i class="sw" style="background:var(--accent)"></i>forecast</span>
     <span><i class="sw" style="background:var(--accent);opacity:.25"></i>measured uncertainty</span>
     <span><i class="sw" style="background:#15181d"></i>observed seed</span>` +
    (haveActual ? `<span><i class="sw" style="background:var(--actual)"></i>actual</span>` : "");
}

/* ---------------------------------------------------------------- table */
function table(stops) {
  const head = MODE === "replay"
    ? "<tr><th>#</th><th>Station</th><th>Dep</th><th>Forecast</th><th>Actual</th><th>Error</th></tr>"
    : "<tr><th>#</th><th>Station</th><th>Dep</th><th>Forecast</th><th>&plusmn;</th></tr>";
  const rows = stops.map((s) => {
    const f = s.forecast === null ? "&mdash;" : s.forecast.toFixed(1);
    if (MODE === "replay") {
      const a = s.actual === null ? "&mdash;" : s.actual.toFixed(1);
      const e = (s.forecast === null || s.actual === null)
        ? "&mdash;" : (s.forecast - s.actual).toFixed(1);
      return `<tr class="${s.is_seed ? "seed" : ""}"><td>${s.stop}</td>
        <td>${s.station}</td><td>${s.departure}</td><td>${f}</td>
        <td>${a}</td><td>${e}</td></tr>`;
    }
    return `<tr class="${s.is_seed ? "seed" : ""}"><td>${s.stop}</td>
      <td>${s.station}</td><td>${s.departure}</td><td>${f}</td>
      <td>${s.band === null ? "&mdash;" : "&plusmn;" + s.band.toFixed(1)}</td></tr>`;
  }).join("");
  $("tbl").innerHTML = head + rows;
}

function stats(res) {
  const fc = res.stops.filter((s) => s.forecast !== null);
  const last = fc[fc.length - 1];
  let html = "";
  if (last) {
    html += `<div class="stat"><b>${last.forecast.toFixed(1)} min</b>
      <span>at ${last.station}</span></div>`;
    if (last.band !== null) {
      html += `<div class="stat"><b>&plusmn;${last.band.toFixed(1)}</b>
        <span>measured error</span></div>`;
    }
  }
  if (MODE === "replay" && res.mae !== null && res.mae !== undefined) {
    const better = res.skill !== null && res.skill > 0;
    html += `<div class="stat"><b>${res.mae.toFixed(2)}</b><span>model MAE</span></div>
             <div class="stat"><b>${res.persistence_mae.toFixed(2)}</b>
               <span>persistence MAE</span></div>
             <div class="stat"><b style="color:${better ? "var(--ok)" : "var(--warn)"}">
               ${res.skill === null ? "&mdash;" : (res.skill > 0 ? "+" : "") + res.skill.toFixed(2)}
               </b><span>skill vs persistence</span></div>`;
    if (!better) {
      msg(`On this run persistence is the better rule. The model encodes a
           &ldquo;delay will accumulate&rdquo; prior: it beats carry-forward on
           services that are actually delayed and loses on punctual ones.`, "note");
    }
  }
  $("stats").innerHTML = html;
}

/* ----------------------------------------------------------------- flow */
async function loadDates() {
  const dates = await api("/api/dates");
  $("date").innerHTML = dates.map((d) => `<option>${d}</option>`).join("");
  $("date").value = dates[dates.length - 1];
  await loadServices();
}

async function loadServices() {
  const d = $("date").value;
  try {
    const svc = await api(`/api/services?date=${d}`);
    $("service").innerHTML = svc.map((s) =>
      `<option value="${s.train_id}">${s.train_id} &middot; ${s.first_dep}
       &middot; ${s.stops} stops &middot; Tier ${s.tier}</option>`).join("");
    msg("");
  } catch (e) {
    $("service").innerHTML = "";
    msg(e.message);
  }
}

function setMode(m) {
  MODE = m;
  $("tab-manual").setAttribute("aria-selected", m === "manual");
  $("tab-replay").setAttribute("aria-selected", m === "replay");
  $("manualbox").style.display = m === "manual" ? "" : "none";
  $("run").textContent = m === "manual" ? "Forecast" : "Replay this run";
  $("modehint").textContent = m === "manual"
    ? "Other services are assumed on time — the operator does not know their delays, and no measurable propagation between services was found."
    : "Seeds the three real observed delays and compares the forecast with what actually happened.";
  $("msg").innerHTML = "";
}

async function run() {
  const train_id = $("service").value, date = $("date").value;
  if (!train_id) return;
  $("run").disabled = true;
  try {
    let res;
    if (MODE === "manual") {
      const seeds = [...document.querySelectorAll("#seeds input")]
        .map((i) => parseFloat(i.value || "0"));
      res = await api("/api/forecast", { train_id, date, seeds });
      msg("");
    } else {
      res = await api("/api/replay", { train_id, date });
    }
    stats(res); chart(res.stops); table(res.stops);
  } catch (e) {
    msg(e.message); $("stats").innerHTML = "";
    $("chart").innerHTML = ""; $("tbl").innerHTML = ""; $("legend").innerHTML = "";
  } finally {
    $("run").disabled = false;
  }
}

(async function init() {
  try {
    const info = await api("/api/info");
    if (!info.loaded) { msg(`Model bundle not loaded: ${info.error}`); return; }
    if (info.synthetic) {
      msg("<b>Synthetic bundle loaded.</b> These weights are random and the "
        + "numbers are meaningless. Replace <code>bundle/</code> with the "
        + "contents of <code>09_deploy/</code>.", "err");
    }
    SEED_LEN = info.seed_len;
    $("seedn").textContent = SEED_LEN;
    $("seeds").innerHTML = Array.from({ length: SEED_LEN }, (_, i) =>
      `<input type="number" step="1" value="0" aria-label="stop ${i + 1} delay">`).join("");
    $("modelinfo").textContent =
      `${info.architecture.toUpperCase()} · ${info.parameters.toLocaleString()} params · `
      + `hidden ${info.hidden}, ${info.heads} heads, ${info.layers} layers · `
      + `trained on ${info.trained_on_days} days · val rollout MAE ${info.val_rollout_mae}`;
    await loadDates();
  } catch (e) { msg(e.message); }

  $("date").onchange = loadServices;
  $("run").onclick = run;
  $("tab-manual").onclick = () => setMode("manual");
  $("tab-replay").onclick = () => setMode("replay");
  setMode("manual");
})();
