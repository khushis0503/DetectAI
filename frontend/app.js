// DetectAI - front-end logic: sends the text to the API and draws the results.

const API = "";   // same server that serves this page

// Sample texts: real texts from our held-out TEST set (never used for training)
const SAMPLES = {
  casual: "To further clarify: pi is a constant that would exist regardless if we had a word, symbol, or number for it. It's not so much an idea or concept, it just is. When you multiply, you're calculating what n of something is -- two of 3 is six; five of 2 is ten, etc. We use pi because it never changes; 3.14159.. of a circle's radius^squared will necessarily always yield the circle's area. Nobody decided that, it's just the way it is.",
  formal: "China said on Thursday its jailing of leading dissident Wang Dan for 11 years was not a human rights issue but an ordinary legal matter that would not hurt Beijing's sensitive relationship with the United States. A Chinese court on Wednesday took less than four hours to condemn former student leader Wang for plotting against the government, in a verdict that prompted quick expressions of deep concern from Washington. Chinese Foreign Ministry spokesman Shen Guofang said the trial would not cause further damage to cross-Pacific ties, long sorely tested by disputes over Beijing's human rights record.",
  chatgpt: "Steam is a digital distribution platform developed by Valve Corporation, a company best known for its video game franchises such as Half-Life and Portal. It was released in 2003 and is available for Microsoft Windows, macOS, and Linux. Steam allows users to purchase, download, and play games, as well as manage their game libraries and socialize with other users. It also includes features such as automatic updates, cloud saving, and support for virtual reality. In addition to games, Steam also offers a wide range of other content, including software, videos, and music. Steam has become one of the largest digital distribution platforms for PC gaming, with a library of over 30,000 games and millions of active users. It has also been credited with helping to popularize digital distribution and the concept of digital ownership of games.",
};

const COLORS = { human: "#2a78d6", ai: "#eb6834", unc: "#8a8f99" };
let thresholds = { likely_ai: 0.75, likely_human: 0.25 };
let deepLink = false;   // true when opened via a #sample= / #text= link (no animated scrolling)

const $ = (id) => document.getElementById(id);
const pct = (p) => `${Math.round(p * 100)}%`;
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const zone = (p) => (p >= thresholds.likely_ai ? "ai" : p <= thresholds.likely_human ? "human" : "unc");

// ---------------------------------------------------------------- input
const input = $("textInput");
function countWords() {
  const n = input.value.trim() ? input.value.trim().split(/\s+/).length : 0;
  const c = $("wordCounter");
  c.textContent = `${n} word${n === 1 ? "" : "s"}` + (n > 0 && n < 50 ? " · under 50 words: less reliable" : "");
  c.classList.toggle("low", n > 0 && n < 50);
}
input.addEventListener("input", countWords);
document.querySelectorAll("[data-sample]").forEach((b) =>
  b.addEventListener("click", () => { input.value = SAMPLES[b.dataset.sample]; countWords(); input.focus(); }));
$("clearBtn").addEventListener("click", () => {
  input.value = ""; countWords(); $("results").hidden = true; $("errorBox").hidden = true; input.focus();
});
$("analyzeBtn").addEventListener("click", analyze);
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze(); });

async function analyze() {
  const btn = $("analyzeBtn");
  $("errorBox").hidden = true;
  btn.disabled = true; btn.classList.add("loading"); btn.querySelector(".btn-label").textContent = "Analyzing…";
  try {
    const res = await fetch(`${API}/api/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: input.value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    render(data);
  } catch (err) {
    const box = $("errorBox");
    box.textContent = err.message.includes("fetch") ? "Cannot reach the DetectAI server. Is run.bat still running?" : err.message;
    box.hidden = false;
  } finally {
    btn.disabled = false; btn.classList.remove("loading"); btn.querySelector(".btn-label").textContent = "Analyze text";
  }
}

// ---------------------------------------------------------------- results
function render(d) {
  thresholds = d.thresholds;
  $("results").hidden = false;

  // short-text warning
  $("shortWarning").hidden = !d.short_text_warning;
  $("shortWarningDetail").textContent =
    `Only ${d.n_words} words. Short texts contain little evidence, so the confidence below has been reduced. 50+ words is better.`;

  // verdict card
  const z = d.verdict === "Likely AI" ? "ai" : d.verdict === "Likely Human" ? "human" : "unc";
  const card = $("verdictCard");
  card.className = `card verdict-card is-${z}`;
  $("verdictText").textContent = d.verdict;
  $("verdictSub").textContent = {
    ai: "The writing patterns are much closer to ChatGPT-generated text.",
    human: "The writing patterns are much closer to human-written text.",
    unc: "The evidence is mixed — the text could be either. Look at the details below.",
  }[z];
  $("probText").textContent = pct(d.ai_probability);
  const c = d.confidence;
  $("confText").textContent = `${c >= 0.66 ? "High" : c >= 0.33 ? "Medium" : "Low"} · ${pct(c)}`;
  $("confFill").style.width = pct(c);
  $("wordsText").textContent = d.n_words;
  $("modelUsed").textContent = `Model: ${d.model_used} · analysed in ${d.seconds}s`;

  drawGauge(d.ai_probability);
  drawHighlights(d);
  drawBars($("sentences"), d.sentences.map((s) => ({ p: s.ai_probability, text: s.text })));

  const hasBreakdown = d.breakdown && d.breakdown.length > 1;
  $("breakdownCard").hidden = !hasBreakdown;
  if (hasBreakdown) drawBars($("breakdown"), d.breakdown.map((b) =>
    ({ p: b.ai_probability, label: `${b.label} · ${b.verdict} · ${b.n_words} words`, text: b.preview + (b.preview.length >= 140 ? "…" : "") })));

  drawStyle(d.style, d.style_reference);
  $("results").scrollIntoView({ behavior: deepLink ? "instant" : "smooth", block: "start" });
}

// Semicircle gauge with the three verdict zones
function drawGauge(p) {
  const cx = 160, cy = 160, r = 120, w = 30;
  const pt = (v, rad) => {           // v in [0,1] -> point on the arc (0 = left, 1 = right)
    const a = Math.PI * (1 - v);
    return [cx + rad * Math.cos(a), cy - rad * Math.sin(a)];
  };
  const arc = (v0, v1, color) => {
    const [x0, y0] = pt(v0, r), [x1, y1] = pt(v1, r);
    return `<path d="M${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1}" stroke="${color}" stroke-width="${w}" fill="none"/>`;
  };
  const th = thresholds.likely_human, ta = thresholds.likely_ai, gap = 0.006;
  // short pointer crossing the arc (keeps the centre free for the number)
  const [px0, py0] = pt(p, r - w / 2 - 22), [px1, py1] = pt(p, r + w / 2 + 6);
  const tickLabel = (v, txt) => { const [x, y] = pt(v, r + w / 2 + 14); return `<text x="${x}" y="${y}" font-size="13" fill="#5b6270" text-anchor="middle">${txt}</text>`; };
  $("gauge").innerHTML = `
    ${arc(0, th - gap, COLORS.human)}${arc(th + gap, ta - gap, COLORS.unc)}${arc(ta + gap, 1, COLORS.ai)}
    ${tickLabel(th, pct(th))}${tickLabel(ta, pct(ta))}
    <line x1="${px0}" y1="${py0}" x2="${px1}" y2="${py1}" stroke="#fff" stroke-width="12" stroke-linecap="round"/>
    <line x1="${px0}" y1="${py0}" x2="${px1}" y2="${py1}" stroke="#111827" stroke-width="6" stroke-linecap="round"/>
    <text x="${cx}" y="${cy - 14}" font-size="46" font-weight="800" text-anchor="middle" fill="#111827">${pct(p)}</text>
    <text x="${cx}" y="${cy + 12}" font-size="14" text-anchor="middle" fill="#5b6270">chance of AI</text>
    <text x="22" y="${cy + 24}" font-size="13" fill="#5b6270">0%</text>
    <text x="298" y="${cy + 24}" font-size="13" fill="#5b6270" text-anchor="end">100%</text>`;
}

// Text with each word coloured by how much it pushed towards AI (orange) or human (blue)
function drawHighlights(d) {
  $("highlightMethod").textContent =
    `Method: ${d.highlight_method}. Stronger colour = stronger influence. Hover a word to see its score.` +
    (d.highlight_note ? ` ${d.highlight_note}` : "");
  let html = "";
  for (const seg of d.highlighted) {
    if (seg.s === null || Math.abs(seg.s) < 0.06) { html += esc(seg.t).replace(/\n\n/g, "</p><p>"); continue; }
    const a = Math.min(0.85, Math.pow(Math.abs(seg.s), 0.7) * 0.85).toFixed(2);
    const rgb = seg.s > 0 ? "235,104,52" : "42,120,214";
    const dir = seg.s > 0 ? "towards AI" : "towards human";
    html += `<span class="w" style="background:rgba(${rgb},${a})" title="${dir} (${seg.s.toFixed(2)})">${esc(seg.t)}</span>`;
  }
  $("highlighted").innerHTML = `<p>${html}</p>`;
}

// Horizontal probability bars (sentences or paragraphs)
function drawBars(el, rows) {
  el.innerHTML = rows.map((r) => {
    const z = zone(r.p);
    return `<div class="bar-row">
      <div class="bar-pct ${z}">${pct(r.p)}</div>
      <div>
        ${r.label ? `<div class="bar-label">${esc(r.label)}</div>` : ""}
        <div class="bar-track"><div class="bar-fill" style="width:${pct(r.p)};background:${COLORS[z]}"></div></div>
        <div class="bar-text">${esc(r.text)}</div>
      </div></div>`;
  }).join("");
}

// Writing-style panel: this text vs typical human / AI values
const STYLE_GROUPS = [
  ["Sentences", [
    ["sent_len_std", "Sentence-length variation", (v) => v.toFixed(1)],
    ["sent_len_mean", "Average sentence length (words)", (v) => v.toFixed(1)],
    ["commas_per_sent", "Commas per sentence", (v) => v.toFixed(2)],
  ]],
  ["Vocabulary", [
    ["type_token_ratio", "Vocabulary richness (type-token ratio)", (v) => v.toFixed(2)],
    ["hapax_ratio", "Words used only once", pct],
    ["avg_word_len", "Average word length (letters)", (v) => v.toFixed(1)],
    ["stopword_ratio", "Function words (stopwords)", pct],
  ]],
  ["Parts of speech (POS tags)", [
    ["noun_ratio", "Nouns", pct], ["verb_ratio", "Verbs", pct], ["adj_ratio", "Adjectives", pct],
    ["adv_ratio", "Adverbs", pct], ["pron_ratio", "Pronouns", pct],
  ]],
];
function drawStyle(style, ref) {
  let html = `<div class="style-legend"><span><i class="lt" style="background:${COLORS.human}"></i>typical human</span>
    <span><i class="lt" style="background:${COLORS.ai}"></i>typical AI</span><span><i class="ld"></i>this text</span></div>`;
  for (const [title, feats] of STYLE_GROUPS) {
    html += `<div class="style-group"><h3>${title}</h3>`;
    for (const [key, name, fmt] of feats) {
      const r = ref[key], v = style[key];
      if (!r) continue;
      const lo = Math.min(r.p05, v), hi = Math.max(r.p95, v), span = hi - lo || 1;
      const pos = (x) => `${(4 + 92 * (x - lo) / span).toFixed(1)}%`;
      html += `<div class="style-row">
        <div class="style-name">${name}</div>
        <div class="style-track" title="human median ${fmt(r.human_median)}, AI median ${fmt(r.ai_median)}">
          <span class="tick human" style="left:${pos(r.human_median)}"></span>
          <span class="tick ai" style="left:${pos(r.ai_median)}"></span>
          <span class="dot" style="left:${pos(v)}"></span>
        </div>
        <div class="style-val">${fmt(v)}</div></div>`;
    }
    html += "</div>";
  }
  $("stylePanel").innerHTML = html;
}

// ---------------------------------------------------------------- model info
async function loadModelInfo() {
  try {
    const info = await (await fetch(`${API}/api/model-info`)).json();
    thresholds = info.thresholds;
    const m = info.ensemble_used ? info.ensemble.test_ensemble : info.main_model_test_metrics;
    const name = info.model_name + (info.ensemble_used ? " + char SVM" : "");
    const badge = $("modelBadge");
    badge.textContent = m ? `${name} · test accuracy ${pct(m.accuracy)}` : name;
    if (!info.thresholds_calibrated) {
      badge.classList.add("warn");
      badge.textContent += " · thresholds not calibrated (run calibrate.py)";
    }
    if (m) {
      $("metricsStrip").innerHTML = `<b>Current model:</b> ${esc(name)} — on our held-out test set:
        accuracy ${pct(m.accuracy)}, precision ${pct(m.precision)}, recall ${pct(m.recall)}, F1 ${pct(m.f1)}, ROC-AUC ${m.roc_auc.toFixed(3)}.
        Verdict thresholds: “Likely AI” from ${pct(thresholds.likely_ai)}, “Likely Human” up to ${pct(thresholds.likely_human)}.`;
    }
  } catch {
    $("modelBadge").textContent = "Server not reachable";
    $("modelBadge").classList.add("warn");
  }
}

// ---------------------------------------------------------------- deep links
// For demos and report screenshots:  /#sample=chatgpt   or   /#text=<url-encoded text>
// optionally with &focus=<element id> to scroll to one section after the analysis.
async function runFromHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const text = params.get("text") || SAMPLES[params.get("sample")];
  if (!text) return;
  deepLink = true;
  input.value = text;
  countWords();
  await analyze();
  const focus = $(params.get("focus") || "");
  if (focus) window.scrollTo({ top: focus.getBoundingClientRect().top + window.scrollY - 16, behavior: "instant" });
  if (params.get("only")) isolate(params.get("only").split(","));
}

// &only=id1,id2 shows just those sections (used to capture clean screenshots of one panel)
function isolate(ids) {
  const keep = new Set();
  for (const id of ids) {
    for (let el = $(id); el && el !== document.body; el = el.parentElement) keep.add(el);
  }
  for (const el of keep) {
    for (const sib of el.parentElement.children) {
      if (!keep.has(sib) && sib.tagName !== "SCRIPT") sib.style.display = "none";
    }
  }
  window.scrollTo(0, 0);
}

countWords();
loadModelInfo().then(runFromHash);
