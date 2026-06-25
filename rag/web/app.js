"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// 與後端 verify_citations 同款：括號式 (p.3,4)/（p.3）或裸式 p.3
const CITE = /[（(]\s*[pP]\.?\s*[\d\s,、]+?\s*[)）]|[pP]\.\s*\d+/g;
function highlightCites(s) {
  return escapeHtml(s).replace(CITE, m => `<span class="cite">${m}</span>`);
}

function showLoading(out, label) {
  const t0 = Date.now();
  out.innerHTML = `<div class="card loading"><div class="spinner"></div>`
    + `<div>${label}</div><div class="elapsed" id="elp">0.0s</div></div>`;
  const el = $("#elp", out);
  const iv = setInterval(() => { el.textContent = ((Date.now() - t0) / 1000).toFixed(1) + "s"; }, 100);
  return () => clearInterval(iv);
}

function renderError(out, data, status) {
  out.innerHTML = `<div class="error">⚠️ ${escapeHtml(data.error || ("HTTP " + status))}</div>`;
}

async function postJSON(path, body) {
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({ error: "回應非 JSON" }));
    return { ok: r.ok, status: r.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: { error: "連不到伺服器（API 在跑嗎？）" } };
  }
}

// ---------- 文件問答 ----------
async function ask(q) {
  const out = $("#rag-out");
  q = (q || $("#rag-q").value).trim();
  if (!q) return;
  $("#rag-q").value = q;
  const btn = $("#rag-btn");
  btn.disabled = true;
  const stop = showLoading(out, "本地 CPU 推論中（約 30–60 秒）…");
  const { ok, status, data } = await postJSON("/ask", { query: q });
  stop();
  btn.disabled = false;
  if (!ok) return renderError(out, data, status);

  let html = `<div class="card"><div class="answer">${highlightCites(data.answer)}</div>`;
  if (data.stripped_pages && data.stripped_pages.length) {
    html += `<div class="note-strip"><span>⚠️</span><span>引用護欄：已剝除 `
      + `${data.stripped_pages.length} 個檢索範圍外頁碼（${data.stripped_pages.join(", ")}），擋 citation 幻覺。</span></div>`;
  }
  if (data.sources && data.sources.length) {
    html += `<div class="sources-title">參考來源</div>`;
    for (const s of data.sources) {
      html += `<div class="source"><span class="source-pg">p.${s.page}</span>`
        + `<span class="source-name">${escapeHtml(s.source)}</span>`
        + `<span class="source-score">score ${s.score}</span></div>`;
    }
  }
  out.innerHTML = html + `</div>`;
}

// ---------- Agent（多輪對話） ----------
let agentHistory = [];   // [{role,content}]；每次請求帶最近幾輪供「它/那家」代名詞解析

function agentTurn(cls, html) {
  const div = document.createElement("div");
  div.className = "turn " + cls;
  div.innerHTML = html;
  $("#agent-out").appendChild(div);
  div.scrollIntoView({ block: "nearest" });
  return div;
}

async function runAgent(q) {
  q = (q || $("#agent-q").value).trim();
  if (!q) return;
  $("#agent-q").value = "";
  const btn = $("#agent-btn");
  btn.disabled = true;

  agentTurn("turn-user", `<div class="bubble">${escapeHtml(q)}</div>`);
  const card = agentTurn("turn-bot",
    `<div class="card loading"><div class="spinner"></div>`
    + `<div>Agent 推論中（多步，每步約 30–60 秒）…</div>`
    + `<div class="elapsed" id="elp-a">0.0s</div></div>`);
  const t0 = Date.now();
  const iv = setInterval(() => {
    const e = $("#elp-a"); if (e) e.textContent = ((Date.now() - t0) / 1000).toFixed(1) + "s";
  }, 100);

  const { ok, status, data } = await postJSON("/agent", { message: q, history: agentHistory });
  clearInterval(iv);
  btn.disabled = false;
  if (!ok) {
    card.innerHTML = `<div class="error">⚠️ ${escapeHtml(data.error || ("HTTP " + status))}</div>`;
    return;
  }

  let html = `<div class="card">`;
  if (data.trace && data.trace.length) {
    html += `<div class="trace-title">工具軌跡（${data.trace.length} 步）</div>`;
    for (const t of data.trace) {
      html += `<div class="step"><div class="step-head"><span class="tool-badge">${escapeHtml(t.tool)}</span>`
        + `<span class="step-args">${escapeHtml(JSON.stringify(t.args))}</span></div>`
        + `<div class="step-result">${escapeHtml(t.result)}</div></div>`;
    }
  }
  html += `<div class="final-answer"><div class="final-label">✅ 最終答案</div>`
    + `<div class="answer">${highlightCites(data.answer)}</div></div></div>`;
  card.innerHTML = html;
  card.scrollIntoView({ block: "nearest" });

  agentHistory.push({ role: "user", content: q }, { role: "assistant", content: data.answer });
  if (agentHistory.length > 6) agentHistory = agentHistory.slice(-6);   // 只留最近 3 輪
}

function clearAgent() {
  agentHistory = [];
  $("#agent-out").innerHTML = "";
  $("#agent-q").value = "";
  $("#agent-q").focus();
}

// ---------- 圖片理解 (VLM) ----------
let vlmDataUrl = null;
function setupVlm() {
  const file = $("#vlm-file"), drop = $("#vlm-drop"), prev = $("#vlm-preview"), ph = $("#vlm-placeholder");
  const load = f => {
    if (!f || !f.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = e => { vlmDataUrl = e.target.result; prev.src = vlmDataUrl; prev.hidden = false; ph.hidden = true; };
    reader.readAsDataURL(f);
  };
  file.addEventListener("change", e => load(e.target.files[0]));
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("drag"); load(e.dataTransfer.files[0]); });
}
async function runVlm() {
  const out = $("#vlm-out");
  if (!vlmDataUrl) { out.innerHTML = `<div class="error">請先上傳圖片。</div>`; return; }
  const btn = $("#vlm-btn");
  btn.disabled = true;
  const stop = showLoading(out, "Gemma 讀圖中（本地 CPU，約 30–90 秒）…");
  const q = $("#vlm-q").value.trim();
  const { ok, status, data } = await postJSON("/vlm", { image_b64: vlmDataUrl, question: q || null });
  stop();
  btn.disabled = false;
  if (!ok) return renderError(out, data, status);
  out.innerHTML = `<div class="card"><div class="answer">${escapeHtml(data.text)}</div></div>`;
}

// ---------- 狀態列 ----------
async function loadHealth() {
  const el = $("#chips");
  try {
    const h = await (await fetch("/health")).json();
    const chips = [`<span class="chip chip-on">● 線上</span>`,
      `<span class="chip">LLM ${escapeHtml(h.gen_model || "?")}</span>`,
      `<span class="chip">VLM ${escapeHtml(h.vlm_model || "?")}</span>`];
    if (h.rerank_model) chips.push(`<span class="chip">rerank ${escapeHtml(String(h.rerank_model).split("/").pop())}</span>`);
    el.innerHTML = chips.join("");
  } catch (e) {
    el.innerHTML = `<span class="chip chip-muted">API 未連線</span>`;
  }
}

// ---------- 初始化 ----------
function setupTabs() {
  $$(".nav-item").forEach(b => b.addEventListener("click", () => {
    $$(".nav-item").forEach(x => x.classList.remove("active"));
    $$(".panel").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    $("#panel-" + b.dataset.tab).classList.add("active");
  }));
}
function setupExamples() {
  $$("#rag-ex .chip-btn").forEach(b => b.addEventListener("click", () => ask(b.textContent)));
  $$("#agent-ex .chip-btn").forEach(b => b.addEventListener("click", () => runAgent(b.textContent)));
}
document.addEventListener("DOMContentLoaded", () => {
  setupTabs(); setupExamples(); setupVlm(); loadHealth();
  $("#rag-btn").addEventListener("click", () => ask());
  $("#rag-q").addEventListener("keydown", e => { if (e.key === "Enter") ask(); });
  $("#agent-btn").addEventListener("click", () => runAgent());
  $("#agent-clear").addEventListener("click", () => clearAgent());
  $("#agent-q").addEventListener("keydown", e => { if (e.key === "Enter") runAgent(); });
  $("#vlm-btn").addEventListener("click", () => runVlm());
});
