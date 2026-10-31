// Viewer: fields on the left, pages on the right, highlight boxes positioned in % so they scale with the image.
const $ = (id) => document.getElementById(id);
let current = null;   // the DocResult being shown
let pollTimer = null;

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  for (const c of children) node.append(c);
  return node;
}

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

// ---- highlighting -------------------------------------------------------------------------------
function clearBoxes() {
  document.querySelectorAll(".box").forEach((b) => b.remove());
  document.querySelectorAll(".field.active").forEach((f) => f.classList.remove("active"));
}

function highlight(regions) {
  clearBoxes();
  let first = null;
  for (const { page, bbox } of regions) {
    const host = document.querySelector(`.page[data-page="${page}"]`);
    if (!host || !bbox) continue;
    const [x0, y0, x1, y1] = bbox;
    const pad = 0.003;
    const box = el("div", { className: "box" });
    Object.assign(box.style, {
      left: `${(x0 - pad) * 100}%`, top: `${(y0 - pad) * 100}%`,
      width: `${(x1 - x0 + 2 * pad) * 100}%`, height: `${(y1 - y0 + 2 * pad) * 100}%`,
    });
    host.append(box);
    first = first || box;
  }
  first?.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ---- rendering ----------------------------------------------------------------------------------
function renderPages(doc) {
  const viewer = $("viewer");
  viewer.replaceChildren();
  for (const p of doc.pages) {
    const img = el("img", { src: `/api/documents/${doc.doc_id}/pages/${p.number}`, alt: `Page ${p.number}`, loading: "lazy" });
    viewer.append(el("div", { className: "page" }, img, el("span", { className: "label", textContent: `Page ${p.number}` })));
    viewer.lastChild.dataset.page = p.number;
  }
}

function renderFields(doc) {
  const list = $("fields");
  list.replaceChildren();
  $("field-count").textContent = `(${doc.fields.length})`;
  let lastPage = null;
  for (const f of doc.fields) {
    if (doc.pages.length > 1 && f.page !== lastPage) {
      list.append(el("li", { className: "page-head", textContent: `Page ${f.page}` }));
      lastPage = f.page;
    }
    const unsure = f.confidence < 0.7 || f.attributes?.value_found_in_lines === false;
    const button = el("button", { type: "button", className: `field${unsure ? " unsure" : ""}` },
      el("span", { className: "name", textContent: f.name.replaceAll("_", " ") }),
      el("span", { className: "type", textContent: `${f.type} · ${Math.round(f.confidence * 100)}%` }),
      el("span", { className: "value", textContent: f.value, dir: "auto" }));
    const attrs = Object.entries(f.attributes || {}).map(([k, v]) => `${k}: ${v}`).join(", ");
    if (attrs) button.title = attrs;
    button.addEventListener("click", () => {
      highlight([{ page: f.page, bbox: f.bbox }]);
      button.classList.add("active");
    });
    list.append(el("li", {}, button));
  }
}

function render(doc) {
  current = doc;
  $("doc-type").textContent = doc.doc_type ? doc.doc_type.replaceAll("_", " ") : "unclassified";
  $("doc-lang").textContent = `language: ${doc.language}`;
  $("doc-engine").textContent = `text from: ${doc.stats?.engine ?? "–"}`;
  $("summary").textContent = doc.summary || "No summary.";
  renderPages(doc);
  renderFields(doc);
  $("chat-log").replaceChildren();
  for (const id of ["overview", "fields-panel", "chat-panel"]) $(id).hidden = false;
  const secs = doc.stats?.total_seconds;
  setStatus(`${doc.filename} · ${doc.fields.length} fields${secs ? ` · ${secs.toFixed(1)} s` : ""}`);
}

// ---- loading ------------------------------------------------------------------------------------
async function openDoc(docId) {
  clearTimeout(pollTimer);
  try {
    const doc = await api(`/api/documents/${docId}`);
    if (doc.status === "processing") {
      setStatus(`Processing ${doc.filename}: ${doc.stage}…`);
      pollTimer = setTimeout(() => openDoc(docId), 1500);
      return;
    }
    if (doc.status === "error") { setStatus(`Couldn't process ${doc.filename}. ${doc.error}`, true); return; }
    render(doc);
    refreshList(docId);
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function refreshList(selected) {
  try {
    const docs = await api("/api/documents");
    const select = $("docs");
    select.replaceChildren(el("option", { value: "", textContent: docs.length ? "Choose…" : "–" }));
    for (const d of docs) select.append(el("option", { value: d.doc_id, textContent: `${d.filename} (${d.status})` }));
    if (selected) select.value = selected;
  } catch { /* the list is a convenience; ignore failures */ }
}

$("file").addEventListener("change", () => { $("file-label").textContent = $("file").files[0]?.name ?? "Choose a PDF or image"; });

$("upload").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("file").files[0];
  if (!file) return;
  const button = e.submitter;
  button.disabled = true;
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("language", $("language").value);
    const { doc_id } = await api("/api/documents", { method: "POST", body: form });
    for (const id of ["overview", "fields-panel", "chat-panel"]) $(id).hidden = true;
    $("viewer").replaceChildren();
    openDoc(doc_id);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    button.disabled = false;
  }
});

$("docs").addEventListener("change", () => { if ($("docs").value) openDoc($("docs").value); });

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question || !current) return;
  const log = $("chat-log");
  log.append(el("div", { className: "msg q", textContent: question, dir: "auto" }));
  $("question").value = "";
  const pending = el("div", { className: "msg a", textContent: "…" });
  log.append(pending);
  try {
    const ans = await api(`/api/documents/${current.doc_id}/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }),
    });
    pending.replaceChildren(el("div", { textContent: ans.answer, dir: "auto" }));
    if (ans.found && ans.boxes.length) {
      const cite = el("button", { type: "button", className: "cite", textContent: `Show source (${ans.line_ids.length} lines)` });
      cite.addEventListener("click", () => highlight(ans.boxes));
      pending.append(cite);
      highlight(ans.boxes);
    }
  } catch (err) {
    pending.textContent = err.message;
  }
});

refreshList();
