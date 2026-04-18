"""Inline HTML/CSS/JS page served by WebInputServer.

Kept as a module-level string to avoid shipping a separate static file.
Two placeholders substituted at serve time:
- ``{token}`` — random session token, required on /prompt and /submit.
- ``{title}`` — browser tab title.
"""

from __future__ import annotations

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --bg: #0f1116; --fg: #e6e6e6; --muted: #9aa4b2;
    --panel: #161a22; --border: #2a3140; --accent: #8ab4ff;
    --ok: #7ee787; --warn: #f0883e;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg: #ffffff; --fg: #1f2328; --muted: #57606a;
      --panel: #f6f8fa; --border: #d0d7de; --accent: #0969da;
      --ok: #116329; --warn: #9a6700;
    }}
  }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--fg);
                font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width: 860px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.25rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border);
            border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
  .panel h2 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;
              color: var(--muted); margin: 0 0 0.5rem; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; margin: 0;
         font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         font-size: 0.9rem; }}
  textarea {{ width: 100%; box-sizing: border-box; min-height: 6rem;
              background: var(--bg); color: var(--fg); border: 1px solid var(--border);
              border-radius: 6px; padding: 0.6rem; font-family: ui-monospace, monospace;
              font-size: 0.95rem; }}
  .row {{ display: flex; gap: 0.5rem; align-items: center; margin-top: 0.75rem; flex-wrap: wrap; }}
  button {{ background: var(--accent); color: white; border: 0; border-radius: 6px;
            padding: 0.5rem 1rem; font-size: 0.95rem; cursor: pointer; }}
  button.secondary {{ background: var(--panel); color: var(--fg); border: 1px solid var(--border); }}
  button:disabled {{ opacity: 0.6; cursor: not-allowed; }}
  .chip {{ display: inline-block; padding: 0.2rem 0.55rem; margin: 0.15rem 0.25rem 0.15rem 0;
           border: 1px solid var(--border); border-radius: 999px; background: var(--bg);
           color: var(--fg); font-size: 0.8rem; cursor: pointer; }}
  .chip:hover {{ border-color: var(--accent); color: var(--accent); }}
  .status {{ font-size: 0.85rem; color: var(--muted); }}
  .status.ok {{ color: var(--ok); }}
  .status.warn {{ color: var(--warn); }}
  .waiting {{ padding: 2rem 1rem; text-align: center; color: var(--muted); font-style: italic; }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <div class="subtitle">Local web input for LazyBridge agents — keep this tab open.</div>

  <div id="app">
    <div class="waiting">Waiting for the next prompt...</div>
  </div>

  <div class="status" id="status">Connecting to server…</div>
</main>

<script>
(function() {{
  const TOKEN = {token_json};
  const app = document.getElementById("app");
  const status = document.getElementById("status");
  let currentSeq = -1;
  let submitting = false;
  let pollTimer = null;

  function setStatus(msg, cls) {{
    status.textContent = msg;
    status.className = "status" + (cls ? " " + cls : "");
  }}

  function escapeHtml(s) {{
    return s.replace(/[&<>"']/g, c => ({{
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;"
    }})[c]);
  }}

  function render(p) {{
    currentSeq = p.seq;
    const quick = (p.quick_commands || []).map(c =>
      `<span class="chip" data-cmd="${{escapeHtml(c)}}">${{escapeHtml(c)}}</span>`
    ).join("");
    app.innerHTML = `
      <div class="panel">
        <h2>Prompt</h2>
        <pre>${{escapeHtml(p.prompt || "")}}</pre>
      </div>
      ${{quick ? `<div class="panel"><h2>Quick commands</h2>${{quick}}</div>` : ""}}
      <div class="panel">
        <h2>Your response</h2>
        <textarea id="response" autofocus></textarea>
        <div class="row">
          <button id="submit">Submit</button>
          <button class="secondary" id="clear">Clear</button>
          <span class="status" id="hint">Ctrl/⌘-Enter to submit</span>
        </div>
      </div>
    `;

    const ta = document.getElementById("response");
    const submitBtn = document.getElementById("submit");
    document.getElementById("clear").onclick = () => {{ ta.value = ""; ta.focus(); }};
    submitBtn.onclick = () => submit(ta.value);
    ta.addEventListener("keydown", (e) => {{
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {{
        e.preventDefault(); submit(ta.value);
      }}
    }});
    document.querySelectorAll(".chip").forEach(el => {{
      el.addEventListener("click", () => {{
        const existing = ta.value.trim();
        ta.value = existing ? existing + "\\n" + el.dataset.cmd : el.dataset.cmd;
        ta.focus();
      }});
    }});
  }}

  function renderWaiting(msg) {{
    app.innerHTML = `<div class="waiting">${{escapeHtml(msg || "Waiting for the next prompt...")}}</div>`;
  }}

  async function submit(text) {{
    if (submitting) return;
    submitting = true;
    const btn = document.getElementById("submit");
    if (btn) btn.disabled = true;
    setStatus("Submitting…");
    try {{
      const r = await fetch("/submit?t=" + encodeURIComponent(TOKEN), {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ seq: currentSeq, response: text }}),
      }});
      if (!r.ok) {{
        setStatus("Submit rejected: " + r.status, "warn");
        if (btn) btn.disabled = false;
        submitting = false;
        return;
      }}
      setStatus("Submitted ✓ — waiting for next prompt…", "ok");
      renderWaiting("Submitted. Waiting for the next prompt…");
    }} catch (e) {{
      setStatus("Submit failed: " + e, "warn");
      if (btn) btn.disabled = false;
    }} finally {{
      submitting = false;
    }}
  }}

  async function poll() {{
    try {{
      const r = await fetch("/prompt?t=" + encodeURIComponent(TOKEN));
      if (!r.ok) {{
        setStatus("Server error: " + r.status, "warn");
        return;
      }}
      const data = await r.json();
      if (data.closed) {{
        setStatus("Server closed. You can shut this tab.", "warn");
        renderWaiting("Session ended.");
        if (pollTimer) {{ clearInterval(pollTimer); pollTimer = null; }}
        return;
      }}
      if (data.prompt !== null && data.seq !== currentSeq) {{
        render(data);
        setStatus("New prompt received.", "ok");
      }} else if (data.prompt === null) {{
        setStatus("Idle — waiting for next prompt…");
      }}
    }} catch (e) {{
      setStatus("Connection error: " + e, "warn");
    }}
  }}

  pollTimer = setInterval(poll, 500);
  poll();
}})();
</script>
</body>
</html>
"""
