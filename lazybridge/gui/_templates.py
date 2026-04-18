"""Inlined HTML/CSS/JS shell for the shared GuiServer.

One page, sidebar + main area.  Polls ``/api/panels`` for the list and
fetches ``/api/panel/<id>`` when a panel is selected.  Each panel kind is
rendered by a dedicated JS function below; adding a new kind only requires
an entry in ``PANEL_RENDERERS`` and a matching server-side ``Panel``
subclass.
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
    --ok: #7ee787; --warn: #f0883e; --bad: #ff6b6b;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg: #ffffff; --fg: #1f2328; --muted: #57606a;
      --panel: #f6f8fa; --border: #d0d7de; --accent: #0969da;
      --ok: #116329; --warn: #9a6700; --bad: #cf222e;
    }}
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--fg);
                font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
                height: 100vh; }}
  body {{ display: grid; grid-template-columns: 240px 1fr; }}
  aside {{ border-right: 1px solid var(--border); overflow-y: auto; padding: 1rem; }}
  aside h1 {{ font-size: 1rem; margin: 0 0 1rem; }}
  aside h2 {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
             color: var(--muted); margin: 1rem 0 0.25rem; }}
  aside ul {{ list-style: none; padding: 0; margin: 0; }}
  aside li {{ padding: 0.35rem 0.5rem; border-radius: 4px; cursor: pointer;
             font-size: 0.88rem; }}
  aside li:hover {{ background: var(--panel); }}
  aside li.active {{ background: var(--accent); color: white; }}
  aside .empty {{ color: var(--muted); font-size: 0.8rem; font-style: italic; }}
  main {{ overflow-y: auto; padding: 1.5rem 2rem; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border);
            border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
  .panel h2 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;
              color: var(--muted); margin: 0 0 0.75rem; }}
  .panel h3 {{ font-size: 0.95rem; margin: 0 0 0.5rem; }}
  .row {{ display: flex; gap: 0.5rem; align-items: center; margin: 0.5rem 0; flex-wrap: wrap; }}
  label {{ display: block; font-size: 0.8rem; color: var(--muted); margin-top: 0.75rem; }}
  input[type=text], input[type=number], textarea, select {{
    width: 100%; background: var(--bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 0.45rem 0.6rem; font-size: 0.92rem;
    font-family: inherit;
  }}
  textarea {{ min-height: 5rem; font-family: ui-monospace, monospace; font-size: 0.9rem; }}
  textarea.large {{ min-height: 9rem; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; margin: 0;
         font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         font-size: 0.88rem; background: var(--bg); padding: 0.75rem;
         border: 1px solid var(--border); border-radius: 6px;
         max-height: 30rem; overflow-y: auto; }}
  button {{ background: var(--accent); color: white; border: 0; border-radius: 6px;
            padding: 0.45rem 0.9rem; font-size: 0.92rem; cursor: pointer; }}
  button.secondary {{ background: var(--panel); color: var(--fg); border: 1px solid var(--border); }}
  button:disabled {{ opacity: 0.6; cursor: not-allowed; }}
  .status {{ font-size: 0.82rem; color: var(--muted); }}
  .status.ok {{ color: var(--ok); }}
  .status.warn {{ color: var(--warn); }}
  .status.bad {{ color: var(--bad); }}
  .empty-main {{ color: var(--muted); font-style: italic; padding: 3rem; text-align: center; }}
  .pill {{ display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
           border: 1px solid var(--border); font-size: 0.75rem; margin-right: 0.25rem;
           color: var(--muted); }}
  .checkbox-list label {{ display: flex; align-items: center; gap: 0.5rem;
                          margin: 0.25rem 0; font-size: 0.9rem; color: var(--fg);
                          cursor: pointer; }}
  .tabs {{ display: flex; gap: 0.25rem; border-bottom: 1px solid var(--border);
          margin: 1rem 0 0.75rem; }}
  .tab {{ padding: 0.5rem 0.8rem; cursor: pointer; font-size: 0.9rem;
         border-bottom: 2px solid transparent; }}
  .tab.active {{ border-color: var(--accent); color: var(--accent); font-weight: 500; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
<aside>
  <h1>LazyBridge GUI</h1>
  <div id="sidebar"></div>
  <div class="status" id="sb-status" style="margin-top:1rem">Loading…</div>
</aside>

<main id="main">
  <div class="empty-main">Select a panel from the sidebar.</div>
</main>

<script>
  // Bootstrap only — the full client lives in /static/app.js so it gets
  // IDE syntax highlighting and isn't tangled up in Python f-string
  // curly-brace escaping (audit L6).
  window.LB_TOKEN = {token_json};
</script>
<script src="/static/app.js?t={token_raw}"></script>
<script data-disabled="original-inline-client" type="text/plain">
(function() {{
  const TOKEN = {token_json};
  const sidebar = document.getElementById("sidebar");
  const sbStatus = document.getElementById("sb-status");
  const main = document.getElementById("main");

  let activePanelId = null;
  let panelList = [];

  function qs(path, params) {{
    const q = new URLSearchParams({{t: TOKEN, ...(params || {{}})}});
    return path + "?" + q.toString();
  }}

  async function apiGet(path) {{
    const r = await fetch(qs(path));
    if (!r.ok) throw new Error("GET " + path + " -> " + r.status);
    return r.json();
  }}

  async function apiPost(path, body) {{
    const r = await fetch(qs(path), {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(body || {{}}),
    }});
    if (!r.ok) {{
      const txt = await r.text();
      throw new Error("POST " + path + " -> " + r.status + ": " + txt);
    }}
    return r.json();
  }}

  function escapeHtml(s) {{
    if (s === null || s === undefined) return "";
    return String(s).replace(/[&<>"']/g, c => ({{
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;"
    }})[c]);
  }}

  function renderSidebar(panels) {{
    const groups = {{}};
    for (const p of panels) {{
      groups[p.group] = groups[p.group] || [];
      groups[p.group].push(p);
    }}
    let html = "";
    if (panels.length === 0) {{
      html = '<div class="empty">No panels registered yet. Call <code>.gui()</code> on an agent, tool, or session.</div>';
    }} else {{
      for (const [group, items] of Object.entries(groups)) {{
        html += `<h2>${{escapeHtml(group)}}</h2><ul>`;
        for (const p of items) {{
          const cls = p.id === activePanelId ? "active" : "";
          html += `<li class="${{cls}}" data-id="${{escapeHtml(p.id)}}">${{escapeHtml(p.label)}}</li>`;
        }}
        html += "</ul>";
      }}
    }}
    sidebar.innerHTML = html;
    sidebar.querySelectorAll("li[data-id]").forEach(li => {{
      li.addEventListener("click", () => loadPanel(li.dataset.id));
    }});
  }}

  async function refreshPanelList() {{
    try {{
      const data = await apiGet("/api/panels");
      panelList = data.panels;
      renderSidebar(panelList);
      sbStatus.textContent = panelList.length + " panel(s)";
      sbStatus.className = "status ok";
      // Human panels change state server-side (prompt arrives without any
      // client action).  Pipeline and agent panels do the same while a
      // run is in flight (label ends with "· running").  Auto-refresh
      // the active one on every poll in those cases.
      if (activePanelId) {{
        const active = panelList.find(p => p.id === activePanelId);
        if (active && (active.kind === "human"
                       || ((active.kind === "pipeline" || active.kind === "agent")
                           && active.label.endsWith("· running")))) {{
          loadPanel(activePanelId);
        }}
      }}
    }} catch (e) {{
      sbStatus.textContent = "Connection error: " + e.message;
      sbStatus.className = "status bad";
    }}
  }}

  async function loadPanel(id) {{
    activePanelId = id;
    renderSidebar(panelList);
    main.innerHTML = '<div class="empty-main">Loading…</div>';
    try {{
      const state = await apiGet("/api/panel/" + encodeURIComponent(id));
      const renderer = PANEL_RENDERERS[state.kind] || renderGeneric;
      main.innerHTML = "";
      renderer(main, state);
    }} catch (e) {{
      main.innerHTML = '<div class="empty-main status bad">Failed to load: ' + escapeHtml(e.message) + '</div>';
    }}
  }}

  async function action(panelId, actionName, args) {{
    return apiPost("/api/panel/" + encodeURIComponent(panelId) + "/action",
                   {{action: actionName, args: args || {{}}}});
  }}

  // ------------------------------------------------------------------
  // Panel renderers
  // ------------------------------------------------------------------

  function makeSection(title) {{
    const div = document.createElement("div");
    div.className = "panel";
    div.innerHTML = `<h2>${{escapeHtml(title)}}</h2>`;
    return div;
  }}

  function renderGeneric(root, state) {{
    const div = makeSection(state.kind + " — " + state.label);
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(state, null, 2);
    div.appendChild(pre);
    root.appendChild(div);
  }}

  function renderAgent(root, state) {{
    // ---------- Inspect / Edit ----------
    const inspect = makeSection("Inspect / edit — " + state.label);
    inspect.innerHTML += `
      <label>Name (read-only)</label>
      <input type="text" value="${{escapeHtml(state.name)}}" disabled>
      <label>Provider (read-only)</label>
      <input type="text" value="${{escapeHtml(state.provider)}}" disabled>
      <label>Model</label>
      <input type="text" id="agent-model" value="${{escapeHtml(state.model || "")}}">
      <div class="row">
        <button id="agent-save-model">Save model</button>
        <span class="status" id="agent-model-status"></span>
      </div>
      <label>System prompt</label>
      <textarea id="agent-system" class="large">${{escapeHtml(state.system || "")}}</textarea>
      <div class="row">
        <button id="agent-save-system">Save system prompt</button>
        <span class="status" id="agent-system-status"></span>
      </div>
      <label>Tools enabled on this agent</label>
      <div class="checkbox-list" id="agent-tools"></div>
      <span class="status" id="agent-tools-status"></span>
      <label>Provider-native tools</label>
      <div class="checkbox-list" id="agent-native-tools"></div>
      <span class="status" id="agent-native-tools-status"></span>
    `;
    root.appendChild(inspect);

    // Native tools
    const nativeBox = inspect.querySelector("#agent-native-tools");
    const nativeEnabled = new Set(state.native_tools || []);
    const nativeAvail = state.available_native_tools || [];
    if (nativeAvail.length === 0) {{
      nativeBox.innerHTML = '<div class="status">No provider-native tools available in this build.</div>';
    }} else {{
      for (const n of nativeAvail) {{
        const checked = nativeEnabled.has(n) ? "checked" : "";
        nativeBox.innerHTML += `<label><input type="checkbox" data-native="${{escapeHtml(n)}}" ${{checked}}>
          <span><strong>${{escapeHtml(n)}}</strong></span>
        </label>`;
      }}
    }}
    nativeBox.addEventListener("change", async (e) => {{
      if (e.target.tagName !== "INPUT") return;
      const name = e.target.dataset.native;
      const checked = e.target.checked;
      const st = inspect.querySelector("#agent-native-tools-status");
      st.textContent = "Updating…"; st.className = "status";
      try {{
        await action(state.id, "toggle_native_tool", {{name, enabled: checked}});
        st.textContent = (checked ? "Enabled " : "Disabled ") + name; st.className = "status ok";
      }} catch (err) {{
        st.textContent = "Failed: " + err.message; st.className = "status bad";
        e.target.checked = !checked;
      }}
    }});

    // Export-as-Python button — puts the current agent state into a snippet
    // the user can paste into code.  Addresses the "GUI edits die with the
    // process" persistence gap.
    const exportSection = makeSection("Export as Python");
    exportSection.innerHTML += `
      <div class="row">
        <button id="agent-export">Generate snippet</button>
        <button id="agent-copy" class="secondary" disabled>Copy</button>
        <span class="status" id="agent-export-status"></span>
      </div>
      <pre id="agent-snippet">—</pre>
    `;
    root.appendChild(exportSection);

    exportSection.querySelector("#agent-export").addEventListener("click", async () => {{
      const st = exportSection.querySelector("#agent-export-status");
      const out = exportSection.querySelector("#agent-snippet");
      const copyBtn = exportSection.querySelector("#agent-copy");
      st.textContent = "Generating…"; st.className = "status";
      try {{
        const res = await action(state.id, "export_python", {{}});
        out.textContent = res.snippet;
        copyBtn.disabled = false;
        st.textContent = "Snippet ready"; st.className = "status ok";
      }} catch (e) {{
        out.textContent = e.message;
        st.textContent = "Failed"; st.className = "status bad";
      }}
    }});
    exportSection.querySelector("#agent-copy").addEventListener("click", async () => {{
      const st = exportSection.querySelector("#agent-export-status");
      const text = exportSection.querySelector("#agent-snippet").textContent;
      try {{
        await navigator.clipboard.writeText(text);
        st.textContent = "Copied to clipboard ✓"; st.className = "status ok";
      }} catch (e) {{
        st.textContent = "Copy failed: " + e.message; st.className = "status bad";
      }}
    }});

    inspect.querySelector("#agent-save-model").addEventListener("click", async () => {{
      const val = inspect.querySelector("#agent-model").value.trim();
      const st = inspect.querySelector("#agent-model-status");
      if (!val) {{ st.textContent = "Model is empty"; st.className = "status warn"; return; }}
      st.textContent = "Saving…"; st.className = "status";
      try {{
        await action(state.id, "update_model", {{value: val}});
        st.textContent = "Saved ✓"; st.className = "status ok";
      }} catch (e) {{
        st.textContent = "Save failed: " + e.message; st.className = "status bad";
      }}
    }});

    // Tools checklist
    const toolsBox = inspect.querySelector("#agent-tools");
    const enabled = new Set((state.tools || []).map(t => t.name));
    const scope = state.available_tools || [];
    if (scope.length === 0) {{
      toolsBox.innerHTML = '<div class="status">No tools in scope. Add tools to agents in the same session to see them here.</div>';
    }} else {{
      for (const t of scope) {{
        const id = "tool-opt-" + t.name;
        const checked = enabled.has(t.name) ? "checked" : "";
        toolsBox.innerHTML += `<label><input type="checkbox" id="${{id}}" data-name="${{escapeHtml(t.name)}}" ${{checked}}>
          <span><strong>${{escapeHtml(t.name)}}</strong> — <span class="status">${{escapeHtml(t.description || "")}}</span></span>
        </label>`;
      }}
    }}

    inspect.querySelector("#agent-save-system").addEventListener("click", async () => {{
      const val = inspect.querySelector("#agent-system").value;
      const st = inspect.querySelector("#agent-system-status");
      st.textContent = "Saving…"; st.className = "status";
      try {{
        await action(state.id, "update_system", {{value: val}});
        st.textContent = "Saved ✓"; st.className = "status ok";
      }} catch (e) {{
        st.textContent = "Save failed: " + e.message; st.className = "status bad";
      }}
    }});

    toolsBox.addEventListener("change", async (e) => {{
      if (e.target.tagName !== "INPUT") return;
      const name = e.target.dataset.name;
      const checked = e.target.checked;
      const st = inspect.querySelector("#agent-tools-status");
      st.textContent = "Updating…"; st.className = "status";
      try {{
        await action(state.id, "toggle_tool", {{name, enabled: checked}});
        st.textContent = (checked ? "Enabled " : "Disabled ") + name; st.className = "status ok";
      }} catch (err) {{
        st.textContent = "Failed: " + err.message; st.className = "status bad";
        e.target.checked = !checked;
      }}
    }});

    // ---------- Test (async, non-blocking) ----------
    const test = makeSection("Test — runs live against " + state.provider);
    test.innerHTML += `
      <label>Mode</label>
      <select id="agent-mode">
        <option value="chat">chat (single turn)</option>
        <option value="loop">loop (tool-calling loop)</option>
        <option value="text">text (chat.content shortcut)</option>
      </select>
      <label>Message</label>
      <textarea id="agent-msg" class="large" placeholder="Write a prompt, then press Run…"></textarea>
      <div class="row">
        <button id="agent-run">Run</button>
        <button id="agent-clear" class="secondary">Clear last result</button>
        <span class="status" id="agent-run-status"></span>
      </div>
      <label>Response</label>
      <pre id="agent-response" class="status">—</pre>
    `;
    root.appendChild(test);
    const lastTest = state.last_test || null;
    const btnRun = test.querySelector("#agent-run");
    const btnClear = test.querySelector("#agent-clear");
    const st = test.querySelector("#agent-run-status");
    const out = test.querySelector("#agent-response");
    const modeEl = test.querySelector("#agent-mode");
    const msgEl = test.querySelector("#agent-msg");
    if (lastTest) {{
      if (lastTest.message) msgEl.value = lastTest.message;
      if (lastTest.mode) modeEl.value = lastTest.mode;
      if (lastTest.status === "running") {{
        btnRun.disabled = true;
        st.textContent = "Running " + lastTest.mode + "… (non-blocking)";
        st.className = "status";
        out.textContent = "…";
      }} else if (lastTest.status === "done") {{
        const dt = ((lastTest.finished_at - lastTest.started_at)).toFixed(1);
        const u = lastTest.usage || {{}};
        const cost = u.cost_usd != null ? ` • $${{u.cost_usd.toFixed(6)}}` : "";
        st.textContent = `Done in ${{dt}}s • ${{u.input_tokens || 0}} in / ${{u.output_tokens || 0}} out${{cost}}`;
        st.className = "status ok";
        out.textContent = lastTest.content
          || (lastTest.parsed != null ? JSON.stringify(lastTest.parsed, null, 2) : "—");
      }} else if (lastTest.status === "error") {{
        st.textContent = "Failed"; st.className = "status bad";
        out.textContent = lastTest.error || "(no error message)";
      }}
    }}
    btnRun.addEventListener("click", async () => {{
      const mode = modeEl.value;
      const msg = msgEl.value;
      if (!msg.trim()) {{ st.textContent = "Message is empty"; st.className = "status warn"; return; }}
      btnRun.disabled = true;
      st.textContent = "Starting…"; st.className = "status";
      out.textContent = "…";
      try {{
        await action(state.id, "test", {{mode, message: msg}});
        // Refresh immediately so the running state is visible; SSE and
        // auto-refresh keep it up to date until completion.
        loadPanel(state.id);
      }} catch (e) {{
        st.textContent = "Start failed: " + e.message; st.className = "status bad";
        btnRun.disabled = false;
      }}
    }});
    btnClear.addEventListener("click", async () => {{
      try {{
        await action(state.id, "clear_test", {{}});
        loadPanel(state.id);
      }} catch (e) {{ st.textContent = e.message; st.className = "status bad"; }}
    }});
  }}

  function renderTool(root, state) {{
    const inspect = makeSection("Inspect — " + state.label);
    inspect.innerHTML += `
      <label>Name</label><input type="text" value="${{escapeHtml(state.name)}}" disabled>
      <label>Description</label><textarea disabled>${{escapeHtml(state.description || "")}}</textarea>
      ${{state.guidance ? `<label>Guidance (for LLM)</label><textarea disabled>${{escapeHtml(state.guidance)}}</textarea>` : ""}}
      <label>Schema mode</label><input type="text" value="${{escapeHtml(state.schema_mode || "")}}" disabled>
      <div class="row">
        ${{state.is_pipeline_tool ? '<span class="pill">pipeline tool</span>' : ""}}
        ${{state.is_delegate ? '<span class="pill">from_agent</span>' : ""}}
      </div>
      <label>JSON Schema</label>
      <pre>${{escapeHtml(JSON.stringify(state.parameters || {{}}, null, 2))}}</pre>
    `;
    root.appendChild(inspect);

    // --------- Test ---------
    const test = makeSection("Test — invoke the tool");
    const props = (state.parameters && state.parameters.properties) || {{}};
    const required = (state.parameters && state.parameters.required) || [];
    let fieldsHtml = "";
    if (Object.keys(props).length === 0) {{
      fieldsHtml = '<div class="status">No parameters — press Invoke to run with an empty payload.</div>';
    }} else {{
      for (const [k, schema] of Object.entries(props)) {{
        const typ = schema.type || "string";
        const req = required.includes(k) ? " *" : "";
        let field;
        if (typ === "boolean") {{
          field = `<input type="checkbox" data-name="${{escapeHtml(k)}}" data-jtype="boolean">`;
        }} else if (typ === "integer" || typ === "number") {{
          field = `<input type="number" step="${{typ === 'integer' ? '1' : 'any'}}" data-name="${{escapeHtml(k)}}" data-jtype="${{typ}}">`;
        }} else if (typ === "array" || typ === "object") {{
          field = `<textarea data-name="${{escapeHtml(k)}}" data-jtype="${{typ}}" placeholder="JSON ${{typ}}…"></textarea>`;
        }} else {{
          field = `<input type="text" data-name="${{escapeHtml(k)}}" data-jtype="string">`;
        }}
        const desc = schema.description ? ` — <span class="status">${{escapeHtml(schema.description)}}</span>` : "";
        fieldsHtml += `<label>${{escapeHtml(k)}}${{req}} <span class="status">(${{escapeHtml(typ)}})</span>${{desc}}</label>${{field}}`;
      }}
    }}
    test.innerHTML += fieldsHtml + `
      <div class="row">
        <button id="tool-invoke">Invoke</button>
        <span class="status" id="tool-invoke-status"></span>
      </div>
      <label>Result</label>
      <pre id="tool-result">—</pre>
    `;
    root.appendChild(test);

    test.querySelector("#tool-invoke").addEventListener("click", async () => {{
      const args = {{}};
      test.querySelectorAll("[data-name]").forEach(el => {{
        const k = el.dataset.name;
        const jtype = el.dataset.jtype;
        let v;
        if (jtype === "boolean") v = el.checked;
        else if (jtype === "integer") v = el.value === "" ? null : parseInt(el.value, 10);
        else if (jtype === "number") v = el.value === "" ? null : parseFloat(el.value);
        else if (jtype === "array" || jtype === "object") {{
          const raw = el.value.trim();
          v = raw === "" ? null : JSON.parse(raw);
        }} else {{
          v = el.value;
        }}
        if (v !== null && v !== "") args[k] = v;
      }});
      const btn = test.querySelector("#tool-invoke");
      const st = test.querySelector("#tool-invoke-status");
      const out = test.querySelector("#tool-result");
      btn.disabled = true;
      st.textContent = "Invoking…"; st.className = "status";
      out.textContent = "…";
      const t0 = Date.now();
      try {{
        const res = await action(state.id, "invoke", {{args}});
        const dt = ((Date.now() - t0) / 1000).toFixed(2);
        out.textContent = typeof res.result === "string" ? res.result : JSON.stringify(res.result, null, 2);
        st.textContent = "Done in " + dt + "s"; st.className = "status ok";
      }} catch (e) {{
        out.textContent = e.message;
        st.textContent = "Failed"; st.className = "status bad";
      }} finally {{
        btn.disabled = false;
      }}
    }});
  }}

  function renderPipeline(root, state) {{
    const inspect = makeSection("Pipeline — " + state.label);
    const tags = [];
    if (state.mode) tags.push(`<span class="pill">${{escapeHtml(state.mode)}}</span>`);
    if (state.combiner) tags.push(`<span class="pill">combiner: ${{escapeHtml(state.combiner)}}</span>`);
    if (state.concurrency_limit != null) tags.push(`<span class="pill">concurrency ≤ ${{state.concurrency_limit}}</span>`);
    if (state.step_timeout != null) tags.push(`<span class="pill">step timeout ${{state.step_timeout}}s</span>`);
    inspect.innerHTML += `
      <div class="row">${{tags.join(" ")}}</div>
      <label>Description</label>
      <textarea disabled>${{escapeHtml(state.description || "")}}</textarea>
      ${{state.guidance ? `<label>Guidance</label><textarea disabled>${{escapeHtml(state.guidance)}}</textarea>` : ""}}
      <label>Participants (${{(state.participants || []).length}})</label>
    `;
    const ul = document.createElement("ul");
    ul.style.paddingLeft = "1rem";
    for (const p of state.participants || []) {{
      const li = document.createElement("li");
      li.style.margin = "0.2rem 0";
      const tag = p.kind === "agent"
        ? `<span class="pill">agent</span> <strong>${{escapeHtml(p.name)}}</strong>
           <span class="status">${{escapeHtml(p.provider || "")}}/${{escapeHtml(p.model || "")}}</span>`
        : `<span class="pill">${{escapeHtml(p.kind)}}</span> <strong>${{escapeHtml(p.name)}}</strong>`;
      li.innerHTML = tag;
      if (p.panel_id) {{
        li.style.cursor = "pointer";
        li.addEventListener("click", () => loadPanel(p.panel_id));
        li.title = "Open " + p.panel_id;
      }}
      ul.appendChild(li);
    }}
    inspect.appendChild(ul);
    root.appendChild(inspect);

    // --------- Test ---------
    const test = makeSection("Test — runs the pipeline live");
    test.innerHTML += `
      <label>Task</label>
      <textarea id="pipeline-task" class="large" placeholder="Write the initial task and hit Run…"></textarea>
      <div class="row">
        <button id="pipeline-run">Run</button>
        <button id="pipeline-clear" class="secondary">Clear last run</button>
        <span class="status" id="pipeline-status"></span>
      </div>
      <label>Progress</label>
      <div id="pipeline-timeline" class="panel" style="margin:0;background:var(--bg)">
        <div class="status">No run yet.</div>
      </div>
      <label>Result</label>
      <pre id="pipeline-result">—</pre>
    `;
    root.appendChild(test);

    // Render the "last_run" payload attached to the state (if any).
    const lastRun = state.last_run || null;
    const participantNames = state.participant_names || [];
    renderPipelineTimeline(test.querySelector("#pipeline-timeline"), participantNames, lastRun);
    const resultPre = test.querySelector("#pipeline-result");
    const statusEl = test.querySelector("#pipeline-status");
    if (lastRun) {{
      resultPre.textContent = lastRun.result != null
        ? (typeof lastRun.result === "string" ? lastRun.result : JSON.stringify(lastRun.result, null, 2))
        : (lastRun.error || "…");
      if (lastRun.status === "running") {{
        statusEl.textContent = "Running…"; statusEl.className = "status";
      }} else if (lastRun.status === "done") {{
        const dt = ((lastRun.finished_at - lastRun.started_at)).toFixed(1);
        statusEl.textContent = "Done in " + dt + "s"; statusEl.className = "status ok";
      }} else if (lastRun.status === "error") {{
        statusEl.textContent = "Failed"; statusEl.className = "status bad";
      }}
    }}

    test.querySelector("#pipeline-run").addEventListener("click", async () => {{
      const task = test.querySelector("#pipeline-task").value;
      if (!task.trim()) {{
        statusEl.textContent = "Task is empty"; statusEl.className = "status warn";
        return;
      }}
      statusEl.textContent = "Starting…"; statusEl.className = "status";
      resultPre.textContent = "…";
      try {{
        await action(state.id, "run", {{task}});
        // Refresh once so the running state becomes visible right away;
        // subsequent refreshes happen via the auto-refresh hook below.
        loadPanel(state.id);
      }} catch (e) {{
        resultPre.textContent = e.message;
        statusEl.textContent = "Start failed"; statusEl.className = "status bad";
      }}
    }});
    test.querySelector("#pipeline-clear").addEventListener("click", async () => {{
      try {{
        await action(state.id, "clear_run", {{}});
        loadPanel(state.id);
      }} catch (e) {{ statusEl.textContent = e.message; statusEl.className = "status bad"; }}
    }});
  }}

  // Render a per-participant timeline from the last_run event buffer.
  function renderPipelineTimeline(root, names, lastRun) {{
    if (!lastRun) {{
      root.innerHTML = '<div class="status">No run yet.</div>';
      return;
    }}
    const events = lastRun.events || [];
    // Build a map: agent_name → {{ started_at, finished_at, stop_reason, n_steps, status }}
    const byAgent = {{}};
    for (const n of names) {{
      byAgent[n] = {{ name: n, status: "pending" }};
    }}
    for (const ev of events) {{
      const bucket = byAgent[ev.agent_name] || (byAgent[ev.agent_name] = {{ name: ev.agent_name }});
      if (ev.event_type === "agent_start") {{
        bucket.started_at = ev.ts;
        bucket.status = bucket.status === "finished" ? "restarted" : "running";
      }} else if (ev.event_type === "agent_finish") {{
        bucket.finished_at = ev.ts;
        bucket.status = "finished";
        if (ev.stop_reason !== undefined) bucket.stop_reason = ev.stop_reason;
        if (ev.n_steps !== undefined) bucket.n_steps = ev.n_steps;
      }}
    }}
    const rows = Object.values(byAgent).map(b => {{
      const dur = (b.started_at && b.finished_at) ? (b.finished_at - b.started_at).toFixed(1) + "s" : "";
      const icon = b.status === "finished" ? "✓" : b.status === "running" ? "…" : "·";
      const cls = b.status === "finished" ? "ok" : b.status === "running" ? "" : "";
      const extras = [dur, b.stop_reason, b.n_steps != null ? (b.n_steps + " step(s)") : null]
        .filter(x => x).map(x => `<span class="status">${{escapeHtml(String(x))}}</span>`).join(" ");
      return `<div class="row" style="margin:0.15rem 0">
        <span style="font-family:ui-monospace,monospace;min-width:1.5rem">${{icon}}</span>
        <strong>${{escapeHtml(b.name)}}</strong>
        <span class="status ${{cls}}">${{escapeHtml(b.status)}}</span>
        ${{extras}}
      </div>`;
    }}).join("");
    const captured = lastRun.captured_from_session ? "" :
      '<div class="status warn" style="margin-top:0.5rem">Session-less pipeline: per-step timeline unavailable, showing final result only.</div>';
    root.innerHTML = rows + captured;
  }}

  function renderSession(root, state) {{
    const inspect = makeSection("Session — " + state.label);
    inspect.innerHTML += `
      <label>ID</label><input type="text" value="${{escapeHtml(state.id)}}" disabled>
      <label>Tracking</label><input type="text" value="${{escapeHtml(state.tracking)}}" disabled>
      <label>Agents (${{(state.agents || []).length}})</label>
      <ul>${{(state.agents || []).map(a => `<li>${{escapeHtml(a.name)}} <span class="status">${{escapeHtml(a.provider)}}/${{escapeHtml(a.model)}}</span></li>`).join("") || "<li class=\\"status\\">none</li>"}}</ul>
      <label>Store keys (${{(state.store_keys || []).length}})</label>
      <pre>${{escapeHtml((state.store_keys || []).join("\\n") || "—")}}</pre>
    `;
    root.appendChild(inspect);
  }}

  function renderHuman(root, state) {{
    const inspect = makeSection("Human input — " + state.name);
    if (state.closed) {{
      inspect.innerHTML += '<div class="status warn">This human panel is closed.</div>';
      root.appendChild(inspect);
      return;
    }}
    if (state.prompt === null || state.prompt === undefined) {{
      inspect.innerHTML += '<div class="status">Idle — waiting for the next prompt…</div>';
      root.appendChild(inspect);
      return;
    }}
    const quick = (state.quick_commands || []).map(c =>
      `<span class="pill" data-cmd="${{escapeHtml(c)}}" style="cursor:pointer">${{escapeHtml(c)}}</span>`
    ).join(" ");
    inspect.innerHTML += `
      <label>Previous output</label>
      <pre>${{escapeHtml(state.prompt)}}</pre>
      ${{quick ? `<label>Quick commands</label><div class="row">${{quick}}</div>` : ""}}
      <label>Your response</label>
      <textarea id="human-response" class="large"></textarea>
      <div class="row">
        <button id="human-submit">Submit (⌘/Ctrl-Enter)</button>
        <span class="status" id="human-status">Seq ${{state.seq}}</span>
      </div>
    `;
    root.appendChild(inspect);
    const ta = inspect.querySelector("#human-response");
    ta.focus();
    ta.addEventListener("keydown", e => {{
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {{
        e.preventDefault();
        submitBtn.click();
      }}
    }});
    inspect.querySelectorAll("[data-cmd]").forEach(el => {{
      el.addEventListener("click", () => {{
        const existing = ta.value.trim();
        ta.value = existing ? existing + "\\n" + el.dataset.cmd : el.dataset.cmd;
        ta.focus();
      }});
    }});
    const submitBtn = inspect.querySelector("#human-submit");
    submitBtn.addEventListener("click", async () => {{
      const st = inspect.querySelector("#human-status");
      submitBtn.disabled = true;
      st.textContent = "Submitting…"; st.className = "status";
      try {{
        const res = await action(state.id, "submit", {{seq: state.seq, response: ta.value}});
        if (res.accepted) {{
          st.textContent = "Submitted ✓"; st.className = "status ok";
          // The panel will become idle again on the next poll.
          setTimeout(() => loadPanel(state.id), 400);
        }} else {{
          st.textContent = "Submission rejected (stale prompt)"; st.className = "status warn";
          submitBtn.disabled = false;
        }}
      }} catch (e) {{
        st.textContent = "Failed: " + e.message; st.className = "status bad";
        submitBtn.disabled = false;
      }}
    }});
  }}

  function renderRouter(root, state) {{
    const inspect = makeSection("Router — " + state.label);
    const rows = (state.routes || []).map(r => {{
      const prov = r.provider ? ` <span class="status">${{escapeHtml(r.provider)}}/${{escapeHtml(r.model || "")}}</span>` : "";
      const link = r.panel_id
        ? `<span class="pill" data-panel="${{escapeHtml(r.panel_id)}}" style="cursor:pointer">→ ${{escapeHtml(r.agent_name)}}</span>`
        : `<strong>${{escapeHtml(r.agent_name)}}</strong>`;
      return `<li><code>${{escapeHtml(r.key)}}</code> → ${{link}}${{prov}}</li>`;
    }}).join("");
    inspect.innerHTML += `
      <label>Default key</label>
      <input type="text" value="${{escapeHtml(state.default || "")}}" disabled>
      <label>Routes (${{(state.routes || []).length}})</label>
      <ul style="padding-left:1rem">${{rows || "<li class=\\"status\\">none</li>"}}</ul>
      <label>Condition</label>
      <pre>${{escapeHtml(state.condition || "")}}</pre>
    `;
    root.appendChild(inspect);
    inspect.querySelectorAll("[data-panel]").forEach(el => {{
      el.addEventListener("click", () => loadPanel(el.dataset.panel));
    }});

    // --------- Test ---------
    const test = makeSection("Test — route a value");
    test.innerHTML += `
      <label>Value to route</label>
      <input type="text" id="router-value" placeholder="value passed to router.route(…)">
      <div class="row">
        <button id="router-route">Which agent?</button>
        <button id="router-route-run" class="secondary">Route &amp; run</button>
        <span class="status" id="router-status"></span>
      </div>
      <label>Optional prompt (for Route &amp; run)</label>
      <textarea id="router-prompt" placeholder="Prompt passed to the routed agent's .chat()"></textarea>
      <label>Result</label>
      <pre id="router-result">—</pre>
    `;
    root.appendChild(test);

    test.querySelector("#router-route").addEventListener("click", async () => {{
      const value = test.querySelector("#router-value").value;
      const st = test.querySelector("#router-status");
      const out = test.querySelector("#router-result");
      st.textContent = "Routing…"; st.className = "status";
      out.textContent = "…";
      try {{
        const res = await action(state.id, "route", {{value}});
        out.textContent = JSON.stringify(res, null, 2);
        st.textContent = "Matched key: " + (res.matched_key || "(default)"); st.className = "status ok";
      }} catch (e) {{
        out.textContent = e.message;
        st.textContent = "Failed"; st.className = "status bad";
      }}
    }});

    test.querySelector("#router-route-run").addEventListener("click", async () => {{
      const value = test.querySelector("#router-value").value;
      const prompt = test.querySelector("#router-prompt").value;
      const st = test.querySelector("#router-status");
      const out = test.querySelector("#router-result");
      if (!prompt.trim()) {{ st.textContent = "Prompt is empty"; st.className = "status warn"; return; }}
      st.textContent = "Routing &amp; running…"; st.className = "status";
      out.textContent = "…";
      const t0 = Date.now();
      try {{
        const res = await action(state.id, "route_and_run", {{value, prompt}});
        const dt = ((Date.now() - t0) / 1000).toFixed(1);
        out.textContent = res.content || JSON.stringify(res, null, 2);
        const u = res.usage || {{}};
        const cost = u.cost_usd != null ? ` • $${{u.cost_usd.toFixed(6)}}` : "";
        st.textContent = `${{res.agent_name}} · ${{dt}}s · ${{u.input_tokens || 0}} in / ${{u.output_tokens || 0}} out${{cost}}`;
        st.className = "status ok";
      }} catch (e) {{
        out.textContent = e.message;
        st.textContent = "Failed"; st.className = "status bad";
      }}
    }});
  }}

  function renderStore(root, state) {{
    const inspect = makeSection("Store — " + state.label);
    const rows = (state.entries || []).map(e => `
      <tr>
        <td style="padding:0.25rem 0.75rem 0.25rem 0"><code>${{escapeHtml(e.key)}}</code></td>
        <td style="padding:0.25rem 0.75rem 0.25rem 0;max-width:32rem;word-break:break-word">${{escapeHtml(e.preview)}}</td>
        <td style="padding:0.25rem 0;color:var(--muted);font-size:0.75rem">${{escapeHtml(e.agent_id || "")}}</td>
      </tr>
    `).join("");
    inspect.innerHTML += `
      <label>Backend</label>
      <input type="text" value="${{escapeHtml(state.backend)}}" disabled>
      <label>Entries (${{state.key_count}})</label>
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:0.88rem">
          <thead><tr><th align="left">Key</th><th align="left">Preview</th><th align="left">Agent</th></tr></thead>
          <tbody>${{rows || `<tr><td colspan=3 class="status">empty</td></tr>`}}</tbody>
        </table>
      </div>
      <div class="row"><button id="store-read-all" class="secondary">Read all as JSON</button></div>
    `;
    root.appendChild(inspect);

    const readAllOut = document.createElement("pre");
    readAllOut.textContent = "";
    inspect.appendChild(readAllOut);
    inspect.querySelector("#store-read-all").addEventListener("click", async () => {{
      readAllOut.textContent = "…";
      try {{
        const res = await action(state.id, "read_all", {{}});
        readAllOut.textContent = JSON.stringify(res.all, null, 2);
      }} catch (e) {{
        readAllOut.textContent = e.message;
      }}
    }});

    // --------- Test (read / write / delete) ---------
    const test = makeSection("Read / write / delete");
    test.innerHTML += `
      <label>Key</label>
      <input type="text" id="store-key">
      <label>Value</label>
      <textarea id="store-value" placeholder="Value to write…"></textarea>
      <div class="row">
        <label style="display:flex;align-items:center;gap:0.35rem;margin:0">
          <input type="checkbox" id="store-as-json"> parse as JSON
        </label>
      </div>
      <div class="row">
        <button id="store-read">Read</button>
        <button id="store-write">Write</button>
        <button id="store-delete" class="secondary">Delete</button>
        <span class="status" id="store-status"></span>
      </div>
      <label>Result</label>
      <pre id="store-result">—</pre>
    `;
    root.appendChild(test);

    const keyEl = test.querySelector("#store-key");
    const valueEl = test.querySelector("#store-value");
    const jsonEl = test.querySelector("#store-as-json");
    const resultEl = test.querySelector("#store-result");
    const statusEl = test.querySelector("#store-status");

    test.querySelector("#store-read").addEventListener("click", async () => {{
      if (!keyEl.value.trim()) {{ statusEl.textContent = "Key is empty"; statusEl.className = "status warn"; return; }}
      statusEl.textContent = "Reading…"; statusEl.className = "status";
      try {{
        const res = await action(state.id, "read", {{key: keyEl.value.trim()}});
        resultEl.textContent = JSON.stringify(res.value, null, 2);
        statusEl.textContent = "Read ✓"; statusEl.className = "status ok";
      }} catch (e) {{
        resultEl.textContent = e.message;
        statusEl.textContent = "Failed"; statusEl.className = "status bad";
      }}
    }});
    test.querySelector("#store-write").addEventListener("click", async () => {{
      if (!keyEl.value.trim()) {{ statusEl.textContent = "Key is empty"; statusEl.className = "status warn"; return; }}
      statusEl.textContent = "Writing…"; statusEl.className = "status";
      try {{
        const res = await action(state.id, "write",
                                 {{key: keyEl.value.trim(), value: valueEl.value, as_json: jsonEl.checked}});
        statusEl.textContent = "Wrote ✓ " + res.key; statusEl.className = "status ok";
        loadPanel(state.id);  // refresh list
      }} catch (e) {{
        resultEl.textContent = e.message;
        statusEl.textContent = "Failed"; statusEl.className = "status bad";
      }}
    }});
    test.querySelector("#store-delete").addEventListener("click", async () => {{
      if (!keyEl.value.trim()) {{ statusEl.textContent = "Key is empty"; statusEl.className = "status warn"; return; }}
      statusEl.textContent = "Deleting…"; statusEl.className = "status";
      try {{
        const res = await action(state.id, "delete", {{key: keyEl.value.trim()}});
        statusEl.textContent = res.ok ? ("Deleted ✓ " + res.key) : ("Not found: " + res.reason);
        statusEl.className = res.ok ? "status ok" : "status warn";
        loadPanel(state.id);
      }} catch (e) {{
        resultEl.textContent = e.message;
        statusEl.textContent = "Failed"; statusEl.className = "status bad";
      }}
    }});
  }}

  function renderMemory(root, state) {{
    const inspect = makeSection("Memory — " + state.label);
    const ratio = state.max_context_tokens > 0
      ? Math.min(100, Math.round(state.token_estimate / state.max_context_tokens * 100))
      : 0;
    const barColor = ratio >= 100 ? "var(--bad)" : ratio >= 75 ? "var(--warn)" : "var(--ok)";
    inspect.innerHTML += `
      <label>Strategy</label>
      <input type="text" value="${{escapeHtml(state.strategy)}}" disabled>
      <label>Budget — ${{state.token_estimate}} / ${{state.max_context_tokens}} tokens (${{ratio}}%)</label>
      <div style="height:0.5rem;background:var(--bg);border:1px solid var(--border);border-radius:999px;overflow:hidden">
        <div style="height:100%;width:${{ratio}}%;background:${{barColor}};transition:width 0.3s"></div>
      </div>
      <div class="row">
        <span class="pill">${{state.turn_count}} turn(s)</span>
        <span class="pill">${{state.message_count}} message(s)</span>
        ${{state.window_size ? `<span class="pill">window = ${{state.window_size}}</span>` : ""}}
        ${{state.has_compressor ? '<span class="pill">LLM compressor</span>' : '<span class="pill">rule-based</span>'}}
      </div>
      ${{state.summary ? `<label>Compressed summary</label><pre>${{escapeHtml(state.summary)}}</pre>` : '<div class="status" style="margin-top:0.5rem">No compression yet.</div>'}}
      <label>History (${{state.history.length}})</label>
    `;
    const hist = document.createElement("div");
    hist.style.maxHeight = "24rem";
    hist.style.overflowY = "auto";
    hist.innerHTML = (state.history || []).map(m => `
      <div style="margin:0.4rem 0;padding:0.5rem;border-left:3px solid var(--border);background:var(--bg);border-radius:4px">
        <div class="status" style="font-size:0.75rem;text-transform:uppercase">${{escapeHtml(m.role)}}</div>
        <pre style="padding:0.3rem 0 0;border:0;background:transparent">${{escapeHtml(m.preview)}}</pre>
      </div>
    `).join("") || '<div class="status">empty</div>';
    inspect.appendChild(hist);
    root.appendChild(inspect);

    // --------- Test/edit ---------
    const test = makeSection("Actions");
    test.innerHTML += `
      <div class="row">
        <button id="mem-compress">Force recompress</button>
        <button id="mem-export" class="secondary">Export history (JSON)</button>
        <button id="mem-clear" class="secondary">Clear</button>
        <span class="status" id="mem-status"></span>
      </div>
      <label>Output</label>
      <pre id="mem-output">—</pre>
    `;
    root.appendChild(test);

    const statusEl = test.querySelector("#mem-status");
    const outEl = test.querySelector("#mem-output");
    test.querySelector("#mem-compress").addEventListener("click", async () => {{
      statusEl.textContent = "Compressing…"; statusEl.className = "status";
      try {{
        const res = await action(state.id, "force_compress", {{}});
        outEl.textContent = res.summary || "(no change — below budget)";
        statusEl.textContent = "Done ✓"; statusEl.className = "status ok";
        loadPanel(state.id);
      }} catch (e) {{
        outEl.textContent = e.message;
        statusEl.textContent = "Failed"; statusEl.className = "status bad";
      }}
    }});
    test.querySelector("#mem-export").addEventListener("click", async () => {{
      statusEl.textContent = "Exporting…"; statusEl.className = "status";
      try {{
        const res = await action(state.id, "export_history", {{}});
        outEl.textContent = JSON.stringify(res.history, null, 2);
        statusEl.textContent = "Exported ✓"; statusEl.className = "status ok";
      }} catch (e) {{
        outEl.textContent = e.message;
        statusEl.textContent = "Failed"; statusEl.className = "status bad";
      }}
    }});
    test.querySelector("#mem-clear").addEventListener("click", async () => {{
      statusEl.textContent = "Clearing…"; statusEl.className = "status";
      try {{
        await action(state.id, "clear", {{}});
        statusEl.textContent = "Cleared ✓"; statusEl.className = "status ok";
        loadPanel(state.id);
      }} catch (e) {{
        outEl.textContent = e.message;
        statusEl.textContent = "Failed"; statusEl.className = "status bad";
      }}
    }});
  }}

  const PANEL_RENDERERS = {{
    agent: renderAgent,
    tool: renderTool,
    pipeline: renderPipeline,
    session: renderSession,
    human: renderHuman,
    router: renderRouter,
    store: renderStore,
    memory: renderMemory,
  }};

  // ------------------------------------------------------------------
  // Bootstrap — SSE with polling fallback
  // ------------------------------------------------------------------
  refreshPanelList();
  let sseActive = false;
  let pollTimer = null;
  function startPolling() {{
    if (pollTimer) return;
    pollTimer = setInterval(refreshPanelList, 2000);
  }}
  function stopPolling() {{
    if (pollTimer) {{ clearInterval(pollTimer); pollTimer = null; }}
  }}
  function connectSse() {{
    try {{
      const es = new EventSource(qs("/api/events"));
      es.onopen = () => {{
        sseActive = true;
        stopPolling();
        sbStatus.textContent = "Live · SSE connected"; sbStatus.className = "status ok";
      }};
      es.addEventListener("refresh", (e) => {{
        let payload = null;
        try {{ payload = JSON.parse(e.data); }} catch (_err) {{}}
        if (!payload || payload.type === "list" || payload.type === "closed") {{
          refreshPanelList();
          return;
        }}
        // state change: refetch list so labels stay fresh, plus the
        // specific panel if it's the one currently shown.
        refreshPanelList();
        if (payload.panel_id && payload.panel_id === activePanelId) {{
          loadPanel(activePanelId);
        }}
      }});
      es.onerror = () => {{
        if (sseActive) {{
          sbStatus.textContent = "SSE dropped — falling back to polling"; sbStatus.className = "status warn";
        }}
        sseActive = false;
        es.close();
        startPolling();
        // Try to reconnect after a back-off.
        setTimeout(connectSse, 3000);
      }};
    }} catch (_err) {{
      startPolling();
    }}
  }}
  // Prefer SSE; poll as a fallback until connected (and if SSE breaks).
  startPolling();
  connectSse();
}})();
</script>
</body>
</html>
"""
