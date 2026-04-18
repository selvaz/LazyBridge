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
      <label>Provider / model (read-only)</label>
      <input type="text" value="${{escapeHtml(state.provider + ' / ' + state.model)}}" disabled>
      <label>System prompt</label>
      <textarea id="agent-system" class="large">${{escapeHtml(state.system || "")}}</textarea>
      <div class="row">
        <button id="agent-save-system">Save system prompt</button>
        <span class="status" id="agent-system-status"></span>
      </div>
      <label>Tools enabled on this agent</label>
      <div class="checkbox-list" id="agent-tools"></div>
      <span class="status" id="agent-tools-status"></span>
    `;
    root.appendChild(inspect);

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

    // ---------- Test ----------
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
        <span class="status" id="agent-run-status"></span>
      </div>
      <label>Response</label>
      <pre id="agent-response" class="status">—</pre>
    `;
    root.appendChild(test);

    test.querySelector("#agent-run").addEventListener("click", async () => {{
      const mode = test.querySelector("#agent-mode").value;
      const msg = test.querySelector("#agent-msg").value;
      const btn = test.querySelector("#agent-run");
      const st = test.querySelector("#agent-run-status");
      const out = test.querySelector("#agent-response");
      if (!msg.trim()) {{
        st.textContent = "Message is empty"; st.className = "status warn";
        return;
      }}
      btn.disabled = true;
      st.textContent = "Running " + mode + "…"; st.className = "status";
      out.textContent = "…";
      const t0 = Date.now();
      try {{
        const res = await action(state.id, "test", {{mode, message: msg}});
        const dt = ((Date.now() - t0) / 1000).toFixed(1);
        out.textContent = res.content || JSON.stringify(res.parsed ?? res, null, 2);
        const usage = res.usage || {{}};
        const cost = usage.cost_usd != null ? ` • $${{usage.cost_usd.toFixed(6)}}` : "";
        st.textContent = `Done in ${{dt}}s • ${{usage.input_tokens || 0}} in / ${{usage.output_tokens || 0}} out${{cost}}`;
        st.className = "status ok";
      }} catch (e) {{
        out.textContent = e.message;
        st.textContent = "Failed"; st.className = "status bad";
      }} finally {{
        btn.disabled = false;
      }}
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

  const PANEL_RENDERERS = {{
    agent: renderAgent,
    tool: renderTool,
    session: renderSession,
  }};

  // ------------------------------------------------------------------
  // Bootstrap
  // ------------------------------------------------------------------
  refreshPanelList();
  setInterval(refreshPanelList, 2000);
}})();
</script>
</body>
</html>
"""
