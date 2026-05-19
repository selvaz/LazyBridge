# HIL with a custom UI

The framework-bundled web UI (`ext/hil/human.py:_WebUI`) is
deliberately minimal: stdlib only, no CSRF tokens, no markdown
rendering, no streaming, no session persistence. For production-grade
needs — Slack, a custom Streamlit page, a ticket system, a mobile
app — the framework's extension point is `_UIProtocol`.

Anything implementing `async def prompt(self, task, *, tools,
output_type) -> str` is accepted by `HumanEngine(ui=...)` and gets
wired into `Plan`s, `Session`s, and routing exactly like the built-in
UIs.

## Source

```python
--8<-- "examples/hil_app/04_custom_ui.py"
```

## Walkthrough

- **`FileWatchUI`** is the minimum-viable example: it writes the
  prompt to `prompt.txt` and waits for the human to write the answer
  to `response.txt`. Trivial in itself — but the *shape* is what
  matters: any I/O surface you can poll or subscribe to can be a
  `HumanEngine` UI.
- **`asyncio.get_running_loop().run_in_executor(None, ...)`** is the
  idiomatic way to wait on blocking I/O without tying up the asyncio
  event loop. The custom UI uses it to poll for the response file
  on a background thread.
- **No inheritance** — `_UIProtocol` is a structural protocol; pass
  any object with a matching `prompt` method.

## When to write your own UI

You should reach for a custom `_UIProtocol` when any of these are
true:

- The bundled web UI's plain-text form isn't expressive enough
  (markdown, images, attachments).
- The human is somewhere other than the same machine (Slack, email,
  ticket queue, mobile push).
- You need streaming partial responses or live progress visibility
  while the agent works.
- You need authentication, CSRF protection, or multi-user state
  separation (the bundled UI is explicitly single-user).
- You want to integrate with an existing web framework (FastAPI,
  Streamlit, Gradio) instead of running an embedded server.

For all of these, the bundled UI is the wrong tool. The framework
*intentionally* keeps `_WebUI` minimal so the extension point
carries the production complexity, not the core.

## See also

- [HumanEngine guide](../guides/mid/human-engine.md) — engine API
  and the `_UIProtocol` contract.
- [HIL as a chat loop](hil-chat-loop.md) — uses the bundled
  `ui="web"` for a single-user local case where the minimal UI is
  enough.
