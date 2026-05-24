# LazyTools — capabilities (`lazytoolkit`)

Reusable **tool providers**, **connector clients**, and **safety wrappers** for
LazyBridge agents. LazyBridge stays a minimal runtime; the concrete,
dependency-carrying tools live here. Anything you add to `Agent(tools=[...])`
(or `PulseAgent(tools=[...])`) that talks to the outside world belongs in
LazyTools.

```bash
pip install lazytoolkit                 # core (just lazybridge)
pip install 'lazytoolkit[gmail]'        # Gmail client + guarded draft/send tools
pip install 'lazytoolkit[telegram]'     # Telegram client + guarded send tool
pip install 'lazytoolkit[mcp]'          # Model Context Protocol connector
pip install 'lazytoolkit[docs]'         # PDF/DOCX/HTML document reading
```

## Import contract

```python
from lazytools.connectors.gmail import GmailTools, GmailClient
from lazytools.connectors.telegram import TelegramTools
from lazytools.connectors.mcp import MCP
from lazytools.connectors.gateway import ExternalToolProvider
from lazytools.documents import read_docs_tools
from lazytools.skills import build_skill, skill_tools
from lazytools.safety import Allowlist, ConfirmationGate, ActionBlocked
```

## What's in the box

| Category | Modules | Guide |
|---|---|---|
| **Connectors** | `connectors/{gmail,telegram,mcp,gateway}` — clients + tool providers that bridge to an external service or protocol | [MCP](../guides/mid/mcp.md) · [External tool gateway](../guides/advanced/external-tools.md) |
| **Documents** | `documents/read_docs` — read `.txt/.md/.pdf/.docx/.html` from a folder/file | — |
| **Skills** | `skills/doc_skills` — build/query portable BM25 doc skills | — |
| **Safety** | `safety/{allowlist,gates}` — reusable allow-list + one-shot confirmation gate | see below |

**Planned** (added when the first module lands, not scaffolded empty): more
connectors — `github`, `slack`, `notion`, `calendar`, `filesystem`, `browser` —
and additional base tools.

## Safety model

Dangerous tools (e.g. `gmail_send`, `telegram_send_message`) are gated by two
independent, composable primitives in `lazytools.safety`:

- **`Allowlist`** — case-insensitive target allow-list (`None` = allow all,
  empty = deny all).
- **`ConfirmationGate`** — one-shot, target-bound grants. Each grant authorizes
  exactly one action and is consumed on use; a grant may be bound to a *scope*
  (the running task id, in LazyPulse) so a concurrent task can never spend it.
  There is no sticky global approval.

A harmless companion always sits alongside the gated action (e.g.
`gmail_create_draft` is never gated; only `gmail_send` is) — the
dry-run-first pattern. Denials raise a typed `ActionBlocked` (a
`PermissionError`).

> **Migrating from ≤0.7.9?** These tools used to live under
> `lazybridge.ext.{mcp,gateway}` and `lazybridge.external_tools.*` (and the
> Gmail/Telegram tools under `lazypulse.adapters.*`). The old paths still work
> with a `DeprecationWarning` until 0.9 — import from `lazytools.*` instead.

Repo: <https://github.com/selvaz/LazyTools>
