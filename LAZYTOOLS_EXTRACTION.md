# LazyTools extraction — implementation plan

> **Scope.** Forward-looking implementation plan for splitting reusable tool
> providers, connector clients, and safety wrappers out of `lazybridge` (and
> `lazypulse`) into a new standalone package **`lazytools`** (PyPI:
> **`lazytoolkit`**). This file is the working tracker; shipped changes land in
> each repo's `CHANGELOG.md`.
>
> **Status legend.** `[ ]` open · `[~]` in progress · `[x]` done · `[-]` skipped (with reason).
>
> **Update protocol.** Each PR ticks at least one checkbox and references the
> phase heading. When a phase's acceptance gate passes, cut a release of the
> affected package(s).

---

## 0. Target architecture (the contract)

```
lazybridge   minimal agent runtime — core abstractions only
lazytools    reusable tool providers + connector clients + safety wrappers
lazypulse    always-on orchestration (tick loop, adapters, policy, ledger)
```

**Hard dependency rules (must hold at every commit):**

| Allowed | Forbidden |
|---|---|
| `lazytools  → lazybridge` | `lazybridge → lazytools` |
| `lazypulse  → lazybridge` | `lazytools  → lazypulse` |
| `lazypulse  → lazytools` *(optional, behind extras)* | circular imports of any kind |

The two **forbidden import** rules are enforced by automated tests (§6), not by
convention. The "core never imports lazytools" rule is the subtle one: because
`lazytools → lazybridge`, any *eager* re-export of `lazytools` from a
`lazybridge` shim creates a circular import **and** violates the rule. Every
backward-compat shim is therefore **lazy** (PEP 562 `__getattr__`, import
inside the function body). See §5.3.

**Core abstractions that stay in `lazybridge` (never moved, never renamed):**
`Agent`, `Engine`/`LLMEngine`/`HumanEngine`, `Tool`/`ToolProvider`, `Envelope`,
`Store`, `Session`/observability, `Plan`, `Guard`/`Verify`, `Memory`.

---

## 1. What moves, what stays (decision record)

> **Note (as built).** The "New home" paths below are the *original plan*
> targets (a flat layout). Phases 0–2 shipped a by-category layout instead —
> `lazytools/connectors/{mcp,gateway,gmail,telegram}`, `lazytools/documents`,
> `lazytools/skills`. See §3 for the as-built tree and §10 for the deviation
> record; the deprecation shims point at these final paths.

### 1.1 From `lazybridge/`

| Source | Symbols | Disposition | New home |
|---|---|---|---|
| `external_tools/read_docs/read_docs.py` | `read_folder_docs`, `read_docs_tools` | **MOVE** | `lazytools/read_docs/read_docs.py` |
| `external_tools/doc_skills/doc_skills.py` | `build_skill`, `query_skill`, `skill_tools`, `skill_builder_tools`, `skill_pipeline` | **MOVE** | `lazytools/doc_skills/doc_skills.py` |
| `external_tools/__init__.py` (+ subpkgs) | namespace | **SHIM** (lazy) → delete in 0.9 | re-export from `lazytools` |
| `ext/mcp/*` | `MCP`, `MCPServer`, transports | **MOVE** *(Decision D1 — resolved)* | `lazytools/mcp/` |
| `ext/gateway.py` | `ExternalToolProvider` (Composio/Pipedream/Arcade) | **MOVE** *(see Decision D2)* | `lazytools/gateway/` |
| `ext/{hil,planners,otel,viz,evals}/*` | HumanEngine, planners, tracing, viz, evals | **KEEP** | maps 1:1 to a core abstraction |
| everything else under `lazybridge/` | core runtime | **KEEP** | — |

These two `external_tools` packages are already documented in-tree as *"worked
examples — not framework primitives"* and are **not** exported from the
top-level `lazybridge` API, so moving them is non-breaking for anyone importing
from `lazybridge` directly.

### 1.2 From `lazypulse/src/lazypulse/adapters/`

The brief's rule: a **Tool** is invoked by the worker mid-run → movable; an
**inbound adapter / policy** produces `InboundMessage` / extends `PulsePolicy`
→ stays in LazyPulse.

| Source | Symbols | Disposition | New home |
|---|---|---|---|
| `gmail/client.py` | `GmailClient`, `GmailService` | **MOVE** | `lazytools/gmail/client.py` |
| `gmail/tools.py` | `GmailTools`, `GmailSendBlocked` | **MOVE** (refactor onto `safety/`) | `lazytools/gmail/tools.py` |
| `gmail/auth.py` | `parse_authentication_results` | **MOVE** *(Decision D3)* | `lazytools/gmail/auth.py` |
| `gmail/inbox.py` | `GmailInbox`, `GmailInboxConfig` | **STAY** | inbound adapter |
| `gmail/policy.py` | `GmailPolicy` | **STAY** | `PulsePolicy` subclass |
| `telegram/client.py` | `TelegramClient`, `TelegramService` | **MOVE** | `lazytools/telegram/client.py` |
| `telegram/tools.py` | `TelegramTools`, `TelegramSendBlocked` | **MOVE** (refactor onto `safety/`) | `lazytools/telegram/tools.py` |
| `telegram/inbox.py` | `TelegramInbox` (+ `Responder.reply`) | **STAY** | inbound adapter |
| `telegram/policy.py` | `TelegramPolicy` | **STAY** | `PulsePolicy` subclass |
| `adapters/webhook.py` | `WebhookAdapter` | **STAY** | inbound adapter |
| `adapters/base.py` | `Adapter`, `Responder` | **STAY** | inbound protocols |

### 1.3 New code created in `lazytools`

| Module | Purpose |
|---|---|
| `safety/allowlist.py` | `Allowlist` — case-insensitive target allow-list (`None` = allow all) |
| `safety/gates.py` | `ConfirmationGate` — one-shot, target-bound grants; no sticky global approval |
| `safety/__init__.py` | `Allowlist`, `ConfirmationGate`, `ActionBlocked` |
| `testing/fake_clients.py` | `FakeGmailService`, `FakeTelegramService` (consolidates per-test fakes) |

---

## 2. Decisions requiring sign-off

> Resolve these **before Phase 1** — they change the move manifest.

- **D1 — `ext/mcp`: RESOLVED → MOVE** to `lazytools/mcp/`. `MCPServer` is a
  `ToolProvider` that connects to external MCP servers, so it belongs with the
  other connectors in `lazytools`. **Caveat (elevated churn):** unlike the doc
  tools, `lazybridge.ext.mcp` is a widely-advertised public path — referenced in
  `docs/` (guides, reference, `pick-tier`, codegen-contract), `skill/SKILL.md`,
  `examples/llm_assistant/05_mcp_allowlisted.py`, and several `tests/unit/*`. It
  needs a robust lazy shim at `lazybridge.ext.mcp` **and** a coordinated docs
  pass (Phase 2). When moved, change its internal import from
  `from lazybridge.tools import Tool` to the public `from lazybridge import Tool`.
  The `mcp = ["mcp>=1.0,<2.0"]` extra moves to `lazytoolkit[mcp]` and drops out
  of `lazybridge`'s `all` extra.
- **D2 — `ext/gateway`:** A connector to commercial integration gateways
  (Composio/Pipedream/Arcade). Fits the bucket-B "connector client" definition
  and brings outbound HTTP. **Recommendation: MOVE** to `lazytools/gateway/`
  with a lazy shim at `lazybridge.ext.gateway`. Lower priority than Gmail/Telegram/docs.
- **D3 — `gmail/auth.py`:** Neutral DKIM/SPF/DMARC parser, currently consumed
  only by Pulse. Moving it makes Pulse's `gmail` extra depend on `lazytoolkit`
  (allowed, but couples them). **Recommendation: MOVE** (reusable). Keep in
  Pulse only if you want the inbox to stand alone with no `lazytools` dependency.
- **D4 — Target repo for `lazytoolkit`: RESOLVED → the `LazyTools` repo** hosts
  `src/lazytools`. *Access note:* this assistant's GitHub scope currently covers
  only `selvaz/{lazybridge,git_test,lazypulse}` and only those three are cloned
  locally — pushing the new package to `selvaz/LazyTools` requires that repo to
  be added to scope (and cloned) first.
- **D5 — Version floors:** `lazytoolkit → lazybridge>=0.7.9,<0.9` and
  `lazypulse[gmail] → lazytoolkit[gmail]`. Confirm the three-way pin window.

---

## 3. The `lazytools` package layout

> **As built — reorganised by category** (see "Deviations" below). Connectors
> are grouped under `connectors/`; document tools and skills get their own
> top-level categories.

```
src/lazytools/
  __init__.py            # __version__ + light re-exports only; NO eager heavy imports
  connectors/            # clients + tool providers that bridge to an external service/protocol
    __init__.py          # no eager imports
    gmail/
      __init__.py        # GmailClient, GmailService, GmailTools, GmailSendBlocked, parse_authentication_results
      client.py
      tools.py
      auth.py
    telegram/
      __init__.py        # TelegramClient, TelegramService, TelegramTools, TelegramSendBlocked
      client.py
      tools.py
    mcp/
      __init__.py        # MCP, MCPServer
      server.py
      transports.py
    gateway/
      __init__.py        # ExternalToolProvider, ExternalToolSpec, ExternalToolClient, ExternalToolError, JsonHttpExternalToolClient
  documents/             # base document-reading tools
    __init__.py
    read_docs.py
  skills/                # build/query portable doc skills
    __init__.py
    doc_skills.py
  safety/
    __init__.py          # Allowlist, ConfirmationGate, ActionBlocked, active_scope, current_scope
    allowlist.py
    gates.py
    context.py           # ambient scope contextvar shared with the orchestrator
  testing/
    __init__.py
    fake_clients.py
  # planned (created when the first module lands, not scaffolded empty):
  #   connectors/{github,slack,notion,calendar,filesystem,browser}, more base tools
```

**`pyproject.toml` (new repo):**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "lazytoolkit"
version = "0.1.0"
description = "Reusable tool providers, connector clients, and safety wrappers for LazyBridge and LazyPulse agents."
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
dependencies = ["lazybridge>=0.7.9,<0.9"]

[project.optional-dependencies]
gmail    = ["google-api-python-client>=2.0,<3.0", "google-auth>=2.0,<3.0", "google-auth-oauthlib>=1.0,<2.0"]
telegram = ["httpx>=0.27,<1.0"]
mcp      = ["mcp>=1.0,<2.0"]
docs     = ["pypdf>=3.0,<7.0", "python-docx>=1.0,<2.0", "trafilatura>=1.6,<3.0"]
test     = ["pytest>=7.0", "pytest-asyncio>=0.23", "pytest-cov>=4.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/lazytools"]
```

Import contract (what users type after `pip install lazytoolkit`):

```python
from lazytools.connectors.gmail import GmailTools, GmailClient
from lazytools.connectors.telegram import TelegramTools
from lazytools.connectors.mcp import MCP
from lazytools.connectors.gateway import ExternalToolProvider
from lazytools.safety import Allowlist, ConfirmationGate, ActionBlocked
from lazytools.documents import read_docs_tools
from lazytools.skills import build_skill, skill_tools
```

---

## 4. The safety layer (reusable patterns)

`GmailTools` and `TelegramTools` today carry **duplicated** machinery: a
`_send_grants: dict[str, int]`, a `_consume_grant` that prefers a target-bound
grant over an any-target one, allow-list normalization, and a
`*SendBlocked(PermissionError)`. This is the reusable safety surface — extract
it once.

**`lazytools/safety/allowlist.py`**

```python
from __future__ import annotations
from collections.abc import Iterable

class Allowlist:
    """Case-insensitive, string-normalized target allow-list.

    ``None`` means "no allow-list configured" → permits everything.
    An empty iterable means "deny everything".
    """
    def __init__(self, allowed: Iterable[object] | None) -> None:
        self._allowed = None if allowed is None else {str(a).lower() for a in allowed}

    def permits(self, target: object) -> bool:
        return self._allowed is None or str(target).lower() in self._allowed
```

**`lazytools/safety/gates.py`**

> **Corrected — grants carry a `scope` dimension.** The real `GmailTools` /
> `TelegramTools` bind grants to `(target, task_id)` and the public surface
> includes `confirm_once(*, task_id=…)` / `confirm_send(*, task_id=…)` (with
> task-bound e2e tests). The plan's original gate dropped that dimension, which
> would have broken the public surface and 5 tests. The gate is kept
> **scope-agnostic** — the caller passes the current scope — so it has no
> orchestration dependency; the ambient value lives in `safety/context.py`
> (`active_scope` / `current_scope`) and is set by `PulseAgent`.

```python
from __future__ import annotations

_ANY = "*"

class ConfirmationGate:
    """One-shot, target-bound confirmation grants for dangerous actions.

    Not a sticky boolean: each grant authorizes exactly one action. Grants are
    matched most-to-least specific: (target, scope) → (target, None) →
    (_ANY, scope) → (_ANY, None). A scope-bound grant is never spendable when no
    scope is supplied. No global "approved forever" state.
    """
    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._grants: dict[tuple[str, str | None], int] = {}

    def grant(self, target: object, *, scope: str | None = None) -> None:
        self._add((str(target).lower(), scope))

    def grant_any(self, *, scope: str | None = None) -> None:
        self._add((_ANY, scope))

    def _add(self, key: tuple[str, str | None]) -> None:
        self._grants[key] = self._grants.get(key, 0) + 1

    def consume(self, target: object, *, scope: str | None = None) -> bool:
        if not self._enabled:
            return True
        target_l = str(target).lower()
        candidates = []
        if scope is not None:
            candidates.append((target_l, scope))
        candidates.append((target_l, None))
        if scope is not None:
            candidates.append((_ANY, scope))
        candidates.append((_ANY, None))
        for key in candidates:
            if self._grants.get(key, 0) > 0:
                self._grants[key] -= 1
                return True
        return False
```

**`lazytools/safety/__init__.py`**

```python
from lazytools.safety.allowlist import Allowlist
from lazytools.safety.gates import ConfirmationGate

class ActionBlocked(PermissionError):
    """Base for dangerous-action denials (allow-list / confirmation)."""

__all__ = ["Allowlist", "ConfirmationGate", "ActionBlocked"]
```

**Safety design invariants** (the acceptance bar for "dangerous tools"):
- Allow-list and confirmation are **independent** gates; a tool may use either or both.
- A confirmation authorizes **exactly one** action and is consumed on use.
- Target-bound grants are preferred over any-target grants.
- No process-global mutable approval state; grants live on the tool instance.
- Denials raise a typed `ActionBlocked` subclass with an audit-friendly message
  (names the action and the reason, never leaks secrets).
- Where appropriate, a tool ships a harmless companion (e.g. `gmail_create_draft`
  is always allowed; only `gmail_send` is gated) — the dry-run-first pattern.

**Refactored `GmailTools` (illustrative; `TelegramTools` is structurally identical):**

```python
from __future__ import annotations
from lazybridge import Tool
from lazytools.gmail.client import GmailService
from lazytools.safety import ActionBlocked, Allowlist, ConfirmationGate

class GmailSendBlocked(ActionBlocked):
    """Raised when ``gmail_send`` is invoked without authorization."""

class GmailTools:
    _is_lazy_tool_provider = True

    def __init__(self, client: GmailService, *, allowed_recipients=None,
                 require_confirmation: bool = True) -> None:
        self._client = client
        self._allowlist = Allowlist(allowed_recipients)
        self._gate = ConfirmationGate(enabled=require_confirmation)

    @property
    def require_confirmation(self) -> bool:        # preserve public attribute
        return self._gate._enabled

    def confirm_once(self) -> None: self._gate.grant_any()
    def confirm_send(self, *, to: str) -> None: self._gate.grant(to)

    def as_tools(self) -> list[Tool]:
        return [
            Tool.wrap(self._create_draft, name="gmail_create_draft",
                      description="Create a Gmail draft (not sent). Args: to, subject, body."),
            Tool.wrap(self._send, name="gmail_send",
                      description="Send an email via Gmail. Requires a one-shot confirmation. Args: to, subject, body."),
        ]

    def _create_draft(self, to: str, subject: str, body: str) -> str:
        return f"draft created: {self._client.create_draft(to=to, subject=subject, body=body).get('id', '<unknown>')}"

    def _send(self, to: str, subject: str, body: str) -> str:
        if not self._allowlist.permits(to):
            raise GmailSendBlocked(f"gmail_send blocked: recipient {to!r} is not in the allow-list")
        if not self._gate.consume(to):
            raise GmailSendBlocked("gmail_send blocked: no outstanding confirmation for this send")
        return f"sent: {self._client.send_message(to=to, subject=subject, body=body).get('id', '<unknown>')}"
```

---

## 5. Phased execution

Each phase ends at an **acceptance gate**. Do not start the next phase until the
gate is green.

### Phase 0 — in-place safety extraction (lowest risk, fully reversible) `[x]`

Land the `safety` helper **inside the current repos first**, before any move.
This validates the API and de-risks Phase 1.

- `[x]` Add `lazypulse/_safety.py` (or a temp module) with `Allowlist` + `ConfirmationGate`.
- `[x]` Refactor `GmailTools` and `TelegramTools` onto it; keep the public
  surface (`confirm_once`, `confirm_send`, `require_confirmation`, `*SendBlocked`).
- `[x]` Existing tests (`test_gmail_tools.py`, `test_telegram.py`) pass **unchanged**.

**Gate 0:** `cd LazyPulse && pytest -q` green; no public-API diff in the tool classes.

### Phase 1 — create `lazytoolkit`, move modules, add lazy shims `[x]`

- `[x]` Scaffold the repo (D4) with the layout in §3.
- `[x]` Move modules per §1; rewrite internal imports
  (`lazypulse.adapters.gmail.client` → `lazytools.gmail.client`;
  `from lazybridge import Tool` is unchanged and correct).
- `[x]` Promote the Phase-0 `safety` helper into `lazytools/safety/`.
- `[x]` Add `lazytools/testing/fake_clients.py`.
- `[x]` Move the tool-level tests (§6 matrix); add the two boundary tests.
- `[x]` **LazyBridge:** replace `external_tools/{read_docs,doc_skills}` bodies,
  `ext/mcp/*`, and `ext/gateway.py` with lazy shims (§5.3); update `pyproject`
  (drop the `tools`, `mcp`, and—if D2 confirmed—gateway-related members from the
  extras and from `all`; remove the two `coverage omit` lines and the "first-class
  ext" mention of `mcp` in the coverage comment).
- `[x]` **LazyPulse:** point `gmail`/`telegram` extras at `lazytoolkit[...]`;
  rewrite inbox imports of `*Service`/`parse_authentication_results` to
  `lazytools.*`; add lazy re-export shims in the two adapter `__init__.py`.

**Gate 1 (all must pass):**
```
# core imports clean with NOTHING optional installed, and cannot see lazytools
python -c "import lazybridge"
pip uninstall -y lazytoolkit && python -c "import lazybridge; print('ok')"
# three suites green
cd LazyBridge && pytest -q
cd LazyTools  && pytest -q
cd LazyPulse  && pytest -q
# boundary guards
cd LazyTools  && pytest -q tests/test_no_lazypulse_import.py
cd LazyBridge && pytest -q tests/unit/test_ext_core_boundary.py
# old import paths still work, with a DeprecationWarning
python -W error::DeprecationWarning -c "from lazybridge.external_tools.read_docs import read_folder_docs" ; echo "expect: DeprecationWarning raised"
```

### Phase 2 — docs & examples `[x]`

- `[x]` LazyBridge docs: "concrete tools live in `lazytools` (`pip install lazytoolkit`)";
  update `docs/guides/core-vs-ext.md`, `docs/guides/basic/tool.md`.
- `[x]` Update examples to import `lazytools.*`:
  `LazyPulse/examples/03_gmail_polling.py`, `07_telegram_polling.py`,
  `LazyBridge/examples/llm_assistant/05_mcp_allowlisted.py`, and any example
  using `external_tools`.
- `[x]` **MCP doc pass** (largest churn): repoint every `from lazybridge.ext.mcp
  import MCP` in `docs/guides/{mid/mcp.md,basic/tool.md}`,
  `docs/reference/extensions.md`, `docs/decisions/pick-tier.md`,
  `docs/for-llms/codegen-contract.md`, and `lazybridge/skill/SKILL.md` to
  `lazytools.mcp`.
- `[x]` `lazytoolkit` README with the import contract from §3.

**Gate 2:** `test_doc_examples_runtime.py` (LazyBridge) green; doc-path link checks pass.

### Phase 3 — remove shims (after one minor release) `[~]`

- `[x]` Delete `lazybridge/external_tools/*` and `lazybridge/ext/{mcp,gateway}`
  shims (`lazybridge 0.9`); updated the core/ext boundary tests + coverage scope.
- `[ ]` Delete the moved-symbol re-exports from the Pulse adapter `__init__.py`.
- `[x]` Record the breaking change in the LazyBridge `CHANGELOG.md` (`[0.9.0]`).
  Pulse `CHANGELOG.md` pending its own shim removal.

**Gate 3:** grep finds no `external_tools` references in shipped code/docs; suites green.

### 5.3 Lazy shim template (the load-bearing detail)

```python
# lazybridge/external_tools/read_docs/__init__.py
"""Deprecated location. Moved to ``lazytools.read_docs`` (pip install lazytoolkit)."""
from __future__ import annotations
import warnings

def __getattr__(name: str):                 # PEP 562 — fires only on attribute access
    warnings.warn(
        "lazybridge.external_tools.read_docs moved to lazytools.read_docs in 0.8; "
        "install 'lazytoolkit' and import from there. This shim is removed in 0.9.",
        DeprecationWarning, stacklevel=2,
    )
    try:
        from lazytools import read_docs as _moved
    except ImportError as exc:
        raise ImportError(
            "lazybridge.external_tools.read_docs now requires 'lazytoolkit' "
            "(pip install 'lazytoolkit[docs]')."
        ) from exc
    return getattr(_moved, name)
```

The import lives **inside** `__getattr__`, so `import lazybridge` never touches
`lazytools` — preserving both the "core never imports lazytools" rule and
freedom from circular imports. Same pattern for `doc_skills`, the optional
`ext/gateway` shim, and the Pulse adapter `__init__` re-exports.

---

## 6. Test migration matrix

| Test (current) | Action | Destination |
|---|---|---|
| `LazyPulse/tests/unit/test_gmail_tools.py` | move | `lazytools/tests/test_gmail_tools.py` |
| `LazyPulse/tests/unit/test_gmail_auth_parser.py` | move (if D3=MOVE) | `lazytools/tests/test_gmail_auth.py` |
| `LazyPulse/tests/unit/test_telegram.py` | **split**: tool cases move, inbox/policy cases stay | both |
| `LazyBridge/tests/unit/test_new_features.py` (read_docs/doc_skills cases) | move | `lazytools/tests/` |
| `LazyBridge/tests/unit/test_mcp.py` | move | `lazytools/tests/test_mcp.py` |
| `LazyBridge/tests/unit/test_audit_{short_term,amend,followup}.py` (ext.mcp imports) | **update imports** | repoint to `lazytools.mcp` (or the lazy shim) |
| `LazyPulse/tests/unit/test_gmail_inbox.py` | keep | — |
| `LazyBridge/tests/unit/test_ext_core_boundary.py` | **extend** | add: `lazybridge` must not `import lazytools` |
| `LazyPulse/tests/unit/test_no_private_imports.py` | **extend / mirror** | add: `lazytools` must not `import lazypulse` |
| — (new) | add | `lazytools/tests/test_safety.py` (allow-list + one-shot grant invariants) |
| — (new) | add | `lazytools/tests/test_no_lazypulse_import.py` (AST/grep guard) |
| — (new, LazyBridge) | add | guard: `import lazybridge` succeeds with `lazytoolkit` uninstalled |

**Boundary guard sketch (`lazytools/tests/test_no_lazypulse_import.py`):**

```python
import ast, pathlib

def test_lazytools_never_imports_lazypulse():
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "lazytools"
    offenders = []
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mod = (getattr(node, "module", None)
                   or (node.names[0].name if isinstance(node, ast.Import) else None))
            if mod and mod.split(".")[0] == "lazypulse":
                offenders.append(f"{py}: {mod}")
    assert not offenders, offenders
```

---

## 7. Acceptance criteria (from the brief) → mechanism

- `[x]` `lazybridge` imports without optional service deps — providers already
  optional; `external_tools` becomes lazy shims. *(Gate 1)*
- `[x]` `lazybridge` / `lazytools` / `lazypulse` suites pass. *(Gates 1–2)*
- `[x]` `Agent(..., tools=[...])` and `PulseAgent(..., tools=[...])` both accept
  `lazytools` providers — they keep `_is_lazy_tool_provider`. *(Gate 1)*
- `[x]` `lazytools` has no import from `lazypulse`. *(boundary test)*
- `[x]` `lazybridge` has no import from `lazytools`. *(boundary test + lazy shims)*
- `[x]` Dangerous tools require allow-list and/or one-shot confirmation; no
  sticky global approval. *(`lazytools.safety` + `test_safety.py`)*
- `[x]` Old imports still work with `DeprecationWarning`, removal documented. *(§5.3, Phase 3)*

---

## 8. Risks & edge cases

1. **Circular import / rule violation (highest risk).** Any eager
   `lazytools` import from a `lazybridge` shim breaks both the dependency rule
   and import itself. *Mitigation:* lazy `__getattr__` shims (§5.3) + the
   "import core with lazytoolkit uninstalled" guard test.
2. **`GmailService` / `TelegramService` Protocols are shared contracts** between
   the moved tools and the staying inboxes. They live with the client in
   `lazytools`; the inbox imports the Protocol back. Structural (duck-typed)
   Protocols mean no runtime coupling beyond the import line.
3. **`require_confirmation` is a public attribute.** Preserved via a property on
   the refactored tools — do not drop it silently.
4. **Three-way version matrix** (`lazypulse → lazytoolkit → lazybridge`). Pin
   conservatively (D5) and add a CI job installing all three together.
5. **Coverage floor (LazyBridge).** `external_tools` is currently in
   `coverage omit`; removing it should *raise* the percentage, but read the new
   floor from a green CI run before touching `fail_under` (pyproject warns about
   local-vs-CI drift).
6. **Docs/examples drift — amplified by the MCP move.** `lazybridge.ext.mcp` is
   referenced across guides, reference docs, `pick-tier`, the codegen contract,
   `skill/SKILL.md`, an example, and several tests. The lazy shim keeps code
   working, but `test_doc_examples_runtime.py` / doc-path checks will fail unless
   the Phase-2 doc pass repoints every reference. Budget MCP as the single
   largest churn item in the whole migration.
7. **`auth.py` coupling (D3).** Moving it adds a Pulse→lazytools dependency for
   the `gmail` extra. Allowed, but make the choice explicitly.

---

## 9. Rollback

- **Phases 0–2 are non-destructive to users:** old import paths keep working via
  shims for a full minor cycle. To roll back, revert the shim commits and the
  `pyproject` extra redirects; the original modules are still in git history.
- **Phase 3 is the only breaking step.** Gate it behind a major-enough version
  bump (`lazybridge 0.9`, `lazypulse` minor) and a `CHANGELOG` migration note.
  If a consumer breaks, restoring the lazy shim (a single small commit) is the
  fix — no data or behavior is lost.

---

## 10. Deviations from this plan (as built — Phases 0–2)

Recorded here per the "fix the plan if you find a concrete error" rule.

1. **Package reorganised by category (§3).** Instead of a flat layout
   (`gmail/`, `telegram/`, `mcp/`, `read_docs/`, …) the connectors are grouped
   under `connectors/` and the document/skill tools get their own categories:
   `connectors/{gmail,telegram,mcp,gateway}`, `documents/read_docs.py`,
   `skills/doc_skills.py`, `safety/`, `testing/`. Rationale: a clearer
   by-type taxonomy that scales as `github`/`slack`/… connectors land. The
   import contract changed accordingly (`lazytools.connectors.gmail`,
   `lazytools.documents`, `lazytools.skills`); shim targets updated to match.

2. **`ConfirmationGate` carries a `scope` dimension (§4).** The plan's gate was
   target-only and dropped the task-binding the real tools have. The shipped
   gate keys grants on `(target, scope)` and stays orchestration-agnostic (the
   caller passes `scope=`). The ambient scope lives in `lazytools/safety/context.py`
   (`active_scope` / `current_scope`). `safety/__init__.py` therefore also
   exports `active_scope` and `current_scope`.

3. **`lazypulse._context` bridges to `lazytools.safety.active_scope`.** So that
   `PulseAgent` (which sets the task id) and the moved tools (which read it via
   `current_scope()`) share one contextvar without `lazytools` importing
   `lazypulse`. The import is **guarded** (`try/except ImportError` with a local
   fallback), so `import lazypulse` still works with `lazytoolkit` uninstalled —
   keeping the `lazypulse → lazytools` dependency optional.

4. **Test matrix corrections (§6).** `test_new_features.py` had no `read_docs`
   cases — only 3 `doc_skills` import-smoke tests; those moved to
   `lazytools/tests/test_doc_skills.py`. The MCP cache tests in
   `test_audit_short_term.py` reuse `FakeTransport` from the (moved) `test_mcp.py`,
   so they were **relocated** to `lazytools/tests/test_mcp.py` rather than left in
   LazyBridge. The remaining MCP/gateway audit tests (`test_audit_amend.py`,
   `test_audit_followup.py`) had their imports repointed to `lazytools.connectors.*`;
   `lazytoolkit` is therefore a **dev/test** dependency of LazyBridge (never a
   runtime one — `import lazybridge` stays clean without it).

5. **Shims have no `__all__`.** PEP 562 `__getattr__` shims provide names
   lazily; declaring `__all__` tripped ruff `F822` (undefined names). Dropped it
   (matches the §5.3 template). The LazyPulse adapter `__init__` shims keep a
   `TYPE_CHECKING` import block so type-checkers and `__all__` still see the
   moved names.

**Gates passed:** Gate 0 (LazyPulse green, tool public surface unchanged);
Gate 1 (`import lazybridge` / `import lazypulse` clean with `lazytoolkit`
uninstalled; three suites green — LazyBridge 1584, LazyTools 95, LazyPulse 176;
both boundary guards green; old import paths warn + resolve); Gate 2
(`test_doc_examples_runtime.py` green; examples + docs repointed to
`lazytools.*`).
