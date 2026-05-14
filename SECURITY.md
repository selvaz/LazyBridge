# Security Policy

## Reporting Vulnerabilities

Report security issues to the maintainers via GitHub Issues (private) or email. Do not open public issues for security vulnerabilities.

## Known Security Considerations

### Tool — Code Execution Risk

**Severity: CRITICAL**

`Tool` wraps arbitrary Python callables. If you expose a tool that runs
shell commands, reads files, or calls `eval`/`exec`, and an LLM controls
the arguments, that is a code-execution surface.

```python
# DANGEROUS — LLM-controlled path argument
import subprocess

from lazybridge import Agent, LLMEngine

def run_cmd(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True).decode()

agent = Agent(engine=LLMEngine("claude-haiku-4-5"), tools=[run_cmd])  # DO NOT DO THIS

# SAFER — whitelist valid operations instead of passing raw commands
ALLOWED_COMMANDS = {"status": "git status", "log": "git log --oneline -5"}
def run_safe_cmd(command: str) -> str:
    """Run a pre-approved git command."""
    if command not in ALLOWED_COMMANDS:
        return f"Unknown command. Choose from: {list(ALLOWED_COMMANDS)}"
    return subprocess.check_output(ALLOWED_COMMANDS[command].split()).decode()
```

### Store — Encryption is opt-in

The base `Store` persists data in plaintext (JSON in memory or SQLite
on disk). For at-rest encryption use `EncryptedStoreAdapter` from the
optional `[encryption]` extra:

```bash
pip install 'lazybridge[encryption]'
```

```python
from cryptography.fernet import Fernet
from lazybridge.store import Store
from lazybridge.store.encryption import EncryptedStoreAdapter

key = Fernet.generate_key()  # persist this in a KMS / sealed-secret system
store = EncryptedStoreAdapter(Store(db="state.sqlite"), key=key)
```

The adapter uses Fernet (AES-128-CBC + HMAC-SHA256) and supports key
rotation via `MultiFernet` semantics (`key=[new_key, old_key]`).

**Threat model coverage:**

- ✅ Protects against an attacker who reads `state.sqlite` off disk.
- ❌ Does NOT protect against an attacker with live process memory —
  in-flight values are plaintext.
- ❌ Does NOT encrypt **keys**, `written_at`, or `agent_id` — an
  attacker with file access still sees the access pattern.
- ❌ Not a substitute for OS-level disk encryption — defence in depth.

Without the adapter, do not store:

- API keys or tokens
- Passwords or credentials
- PII (names, emails, phone numbers)

### LLMGuard — Prompt Injection

When using `LLMGuard` as a content moderator, the guard agent itself can be
subject to prompt injection from the content it is evaluating. Use a separate,
hardened model for moderation and keep the guard's system prompt minimal.

```python
from lazybridge import Agent, LLMGuard
from lazybridge.engines.llm import LLMEngine

# Moderator agent uses a dedicated engine — not the same one handling user input.
moderator = Agent(engine=LLMEngine("anthropic/claude-haiku-4-5"), name="moderator")
guard = LLMGuard(agent=moderator)
```

### Provider API Keys

API keys are read from environment variables by default. Never hardcode keys
in source code or commit them to version control.

```python
# Good — reads from ANTHROPIC_API_KEY env var
agent = Agent.from_provider("anthropic", tier="cheap")

# Bad — key in code
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # don't do this
```

### MCP Servers — Tool Surface Audit

`MCP.stdio()` and `MCP.http()` integrate an external MCP server's tool
catalogue into an Agent.  Each tool advertised by the server becomes a
first-class `Tool` on the Agent — i.e. **anything the LLM decides to
call will run inside your process or via the server's permissions**.

Both factories are **deny-by-default since 0.7.9** — omitting both
`allow=` and `deny=` raises `ValueError` at construction:

- **`MCP.http(...)`** — a remote server is untrusted; the catalogue
  could be anything.  Pass an explicit list of the tools you want
  the LLM to see, e.g. `allow=["create_issue", "list_prs"]`, or
  `allow=["*"]` once you have audited the catalogue.
- **`MCP.stdio(...)`** — also deny-by-default.  Even though the
  subprocess is spawned by your code, the pre-0.7.9 warn-and-proceed
  default was unsafe for filesystem / git / shell MCP servers (the
  LLM could invoke any tool the subprocess advertised).  Pass
  `allow=["*"]` to opt every tool in explicitly after auditing, or
  restrict with a glob (`allow=["fs.read_*"]`) or a `deny=` block
  list.

```python
from lazybridge.ext.mcp import MCP

# Restrict an internal stdio MCP server to read-only filesystem tools.
fs_safe = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
    allow=["fs.list_*", "fs.read_*"],
    deny=["fs.delete_*", "fs.write_*"],
)
```

`allow` / `deny` patterns use `fnmatch` against the **namespaced**
tool name (`"<server-name>.<tool>"`) — write `"github.delete_*"`,
not `"delete_*"`.

### HumanEngine Web UI

`HumanEngine(ui="web")` starts an HTTP server on `localhost` only. It is not
exposed to the network by default. Do not proxy or forward the port to external
interfaces without adding authentication — the form accepts any POST submission.

### Native provider tools — server-side execution

`Agent(native_tools=[NativeTool.WEB_SEARCH, ...])` enables tools the
**LLM provider** executes server-side, **not** LazyBridge.  Two of
them — `NativeTool.CODE_EXECUTION` and `NativeTool.COMPUTER_USE` —
expose attack surface so large that we require an explicit opt-in:

```python
agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    native_tools=[NativeTool.CODE_EXECUTION],
    allow_dangerous_native_tools=True,  # REQUIRED for CODE_EXECUTION / COMPUTER_USE
)
```

Omit the flag and `Agent.__init__` raises `ValueError`.  The flag is
intentionally noisy — once enabled the model can run arbitrary
Python (provider-sandboxed, but still arbitrary) or drive a remote
desktop on the provider's side.  Wrap such agents in a
`SupervisorEngine` if you need an additional human-approval gate
before tool calls execute.

### Agent Fallback Chains

`Agent(fallback=...)` routes to the fallback engine on any error envelope,
including errors caused by invalid inputs or policy violations. Ensure the
fallback agent enforces the same security invariants (guards, output validation)
as the primary agent.
