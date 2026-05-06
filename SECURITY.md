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
def run_cmd(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True).decode()

agent = Agent("anthropic", tools=[run_cmd])  # DO NOT DO THIS

# SAFER — whitelist valid operations instead of passing raw commands
ALLOWED_COMMANDS = {"status": "git status", "log": "git log --oneline -5"}
def run_safe_cmd(command: str) -> str:
    """Run a pre-approved git command."""
    if command not in ALLOWED_COMMANDS:
        return f"Unknown command. Choose from: {list(ALLOWED_COMMANDS)}"
    return subprocess.check_output(ALLOWED_COMMANDS[command].split()).decode()
```

### Store — No Encryption

`Store` persists data in plain text (JSON in memory or SQLite on disk). Do not store:

- API keys or tokens
- Passwords or credentials
- PII (names, emails, phone numbers) without encryption

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
agent = Agent("anthropic")

# Bad — key in code
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # don't do this
```

### HumanEngine Web UI

`HumanEngine(ui="web")` starts an HTTP server on `localhost` only. It is not
exposed to the network by default. Do not proxy or forward the port to external
interfaces without adding authentication — the form accepts any POST submission.

### Agent Fallback Chains

`Agent(fallback=...)` routes to the fallback engine on any error envelope,
including errors caused by invalid inputs or policy violations. Ensure the
fallback agent enforces the same security invariants (guards, output validation)
as the primary agent.
