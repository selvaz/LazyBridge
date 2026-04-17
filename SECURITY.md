# Security Policy

## Reporting Vulnerabilities

Report security issues to the maintainers via GitHub Issues (private) or email. Do not open public issues for security vulnerabilities.

## Known Security Considerations

### LazyTool.load() — Code Execution Risk

**Severity: CRITICAL**

`LazyTool.load(path)` executes the Python file at `path`. This is by design — it allows loading saved tools from disk. However:

- **NEVER** expose `load()` as a LazyTool or let an LLM control the path argument
- **NEVER** load files from untrusted sources
- Use `base_dir=` to restrict the search path when loading user-provided paths

```python
# SAFE — restricted to known directory
tool = LazyTool.load("tools/my_tool.py", base_dir="tools/")

# DANGEROUS — LLM-controlled path
tool = LazyTool.load(user_provided_path)  # DO NOT DO THIS
```

### LazyStore — No Encryption

`LazyStore` stores data in plain text (JSON in memory or SQLite). Do not store:

- API keys or tokens
- Passwords or credentials
- PII (names, emails, phone numbers) without encryption

### LLMGuard — Prompt Injection

When using `LLMGuard` as a content moderator, the guard agent itself can be subject to prompt injection from the content it's evaluating. Use a separate, hardened model for moderation.

### Provider API Keys

API keys are read from environment variables by default. Never hardcode keys in source code or commit them to version control.
