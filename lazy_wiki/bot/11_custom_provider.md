# Custom Providers — Complete Reference

`BaseProvider` is a **stable public extension point**. Implement it to connect any LLM backend — local models, private APIs, custom wrappers — to the full LazyBridge stack.

---

## 1. Stability contract

The following are guaranteed stable across minor versions:

| Surface | What it does |
|---|---|
| `BaseProvider.__init__(api_key, model, **kwargs)` | Constructor — kwargs forwarded to `_init_client` |
| `_init_client(**kwargs)` | Override to build your SDK client |
| `complete(request)` → `CompletionResponse` | Sync completion |
| `stream(request)` → `Iterator[StreamChunk]` | Sync streaming |
| `acomplete(request)` → `CompletionResponse` | Async completion |
| `astream(request)` → `AsyncIterator[StreamChunk]` | Async streaming generator |
| `default_model: str` | Class-level default model name |
| `supported_native_tools: frozenset[NativeTool]` | Declare supported native tools |
| `get_default_max_tokens(model)` | Per-model max tokens cap |
| `_resolve_model(request)` | Helper: request → instance → class default |
| `_compute_cost(model, input_tokens, output_tokens)` | Override for cost tracking |
| `_check_native_tools(tools)` | Filters unsupported native tools with warning |

Breaking changes will follow a deprecation cycle and a minor-version bump.

---

## 2. Minimal implementation

```python
from collections.abc import AsyncIterator, Iterator
from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest, CompletionResponse, StreamChunk, UsageStats
)

class EchoProvider(BaseProvider):
    """Trivial provider that echoes the last user message — no API call."""

    default_model = "echo-1"

    def _init_client(self, **kwargs) -> None:
        pass  # No SDK needed

    def _last_user_message(self, request: CompletionRequest) -> str:
        for msg in reversed(request.messages):
            if msg.role.value == "user":
                from lazybridge.core.types import TextContent
                for block in (msg.content if isinstance(msg.content, list) else []):
                    if isinstance(block, TextContent):
                        return block.text
                if isinstance(msg.content, str):
                    return msg.content
        return ""

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        text = self._last_user_message(request)
        return CompletionResponse(
            content=f"[echo] {text}",
            model=self._resolve_model(request),
            usage=UsageStats(input_tokens=len(text), output_tokens=len(text) + 7),
        )

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        text = f"[echo] {self._last_user_message(request)}"
        for char in text:
            yield StreamChunk(delta=char)
        yield StreamChunk(
            delta="",
            stop_reason="end_turn",
            is_final=True,
            usage=UsageStats(input_tokens=10, output_tokens=len(text)),
        )

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        return self.complete(request)   # sync is fine for trivial providers

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        for chunk in self.stream(request):
            yield chunk
```

Usage:

```python
from lazybridge import LazyAgent

agent = LazyAgent(EchoProvider())
print(agent.chat("hello world").content)   # "[echo] hello world"
```

---

## 3. Real-world example — Ollama (local models)

```python
import asyncio
from collections.abc import AsyncIterator, Iterator
import httpx

from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest, CompletionResponse, StreamChunk, UsageStats
)


class OllamaProvider(BaseProvider):
    """Provider for Ollama local LLM server (https://ollama.ai)."""

    default_model = "llama3"

    def _init_client(self, base_url: str = "http://localhost:11434", **kwargs) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=120)
        self._aclient = httpx.AsyncClient(base_url=self._base_url, timeout=120)

    def _build_messages(self, request: CompletionRequest) -> list[dict]:
        msgs = []
        if request.system:
            msgs.append({"role": "system", "content": request.system})
        for msg in request.messages:
            if isinstance(msg.content, str):
                msgs.append({"role": msg.role.value, "content": msg.content})
            else:
                text = " ".join(
                    b.text for b in msg.content
                    if hasattr(b, "text")
                )
                msgs.append({"role": msg.role.value, "content": text})
        return msgs

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = {
            "model": self._resolve_model(request),
            "messages": self._build_messages(request),
            "stream": False,
        }
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["message"]["content"]
        usage = data.get("eval_count", 0)
        return CompletionResponse(
            content=content,
            model=self._resolve_model(request),
            stop_reason="end_turn",
            usage=UsageStats(output_tokens=usage),
            raw=data,
        )

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        payload = {
            "model": self._resolve_model(request),
            "messages": self._build_messages(request),
            "stream": True,
        }
        with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                import json
                data = json.loads(line)
                delta = data.get("message", {}).get("content", "")
                done = data.get("done", False)
                if done:
                    yield StreamChunk(
                        delta=delta,
                        stop_reason="end_turn",
                        is_final=True,
                        usage=UsageStats(output_tokens=data.get("eval_count", 0)),
                    )
                else:
                    yield StreamChunk(delta=delta)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        payload = {
            "model": self._resolve_model(request),
            "messages": self._build_messages(request),
            "stream": False,
        }
        resp = await self._aclient.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return CompletionResponse(
            content=data["message"]["content"],
            model=self._resolve_model(request),
            stop_reason="end_turn",
            usage=UsageStats(output_tokens=data.get("eval_count", 0)),
            raw=data,
        )

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        import json
        payload = {
            "model": self._resolve_model(request),
            "messages": self._build_messages(request),
            "stream": True,
        }
        async with self._aclient.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                delta = data.get("message", {}).get("content", "")
                done = data.get("done", False)
                if done:
                    yield StreamChunk(
                        delta=delta,
                        stop_reason="end_turn",
                        is_final=True,
                        usage=UsageStats(output_tokens=data.get("eval_count", 0)),
                    )
                else:
                    yield StreamChunk(delta=delta)
```

```python
from lazybridge import LazyAgent

# Point to local Ollama server
agent = LazyAgent(OllamaProvider(model="llama3"))
resp = agent.chat("Explain gradient descent in one sentence.")
print(resp.content)
```

---

## 4. Adding cost tracking

```python
class MyProvider(BaseProvider):
    _PRICES = {
        "my-model-v1":   (0.50,  1.50),   # ($/1M input, $/1M output)
        "my-model-mini": (0.10,  0.30),
    }

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        for key, (inp, out) in self._PRICES.items():
            if key in model:
                return (input_tokens * inp + output_tokens * out) / 1_000_000
        return None
```

Cost is then available on every response:

```python
resp = agent.chat("hello")
print(f"${resp.usage.cost_usd:.6f}")
```

---

## 5. Declaring native tool support

```python
from lazybridge.core.types import NativeTool

class MySearchProvider(BaseProvider):
    supported_native_tools = frozenset({NativeTool.WEB_SEARCH})

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        # _check_native_tools filters and warns about unsupported ones
        active_native = self._check_native_tools(request.native_tools)
        # ... pass active_native to your SDK
```

---

## 6. Registering the provider with the string alias

Built-in string aliases (`"anthropic"`, `"openai"`, etc.) are resolved in
`lazybridge/core/executor.py`. Custom providers **do not need a string alias** —
pass the instance directly:

```python
agent = LazyAgent(MyProvider(api_key="...", model="my-model-v1"))
```

If you want a string alias for your organisation's internal use, subclass and
register via the `PROVIDER_MAP` in `executor.py` — but note this map is **not**
part of the stable contract and may change.

---

## 7. Rules for correct implementation

**You MUST:**
- Return a `CompletionResponse` with `content: str` set (even if empty).
- Yield a final `StreamChunk` with `is_final=True` and `stop_reason` set.
- Use `_resolve_model(request)` inside `complete/stream`, not `self.model` directly.
- Not mutate `request` — it is shared and treated as read-only.

**You SHOULD:**
- Populate `usage.input_tokens` and `usage.output_tokens`.
- Set `raw=<original SDK response>` so callers can access provider-specific fields.
- Implement `_compute_cost` if your API has known pricing.
- Override `get_default_max_tokens` if your model cap differs from 4096.

**You MUST NOT:**
- Block the event loop in `acomplete` / `astream`.
- Catch and swallow exceptions — let them propagate for the retry layer.
- Use `asyncio.run()` inside async methods.
