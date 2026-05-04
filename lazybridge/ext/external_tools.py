"""External tool providers for server-side managed integrations.

This extension adapts tools exposed by an external gateway into normal
LazyBridge :class:`~lazybridge.tools.Tool` objects.  It is intended for
Pipedream/Composio/Arcade/custom backends that already own OAuth,
credential storage, policy, and audit logging.

Secrets must stay in the external gateway.  LazyBridge receives only tool
schemas and sanitized tool results.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from lazybridge.tools import Tool

_JSON_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": True}


@dataclass(frozen=True)
class ExternalToolSpec:
    """A remotely hosted tool definition.

    ``parameters`` is the provider-agnostic JSON Schema object used by
    LazyBridge providers when advertising tools to an LLM.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    strict: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExternalToolSpec:
        """Build a spec from common external registry shapes.

        Accepted inputs:
        - ``{"name", "description", "parameters"}``
        - OpenAI-style ``{"function": {"name", "description", "parameters"}}``
        """
        data: Mapping[str, Any]
        if isinstance(raw.get("function"), Mapping):
            data = raw["function"]  # type: ignore[index]
        else:
            data = raw

        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("External tool spec must include a non-empty string name")

        description = data.get("description")
        if description is None:
            description = f"Call external tool {name}."
        if not isinstance(description, str):
            raise ValueError(f"External tool {name!r} description must be a string")

        parameters = data.get("parameters") or _JSON_OBJECT_SCHEMA
        if not isinstance(parameters, Mapping):
            raise ValueError(f"External tool {name!r} parameters must be a JSON Schema object")

        strict = bool(data.get("strict", raw.get("strict", False)))
        return cls(name=name, description=description, parameters=parameters, strict=strict)


class ExternalToolError(RuntimeError):
    """Raised when an external tool registry or execution call fails."""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@runtime_checkable
class ExternalToolClient(Protocol):
    """Client protocol used by :class:`ExternalToolProvider`.

    Implement this protocol for Pipedream, Composio, Arcade, an internal
    gateway, or a test double.  Implementations should keep credentials
    server-side and should not return secrets in tool results.
    """

    def list_tools(self) -> Iterable[ExternalToolSpec | Mapping[str, Any]]:
        """Return the current tool registry."""
        ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Execute one remote tool."""
        ...


class JsonHttpExternalToolClient:
    """Small stdlib HTTP client for a JSON external-tool gateway.

    Default endpoint contract:
    - ``GET  {base_url}/tools`` returns either ``[{...}]`` or ``{"tools": [{...}]}``.
    - ``POST {base_url}/tools/{name}/call`` with ``{"arguments": {...}}`` returns JSON.

    This class is intentionally narrow.  For Pipedream/Composio/Arcade,
    wrap their SDK/API behind :class:`ExternalToolClient` when their HTTP
    shape differs from the default contract.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        tools_path: str = "/tools",
        call_path_template: str = "/tools/{name}/call",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.tools_path = tools_path
        self.call_path_template = call_path_template

    def list_tools(self) -> list[ExternalToolSpec]:
        payload = self._request("GET", self.tools_path)
        if isinstance(payload, Mapping):
            payload = payload.get("tools", [])
        if not isinstance(payload, list):
            raise ExternalToolError("External tool registry must return a list or {'tools': list}", body=payload)
        return [ExternalToolSpec.from_mapping(item) for item in payload]

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        path = self.call_path_template.format(name=urllib.parse.quote(name, safe=""))
        return self._request("POST", path, {"arguments": dict(arguments)})

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        if not path.startswith("/"):
            path = f"/{path}"
        url = f"{self.base_url}{path}"
        encoded: bytes | None = None
        headers = {"Accept": "application/json", **self.headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        if body is not None:
            encoded = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=encoded, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read()
                if not data:
                    return None
                return json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed_body: Any = json.loads(error_body)
            except json.JSONDecodeError:
                parsed_body = error_body
            raise ExternalToolError(
                f"External tool gateway returned HTTP {exc.code} for {method} {path}",
                status=exc.code,
                body=parsed_body,
            ) from exc
        except urllib.error.URLError as exc:
            raise ExternalToolError(f"External tool gateway request failed for {method} {path}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ExternalToolError(f"External tool gateway returned invalid JSON for {method} {path}") from exc


class ExternalToolProvider:
    """Expose an external tool registry as a LazyBridge tool provider.

    ``Agent(tools=[ExternalToolProvider(client)])`` expands the provider
    into normal :class:`Tool` objects through LazyBridge's existing
    ``_is_lazy_tool_provider`` hook.
    """

    _is_lazy_tool_provider = True

    def __init__(
        self,
        client: ExternalToolClient,
        *,
        specs: Iterable[ExternalToolSpec | Mapping[str, Any]] | None = None,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        name_prefix: str = "",
        strict: bool | None = None,
    ) -> None:
        self.client = client
        self._specs = list(specs) if specs is not None else None
        self.include = set(include or []) or None
        self.exclude = set(exclude or [])
        self.name_prefix = name_prefix
        self.strict = strict

    def list_specs(self) -> list[ExternalToolSpec]:
        raw_specs = self._specs if self._specs is not None else list(self.client.list_tools())
        specs = [
            spec if isinstance(spec, ExternalToolSpec) else ExternalToolSpec.from_mapping(spec) for spec in raw_specs
        ]
        if self.include is not None:
            specs = [spec for spec in specs if spec.name in self.include]
        if self.exclude:
            specs = [spec for spec in specs if spec.name not in self.exclude]
        return specs

    def as_tools(self) -> list[Tool]:
        return [self._tool_from_spec(spec) for spec in self.list_specs()]

    def _tool_from_spec(self, spec: ExternalToolSpec) -> Tool:
        tool_name = f"{self.name_prefix}{spec.name}"
        remote_name = spec.name

        async def _call_external(**kwargs: Any) -> Any:
            return await _call_client(self.client, remote_name, kwargs)

        _call_external.__name__ = tool_name
        _call_external.__doc__ = spec.description

        return Tool.from_schema(
            tool_name,
            spec.description,
            dict(spec.parameters),
            _call_external,
            strict=spec.strict if self.strict is None else self.strict,
        )


async def _call_client(client: ExternalToolClient, name: str, arguments: Mapping[str, Any]) -> Any:
    maybe_async = getattr(client, "acall_tool", None)
    if maybe_async is not None:
        return await maybe_async(name, arguments)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: client.call_tool(name, arguments))


__all__ = [
    "ExternalToolClient",
    "ExternalToolError",
    "ExternalToolProvider",
    "ExternalToolSpec",
    "JsonHttpExternalToolClient",
]
