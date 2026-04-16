"""tool_schema — pipeline for making a Python function LLM-friendly.

Owns everything between a raw Python callable and a provider-ready ToolDefinition:
  - Type annotation → JSON Schema conversion
  - Docstring parameter description extraction
  - Pydantic-backed argument validation + coercion
  - SIGNATURE / LLM / HYBRID compilation modes
  - Compilation fingerprinting (deterministic hash of all inputs)
  - Compiled artifact structure (ToolCompileArtifact)
  - Artifact caching (ArtifactStore / InMemoryArtifactStore)

LazyTool (in lazy_tool.py) is the only consumer of this module's public API.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import re
import threading
import types as _types
import typing
import warnings
import weakref as _weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from typing import Any

from pydantic import (
    BaseModel as _BaseModel,
)
from pydantic import (
    ValidationError as _ValidationError,
)
from pydantic import (
    create_model as _create_model,
)

from lazybridge.core.types import ToolDefinition

_logger = logging.getLogger(__name__)

# Bump these when compile logic or LLM prompt templates change.
_COMPILER_VERSION = "1"
_LLM_PROMPT_VERSION = "1"


# ---------------------------------------------------------------------------
# Public exceptions (re-exported via tools.py)
# ---------------------------------------------------------------------------

class ToolArgumentValidationError(ValueError):
    """Raised when tool call arguments fail validation before execution."""


class ToolSchemaBuildError(RuntimeError):
    """Raised when ToolSchemaBuilder fails to produce a valid ToolDefinition."""


# ---------------------------------------------------------------------------
# ToolSchemaMode
# ---------------------------------------------------------------------------

class ToolSchemaMode(StrEnum):
    """How the ToolBridge generates the canonical ToolDefinition."""

    SIGNATURE = "signature"  # deterministic introspection — default
    LLM = "llm"              # schema generated entirely by an external LLM
    HYBRID = "hybrid"        # SIGNATURE types + LLM descriptions


# ---------------------------------------------------------------------------
# ToolSourceStatus
# ---------------------------------------------------------------------------

class ToolSourceStatus(StrEnum):
    """Records how the final ToolDefinition in a ToolCompileArtifact was produced."""

    BASELINE_ONLY = "baseline_only"
    """SIGNATURE only — no LLM was involved."""

    LLM_ENRICHED = "llm_enriched"
    """HYBRID — SIGNATURE supplied types; LLM enriched descriptions."""

    LLM_INFERRED = "llm_inferred"
    """LLM mode — full schema inferred by the LLM."""

    FALLBACK_TO_BASELINE = "fallback_to_baseline"
    """HYBRID or LLM was requested but fell back to SIGNATURE (schema_llm absent or failed)."""


# ---------------------------------------------------------------------------
# ToolCompileArtifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCompileArtifact:
    """Full output of a single ToolSchemaBuilder compilation run.

    Carries both the result (``definition``) and structured metadata about
    how it was produced.  Useful for drift detection, debugging, and auditing.

    Fields:
        fingerprint        Deterministic 24-char hex identifying all compile inputs.
        compiler_version   Version of the compile logic that produced this artifact.
        prompt_version     Version of the LLM prompt templates (relevant for LLM/HYBRID).
        mode               Compilation mode requested.
        source_status      How the final definition was actually produced.
        definition         The resulting ToolDefinition (provider-ready).
        baseline_definition SIGNATURE-only baseline; set for LLM and HYBRID modes.
        llm_enriched_fields Parameter names whose descriptions were supplied by the LLM.
        warnings           Structured warnings emitted during compilation.
        cache_hit          True when this artifact was retrieved from an ArtifactStore.
    """

    fingerprint: str
    compiler_version: str
    prompt_version: str
    mode: ToolSchemaMode
    source_status: ToolSourceStatus
    definition: ToolDefinition
    baseline_definition: ToolDefinition | None
    llm_enriched_fields: frozenset[str]
    warnings: tuple[str, ...]
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# ArtifactStore protocol + InMemoryArtifactStore
# ---------------------------------------------------------------------------

class ArtifactStore(typing.Protocol):
    """Minimal interface for ToolCompileArtifact caching.

    ``get`` returns None on a cache miss.
    ``put`` stores an artifact (keyed by its ``fingerprint``).
    """

    def get(self, fingerprint: str) -> ToolCompileArtifact | None: ...
    def put(self, artifact: ToolCompileArtifact) -> None: ...


class InMemoryArtifactStore:
    """Thread-safe in-memory ArtifactStore backed by a plain dict.

    Suitable for single-process use.  For multi-process or persistent caching
    implement ArtifactStore with a database or file backend.
    """

    def __init__(self) -> None:
        self._store: dict[str, ToolCompileArtifact] = {}
        self._lock = threading.Lock()

    def get(self, fingerprint: str) -> ToolCompileArtifact | None:
        with self._lock:
            return self._store.get(fingerprint)

    def put(self, artifact: ToolCompileArtifact) -> None:
        with self._lock:
            self._store[artifact.fingerprint] = artifact

    def clear(self) -> None:
        """Remove all cached artifacts."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Phase 1 — compile input + fingerprinting (private)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CompileInput:
    """Canonical, hashable representation of all inputs to schema compilation."""

    func_qualname: str
    func_source_hash: str
    name: str
    description: str
    mode: str           # ToolSchemaMode.value
    strict: bool
    schema_llm_id: str  # stable identifier; "" when schema_llm is None
    compiler_version: str
    prompt_version: str

    def fingerprint(self) -> str:
        """Return a deterministic 24-char hex hash of all compile inputs."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _func_source_hash(func: Callable) -> str:
    """Return a 16-char hex hash of ``func``'s source code (or best fallback)."""
    try:
        src = inspect.getsource(func)
    except OSError:
        try:
            sig_str = str(inspect.signature(func))
        except Exception:
            sig_str = "(...)"
        doc = inspect.getdoc(func) or ""
        src = f"def {func.__name__}{sig_str}: ...\n"
        if doc:
            src += f'    """{doc}"""\n'
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _schema_llm_id(schema_llm: Any) -> str:
    """Return a stable string identifier for ``schema_llm``.

    Uses ``__qualname__`` for functions/methods; ``type(obj).__qualname__``
    for class instances.  Returns "" when schema_llm is None.
    """
    if schema_llm is None:
        return ""
    if hasattr(schema_llm, "__qualname__"):
        return getattr(schema_llm, "__qualname__")
    return type(schema_llm).__qualname__


def _make_compile_input(
    func: Callable,
    name: str,
    description: str,
    mode: ToolSchemaMode,
    strict: bool,
    schema_llm: Any,
) -> _CompileInput:
    return _CompileInput(
        func_qualname=getattr(func, "__qualname__", getattr(func, "__name__", repr(func))),
        func_source_hash=_func_source_hash(func),
        name=name,
        description=description,
        mode=mode.value,
        strict=strict,
        schema_llm_id=_schema_llm_id(schema_llm),
        compiler_version=_COMPILER_VERSION,
        prompt_version=_LLM_PROMPT_VERSION,
    )


# ---------------------------------------------------------------------------
# Schema generation helpers (private)
# ---------------------------------------------------------------------------

_PY_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

# Python 3.10+ native union type (str | int).  None on older versions.
_NATIVE_UNION_TYPE: type | None = getattr(_types, "UnionType", None)


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Recursively convert a Python type annotation to a JSON Schema dict."""
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ()) or ()

    # Detect both typing.Union[X, Y] and native str | int (Python 3.10+)
    is_union = origin is typing.Union or (
        _NATIVE_UNION_TYPE is not None and isinstance(annotation, _NATIVE_UNION_TYPE)
    )
    if is_union and not args:
        args = getattr(annotation, "__args__", ()) or ()

    # Optional[X]  ->  {"anyOf": [schema(X), {"type": "null"}]}
    if is_union:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) < len(args)
        if len(non_none) == 1 and has_none:
            return {"anyOf": [_annotation_to_schema(non_none[0]), {"type": "null"}]}
        return {"anyOf": [_annotation_to_schema(a) for a in args]}

    # Annotated[X, metadata] -> base schema + optional description from metadata
    if hasattr(annotation, "__metadata__"):
        base = _annotation_to_schema(args[0]) if args else {}
        for meta in annotation.__metadata__:
            if isinstance(meta, str) and meta:
                return {**base, "description": meta}
            # pydantic.fields.FieldInfo and similar objects with .description
            if hasattr(meta, "description") and isinstance(getattr(meta, "description"), str):
                desc = meta.description
                if desc:
                    return {**base, "description": desc}
        return base

    # Literal["a", "b"] -> {"type": "string", "enum": ["a", "b"]}
    # type must be declared alongside enum — required by Gemini and OpenAI strict mode.
    if origin is typing.Literal:
        enum_vals = list(args)
        if all(isinstance(v, str) for v in enum_vals):
            return {"type": "string", "enum": enum_vals}
        if all(isinstance(v, int) for v in enum_vals):
            return {"type": "integer", "enum": enum_vals}
        return {"enum": enum_vals}

    # list[X]
    if origin is list:
        return {"type": "array", "items": _annotation_to_schema(args[0])} if args else {"type": "array"}

    # set[X] / frozenset[X] — arrays with uniqueItems
    if origin is set or origin is frozenset:
        base = {"type": "array", "uniqueItems": True}
        if args:
            base["items"] = _annotation_to_schema(args[0])
        return base

    # tuple[X, ...] (homogeneous) / tuple[X, Y, Z] (fixed-length)
    if origin is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _annotation_to_schema(args[0])}
        return {
            "type": "array",
            "prefixItems": [_annotation_to_schema(a) for a in args],
            "items": False,
        }

    # dict[K, V] — simple object
    if origin is dict:
        return {"type": "object"}

    # Primitives
    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}

    # Enum subclass -> {"type": "string"/"integer", "enum": [...]}
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        enum_vals = [e.value for e in annotation]
        if all(isinstance(v, str) for v in enum_vals):
            return {"type": "string", "enum": enum_vals}
        if all(isinstance(v, int) for v in enum_vals):
            return {"type": "integer", "enum": enum_vals}
        return {"enum": enum_vals}

    # Pydantic BaseModel subclass — use its JSON Schema directly.
    # $defs produced by nested models are preserved as-is: Anthropic and
    # OpenAI both support JSON Schema draft 7+ $ref resolution natively.
    # For providers that require a flat schema, pre-flatten with
    # model.model_json_schema(mode='serialization') before passing to ToolBridge.
    if inspect.isclass(annotation) and issubclass(annotation, _BaseModel):
        schema = annotation.model_json_schema(mode="serialization")
        schema.pop("title", None)
        return schema

    # inspect.Parameter.empty, typing.Any, unknown -> {}
    return {}


def _parse_docstring_params(doc: str) -> dict[str, str]:
    """Extract {param: description} from Google-style or Sphinx-style docstrings."""
    if not doc:
        return {}
    params: dict[str, str] = {}

    # Sphinx/reST: ":param [type] name: description"
    for m in re.finditer(r":param(?:\s+\S+)?\s+(\w+)\s*:\s*([^\n:][^\n]*)", doc):
        params[m.group(1)] = m.group(2).strip()
    if params:
        return params

    # Google/NumPy: "Args:\n    name [(type)]: description"
    section = re.search(r"(?:Args|Arguments|Parameters)\s*:\s*\n((?:[ \t]+\S[^\n]*\n?)+)", doc)
    if section:
        for m in re.finditer(r"^[ \t]+(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", section.group(1), re.MULTILINE):
            params[m.group(1)] = m.group(2).strip()

    return params


_arg_model_cache: _weakref.WeakKeyDictionary = _weakref.WeakKeyDictionary()


def _make_arg_model(func: Callable) -> type | None:
    """Build and cache a Pydantic validation model for ``func``'s arguments.

    Returns None for uninspectable callables or zero-arg functions.
    Uses ``extra='forbid'`` unless the function accepts **kwargs.

    Uses a WeakKeyDictionary keyed on (func, annotations_id) so:
    - Entries are removed automatically when functions are garbage-collected.
    - Stale models are not returned when a function's annotations are mutated
      (dynamic tool factories, hot-reload in development).
    """
    # Cache lookup strategy:
    # _arg_model_cache is a WeakKeyDictionary keyed on the function object itself.
    # Each cached value is a (annotations_id, model) tuple.  annotations_id is
    # id(func.__annotations__) at store time.  If the annotations dict is later
    # replaced (e.g. by a hot-reload or dynamic decorator), its id changes and
    # cached[0] != current_annotations_id, so the cache is considered stale and
    # the model is rebuilt.  cache_key[1] is pre-computed once to avoid two
    # getattr calls.
    cache_key = (func, id(getattr(func, "__annotations__", None)))
    cached = _arg_model_cache.get(func)
    if cached is not None and cached[0] == cache_key[1]:
        return cached[1]
    try:
        sig = inspect.signature(func)
        hints = typing.get_type_hints(func, include_extras=True)
    except (TypeError, ValueError, AttributeError) as exc:
        _logger.debug("Cannot inspect signature for %r: %s", getattr(func, "__qualname__", func), exc)
        return None

    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, Any)
        fields[name] = (
            (annotation, ...) if param.default is inspect.Parameter.empty else (annotation, param.default)
        )

    if not fields:
        return None  # zero-arg or pure **kwargs — nothing to validate

    try:
        from pydantic import ConfigDict
        model = _create_model(
            f"_{func.__name__}_args",
            __config__=ConfigDict(
                extra="allow" if accepts_kwargs else "forbid",
                arbitrary_types_allowed=True,
            ),
            **fields,
        )
        _arg_model_cache[func] = (id(getattr(func, "__annotations__", None)), model)
        return model
    except (TypeError, ValueError) as exc:
        _logger.debug("Cannot build arg model for %r: %s", getattr(func, "__qualname__", func), exc)
        return None


def _validate_and_coerce_arguments(func: Callable, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce tool call arguments via a generated Pydantic model.

    Returns a coerced copy of ``arguments`` (``model_dump()`` of the validated
    model).  This ensures type coercion (e.g. int→float) happens before the
    callable receives the payload.

    Raises ToolArgumentValidationError on missing required args, unexpected
    args (when **kwargs is absent), or type/value violations.
    """
    model_cls = _make_arg_model(func)
    if model_cls is None:
        return arguments  # uninspectable or zero-arg — pass through

    try:
        validated = model_cls.model_validate(arguments)  # type: ignore[attr-defined]
        return validated.model_dump()
    except _ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ToolArgumentValidationError(
            f"Invalid arguments for '{func.__name__}': {errors}"
        ) from exc


# ---------------------------------------------------------------------------
# LLM-assisted schema helpers (private)
# ---------------------------------------------------------------------------

class _LLMParamDef(_BaseModel):
    """Single parameter description returned by the LLM in LLM mode."""

    type: str = ""
    description: str = ""
    required: bool = True


class _LLMToolSchema(_BaseModel):
    """Full tool schema returned by the LLM in LLM mode."""

    name: str
    description: str
    params: dict[str, _LLMParamDef] = {}


class _LLMEnrichment(_BaseModel):
    """Descriptions returned by the LLM in HYBRID mode (structure from SIGNATURE)."""

    description: str = ""
    param_descriptions: dict[str, str] = {}


def _call_schema_llm(schema_llm: Any, prompt: str, schema: type) -> Any:
    """Invoke schema_llm and return a validated Pydantic model instance.

    Supports two interfaces:
      - object with ``.json(prompt, schema=...)`` (ModelBridge / LazyLayer)
      - plain callable ``(prompt: str) -> dict``

    Raises ToolSchemaBuildError with context on any failure.
    """
    try:
        if hasattr(schema_llm, "json"):
            result = schema_llm.json(prompt, schema=schema)
            if isinstance(result, dict):
                return schema.model_validate(result)  # type: ignore[attr-defined]
            return result
        raw = schema_llm(prompt)
        return schema.model_validate(raw)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ToolSchemaBuildError(
            f"schema_llm failed to produce a valid tool schema "
            f"(schema={schema.__name__}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# $ref flattening utility
# ---------------------------------------------------------------------------

def _flatten_refs(schema: dict) -> dict:
    """Inline all ``$ref`` / ``$defs`` entries in a JSON Schema, returning a flat copy.

    Only ``#/$defs/<Name>`` references are resolved (JSON Schema draft 7 style).
    Circular references are detected and left as-is to avoid infinite recursion.

    This is an opt-in post-processing step; providers that handle ``$ref``
    natively (Anthropic, OpenAI) do not require it.

    Algorithm
    ---------
    ``_resolve(node, visited)`` walks the schema tree recursively:

    * Lists — recurse into each element.
    * Non-dict leaves — returned unchanged.
    * ``{"$ref": "#/$defs/Foo"}`` nodes — looked up in the top-level ``$defs``
      dict and replaced by a deep-copied, recursively-resolved copy of the
      definition.  If ``Foo`` is already in ``visited`` (a ``frozenset[str]``)
      the ref is left as-is, breaking the cycle.
    * All other dicts — recurse into each value.

    ``visited`` is a ``frozenset`` (immutable), so each recursive call gets its
    own independent view of which defs have been expanded on the current path.
    This is correct for DAG-shaped schemas where a definition can be referenced
    from multiple independent branches without being considered circular.
    """
    defs = schema.get("$defs", {})
    if not defs:
        return schema

    def _resolve(node: Any, visited: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [_resolve(item, visited) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref: str = node["$ref"]
            if ref.startswith("#/$defs/"):
                def_name = ref[len("#/$defs/"):]
                # Circular reference guard: if this def is already being
                # expanded on the current path, leave the $ref intact.
                if def_name not in visited and def_name in defs:
                    # deep-copy so that mutations of the inlined node do not
                    # corrupt the original $defs entry when it is used elsewhere.
                    return _resolve(copy.deepcopy(defs[def_name]), visited | {def_name})
            return node
        return {k: _resolve(v, visited) for k, v in node.items()}

    # Strip "$defs" from the output — all definitions are now inlined.
    # Each top-level value starts with an empty visited set (independent paths).
    flat = {k: _resolve(copy.deepcopy(v), frozenset()) for k, v in schema.items() if k != "$defs"}
    return flat


# ---------------------------------------------------------------------------
# ToolSchemaBuilder
# ---------------------------------------------------------------------------

class ToolSchemaBuilder:
    """Generates a canonical ToolDefinition (or ToolCompileArtifact) from a Python callable.

    Compilation pipeline (``build_artifact`` calls ``_compile``, which routes
    to the mode-specific helpers):

    Phase 1 — Input canonicalisation (``_make_compile_input``)
        Compute stable identifiers for all compile inputs (function qualname,
        source hash, name, description, mode, strict flag, schema_llm id,
        compiler/prompt versions).

    Phase 2 — Fingerprinting (``_CompileInput.fingerprint``)
        SHA-256 of the JSON-serialised inputs, truncated to 24 hex chars.
        Used as the cache key for the ArtifactStore.

    Phase 3 — Cache lookup (``build_artifact``)
        If an ArtifactStore is configured, return the cached artifact
        immediately (with ``cache_hit=True``) to avoid redundant introspection
        or LLM calls.

    Phase 4 — Schema generation (``_build_*_mode`` helpers)
        SIGNATURE: Python type introspection + docstring param extraction.
        LLM:       Full schema inferred by an external LLM; SIGNATURE baseline
                   kept for drift detection.
        HYBRID:    SIGNATURE supplies types; LLM enriches descriptions only.

    Phase 5 — Ref flattening (optional, ``_flatten_refs``)
        If ``flatten_refs=True``, inline all ``$ref``/``$defs`` entries so
        providers that do not support JSON Schema $ref resolution receive a
        flat schema.

    Phase 6 — Invariant checks & warnings
        LLM/HYBRID modes verify that all signature-required parameters appear
        in the LLM output; missing ones are stubbed and a warning is emitted.

    Three modes:

    SIGNATURE (default)
        Pure Python introspection — deterministic, fast, no external deps.

    LLM  *(experimental)*
        Full schema inferred by an external LLM.  Requires ``schema_llm``.

    HYBRID
        SIGNATURE supplies parameter names and types; the LLM enriches
        descriptions.  Requires ``schema_llm``; falls back to SIGNATURE
        with a warning when ``schema_llm`` is None.

    ``schema_llm`` accepts any object with a ``.json(prompt, *, schema)``
    method (ModelBridge / LazyLayer) or a plain callable ``(str) -> dict``.

    Args:
        artifact_store: Optional ArtifactStore used to cache and retrieve
            ToolCompileArtifact objects.  Defaults to None (no caching).
    """

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        *,
        flatten_refs: bool = False,
    ) -> None:
        self._store = artifact_store
        self._flatten_refs = flatten_refs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = False,
        mode: ToolSchemaMode = ToolSchemaMode.SIGNATURE,
        schema_llm: Any | None = None,
    ) -> ToolDefinition:
        """Build and return the ToolDefinition (backward-compatible shortcut).

        Delegates to ``build_artifact()`` and returns ``artifact.definition``.
        """
        return self.build_artifact(
            func,
            name=name,
            description=description,
            strict=strict,
            mode=mode,
            schema_llm=schema_llm,
        ).definition

    def build_artifact(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = False,
        mode: ToolSchemaMode = ToolSchemaMode.SIGNATURE,
        schema_llm: Any | None = None,
    ) -> ToolCompileArtifact:
        """Compile a ToolCompileArtifact from ``func``.

        Checks the ArtifactStore first (if configured).  On a cache miss,
        compiles and stores the artifact before returning it.

        Fingerprinting strategy
        -----------------------
        The fingerprint is computed from a canonical ``_CompileInput`` that
        captures every input that could affect the compiled output: the
        function's qualified name, a hash of its source code, the effective
        name and description, mode, strict flag, a stable id for schema_llm,
        and the compiler/prompt version constants.  SHA-256 over the JSON
        serialisation ensures the fingerprint is both deterministic and
        collision-resistant for practical caches.
        """
        # Resolve effective name/description here so the fingerprint is stable
        # regardless of whether the caller passes explicit overrides.
        effective_name = name or func.__name__
        if description is not None:
            effective_desc = description
        else:
            doc = inspect.getdoc(func) or ""
            # Use only the first line (summary sentence) of the docstring so
            # minor docstring edits don't invalidate the fingerprint.
            effective_desc = doc.splitlines()[0].strip() if doc else ""

        compile_input = _make_compile_input(
            func, effective_name, effective_desc, mode, strict, schema_llm
        )
        # 24-char hex fingerprint uniquely identifies this compile configuration.
        fp = compile_input.fingerprint()

        if self._store is not None:
            cached = self._store.get(fp)
            if cached is not None:
                # Cache hit — return a copy with cache_hit=True.
                # ToolCompileArtifact is a frozen dataclass so dataclasses.replace
                # is used to produce a modified copy without mutating the stored obj.
                from dataclasses import replace as _replace
                return _replace(cached, cache_hit=True)

        # For LLM/HYBRID modes, pass the original (possibly None) description so the
        # LLM result takes precedence over the auto-extracted docstring.
        desc_for_compile = (
            description
            if mode in (ToolSchemaMode.LLM, ToolSchemaMode.HYBRID)
            else effective_desc
        )
        artifact = self._compile(
            func,
            name=effective_name,
            description=desc_for_compile,
            strict=strict,
            mode=mode,
            schema_llm=schema_llm,
            fingerprint=fp,
        )

        if self._flatten_refs:
            # Opt-in: inline all $ref/$defs for providers that don't handle them.
            from dataclasses import replace as _replace
            flat_params = _flatten_refs(artifact.definition.parameters)
            flat_defn = _replace(artifact.definition, parameters=flat_params)
            artifact = _replace(artifact, definition=flat_defn)

        if self._store is not None:
            # Store after flattening so cached artifacts are also ref-free.
            self._store.put(artifact)

        return artifact

    # ------------------------------------------------------------------
    # Internal compilation
    # ------------------------------------------------------------------

    def _compile(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        description: str,
        strict: bool,
        mode: ToolSchemaMode,
        schema_llm: Any | None,
        fingerprint: str,
    ) -> ToolCompileArtifact:
        """Route to the correct mode and wrap the result in a ToolCompileArtifact."""
        compile_warnings: list[str] = []

        if mode == ToolSchemaMode.LLM:
            # LLM mode: the entire schema (name, description, parameter types AND
            # descriptions) is inferred by an external LLM from the function source.
            # A SIGNATURE baseline is always computed first so it can be stored in
            # the artifact for drift detection, even when LLM succeeds.
            if schema_llm is None:
                raise ValueError(
                    "schema_llm is required when mode=ToolSchemaMode.LLM. "
                    "Pass a ModelBridge/LazyLayer instance or a callable (str) -> dict."
                )
            baseline = self._build_signature_mode(func, name=name, description=description, strict=strict)
            try:
                defn, llm_fields, extra_warnings = self._build_llm_mode(
                    func, name=name, description=description, strict=strict, schema_llm=schema_llm
                )
                compile_warnings.extend(extra_warnings)
            except ToolSchemaBuildError as _llm_err:
                # Graceful degradation: if the LLM call fails, fall back to the
                # SIGNATURE baseline rather than raising to the caller.
                msg = (
                    f"ToolSchemaMode.LLM schema_llm call failed ({_llm_err}); "
                    "falling back to SIGNATURE mode."
                )
                warnings.warn(msg, stacklevel=4)
                compile_warnings.append(msg)
                return ToolCompileArtifact(
                    fingerprint=fingerprint,
                    compiler_version=_COMPILER_VERSION,
                    prompt_version=_LLM_PROMPT_VERSION,
                    mode=mode,
                    source_status=ToolSourceStatus.FALLBACK_TO_BASELINE,
                    definition=baseline,
                    baseline_definition=baseline,
                    llm_enriched_fields=frozenset(),
                    warnings=tuple(compile_warnings),
                )
            return ToolCompileArtifact(
                fingerprint=fingerprint,
                compiler_version=_COMPILER_VERSION,
                prompt_version=_LLM_PROMPT_VERSION,
                mode=mode,
                source_status=ToolSourceStatus.LLM_INFERRED,
                definition=defn,
                baseline_definition=baseline,  # kept for audit/drift comparison
                llm_enriched_fields=frozenset(llm_fields),
                warnings=tuple(compile_warnings),
            )

        if mode == ToolSchemaMode.HYBRID:
            # HYBRID mode: SIGNATURE provides authoritative parameter types;
            # the LLM enriches only the description strings.  This is safer
            # than full LLM mode because type information is never hallucinated.
            baseline = self._build_signature_mode(func, name=name, description=description, strict=strict)
            if schema_llm is not None:
                try:
                    defn, llm_fields, extra_warnings = self._build_hybrid_mode(
                        func, name=name, description=description, strict=strict,
                        schema_llm=schema_llm, baseline=baseline,
                    )
                    compile_warnings.extend(extra_warnings)
                except ToolSchemaBuildError as _llm_err:
                    # Graceful degradation: use baseline if enrichment fails.
                    msg = (
                        f"ToolSchemaMode.HYBRID schema_llm call failed ({_llm_err}); "
                        "falling back to SIGNATURE mode."
                    )
                    warnings.warn(msg, stacklevel=4)
                    compile_warnings.append(msg)
                    return ToolCompileArtifact(
                        fingerprint=fingerprint,
                        compiler_version=_COMPILER_VERSION,
                        prompt_version=_LLM_PROMPT_VERSION,
                        mode=mode,
                        source_status=ToolSourceStatus.FALLBACK_TO_BASELINE,
                        definition=baseline,
                        baseline_definition=baseline,
                        llm_enriched_fields=frozenset(),
                        warnings=tuple(compile_warnings),
                    )
                return ToolCompileArtifact(
                    fingerprint=fingerprint,
                    compiler_version=_COMPILER_VERSION,
                    prompt_version=_LLM_PROMPT_VERSION,
                    mode=mode,
                    source_status=ToolSourceStatus.LLM_ENRICHED,
                    definition=defn,
                    baseline_definition=baseline,
                    llm_enriched_fields=frozenset(llm_fields),
                    warnings=tuple(compile_warnings),
                )
            else:
                # schema_llm=None with HYBRID requested — degrade gracefully.
                msg = "ToolSchemaMode.HYBRID requires schema_llm; fallback to SIGNATURE mode."
                warnings.warn(msg, stacklevel=4)
                compile_warnings.append(msg)
                return ToolCompileArtifact(
                    fingerprint=fingerprint,
                    compiler_version=_COMPILER_VERSION,
                    prompt_version=_LLM_PROMPT_VERSION,
                    mode=mode,
                    source_status=ToolSourceStatus.FALLBACK_TO_BASELINE,
                    definition=baseline,
                    baseline_definition=baseline,
                    llm_enriched_fields=frozenset(),
                    warnings=tuple(compile_warnings),
                )

        # SIGNATURE mode: pure Python introspection — deterministic and fast.
        # baseline_definition is None because there is no LLM baseline to compare.
        defn = self._build_signature_mode(func, name=name, description=description, strict=strict)
        return ToolCompileArtifact(
            fingerprint=fingerprint,
            compiler_version=_COMPILER_VERSION,
            prompt_version=_LLM_PROMPT_VERSION,
            mode=mode,
            source_status=ToolSourceStatus.BASELINE_ONLY,
            definition=defn,
            baseline_definition=None,
            llm_enriched_fields=frozenset(),
            warnings=tuple(compile_warnings),
        )

    def _build_signature_mode(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        description: str,
        strict: bool,
    ) -> ToolDefinition:
        """Generate ToolDefinition via Python introspection."""
        sig = inspect.signature(func)

        # include_extras=True preserves Annotated[T, metadata] so we can
        # extract descriptions from e.g. Annotated[int, "The count"].
        try:
            resolved_hints = typing.get_type_hints(func, include_extras=True)
        except (NameError, AttributeError) as exc:
            _logger.debug(
                "Cannot resolve type hints for %r (forward ref or missing attr): %s",
                func.__qualname__, exc,
            )
            resolved_hints = {}
        except Exception as exc:
            _logger.error(
                "Could not resolve type hints for %r: %s. "
                "Parameter types will be untyped.",
                func.__qualname__, exc,
            )
            resolved_hints = {}

        doc_params = _parse_docstring_params(inspect.getdoc(func) or "")

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = resolved_hints.get(param_name, inspect.Parameter.empty)
            prop: dict[str, Any] = (
                {} if annotation is inspect.Parameter.empty else _annotation_to_schema(annotation)
            )
            if "description" not in prop and param_name in doc_params:
                prop["description"] = doc_params[param_name]
            properties[param_name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        json_schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            json_schema["required"] = required
        if strict:
            json_schema["additionalProperties"] = False

        return ToolDefinition(
            name=name,
            description=description,
            parameters=json_schema,
            strict=strict,
        )

    def _build_llm_mode(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        description: str,
        strict: bool,
        schema_llm: Any,
    ) -> tuple[ToolDefinition, list[str], list[str]]:
        """Generate ToolDefinition entirely via LLM.

        Returns ``(definition, llm_param_names, warnings)``.
        """
        compile_warnings: list[str] = []
        source = _get_source_or_stub(func)

        prompt = (
            "Analyze this Python function and generate a tool definition for it.\n\n"
            f"```python\n{source}\n```\n\n"
            "Return a JSON object with:\n"
            '- name: snake_case tool name\n'
            '- description: one clear sentence describing what the tool does\n'
            '- params: object where each key is a parameter name and the value has:\n'
            '  - type: one of "string","integer","number","boolean","array","object"'
            ' (empty string if unknown)\n'
            '  - description: what this parameter is for\n'
            '  - required: true if the parameter has no default value\n'
        )

        result: _LLMToolSchema = _call_schema_llm(schema_llm, prompt, _LLMToolSchema)

        # Signature is authoritative for required params.
        sig_required = _sig_required_params(func)

        properties: dict[str, Any] = {}
        required: list[str] = []
        llm_param_names: list[str] = []

        for param_name, param_def in result.params.items():
            prop: dict[str, Any] = {}
            if param_def.type:
                prop["type"] = param_def.type
            if param_def.description:
                prop["description"] = param_def.description
            properties[param_name] = prop
            llm_param_names.append(param_name)
            if param_def.required or param_name in sig_required:
                required.append(param_name)

        # Ensure signature-required params not mentioned by LLM are still required.
        for sig_req in sig_required:
            if sig_req in properties and sig_req not in required:
                required.append(sig_req)

        # Phase 6 invariant: warn if LLM param set diverges significantly.
        llm_param_set = set(result.params.keys())
        if sig_required and llm_param_set:
            missing_required = sig_required - llm_param_set
            if missing_required:
                msg = (
                    f"LLM did not mention required parameter(s) {sorted(missing_required)!r} "
                    f"for tool '{name}'. They will be marked required but may lack descriptions."
                )
                _logger.warning(msg)
                compile_warnings.append(msg)
                # Add a minimal stub so the JSON Schema is valid:
                # every item in 'required' must have a matching key in 'properties'.
                for sig_req in missing_required:
                    if sig_req not in properties:
                        properties[sig_req] = {}
                    if sig_req not in required:
                        required.append(sig_req)

        json_schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            json_schema["required"] = required
        if strict:
            json_schema["additionalProperties"] = False

        return (
            ToolDefinition(
                name=name or result.name,
                description=description or result.description,
                parameters=json_schema,
                strict=strict,
            ),
            llm_param_names,
            compile_warnings,
        )

    def _build_hybrid_mode(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        description: str,
        strict: bool,
        schema_llm: Any,
        baseline: ToolDefinition,
    ) -> tuple[ToolDefinition, list[str], list[str]]:
        """SIGNATURE types + LLM descriptions.

        Returns ``(definition, enriched_param_names, warnings)``.
        """
        compile_warnings: list[str] = []
        source = _get_source_or_stub(func)

        prompt = (
            "Analyze this Python function and describe it for use as an AI tool.\n\n"
            f"```python\n{source}\n```\n\n"
            "Return a JSON object with:\n"
            "- description: one clear sentence describing what the tool does\n"
            "- param_descriptions: object mapping each parameter name to a brief description\n"
        )

        enrichment: _LLMEnrichment = _call_schema_llm(schema_llm, prompt, _LLMEnrichment)

        # Phase 6: warn if LLM returned an empty description (fall back to baseline).
        llm_desc = enrichment.description.strip()
        if not llm_desc:
            msg = f"LLM returned empty description for tool '{name}'; using SIGNATURE baseline."
            _logger.warning(msg)
            compile_warnings.append(msg)

        props: dict[str, Any] = copy.deepcopy(baseline.parameters.get("properties", {}))
        enriched: list[str] = []
        for param_name, desc in enrichment.param_descriptions.items():
            if param_name in props and desc:
                props[param_name]["description"] = desc
                enriched.append(param_name)

        merged_params = {**baseline.parameters, "properties": props}
        effective_desc = description or llm_desc or baseline.description

        return (
            ToolDefinition(
                name=baseline.name,
                description=effective_desc,
                parameters=merged_params,
                strict=strict,
            ),
            enriched,
            compile_warnings,
        )


# ---------------------------------------------------------------------------
# Helpers shared across build modes
# ---------------------------------------------------------------------------

def _get_source_or_stub(func: Callable) -> str:
    """Return source code or a best-effort stub for ``func``."""
    try:
        return inspect.getsource(func)
    except OSError:
        try:
            sig_str = str(inspect.signature(func))
        except (TypeError, ValueError):
            sig_str = "(...)"
        doc = inspect.getdoc(func) or ""
        src = f"def {func.__name__}{sig_str}: ...\n"
        if doc:
            src += f'    """{doc}"""\n'
        return src


def _sig_required_params(func: Callable) -> set[str]:
    """Return the set of parameter names that are required (no default)."""
    try:
        sig = inspect.signature(func)
        return {
            p for p, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            and p not in ("self", "cls")
        }
    except (TypeError, ValueError):
        return set()


# ---------------------------------------------------------------------------
# Module-level default builder (no artifact store)
# ---------------------------------------------------------------------------

_DEFAULT_BUILDER = ToolSchemaBuilder()
