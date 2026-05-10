# Error recovery cheat-sheet

Condensed pairings of common LazyBridge errors and the canonical
fix.  All errors follow the four-part body documented in
[codegen-contract.md](codegen-contract.md): what failed, what was
passed, what's expected, **and a concrete fix snippet** at the bottom.
This page gathers the most common shapes for quick LLM lookup.

| Error / Pattern | Cause | Fix |
|---|---|---|
| `PlanCompileError: Step 'X' (#n) — context=from_step('typo') references unknown step` | Sentinel points at a step that doesn't exist (typo, forward-ref, or auto-named ``_anon_*`` step) | Replace with the suggested name from the error's "Did you mean 'Y'?" hint, or pass a real ``name=`` to the referent step |
| `PlanCompileError: same parallel band` | ``from_step("sibling")`` inside a parallel band — siblings can't see each other's output | Either move the reader *after* the band, or use ``from_parallel_all("first_in_band")`` to aggregate |
| `PlanCompileError: from_agent('X') — missing 'store='` | The source agent has no ``store=`` so its output can't be persisted for cross-step read | Pass ``store=Store(...)`` to the source Agent, or use ``from_step('step_name')`` instead |
| `ValueError: MCP.stdio('X') requires an explicit allow=[...] or deny=[...]` | 0.7.9 deny-by-default for MCP servers | Pass ``allow=["*"]`` after auditing, or restrict with a glob ``allow=["fs.read_*"]`` |
| `ValueError: Agent(name=...) is required when engine is not LLMEngine` | T7 — non-LLM engines (Plan/Supervisor/Human) can't auto-name from a model string | Add ``name="my_agent"`` to the Agent constructor |
| `ValueError: Agent(model=..., engine=...) — can't pass both` | T6 — ``model=`` is shorthand for ``LLMEngine(model)``; ``engine=`` is the explicit form | Pick one — drop ``model=`` if you want a custom engine |
| `Tool.__init__: mode='auto' was removed in 0.7.9` | 0.7.9 deleted the graceful-fallback ladder | Pass ``mode="signature"`` (introspect, default), ``mode="llm"`` (let an LLM build the schema), or ``mode="hybrid"`` |
| `AttributeError: type object 'Agent' has no attribute 'from_chain'` | 0.7.9 deleted ``Agent.from_chain`` / ``from_engine`` / ``from_model`` / ``from_plan`` / ``from_parallel`` | Use ``Agent.chain(a, b)`` / ``Agent.parallel(...)`` or ``Agent(engine=...)`` for the others |
| `ValueError: LLMEngine(tool_choice='parallel') was removed in 0.7.9` | 0.7.9 deleted this knob — concurrent tools are the default and can't be disabled | Drop the argument (or use ``tool_choice='auto'`` / ``'any'``) |
| `UnsupportedNativeToolError: provider X does not support CODE_EXECUTION` | The provider's ``supported_native_tools`` doesn't cover the requested tool | Either swap to a provider that does (see ``lazybridge.matrix.provider_capabilities()``), or pass ``strict_native_tools=False`` to let the tool be silently dropped |
| `UnsupportedFeatureError: provider X with structured_output + tools` | DeepSeek-legacy / OpenAI-image limits — provider can't combine both | Either drop ``output_type=`` for the tool-using turn, or drop the conflicting tool |
| `ValueError: EncryptedStoreAdapter: stored value is not an lb-enc-v1 token` | Adapter pointed at a plaintext Store (mixed-mode reads are unsafe) | Either decrypt+re-encrypt the legacy rows, or unwrap the adapter to read them |
| `StreamStallError: no token in 90s` | Provider stream went silent past ``stream_idle_timeout`` | Lower the timeout, or wrap the call in retry logic, or switch to non-streaming |
| `ToolTimeoutError: tool X exceeded 30s` | Tool wall-clock exceeded ``tool_timeout`` | Raise the timeout on the ``LLMEngine``, or make the tool faster |
| `RuntimeError: async def functions are not natively supported` | ``pytest-asyncio`` not installed in the test env | ``pip install -e '.[test]'`` (see [CONTRIBUTING.md](../../CONTRIBUTING.md)) |
| `ModuleNotFoundError: lazybridge.external_tools.report_builder` | Reporting moved to ``selvaz/LazyReport`` in 0.7.9 | ``pip install lazybridge-reports``, then ``import lazybridge_reports`` |
| `ImportError: EncryptedStoreAdapter requires 'cryptography'` | Opt-in extra not installed | ``pip install 'lazybridge[encryption]'`` |
| `PackageNotFoundError: lazybridge` reads as version 0.7.0 | Stale dev install + new source tree | ``pip install -e .`` to refresh the editable install's metadata |

## When the error doesn't appear in the table

1. Read the bottom of the error message verbatim — the four-part body
   ends with a literal **fix snippet**, often the exact replacement
   line of code.
2. Cross-reference the symbol against [`lazybridge/llms.json`](../../lazybridge/llms.json) — the
   ``avoid`` list and ``renames`` map cover every 0.7.9 deletion.
3. If the symbol is from a 0.4-era import path
   (``LazyAgent``/``LazyTool``/``LazySession``/``LazyContext``), see
   the [migration guide](../migrations/0.7-to-0.79.md).
