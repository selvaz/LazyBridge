# Examples

## Research Pipeline — the flagship comparison

Three implementations of the same pipeline:
- **Researcher** (Anthropic): searches for company info using tool calling
- **Writer** (OpenAI): produces a structured Pydantic report
- **Orchestrator** (Anthropic): drives the pipeline for multiple companies

| File | Approach | Lines | Dependencies |
|---|---|---|---|
| `research_pipeline_raw_sdk.py` | Anthropic + OpenAI SDK, fully manual | ~110 | 2 packages |
| `research_pipeline_langgraph.py` | LangGraph + LangChain | ~65 | 4 packages |
| `research_pipeline_lazybridge.py` | LazyBridge | ~35 | 1 package |

### What the comparison shows

| | Raw SDK | LangGraph | LazyBridge |
|---|---|---|---|
| Manual tool loops | 3 `while` loops | 0 | 0 |
| Explicit state declaration | no | yes (TypedDict) | no |
| Separate LLM clients / classes | 2 | 2 | 1 unified API |
| Multi-provider swap | rewrite | change class | change string |
| Session tracking | absent | LangSmith (external) | built-in |

### Running

Each file is self-contained. Set your API keys and run:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

python examples/research_pipeline_lazybridge.py
```
