# quant_agent — Pre-configured quantitative analysis agent

A fully-equipped quantitative financial analysis agent with a hybrid architecture:
complex tools get dedicated sub-agent pipelines (agent_tool) for intelligent
parameter construction; simple tools use direct tool calling for efficiency.

## Install

```bash
pip install lazybridge[stats,downloader]
```

## Quick start

```python
from lazybridge.external_pipelines.quant_agent import quant_agent

agent, rt = quant_agent("anthropic")
resp = agent("Download SPY, AAPL, and MSFT. Analyze their volatility.")
print(resp.text())
rt.close()
```

## Architecture

```
quant_agent (skills: methodology, tool guide, downloader guide)
│
│ agent_tool pipelines (2x LLM calls — NL→structured params→function):
├── analyze            (complex: mode resolution, target inference)
├── fit_model          (complex: family selection, param tuning)
├── download_tickers   (complex: ticker list from NL description)
├── query_data         (complex: NL→SQL translation)
│
│ plain tools (1x LLM call — direct arg filling):
├── discover_data       (no params)
├── discover_analyses   (simple: optional dataset + limit)
├── register_dataset    (simple: user provides explicit values)
├── list_universe       (simple: optional filter)
├── search_tickers      (simple: single query string)
├── profile_dataset     (simple: single name)
├── forecast_model      (simple: run_id + steps)
├── run_diagnostics     (simple: name + column)
├── compare_models      (simple: list of run_ids)
├── get_run             (simple: single run_id)
├── list_runs           (simple: optional dataset + limit)
├── list_artifacts      (simple: run_id + optional type)
└── get_plot            (simple: run_id + name)
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `provider` | `"anthropic"` | LLM provider |
| `model` | provider default | Model override |
| `artifacts_dir` | `"artifacts"` | Where to store analysis artifacts and ticker cache |
| `cache_dir` | `{artifacts_dir}/ticker_cache` | Ticker data cache |
| `name` | `"quant_analyst"` | Agent name |
| `system` | `QUANT_SYSTEM_PROMPT` | System prompt override |

## Returns

`(agent, runtime)` tuple:
- `agent` — an `Agent` with all tools pre-bound
- `runtime` — a `StatRuntime` instance. Call `rt.close()` when done.

## Dependencies

Requires both `stat_runtime` and `data_downloader` extensions:
- `lazybridge.external_tools.stat_runtime` — statistical analysis runtime
- `lazybridge.external_tools.data_downloader` — market data ingestion
