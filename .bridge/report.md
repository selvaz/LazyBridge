# LazyBridge provider model refresh report

## Summary

- Updated OpenAI tier routing so `expensive` selects `gpt-6.2-sol` and `top` selects `gpt-6.2-astra`.
- Added GPT-6.2 Sol/Astra Standard token rates from the official OpenAI pricing page, retrieved 2026-11-18, while retaining older model pricing and explicit selection.
- Updated Anthropic's `top` tier to `claude-fable-5-2`, added its official pricing and cache-read rate, and aligned fallback and forced-tool-choice behavior.
- Refreshed the cross-provider tier catalogue, bundled JSON metadata, user documentation, skill references, changelog, and regression tests.

## Verification

- Provider regression test copied outside the repository test tree: 3 passed.
- `ruff check lazybridge tests`: passed.
- `ruff format --check lazybridge tests`: passed.
- `python -m compileall -q lazybridge tests`: passed.
- `python -m json.tool lazybridge/llms.json`: passed.
- Targeted and full `python -m pytest -q` runs are blocked because `pytest-asyncio` is not installed. Installing the test extra was attempted, but the environment's package-index tunnel returned HTTP 403 while resolving `hatchling`.

## Pricing sources

- OpenAI: https://developers.openai.com/api/docs/pricing (retrieved 2026-11-18)
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing (retrieved 2026-11-18)
