# Card 1: Research desk

**One lead agent. Its tools are a pipeline of agents and the agents themselves.**

You ask a question. The lead (Claude Code, Opus) runs a deterministic research
pipeline: two researchers search the web in parallel, one for and one against,
then a reviewer audits them and a writer, a token-based API model, produces a cited brief. If a gap
still matters, the lead calls a specialist directly to dig deeper.

<div class="lb-card-figure" markdown="0">
--8<-- "docs/cards/research-desk.svg"
</div>

## How it is built

1. **Prompts first.** Every instruction an LLM sees is a constant at the top.
2. **Two specialists.** `deep_research` and `fact_check` are agents on two
   different engines, read-only, with web search.
3. **A pipeline that is itself an agent.** `Agent(engine=Plan(...))` runs the
   same two specialists as steps: `support` and `challenge` in parallel
   (`parallel=True`), then `review` reads both (`from_parallel_all`), then
   `summary` reads the research and the review (`from_step`). The summary is
   written by DeepSeek over its API: subscription engines and token-based
   models are peers in the same pipeline.
4. **Stack them as tools.** The lead gets `tools=[research_brief, sonnet, terra]`.
   No glue code: an agent, or a whole pipeline, is just a tool.

## The code

```python
--8<-- "examples/cards/research_desk.py"
```

## Run it

```console
python research_desk.py "Should a small team start with SQLite or Postgres?"
```

Install everything with [one command](index.md#install-everything-with-one-command).
You must [sign in to Claude Code and Codex](index.md#sign-in-to-claude-code-and-codex-required)
once before the first run; with a subscription login they use your plan's quota.
