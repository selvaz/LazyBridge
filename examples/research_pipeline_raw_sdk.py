"""
Research Pipeline — Raw SDK (Anthropic + OpenAI)
=================================================
Researcher agent (Anthropic) searches for company info using tool calling.
Writer agent (OpenAI) produces a structured report.
An orchestrator (Anthropic) drives the full pipeline for multiple companies.

Install: pip install anthropic openai pydantic
"""

import anthropic
from openai import OpenAI
from pydantic import BaseModel

# ── Mock tool ──────────────────────────────────────────────────────────────────

def search_company(name: str) -> str:
    db = {
        "Stripe": "Payments infra, founded 2010, valuation $65 B",
        "Plaid":  "Banking data network, founded 2013, valuation $13 B",
        "Brex":   "Corporate cards & spend management, founded 2017, valuation $12 B",
    }
    return db.get(name, f"No data for {name}")


class Report(BaseModel):
    title: str
    executive_summary: str
    companies: list[str]


# ── Clients ────────────────────────────────────────────────────────────────────

anthropic_client = anthropic.Anthropic()
openai_client    = OpenAI()

tool_schema = [{
    "name": "search_company",
    "description": "Search company information by name.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}]

# ── Helper: researcher loop (Anthropic) ───────────────────────────────────────
# text + tool_use blocks MUST be in the same array — Anthropic API requirement.
# The loop runs until stop_reason == "end_turn" (no more tool calls).

def run_researcher(task: str) -> str:
    messages = [{"role": "user", "content": task}]
    while True:
        resp = anthropic_client.messages.create(
            model="claude-opus-4-6", max_tokens=1024,
            tools=tool_schema, messages=messages,
        )
        assistant_blocks = []
        for block in resp.content:
            if block.type == "text":
                assistant_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_blocks.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_blocks})

        if resp.stop_reason == "end_turn":
            return next((b.text for b in resp.content if b.type == "text"), "")

        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": search_company(**block.input),
                })
        messages.append({"role": "user", "content": tool_results})


# ── Helper: writer (OpenAI structured output) ─────────────────────────────────

def run_writer(research: str) -> Report:
    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a business analyst writer."},
            {"role": "user", "content": f"Write a structured report based on:\n\n{research}"},
        ],
        response_format=Report,
    )
    return completion.choices[0].message.parsed


# ── Pipeline function (researcher → writer) ───────────────────────────────────

def run_research_pipeline(task: str) -> str:
    research = run_researcher(task)
    report   = run_writer(research)
    return report.model_dump_json()


# ── Orchestrator tool schema ───────────────────────────────────────────────────

orchestrator_tool_schema = [{
    "name": "run_research_pipeline",
    "description": "Research companies and produce a structured report.",
    "input_schema": {
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    },
}]

# ── Orchestrator loop (third manual while loop) ───────────────────────────────

orch_messages = [{
    "role": "user",
    "content": "Research Stripe, Plaid, and Brex. Call the pipeline for each company.",
}]

while True:
    orch_resp = anthropic_client.messages.create(
        model="claude-opus-4-6", max_tokens=2048,
        tools=orchestrator_tool_schema, messages=orch_messages,
    )
    orch_blocks = []
    for block in orch_resp.content:
        if block.type == "text":
            orch_blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            orch_blocks.append({
                "type": "tool_use", "id": block.id,
                "name": block.name, "input": block.input,
            })
    orch_messages.append({"role": "assistant", "content": orch_blocks})

    if orch_resp.stop_reason == "end_turn":
        for block in orch_resp.content:
            if block.type == "text":
                print(block.text)
        break

    orch_results = []
    for block in orch_resp.content:
        if block.type == "tool_use":
            orch_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": run_research_pipeline(**block.input),
            })
    orch_messages.append({"role": "user", "content": orch_results})
