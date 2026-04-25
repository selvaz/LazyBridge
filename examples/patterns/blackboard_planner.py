"""``make_blackboard_planner`` — un planner senza DAG, basato su una to-do list.

Alternativa più semplice (e meno precisa) a :func:`plan_tool.make_planner`:
invece di forzare l'LLM a comporre un Plan strutturato (DAG con sentinels,
``parallel=true``, ``from_step``…), gli si dà un blocco-appunti e quattro
funzioni elementari per gestirlo.

Workflow tipico
---------------
1. L'LLM riceve la richiesta dell'utente.
2. Chiama ``set_plan(reasoning, tasks=[...])`` per scrivere la to-do list.
3. Loop: ``get_next() → call_sub_agent(task) → mark_done(idx, summary)``.
4. Quando tutti i task sono spuntati, sintetizza la risposta finale.

Differenze vs. ``make_planner`` (DAG)
------------------------------------
+ Più semplice da capire e da promptare.
+ L'LLM riordina / aggiunge task al volo (basta ``set_plan`` di nuovo).
+ Niente sintassi DAG, niente Pydantic models nidificate.
- Nessun parallelismo automatico — l'LLM esegue i task uno alla volta.
- Nessuna validazione strutturale — l'LLM può "dimenticare" di marcare
  done o saltare task; bisogna confidare nelle istruzioni di sistema.
- Non è "preciso" come un Plan, ma per task che non richiedono fan-out
  o coordinazione fine, funziona benissimo.

Quando usarlo
-------------
- Task aperti dove la struttura emerge eseguendo (research esplorativa).
- Quando il piano cambia spesso in base ai risultati.
- Quando vuoi che l'LLM resti "in controllo" del flow.

Quando preferire ``make_planner``
---------------------------------
- Hai bisogno di parallelismo nativo (``parallel=true`` su step DAG).
- Validazione del piano prima dell'esecuzione (``PlanCompiler``).
- Verifica giudice integrata (``verify=``).
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from lazybridge import Agent, LLMEngine, Tool


# ---------------------------------------------------------------------------
# Stato interno
# ---------------------------------------------------------------------------


@dataclass
class _BlackboardState:
    """Stato della blackboard. Una sola attiva per planner agent."""

    plan_id: str = ""
    reasoning: str = ""
    tasks: list[str] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    revision_count: int = 0
    created_at: float = field(default_factory=time.time)


def _format_state(state: _BlackboardState) -> str:
    if not state.tasks:
        return f"Plan {state.plan_id}: empty (call set_plan to start)."
    done_count = sum(state.done)
    lines = [
        f"Plan {state.plan_id} ({done_count}/{len(state.tasks)} done"
        + (f", revision {state.revision_count}" if state.revision_count else "")
        + ")",
        f"Reasoning: {state.reasoning}",
        "Tasks:",
    ]
    for i, (task, done, result) in enumerate(zip(state.tasks, state.done, state.results)):
        mark = "[x]" if done else "[ ]"
        if done and result:
            lines.append(f"  {i}. {mark} {task}\n       → {result}")
        else:
            lines.append(f"  {i}. {mark} {task}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def make_blackboard_tools() -> list[Tool]:
    """Quattro tool che condividono una blackboard via closure.

    Restituisce ``[set_plan, get_plan, mark_done, get_next]``. Stato singolo
    per istanza — costruisci una nuova istanza per ogni planner agent (o per
    ogni sessione) se vuoi blackboard separate.
    """
    state = _BlackboardState()

    def set_plan(reasoning: str, tasks: list[str]) -> str:
        """Sostituisce il piano corrente con una nuova to-do list.

        Args:
            reasoning: Perché questa lista; cosa pensi di ottenere alla fine.
            tasks: Lista flat di istruzioni in linguaggio naturale, in
                ordine di esecuzione preferito. Ogni task è auto-contenuto.

        Returns:
            Stato della blackboard dopo l'aggiornamento.
        """
        if not reasoning or not reasoning.strip():
            return "REJECTED: reasoning is required (briefly: why this plan)."
        if not tasks:
            return "REJECTED: tasks list must not be empty."
        if state.tasks:
            state.revision_count += 1
        else:
            state.plan_id = uuid.uuid4().hex[:8]
        state.reasoning = reasoning.strip()
        state.tasks = list(tasks)
        state.done = [False] * len(tasks)
        state.results = [""] * len(tasks)
        return _format_state(state)

    def get_plan() -> str:
        """Restituisce lo stato corrente della blackboard."""
        return _format_state(state)

    def mark_done(task_index: int, result_summary: str) -> str:
        """Spunta un task e memorizza un riassunto del risultato.

        Args:
            task_index: Posizione del task (0-based).
            result_summary: Breve sintesi (1-3 frasi) del risultato — non
                il testo completo, solo l'essenziale per i task successivi.

        Returns:
            Stato aggiornato + indicazione del prossimo task TODO se esiste.
        """
        if not state.tasks:
            return "REJECTED: no plan set; call set_plan first."
        if not (0 <= task_index < len(state.tasks)):
            return (
                f"REJECTED: task_index={task_index} out of range "
                f"(valid: 0..{len(state.tasks) - 1})."
            )
        if state.done[task_index]:
            return f"REJECTED: task {task_index} already done."
        if not result_summary or not result_summary.strip():
            return "REJECTED: result_summary is required (1-3 sentences)."
        state.done[task_index] = True
        state.results[task_index] = result_summary.strip()
        nxt = next(
            (i for i, d in enumerate(state.done) if not d),
            None,
        )
        tail = (
            f"\nNext TODO: {nxt}. {state.tasks[nxt]}"
            if nxt is not None
            else "\nAll tasks done — synthesise the final answer for the user."
        )
        return _format_state(state) + tail

    def get_next() -> str:
        """Restituisce il prossimo task non spuntato."""
        if not state.tasks:
            return "no plan set; call set_plan first."
        nxt = next((i for i, d in enumerate(state.done) if not d), None)
        if nxt is None:
            return "all tasks done — synthesise the final answer for the user."
        return f"Next TODO: {nxt}. {state.tasks[nxt]}"

    return [
        Tool(set_plan, mode="signature"),
        Tool(get_plan, mode="signature"),
        Tool(mark_done, mode="signature"),
        Tool(get_next, mode="signature"),
    ]


# ---------------------------------------------------------------------------
# Guidance per il system prompt del planner
# ---------------------------------------------------------------------------

BLACKBOARD_PLANNER_GUIDANCE = """\
# How to handle a request

You have a set of specialist sub-agents available as direct tools, plus
four blackboard tools for managing a to-do list:

- ``set_plan(reasoning, tasks)``     — write the initial plan.
- ``get_plan()``                     — read the current state with checkmarks.
- ``mark_done(task_index, summary)`` — tick a task and record a brief result.
- ``get_next()``                     — convenience: get the next TODO item.

## Workflow

1. **Trivial query** — answer directly. No tools.
2. **One sub-agent suffices** — call that sub-agent directly. No blackboard.
3. **Multi-step work** — use the blackboard:
   a. ``set_plan(reasoning="...", tasks=["...", "...", "..."])`` —
      flat list, in execution order, each task self-contained.
   b. Loop until done:
      - call the right sub-agent for the next task,
      - call ``mark_done(idx, "<short summary of the result>")``.
   c. Once everything is ticked, synthesise the final answer for the user.

## Rules

- ``reasoning`` is required when you set a plan — briefly say why this list
  of tasks fits the request.
- Keep the task list **coarse** (3-6 items). If you find yourself emitting
  10+ tasks, you're over-decomposing.
- ``mark_done`` summaries should be **1-3 sentences** — just enough for
  later tasks to consume. Don't paste full sub-agent outputs.
- You can **revise the plan mid-flow**: call ``set_plan`` again with a
  new list (revision counter increments; previous done state is dropped).
  Use this when a result changes what should happen next.
- If unsure, call ``get_plan()`` to refresh your context.

## Worked example

User: "Research recent agent frameworks and write a one-paragraph summary."

You:
1. ``set_plan(
       reasoning="Two specialists: research finds the frameworks, writer turns
                  the findings into prose. Two tasks; the writer needs the
                  research output as input.",
       tasks=[
           "Research the most discussed AI agent frameworks of 2026.",
           "Write a one-paragraph summary based on the research findings.",
       ],
   )``
2. ``research("Research the most discussed AI agent frameworks of 2026.")``
3. ``mark_done(0, "Top 5 frameworks in 2026: LazyBridge, LangGraph, CrewAI, AutoGen, Smol-Agents.")``
4. ``writer("Write a one-paragraph summary based on the research findings.")``
   (the writer reads its task and uses what's in your context — the prior
   research result.)
5. ``mark_done(1, "Wrote 4-sentence summary covering trade-offs.")``
6. Reply to the user with the writer's paragraph.

## Pitfalls

- **Don't forget to mark_done.** The blackboard is your working memory;
  if you skip ``mark_done`` you'll re-run tasks or lose state.
- **Don't dump full sub-agent output into ``mark_done`` summaries.** Keep
  it short — it's a hint to your future self, not the answer.
- **Don't blackboard a one-shot task.** If one sub-agent call answers the
  question, just call it.
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_blackboard_planner(
    agents: list[Agent],
    *,
    model: str = "claude-opus-4-7",
    system: Optional[str] = None,
    name: str = "blackboard_planner",
    verbose: bool = False,
    verify: Optional[Agent] = None,
    max_verify: int = 3,
) -> Agent:
    """Build a blackboard-style planner :class:`Agent` over the given sub-agents.

    The returned agent has:
      - each sub-agent in ``agents`` as a direct tool,
      - four blackboard tools (``set_plan``, ``get_plan``, ``mark_done``,
        ``get_next``) for managing a flat to-do list shared across the turn.

    Args:
        agents: Sub-agents the planner may call. Each must have a unique
            ``.name``.
        model: Provider model id for the planner LLM.
        system: Override the default system prompt
            (``"You are a generalist assistant.\\n\\n" + BLACKBOARD_PLANNER_GUIDANCE``).
        name: Display name for the planner.
        verbose: Print event traces to stdout.
        verify: Optional judge :class:`Agent` (LazyBridge ``verify=`` loop).
        max_verify: Max judge attempts.

    Raises:
        ValueError: if ``agents`` is empty or contains duplicate names.
    """
    if not agents:
        raise ValueError("make_blackboard_planner: agents list must not be empty")
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError(
            f"make_blackboard_planner: agents must have unique names; got {names}"
        )

    bb_tools = make_blackboard_tools()

    if system is None:
        system = "You are a generalist assistant.\n\n" + BLACKBOARD_PLANNER_GUIDANCE

    return Agent(
        engine=LLMEngine(model, system=system),
        tools=[*agents, *bb_tools],
        name=name,
        verbose=verbose,
        verify=verify,
        max_verify=max_verify,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    def web_search(query: str) -> str:
        """Look up current facts (stub)."""
        return f"[stub web result for {query!r}]"

    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    research = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Look up facts via web_search."),
        tools=[web_search],
        name="research",
        description="Web lookups for current facts. No math.",
    )
    math = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Solve arithmetic with add."),
        tools=[add],
        name="math",
        description="Arithmetic only.",
    )
    writer = Agent(
        engine=LLMEngine("claude-opus-4-7", system="Synthesise prior results into prose."),
        name="writer",
        description="Turns prior results into a short paragraph.",
    )

    planner = make_blackboard_planner([research, math, writer], verbose=True)

    queries = [
        "What does FAANG stand for?",                                         # trivial
        "What is 17 * 23 + 5?",                                               # one agent
        "Research quantum networking and write a one-paragraph brief.",       # multi-step
    ]
    for q in queries:
        print(f"\n>>> {q}")
        print(planner(q).text())


if __name__ == "__main__":
    main()
