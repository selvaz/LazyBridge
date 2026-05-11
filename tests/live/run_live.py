"""Live pipeline rungs — eseguibili uno ad uno in Spyder.

Come usarlo:
  - Esegui prima la cella "Setup" (Ctrl+Enter)
  - Poi esegui qualsiasi rung con Ctrl+Enter
  - Imposta VIZ_OPEN = True nel Setup per aprire il browser su ogni rung

Modelli:
  - MODEL_CHEAP    = claude-haiku-4-5  (default)
  - MODEL_CAPABLE  = claude-haiku-4-5

Override con variabili d'ambiente:
  os.environ["LB_LIVE_MODEL"]          = "deepseek-chat"
  os.environ["LB_LIVE_MODEL_CAPABLE"]  = "gpt-4o-mini"
  os.environ["OPENAI_API_KEY"]         = "sk-..."
  os.environ["DEEPSEEK_API_KEY"]       = "sk-..."
"""

# %% Setup — esegui sempre prima
from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from lazybridge import (
    Agent,
    LLMEngine,
    Memory,
    Plan,
    Session,
    Step,
    Store,
    Tool,
    from_agent,
    from_parallel_all,
    from_prev,
    from_start,
    from_step,
)
from lazybridge.ext.viz import Visualizer

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

MODEL_CHEAP   = os.getenv("LB_LIVE_MODEL",         "claude-haiku-4-5")
MODEL_CAPABLE = os.getenv("LB_LIVE_MODEL_CAPABLE", "claude-haiku-4-5")

# True = apre il browser per ogni rung (utile in sessioni di debug interattive)
VIZ_OPEN = False

# Spyder/IPython: nest_asyncio evita "Event loop is closed" da httpx cleanup
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helper: ogni rung parte con Session + Visualizer già avviato
# ---------------------------------------------------------------------------

@contextmanager
def _rung(label: str):
    """Context manager per ogni rung: Session + Visualizer + cleanup automatico."""
    tmp = Path(tempfile.mkdtemp())
    sess = Session(db=str(tmp / "live.db"), console=True)
    with Visualizer(sess, auto_open=VIZ_OPEN) as viz:
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"  viz  → {viz.url}")
        print(f"  db   → {tmp / 'live.db'}")
        print(f"{'─'*60}")
        yield sess, tmp
    sess.close()


print("Setup OK")
print(f"  models : {MODEL_CHEAP} / {MODEL_CAPABLE}")
print(f"  VIZ_OPEN = {VIZ_OPEN}  (imposta True per aprire il browser)")


# %% Rung 1 — single agent (bare call)
# Verifica: provider auth, executor, Envelope metadata
with _rung("Rung 1 — single agent") as (sess, tmp):
    agent = Agent(
        engine=LLMEngine(MODEL_CHEAP, system="You are terse. Follow instructions exactly."),
        session=sess,
        name="rung1",
    )
    env = agent("Reply with exactly the word: PONG")

    assert "PONG" in env.text().upper(), f"Got: {env.text()!r}"
    assert env.metadata.input_tokens > 0
    assert env.metadata.cost_usd >= 0

    print(f"✓ text       : {env.text()!r}")
    print(f"  tokens     : {env.metadata.input_tokens}in / {env.metadata.output_tokens}out")
    print(f"  cost       : ${env.metadata.cost_usd:.6f}")
    print(f"  latency_ms : {env.metadata.latency_ms:.0f}")


# %% Rung 2 — agent with tool
# Verifica: tool schema, dispatch, result injection, eventi tool_call/tool_result
with _rung("Rung 2 — agent with tool") as (sess, tmp):
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    agent = Agent(
        engine=LLMEngine(MODEL_CHEAP, system="Use tools when asked. Reply concisely."),
        tools=[Tool.wrap(multiply, name="multiply")],
        session=sess,
        name="rung2",
    )
    env = agent("What is 6 multiplied by 7? Use the multiply tool.")

    assert "42" in env.text(), f"Got: {env.text()!r}"

    event_types = [e["event_type"] for e in sess.events.query()]
    assert "tool_call"   in event_types, f"No tool_call: {event_types}"
    assert "tool_result" in event_types, f"No tool_result: {event_types}"

    print(f"✓ text   : {env.text()!r}")
    print(f"  events : {event_types}")


# %% Rung 3 — structured output
# Verifica: Pydantic output parsing, payload tipizzato
with _rung("Rung 3 — structured output") as (sess, tmp):
    class Coords(BaseModel):
        x: int
        y: int

    agent = Agent(
        engine=LLMEngine(MODEL_CHEAP, system="Return only the requested JSON fields."),
        output=Coords,
        session=sess,
        name="rung3",
    )
    env = agent("Return x=4 y=9.")

    assert isinstance(env.payload, Coords), f"Expected Coords, got {type(env.payload)}"
    assert env.payload.x == 4, f"x={env.payload.x}"
    assert env.payload.y == 9, f"y={env.payload.y}"

    print(f"✓ payload : {env.payload}")


# %% Rung 4 — sequential Plan + event tracking
# Verifica: Plan compile, from_prev, AGENT_START/FINISH nel session log
with _rung("Rung 4 — sequential Plan") as (sess, tmp):
    fetch = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Reply only with: DATA:42"),
        name="fetch",
        session=sess,
    )
    analyse = Agent(
        engine=LLMEngine(
            MODEL_CAPABLE,
            system="Extract the integer after 'DATA:' and reply with only that integer.",
        ),
        name="analyse",
        session=sess,
    )
    pipeline = Agent(
        engine=Plan(Step("fetch"), Step("analyse", task=from_prev)),
        tools=[fetch, analyse],
        session=sess,
        name="pipeline",
    )
    env = pipeline("run")

    assert "42" in env.text(), f"Got: {env.text()!r}"

    event_types = [e["event_type"] for e in sess.events.query()]
    assert "agent_start"  in event_types
    assert "agent_finish" in event_types
    assert len(event_types) >= 4

    print(f"✓ text   : {env.text()!r}")
    print(f"  events : {event_types}")
    print(f"  cost   : ${env.metadata.cost_usd:.6f}")


# %% Rung 5 — parallel plan steps
# Verifica: parallel band, from_parallel_all fan-in, data flow ra→merger e rb→merger
with _rung("Rung 5 — parallel steps") as (sess, tmp):
    ra = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Follow the task instruction exactly."),
        name="ra", session=sess,
    )
    rb = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Follow the task instruction exactly."),
        name="rb", session=sess,
    )
    merger = Agent(
        engine=LLMEngine(
            MODEL_CAPABLE,
            system="You receive labeled results from two agents. List what each agent returned.",
        ),
        name="merger", session=sess,
    )
    pipeline = Agent(
        engine=Plan(
            Step("ra",     task="Reply with exactly one word: ALPHA", parallel=True),
            Step("rb",     task="Reply with exactly one word: BETA",  parallel=True),
            Step("merger", task=from_parallel_all("ra")),
        ),
        tools=[ra, rb, merger],
        session=sess,
        name="par_pipeline",
    )
    env = pipeline("run")

    assert env.text(), "merger returned empty"

    events = sess.events.query()
    tool_results  = [e for e in events if e["event_type"] == "tool_result"]
    branch_results = {
        e["payload"].get("branch_id"): e["payload"].get("result", "")
        for e in tool_results
        if e["payload"].get("branch_id") in ("ra", "rb")
    }
    assert "ra" in branch_results, f"No result from branch ra"
    assert "rb" in branch_results, f"No result from branch rb"

    print(f"✓ ra output : {branch_results['ra'][:60]!r}")
    print(f"  rb output : {branch_results['rb'][:60]!r}")
    print(f"  merger    : {env.text()[:100]!r}")
    print(f"  cost      : ${env.metadata.cost_usd:.6f}")


# %% Rung 6 — agent as tool (nested agent)
# Verifica: agent-as-tool wrapping, cost roll-up attraverso il confine tool
with _rung("Rung 6 — agent as tool") as (sess, tmp):
    translator = Agent(
        engine=LLMEngine(
            MODEL_CHEAP,
            system="Translate the given text to French. Reply with only the translation.",
        ),
        name="translator",
        session=sess,
    )
    outer = Agent(
        engine=LLMEngine(MODEL_CHEAP, system="Use the translator tool when asked to translate."),
        tools=[translator],
        session=sess,
        name="outer",
    )
    env = outer("Translate 'hello' to French using the translator tool.")

    assert any(w in env.text().lower() for w in ["bonjour", "salut", "allô"]), (
        f"Got: {env.text()!r}"
    )

    print(f"✓ translation : {env.text()!r}")
    print(f"  total tokens: {env.metadata.input_tokens}in / {env.metadata.output_tokens}out")


# %% Rung 7 — Agent.chain + Visualizer (replay after run)
# Verifica: Visualizer lifecycle, SSE, graph schema, event log, replay constructor
with _rung("Rung 7 — chain + Visualizer") as (sess, tmp):
    def web_search(query: str) -> str:
        """Search the web and return a short summary."""
        time.sleep(0.2)
        return f"[stub] top result for '{query}': Python is a popular language."

    researcher = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Find one key fact. Use the web_search tool."),
        tools=[Tool.wrap(web_search, name="web_search")],
        name="researcher",
        session=sess,
    )
    writer = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Write one sentence summarising the findings."),
        name="writer",
        session=sess,
    )
    env = Agent.chain(researcher, writer)("What is Python used for?")

    assert env.text(), "writer returned empty"

    events     = sess.events.query()
    event_types = {e["event_type"] for e in events}
    assert "agent_start"  in event_types
    assert "agent_finish" in event_types
    assert len(events) > 4

    # replay: costruisce un Visualizer offline dal DB
    viz_replay = Visualizer.replay(str(tmp / "live.db"), auto_open=False)
    assert viz_replay is not None

    print(f"✓ result  : {env.text()[:100]!r}")
    print(f"  events  : {sorted(event_types)}")
    print(f"  n events: {len(events)}")
    print(f"  cost    : ${env.metadata.cost_usd:.6f}")
    print(f"  replay  : Visualizer.replay(r'{tmp / 'live.db'}').open()")


# %% Rung 8 — from_step + from_start sentinels
# from_step("name") = output di uno step specifico
# from_start = envelope originale come context
with _rung("Rung 8 — from_step + from_start") as (sess, tmp):
    TOPIC = "Python programming language"

    expand = Agent(
        engine=LLMEngine(
            MODEL_CAPABLE,
            system="List exactly 3 key facts about the topic. One per line, no numbering.",
        ),
        name="expand",
        session=sess,
    )
    distil = Agent(
        engine=LLMEngine(
            MODEL_CAPABLE,
            system=(
                "You receive 3 facts as your task and the original topic as context. "
                "Write one sentence that captures the essence of the topic using those facts."
            ),
        ),
        name="distil",
        session=sess,
    )
    pipeline = Agent(
        engine=Plan(
            Step("expand"),
            Step("distil", task=from_step("expand"), context=from_start),
        ),
        tools=[expand, distil],
        session=sess,
        name="sentinel_pipeline",
    )
    env = pipeline(TOPIC)

    assert env.text(), "distil returned empty"

    events = sess.events.query()
    starts = [
        e for e in events
        if e["event_type"] == "agent_start" and e["payload"].get("agent_name") == "distil"
    ]
    assert starts, "distil non è stato avviato"
    distil_task = starts[0]["payload"].get("task", "")
    assert distil_task != TOPIC, "from_step non ha risolto: distil ha ricevuto il task originale invariato"

    expand_finish = next(
        (e for e in events if e["event_type"] == "agent_finish" and e["payload"].get("agent_name") == "expand"),
        None,
    )
    expand_out = expand_finish["payload"].get("payload", "") if expand_finish else "?"

    print(f"✓ expand output : {str(expand_out)[:80]!r}")
    print(f"  distil task   : {distil_task[:80]!r}  ← from_step(expand) ✓")
    print(f"  final output  : {env.text()[:100]!r}")


# %% Rung 9 — Store + checkpoint (crash-resume)
# Verifica che un Plan con Store salti gli step già completati al secondo run
with _rung("Rung 9 — Store checkpoint") as (sess, tmp):
    db_path  = str(tmp / "checkpoint.db")
    store    = Store(db=db_path)
    _call_log: list[str] = []

    def tracked_fetch(topic: str) -> str:
        """Fetch data — registra ogni invocazione come side effect."""
        _call_log.append(topic)
        return f"Facts about {topic}: widely used, open-source, well-documented."

    fetcher = Agent(
        engine=LLMEngine(
            MODEL_CAPABLE,
            system="Use tracked_fetch to get facts, then summarise in one sentence.",
        ),
        tools=[Tool.wrap(tracked_fetch, name="tracked_fetch")],
        name="fetcher",
        session=sess,
        store=store,
    )
    writer2 = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Write a two-sentence report from the findings."),
        name="writer2",
        session=sess,
        store=store,
    )
    _CKPT_KEY = "rung9_pipeline"
    pipeline = Agent(
        engine=Plan(
            Step("fetcher"), Step("writer2", task=from_prev),
            store=store, checkpoint_key=_CKPT_KEY,
        ),
        tools=[fetcher, writer2],
        store=store,
        session=sess,
        name="checkpoint_pipeline",
    )

    env1 = pipeline("Python")
    calls_run1 = len(_call_log)
    assert env1.text(), "run1 vuoto"
    assert calls_run1 >= 1, "tracked_fetch non chiamato al run1"

    # Secondo run: stesso DB + resume=True → fetcher saltato dal checkpoint
    sess2 = Session(db=str(tmp / "live2.db"), console=False)
    store2  = Store(db=db_path)
    fetcher2 = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Use tracked_fetch to get facts, then summarise."),
        tools=[Tool.wrap(tracked_fetch, name="tracked_fetch")],
        name="fetcher",
        session=sess2,
        store=store2,
    )
    writer3 = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Write a two-sentence report from the findings."),
        name="writer2",
        session=sess2,
        store=store2,
    )
    pipeline2 = Agent(
        engine=Plan(
            Step("fetcher"), Step("writer2", task=from_prev),
            store=store2, checkpoint_key=_CKPT_KEY, resume=True,
        ),
        tools=[fetcher2, writer3],
        store=store2,
        session=sess2,
        name="checkpoint_pipeline",
    )
    env2 = pipeline2("Python")
    calls_run2 = len(_call_log)
    sess2.close()

    assert calls_run2 == calls_run1, (
        f"tracked_fetch ri-eseguito! run1={calls_run1} → run2={calls_run2}"
    )

    print(f"✓ run1 output  : {env1.text()[:80]!r}")
    print(f"  run2 output  : {env2.text()[:80]!r}")
    print(f"  tracked_fetch: {calls_run1} chiamata/e → {calls_run2} (nessun re-run ✓)")


# %% Rung 10 — routes_by (structured routing via Literal field)
# Il classifier produce un output strutturato il cui campo Literal guida il routing
with _rung("Rung 10 — routes_by routing") as (sess, tmp):
    class SentimentDecision(BaseModel):
        sentiment: Literal["positive", "negative"]
        brief: str

    classifier = Agent(
        engine=LLMEngine(
            MODEL_CAPABLE,
            system=(
                "Classify the sentiment of the review. "
                "Return JSON with 'sentiment' ('positive' or 'negative') and 'brief' (one sentence)."
            ),
        ),
        output=SentimentDecision,
        name="classifier",
        session=sess,
    )
    positive_handler = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Write a warm one-sentence thank-you for a positive review."),
        name="positive",
        session=sess,
    )
    negative_handler = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Write a professional one-sentence apology for a negative review."),
        name="negative",
        session=sess,
    )
    finalizer = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Output the text you receive unchanged."),
        name="finalizer",
        session=sess,
    )
    pipeline = Agent(
        engine=Plan(
            # after_branches skips the sibling branch and jumps directly to finalizer
            Step("classifier", routes_by="sentiment", output=SentimentDecision, after_branches="finalizer"),
            Step("positive", task=from_step("classifier")),
            Step("negative", task=from_step("classifier")),
            Step("finalizer", task=from_prev),
        ),
        tools=[classifier, positive_handler, negative_handler, finalizer],
        session=sess,
        name="routing_pipeline",
    )

    env_pos = pipeline("Excellent product! Works perfectly, fast delivery, very happy.")

    assert env_pos.text(), "routing pipeline vuota"

    events = sess.events.query()
    agent_names_ran = {
        e["payload"].get("agent_name")
        for e in events
        if e["event_type"] == "agent_finish"
    }
    assert "negative" not in agent_names_ran, (
        f"negative non doveva girare per un review positivo! agents ran: {agent_names_ran}"
    )
    assert "positive" in agent_names_ran, f"positive non è girato! agents ran: {agent_names_ran}"

    clf_finish = next(
        (e for e in events if e["event_type"] == "agent_finish" and e["payload"].get("agent_name") == "classifier"),
        None,
    )
    assert clf_finish, "classifier non ha completato"

    print(f"✓ classifier  : {clf_finish['payload'].get('payload', {})}")
    print(f"  agents ran  : {sorted(agent_names_ran)}")
    print(f"  final output: {env_pos.text()[:120]!r}")
    print(f"  cost        : ${env_pos.metadata.cost_usd:.6f}")


# %% Rung 11 — Memory (conversazione multi-turn)
# Verifica che un agente ricordi il turno precedente senza che il task lo ripeta
with _rung("Rung 11 — Memory multi-turn") as (sess, tmp):
    memory = Memory(strategy="sliding")
    assistant = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="You are a helpful assistant. Be concise."),
        memory=memory,
        session=sess,
        name="assistant",
    )

    env_t1 = assistant("My favourite colour is ultramarine blue. Just say 'Got it'.")
    assert env_t1.text(), f"t1 vuoto"

    env_t2 = assistant("What colour did I just mention?")
    assert any(w in env_t2.text().lower() for w in ["ultramarine", "blue"]), (
        f"Memory non funziona — t2: {env_t2.text()!r}"
    )

    print(f"✓ t1 : {env_t1.text()!r}")
    print(f"  t2 : {env_t2.text()!r}  ← ricorda 'ultramarine' ✓")


# %% Rung 12 — from_agent (cross-run Store lookup)
# Run 1 → researcher raccoglie dati, output salvato in Store
# Run 2 → writer legge quell'output via from_agent("researcher")
with _rung("Rung 12 — from_agent sentinel") as (sess, tmp):
    db_fa      = str(tmp / "agent_store.db")
    shared_store = Store(db=db_fa)

    researcher_fa = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="List exactly 3 facts. One per line."),
        name="researcher",
        session=sess,
        store=shared_store,
    )
    pipeline_run1 = Agent(
        engine=Plan(Step("researcher")),
        tools=[researcher_fa],
        store=shared_store,
        session=sess,
        name="run1",
    )
    env_run1 = pipeline_run1("quantum computing")
    assert env_run1.text(), "run1 vuoto"

    writer_fa = Agent(
        engine=LLMEngine(MODEL_CAPABLE, system="Write a one-paragraph summary from the research provided."),
        name="writer",
        session=sess,
        store=shared_store,
    )
    pipeline_run2 = Agent(
        engine=Plan(Step("writer", task=from_agent("researcher"))),
        tools=[researcher_fa, writer_fa],
        store=shared_store,
        session=sess,
        name="run2",
    )
    env_run2 = pipeline_run2("write the report")

    assert env_run2.text(), "run2 vuoto"
    assert any(w in env_run2.text().lower() for w in ["quantum", "comput"]), (
        f"from_agent non ha passato i dati: {env_run2.text()!r}"
    )

    print(f"✓ researcher : {env_run1.text()[:80]!r}")
    print(f"  writer     : {env_run2.text()[:120]!r}  ← letto da Store ✓")


# %% Rung 13 — multi-provider
# Stesso task su provider diversi (skip automatico se API key non impostata)
with _rung("Rung 13 — multi-provider") as (sess, tmp):
    TEST_OPENAI   = bool(os.getenv("OPENAI_API_KEY"))
    TEST_DEEPSEEK = bool(os.getenv("DEEPSEEK_API_KEY"))

    PROVIDERS = [("Anthropic", MODEL_CAPABLE)]
    if TEST_OPENAI:
        PROVIDERS.append(("OpenAI", "gpt-4o-mini"))
    if TEST_DEEPSEEK:
        PROVIDERS.append(("DeepSeek", "deepseek-chat"))

    TASK = "Name the three primary colours. Reply with a comma-separated list only."

    results = {}
    for label, model_id in PROVIDERS:
        agent_p = Agent(
            engine=LLMEngine(model_id, system="Follow instructions exactly. Be concise."),
            session=sess,
            name=f"agent_{label.lower()}",
        )
        env_p = agent_p(TASK)
        assert env_p.text(), f"{label}: risposta vuota"
        results[label] = {
            "text":    env_p.text(),
            "in":      env_p.metadata.input_tokens,
            "out":     env_p.metadata.output_tokens,
            "cost":    env_p.metadata.cost_usd,
            "ms":      env_p.metadata.latency_ms,
        }

    print(f"✓ {'Provider':<12} {'Answer':<35} {'tok_in':>6} {'tok_out':>7} {'cost':>10} {'ms':>7}")
    print(f"  {'─'*75}")
    for lbl, r in results.items():
        print(f"  {lbl:<12} {r['text'][:33]:<35} {r['in']:>6} {r['out']:>7} ${r['cost']:>8.6f} {r['ms']:>7.0f}")
    if not TEST_OPENAI:
        print("  (OpenAI saltato   — imposta OPENAI_API_KEY)")
    if not TEST_DEEPSEEK:
        print("  (DeepSeek saltato — imposta DEEPSEEK_API_KEY)")
