# LazyBridge — Investor Overview

*Documento riservato. Versione del codebase: pre-1.0, 1099 test verdi, branch
`claude/audit-architecture-competitors-9InKx`.*

## 1. Executive summary

**LazyBridge è un framework Python per costruire sistemi LLM** —
agenti, pipeline multi-step, copiloti — **eliminando il plumbing**.
Una sola API (`Agent("model")(task)`) nasconde provider, async, retry,
tool-calling, structured output. La complessità di produzione
(pipeline tipizzate, crash-resume, observability OTel,
human-in-the-loop) è additive: niente DSL, niente decorator, niente
classi base, niente `asyncio.run` in vista.

**Stato oggi (maggio 2026):**

- 1099 test verdi
- 4 provider ufficiali (Anthropic, OpenAI, Google, DeepSeek) + LiteLLM
  bridge per 100+ provider
- 5 estensioni `stable`, 1 `alpha`, 6 esempi di dominio
- OpenTelemetry GenAI Semantic Conventions native
- Documentazione dual-track (sito MkDocs + Claude Skill bundle)
  generata da una singola sorgente
- Versione pre-1.0 dichiarata "beta" con onestà

## 2. Il problema

Costruire applicazioni LLM in produzione richiede oggi tre cose
contemporaneamente:

1. Capire l'SDK di ciascun provider (Anthropic, OpenAI, Google sono
   tutti diversi su tool-calling, streaming, structured output).
2. Gestire async, retry, rate-limit, schema discovery, tool-call shape.
3. Aggiungere l'infrastruttura: tracing, persistenza, ripresa da crash,
   valutazione, human-in-the-loop.

I framework esistenti si dispongono su due estremi:

- **Astrazione massima** (LangChain, AutoGen): facile iniziare,
  debugging un incubo. Decine di classi, callback chain, "Runnable",
  "Executor" — il developer paga in trasparenza.
- **Vicinanza all'SDK del provider** (SDK ufficiali, custom code):
  debuggable, ma il developer riscrive lo stesso boilerplate
  (retry, schema, async loops, tool registration) ad ogni progetto.

**Manca un framework che sia trasparente _e_ conciso.**

## 3. La soluzione: filosofia in cinque principi

### 3.1 Una sola API, façade sincrona

```python
from lazybridge import Agent
print(Agent("claude-opus-4-7")("write a haiku"))
```

Niente `async def`, niente `await`, niente `asyncio.run`. La façade
gestisce il loop. Il developer avanzato che vuole asyncio chiama
`agent.arun(...)` — ma non è obbligato.

### 3.2 Tool-is-Tool

Funzioni, Agent, Agent-of-Agent: tutti sono `Tool`. Composizione
uniforme.

```python
researcher = Agent("claude-opus-4-7", tools=[web_search])
analyst    = Agent("gpt-5", tools=[researcher])   # un Agent È un Tool
```

Niente "AgentExecutor", "RunnableLambda", "ChainOfThought wrapper" da
imparare. Una primitiva, infinite composizioni.

### 3.3 Pipeline tipizzate, validate a costruzione

Per applicazioni multi-step:

```python
from lazybridge import Plan, Step, from_prev

plan = Plan(
    Step(searcher, name="search"),
    Step(analyst,  name="analyse", context=from_prev),
    Step(writer,   name="write",   context=from_prev),
)
```

`Plan(...)` valida il DAG **prima** di chiamare l'LLM. Step inesistente
in un `from_step()`? `PlanCompileError` immediato. Forward reference?
Errore. Parallel-band misuse? Errore. **Niente fallimenti runtime su
typo che costano un giro completo di token.**

### 3.4 Routing dichiarativo, non magico

```python
Step(searcher, name="search", output=Hits,
     routes={"apology": when.field("items").empty()})
```

Nessun campo `next` magico nello schema. Nessuna lambda nascosta. Il
routing è visibile sulla firma dello `Step`.

### 3.5 I production knob sono additive, non riscritture

Stessa pipeline. La produzione si attiva via parametri:

```python
plan = Plan(
    Step(searcher, name="search"),
    ...,
    store=Store(db="x.sqlite"),
    checkpoint_key="job-42",
    resume=True,            # crash recovery CAS-protected
    on_concurrent="fork",   # parallel execution con keyspace isolato
)
```

## 4. Profondità tecnica

### 4.1 Capacità presenti

| Capacità | Tier |
|---|---|
| 4 provider ufficiali (Anthropic, OpenAI, Google, DeepSeek) | core |
| LiteLLM bridge (100+ provider) | core (opt-in) |
| Plan engine con DAG validato compile-time | core |
| Crash-resume CAS-protected (`Store` + `checkpoint_key`) | core |
| Parallel bands con join atomico ("all or nothing") | core |
| Tool-calling automatico con schema da type-hints | core |
| Structured output (Pydantic) cross-provider | core |
| `Memory` + `Store` + `Session` (state layers separati) | core |
| 6 exporter (Console, JSON, Callback, Filtered, Structured, OTel) | core |
| OpenTelemetry GenAI Semantic Conventions native | ext.otel — stable |
| MCP client (stdio + Streamable HTTP) | ext.mcp — stable |
| `HumanEngine` + `SupervisorEngine` | ext.hil — stable |
| `EvalSuite` + `llm_judge` + assertion helpers | ext.evals — stable |
| `verify=` judge/retry loop | core |
| `when` predicate DSL (routing dichiarativo) | core |
| `Plan.run_many` fan-out (zero `ThreadPool`/`asyncio`) | core |
| Planner factories (DAG builder, blackboard) | ext.planners — alpha |
| 6 esempi di dominio (quant, stat, doc skills, …) | ext — domain |

### 4.2 Architettura

LazyBridge è strutturato in due regimi nello stesso pacchetto PyPI:

- **Core** (`lazybridge/`): API pubblica, beta, breaking changes
  documentati nel CHANGELOG.
- **Extensions** (`lazybridge.ext.*`): tag esplicito per modulo —
  `stable` / `beta` / `alpha` / `domain`.

Il tier `domain` è la nostra dichiarazione onesta: alcuni moduli
(es. `quant_agent`, `stat_runtime`) sono **esempi worked**, non parte
del contratto framework. Spediti per documentare i pattern, non
garantiti.

Regola architettonica: **core non importa mai da ext**. Cycle
prevention via import-linter.

### 4.3 Observability

Ogni run produce un `EventLog`. Gli `Exporter` lo trasformano:

- `OTelExporter` mappa su OpenTelemetry GenAI Semantic Conventions
  (`gen_ai.system`, `gen_ai.usage.input_tokens`,
  `gen_ai.tool.call.id`). **L'unico framework agentico in conformità
  nativa al 2026** — gli altri richiedono mapping manuale.
- `JsonFileExporter` per replay/audit.
- `StructuredLogExporter` per stack di logging esistenti.

`Envelope.metadata.nested_*` aggrega cost / token / latency
**transitivamente** attraverso composizioni Agent-of-Agent. Il costo di
un sistema di 3 livelli si legge dal top-level `Envelope`, senza
walking manuale.

### 4.4 Resilienza

- **Hybrid back-pressure** sul bus eventi: critical events (errori,
  decisioni gate) bloccano; telemetria si droppa con counter visibile.
- **MCP cache TTL** (default 60s) evita storms su gateway esterni.
- **Verify retry**: judge-based con feedback rifornito al modello.
- **CAS-protected checkpoint** (compare-and-swap): impedisce
  doppio-write su run concorrenti con stesso `checkpoint_key`.
  Fallisce esplicitamente (`ConcurrentPlanRunError`) o forka
  automaticamente con namespace isolato.
- **Atomicità dei parallel band**: se un branch fallisce, *nessuno*
  scrive nello store — il resume rilancia il band intero senza
  doppi side-effect.

## 5. Lo spirito

LazyBridge è costruito da chi **odia il plumbing**. Tre regole guidano
ogni decisione:

1. **Se richiede `async`/`await`/`asyncio.run` nell'esempio
   user-facing, va nascosto.** La façade `Agent` è sincrona.
   `Plan.run_many` evita `ThreadPoolExecutor`. Ogni rendering di
   documento, ogni test, ogni recipe segue questa regola — verificato
   con grep CI.

2. **Se richiede una `lambda` nello `Step(...)`, ho fallito il
   design.** Il `when` DSL (`when.field("items").empty()`) è stato
   aggiunto per cancellare l'ultimo lambda dagli esempi.

3. **Se richiede una classe base, ho fallito il design.** `Agent` è
   una funzione. `Tool` è una funzione tipizzata. `Step` è un
   namedtuple. L'utente non eredita; compone.

Il risultato è che il codice user-facing **legge come pseudocodice**:

```python
plan = Plan(
    Step(searcher,  name="search",
         routes={"apology": when.field("items").empty()}),
    Step(analyst,   name="analyse", context=from_prev),
    Step(writer,    name="write",   context=from_prev),
    Step(apology,   name="apology"),
)
result = Agent.from_engine(plan)("AI trends April 2026")
```

Niente `@tool`, `@chain`, `@agent`. Niente `Runnable`, `Executor`,
`Chain.from_*`. Pochi nomi, ognuno significativo.

## 6. Posizionamento di mercato

Il mercato agentico 2026 è affollato. Mappa onesta:

| Framework | Forza | Debolezza vs LazyBridge |
|---|---|---|
| **LangChain / LangGraph** | Ecosystem, integrations | Boilerplate massivo, debugging difficile, classi proliferate |
| **LlamaIndex** | RAG-focus | Stessa astrazione pesante; non general-purpose |
| **CrewAI** | "Roles" mental model | Magico (delega implicita), non Python-idiomatic |
| **Pydantic AI** | Tipizzazione, vicino alla nostra filosofia | Single-agent focused, niente Plan/checkpoint, niente HIL |
| **AutoGen / Semantic Kernel** | Enterprise (Microsoft) | DSL pesante, paradigm chiuso |
| **Anthropic Agent SDK** | Native to Claude | Single-provider, single-pattern |

**Dove LazyBridge vince:**

- Unico con **DAG compile-time validato** + **crash-resume
  CAS-protected** nello stesso pacchetto.
- Lo zero-boilerplate è radicale: niente `async def` nei test, nei
  recipe, negli esempi. Verificato grep-based in CI.
- Multi-provider nativo (4 + LiteLLM) senza essere catch-all.
- OTel GenAI compliant da subito.

**Dove non vinciamo (oggi):**

- Ecosistema integrazioni (LangChain ne ha migliaia).
- Brand/marketing (Microsoft e LangChain hanno team interi).
- Tool ecosystem proprio (mitigato da MCP che apre l'intero
  ecosistema MCP).

## 7. Stato e milestone

| Milestone | Stato |
|---|---|
| Core API stable in spirito | Done |
| 4 provider + LiteLLM bridge | Done |
| Plan engine + sentinel + crash-resume | Done |
| OTel GenAI Semantic Conventions | Done |
| 5 stable extensions | Done |
| 1099 test verdi | Done |
| Documentazione dual-track (umano + LLM Skill) | Done |
| Versione 1.0 (semver strict) | In corso |
| Prima release pubblica con case study | In corso |
| Ecosystem 3rd-party tool/integration | Da fare |

## 8. Roadmap (12-18 mesi)

**Q2 2026** (subito):
- Release 1.0 pubblica
- Hosting documentazione (selvaz.github.io/LazyBridge già live)
- 3 case study reali (quant agent, doc skills, dataset analyser)

**Q3 2026:**
- `lazybridge.ext.rag` (chiusura del cerchio agentic + RAG)
- Plan serialization completa (sub-Plan distribuiti)
- Web Console minima per `Session` replay

**Q4 2026:**
- Plugin architecture per provider 3rd-party verificati
- Distributed Plan execution (Ray/Dask backend opzionale)
- SaaS landing per dev onboarding

**H1 2027:**
- Enterprise SSO + audit log standardizzati
- Compliance pack (HIPAA/SOC2-friendly logging)
- Marketplace skills/MCP server curati

## 9. Allocazione capitali

| Voce | % | Cosa risolve |
|---|---|---|
| Developer Relations & content | 30% | Adozione: i framework agentici si guadagnano per word-of-mouth dev |
| 2 maintainer full-time | 35% | Velocità feature + responsiveness su issue |
| Infrastruttura SaaS (replay, registry) | 20% | Monetizzazione + lock-in soft |
| Compliance / sicurezza | 10% | Sblocca enterprise |
| Cuscinetto runway | 5% | Margine |

## 10. Perché ora

- Il mercato ha appena attraversato il "framework wars" del
  2024-2025. Le opzioni mature mostrano già le loro fragilità
  (LangChain ha riscritto core 3 volte).
- **OpenTelemetry GenAI** è uscito stabile fine 2025. Chi lo adotta
  nativo ha vantaggio tooling.
- **MCP** (Anthropic, ora multi-vendor) è standard de facto. Il
  nostro client è production-ready oggi.
- I dev senior chiedono **trasparenza**, non magia. Il segmento
  "Pydantic AI users" cresce rapidamente — è il nostro target naturale.

## 11. Rischi onesti

1. **Single maintainer.** Bus factor 1. Mitigazione: priorità alta del
   piano d'investimento (2 maintainer = primo capitolo).
2. **Adozione lenta.** I framework si guadagnano lentamente; LangChain
   ci ha messo 18 mesi prima di esplodere. Mitigazione: contenuto +
   integration early.
3. **Provider drift.** Ogni provider cambia API; manteniamo 4
   provider ufficiali + LiteLLM bridge come contromisura.
4. **Anthropic / OpenAI rilasciano un loro framework dominante.**
   Già successo (Anthropic Agent SDK). Mitigazione: il
   multi-provider è il nostro fossato; un SDK provider-locked non è
   un competitor diretto.
5. **Pre-1.0 instability.** Stiamo allargando la SemVer surface;
   rischio breaking change documentato nel CHANGELOG. Il commitment
   al `stable` tier dimostra serietà nel definire boundaries.

## 12. Demo immediata

```bash
pip install lazybridge[mcp,otel,evals]
```

```python
from lazybridge import Agent, Plan, Step, Session, when, from_prev
from lazybridge.ext.mcp import MCP
from lazybridge.ext.otel import OTelExporter

fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

with Session(exporters=[OTelExporter(endpoint="http://jaeger:4318")]) as sess:
    plan = Plan(
        Step(Agent("claude-opus-4-7", tools=[fs]), name="read"),
        Step(Agent("claude-opus-4-7"),             name="summarise",
             context=from_prev),
    )
    print(Agent.from_engine(plan, session=sess)("summarise the README"))
```

Cinque righe di pipeline produttiva: tool federati via MCP, tracing
OTel completo, cost roll-up automatico, crash-resume disponibile via
un parametro. **Niente boilerplate.**

## 13. Contatti

[Spazio da personalizzare con dati founder, repository pubblico,
documentazione, case study di riferimento.]

---

*La forza del progetto non è che sia "il framework più completo" — è
che ogni riga di codice user-facing è giustificata. Niente plumbing
nascosto, niente magia da imparare. Una soluzione costruita da chi
deve usarla in produzione, non da chi la deve vendere.*
