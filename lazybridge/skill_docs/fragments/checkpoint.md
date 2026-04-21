## signature
Plan(
    *steps,
    store: Store,
    checkpoint_key: str,
    resume: bool = False,
) -> Engine

# Persisted shape at store[checkpoint_key]:
#   {
#     "next_step": str | None,
#     "kv": {"writes_key": payload, ...},
#     "completed_steps": [str],
#     "status": "running" | "failed" | "done",
#   }

## rules
- Checkpoint scatta dopo ogni step riuscito e dopo ogni step fallito.
- Success path: ``status="running"`` (step successivo pendente) →
  ``status="done"`` quando ``next_step is None``.
- Fail path: lo step fallito NON viene aggiunto a ``completed_steps``;
  il checkpoint salva ``next_step=<nome step fallito>`` + ``status="failed"``.
  Un successivo run con ``resume=True`` ri-parte da quello step.
- Success + ``resume=True`` + ``status="done"`` → short-circuit: il Plan
  ritorna un Envelope con payload = ``kv`` cached, senza rieseguire.
- Il checkpoint è JSON-encoded via ``Store.write``; gli step ``writes=``
  devono essere JSON-serialisable (string, dict, Pydantic model via
  ``.model_dump()``).

## narrative
Checkpoint/resume converte Plan da "workflow che può crashare" a
"workflow idempotente attraverso crash". Il meccanismo è intenzionalmente
minimale: nessuna history di Envelope, nessuna ricostruzione di stato
in-memory. Quello che sopravvive a un restart è solo
``writes``-bucket + puntatore allo step successivo + status.

Tre scenari d'uso tipici:

* **Run lungo e costoso** — una pipeline a 6 step con ciascun step che
  costa $0.50 in token. Un crash alla 5/6 deve riprendere dalla 5, non
  dalla 1.
* **Eventi esterni** — un Plan che attende un webhook. Il processo si
  spegne; un altro processo con lo stesso ``store`` + ``checkpoint_key``
  + ``resume=True`` riprende dove l'altro aveva lasciato.
* **Dev loop** — stai iterando su uno step specifico; i precedenti
  sono già computati e persistiti, non ha senso rifarli ogni volta.

La regola di pollice: usa checkpoint quando il costo di rieseguire gli
step precedenti supera il costo della complicazione di storage.

## example
```python
from lazybridge import Agent, Plan, Step, Store

store = Store(db="pipeline.sqlite")

def build_plan():
    return Plan(
        Step(researcher, name="search",  writes="hits"),
        Step(ranker,     name="rank",    writes="ranked"),
        Step(writer,     name="write",   writes="draft"),
        store=store,
        checkpoint_key="pipeline",
        resume=True,
    )

# Run 1 — crash dopo rank: status="failed", next_step="write".
try:
    Agent.from_engine(build_plan())("AI trends")
except KeyboardInterrupt:
    pass

# Run 2 — resume dallo step fallito, non rifà search+rank.
Agent.from_engine(build_plan())("AI trends")

# Run 3 — il plan è già "done": short-circuit, ritorna kv cached.
result = Agent.from_engine(build_plan())("AI trends")
print(result.payload)  # {"hits": ..., "ranked": ..., "draft": ...}
```

## pitfalls
- Cambiare la definizione del ``Plan`` (aggiungere / rimuovere step,
  rinominare) e riprendere da un checkpoint vecchio è un errore: il
  ``next_step`` salvato può non esistere più. Invalida il checkpoint
  (``store.delete(checkpoint_key)``) dopo refactor dei step.
- Non-JSON-serialisable ``writes`` (es. un file handle) si rompono in
  silenzio (vengono convertiti a stringa via ``default=str``).
- Il resume non ri-inietta session / exporter del run originario; passa
  gli stessi ``session=`` + ``store=`` su ogni run per continuità.

## see-also
[plan](plan.md), [store](store.md),
decision tree: [checkpoint](../decisions/checkpoint.md)
