# Assessment di implementazione — LazyBridge (Luglio 2026)

> Audit indipendente della repository `LazyBridge` (package Python `lazybridge`, versione
> 1.0.1, ~24.900 LOC sorgente in 69 file `.py`, 117 file di test). Metodo: lettura del codice
> reale (core letto integralmente; provider/engines/ext auditati file per file), esecuzione
> reale della suite di test con coverage, verifica incrociata piani/documenti vs codice.
> Data: 2026-07-08. Riferimenti sempre in forma `file:riga` sul commit corrente (`acbcc94`).

---

## 1. Panoramica e stato di salute

LazyBridge è un framework multi-provider per agenti LLM ("Agent = Engine + Tools + State"):
provider nativi Anthropic/OpenAI/Google/DeepSeek + bridge LiteLLM + LM Studio
(`lazybridge/core/providers/`), motore di orchestrazione deterministico `Plan` con
checkpoint/resume (`lazybridge/engines/plan/`), `ReplanEngine`, guardrails con LLM-judge
anti-injection (`lazybridge/guardrails.py`), HIL (`lazybridge/ext/hil/`), observability
(`Session`/`EventLog` SQLite con writer batched, `lazybridge/session.py`), store persistente
con adapter di cifratura Fernet (`lazybridge/store/`), visualizer web (`lazybridge/ext/viz/`).

**Giudizio complessivo: codebase in salute molto buona, sopra la media.** La suite (1937
test) è verde in 27 s, coverage totale reale 80.4%, zero TODO/FIXME/HACK nel sorgente, error
message a formato standard, decisioni di design tracciate. Le debolezze reali sono
concentrate in: (a) percorsi streaming dei provider (poco coperti e con leak di risorse su
early-exit), (b) la Web UI human-in-the-loop (priva delle protezioni che il viz server dello
stesso repo ha), (c) un paio di bug di resume/cancellazione nel runtime Plan/LLMEngine,
(d) drift documentale tra i file di piano (fermi a "0.7.9 alpha") e la realtà 1.0.1 stable,
(e) extra opzionali (encryption/OTel/litellm) mai esercitati in CI.

| Dimensione | Voto | Motivazione |
|---|---|---|
| Correttezza | **B+** | 1937 test verdi; però: `BaseException` dei tool inghiottite nel tool-loop (`llm.py:942-947`), stream provider non chiusi su early-exit (§3 M-7), resume da `plan_state` ignora `status` (`_plan.py:367-375`, M-13) |
| Sicurezza | **B** | Ottimo di base (redazione secrets di default, LLMGuard anti-injection con scrub tag, viz server con token+CSP+traversal-proof, CodeQL settimanale); ma Web UI HIL senza auth né cap sul body (`human.py:461-509`) |
| Test | **B+** | Coverage 80.4% (gate 73%), regression test per ogni bug-fix storico; però encryption/OTel/litellm skippano SEMPRE in CI (22 skip), replan 51%, google 58%, openai 65% |
| Docs | **A-** | Docs eccezionalmente coerenti col codice (nav mkdocs 102/102, esempi README tutti validi — verificato); drift residuo: `llms.json` fermo a 0.7.9/alpha con contratti errati, IMPLEMENTATION.md contraddice la 1.0.1, 2 tabelle native-tools discordanti |
| Manutenibilità | **A-** | Architettura core/ext pulita con boundary testato (`test_core_ext_boundary.py`), mypy strict a tier, bridge sync/async unificato (`_asyncbridge.py`); funzioni >250 righe nei runtime (`llm.py:_loop` ~308, `_compiler.py:validate` ~419) e ~120 righe di checkpoint duplicate tra plan e replan |

---

## 2. Stato dell'implementazione

### Completo e verificato nel codice
- **Core pubblico** (`lazybridge/__init__.py:118-248`): `Agent`, `ParallelAgent`, `Envelope`
  generico, `Tool`/`Tool.wrap`/`tool()`, sentinelle (`from_prev`, `from_step`, `from_agent`,
  `from_memory`, `from_parallel[_all]`, `from_start`), `AgentPool`, `conclude`, `when`,
  `Memory`, `Store`, `Session`/`EventLog`/exporters, guardrails, `MockAgent`.
- **Provider**: 6 adapter in `lazybridge/core/providers/` con capability `ClassVar`,
  registry lazy (`_registry.py`), tier alias (`cheap`/`medium`/`top`), costi con cache-token
  per-provider (semantiche diverse ma ciascuna corretta — verificato).
- **Engines**: `LLMEngine` (tool-loop, stall-detection, retry/backoff, prompt caching),
  `Plan`/`Step` (compiler con errori a formato standard e "Did you mean", checkpoint CAS,
  parallel band, routing), `ReplanEngine`.
- **Ext**: evals, hil (Human/Supervisor, CLI+Web UI), otel exporter, planners, viz
  (server con token auth, replay, exporter).
- **Fasi 1–6 di IMPLEMENTATION.md**: le deletion dichiarate sono reali — nessun
  `from_chain`/`from_model`/`_UNSET`/`mode="auto"`/`tool_choice="parallel"` nel sorgente
  (verificato via grep); `EncryptedStoreAdapter` esiste (`store/encryption.py`) con 23 test;
  formato errori standard applicato; tabella provider generata da `lazybridge/matrix.py`.
- **CI/CD**: 5 workflow (`test.yml` pre-commit+ruff+mypy 3.11/3.12+unit 3.11-3.13,
  `docs.yml` mkdocs --strict + assert llms.txt, `codeql.yml` settimanale, `integration.yml`
  live nightly per 4 provider, `release.yml` con OIDC Trusted Publishing, gate lint/type/
  unit/docs, verifica tag==versione, GitHub Release da CHANGELOG). Dependabot con ignore
  deliberato dei major dei provider SDK. Impostazione di livello professionale.

### Parziale
- **Coverage streaming provider**: `google.py` 58%, `openai.py` 65% (Responses-API streaming
  1091-1330 quasi scoperto), `anthropic.py` 69%, `deepseek.py` 69%. `replan/_engine.py` 51%.
- **`EncryptedStoreAdapter` e `OTelExporter`**: test presenti ma condizionati agli extra;
  nel run di default coverage 31% e 0% rispettivamente (in CI: sempre skip — vedi M-1).
- IMPLEMENTATION.md Phase 5: pubblicazione PyPI di `lazybridge-reports` e CI del repo
  gemello dichiarate "user-side", non verificabili da qui (caselle ancora aperte
  `IMPLEMENTATION.md:302-303,321`).

### Dichiarato ma mancante (gap piano → codice)
- **Phase 7 — `agents.md` config file + SuperTool/Skill** (`PROJECT_LAYOUT.md`,
  `SUPERTOOL_PLAN.md`, `IMPLEMENTATION.md:338-393`): dichiarato onestamente
  "design-complete, NOT STARTED". Confermato: nessun `load_agents`, nessun
  `LLMEngine.for_agent`, nessun loader `SKILL.md` nel package. I documenti di design sono
  coerenti tra loro (ladder di precedenza, sentinella `UNSET`, merge OVERRIDE/COMPOSE).
- **Phase 8 — Media output** (`MEDIA_OUTPUT.md`, `IMPLEMENTATION.md:397-431`): nessun
  `MediaRef`, nessun `image_gen`/`speech`/`video_gen` nel codice. Confermato "zero code".
- **LAZYTOOLS_EXTRACTION.md**: estrazione completata — i tool kit e l'MCP connector vivono
  nel package esterno `lazytoolkit`, gli shim sono stati rimossi in 0.9 come dichiarato;
  resta un esempio che dipende dal package esterno
  (`examples/llm_assistant/05_mcp_allowlisted.py:3`, `from lazytools.connectors.mcp
  import MCP` — import name `lazytools`, PyPI `lazytoolkit`), correttamente documentato.
- **Cross-cutting principles** (`IMPLEMENTATION.md:434-438`): il lint CI che collega i
  marker `(bug fix)` del CHANGELOG a un test non esiste in `.github/workflows/`.
- Item dichiarati "Planned" e correttamente non implementati (nessuna falsa promessa):
  provider fallback chains automatiche e redazione PII oltre le credenziali
  (`docs/index.md`, maturity table; `base.py:402-410` `_FALLBACKS` è dichiarato
  "Not implemented").

---

## 3. Issue trovate — ordinate per severità

Nessuna issue **CRITICA** (RCE, perdita dati sistematica, secrets hardcoded, injection
SQL — tutte le query Store/EventLog sono parametrizzate; nessun pickle/yaml.load unsafe).

Conteggio post-revisione: **0 CRITICA · 3 ALTA · 14 MEDIA · 21 BASSA** (le ex A-1 e A-5
sono state declassate a M-13 e M-14 — vedi Nota di revisione in fondo).

### ALTA

**A-2 — Nel tool-loop di LLMEngine una `BaseException` catturata da gather diventa risultato "riuscito" del tool**
- File: `lazybridge/engines/llm.py:902-947`.
- `asyncio.gather(..., return_exceptions=True)` (:902) cattura anche le `BaseException`
  sollevate dentro un tool: `CancelledError` interna e le `BaseException` custom del
  framework (`PlanPaused`, `_types.py:219`, è deliberatamente `BaseException`). Il re-raise
  esplicito (:913-915) gestisce solo `ConcludeSignal`; la classificazione a valle testa
  `isinstance(tr, Exception)` (:942), falso per le `BaseException`, che finiscono in
  `else: content = str(tr); is_err = False` (:945-947). Un tool cancellato o una
  `PlanPaused` vagante vengono quindi presentati al modello come risultato normale
  non-errore (per `CancelledError` la stringa è vuota) e il loop continua.
- Precisazione (verifica adversariale): `KeyboardInterrupt`/`SystemExit` non rientrano
  davvero in questo scenario — `Task.__step` le ri-solleva anche nell'event loop, che
  quindi si interrompe comunque. Il caso concreto è `CancelledError` interna / `PlanPaused`.

**A-3 — La Web UI human-in-the-loop non ha autenticazione (asimmetria col viz server)**
- File: `lazybridge/ext/hil/human.py:505-509`
  (`ThreadingHTTPServer(("127.0.0.1", self._port), _Handler)`).
- Il server HIL espone la task pendente (GET) e accetta la decisione umana (POST) senza
  alcun token: qualunque processo/utente locale può leggere il prompt e inviare una
  decisione contraffatta — l'unico controllo è `_epoch` (:468-479), un intero piccolo e
  indovinabile. Il viz server dello stesso repo usa `secrets.token_urlsafe(24)` +
  `hmac.compare_digest` (`ext/viz/server.py:117`) proprio perché localhost non è un confine
  di fiducia su host multiutente. Qui la superficie *approva azioni dell'agente*: è la più
  sensibile delle due.
- Riproduzione: avviare `HumanEngine` in modalità web; da un altro utente della macchina
  `curl -X POST -d "_epoch=<N>&response=approve" http://127.0.0.1:<porta>/`.

**A-4 — `do_POST` della Web UI HIL legge un Content-Length illimitato e non validato**
- File: `lazybridge/ext/hil/human.py:461-462` —
  `length = int(self.headers.get("Content-Length", 0))` poi `self.rfile.read(length)` senza
  cap: una singola richiesta può forzare allocazioni multi-GB (DoS memoria); un header non
  numerico solleva `ValueError` non gestita nel handler. Il viz server protegge esattamente
  questo caso (`ext/viz/server.py:244-247`): l'omissione qui è un'incoerenza oggettiva.

### MEDIA

**M-1 — Gli extra opzionali non sono mai testati in CI (encryption/OTel/litellm sempre skip)**
- File: `.github/workflows/test.yml:108-110` — il job unit installa solo
  `.[anthropic,openai,google,test]`. Di conseguenza `tests/unit/test_store_encryption.py`
  (tutto il file), `test_otel_exception_logging.py`, parti di `test_audit_short_term.py`,
  `test_resolution_fixes.py`, `test_audit_amend.py`, `test_audit_deep.py`,
  `test_multimodal_images.py:335` skippano **sempre** (verificato: `pytest -rs` → 22 skip,
  quasi tutti per moduli opzionali). `EncryptedStoreAdapter` (componente di sicurezza) e
  `OTelExporter` (0% coverage nel run standard) non hanno gate di regressione automatico.
  Installando `cryptography` + `opentelemetry-sdk` gli 88 test interessati passano
  (verificato in questo audit).

**M-2 — IMPLEMENTATION.md descrive lo stato "0.7.9, niente v1.0" mentre il package è 1.0.1 stable**
- File: `IMPLEMENTATION.md:324,332-334,451` ("staying on 0.7.9 per user direction",
  "`__stability__='stable'` — deferred", "Tag v1.0.0 — not happening on this branch") vs
  `pyproject.toml:7` (`version = "1.0.1"`), `lazybridge/__init__.py:109`
  (`__stability__ = "stable"`), CHANGELOG 1.0.1 del 2026-07-06. Il tracker contraddice la
  realtà per chiunque lo usi come fonte di stato (il disclaimer in testa lo ammette, ma
  tre sezioni intere sono ormai false).

**M-3 — Messaggi di deprecazione "will be removed in 1.0" ancora attivi in 1.0.1**
- File: `lazybridge/__init__.py:270-296` — `Task`, `CacheConfig`, `PROVIDER_ALIASES`
  emettono `DeprecationWarning` che promette la rimozione "in 1.0", ma la 1.0.1 li spedisce
  ancora. Il messaggio ora è fuorviante: o si rimuovono gli alias (breaking) o si
  riformula la scadenza (es. "in 2.0").

**M-4 — `lazybridge/llms.json` (spedito nel wheel, pensato per i code-generator LLM) è fermo a 0.7.9/alpha e documenta contratti sbagliati**
- File: `lazybridge/llms.json:3-4` — `"version": "0.7.9", "stability": "alpha"`.
  Inoltre: `canonical_patterns.parallel_fanout` (:14) afferma `env.payload → list[Envelope]`,
  ma `ParallelAgent._join_branches` (agent.py:1360-1367) imposta `payload=joined` (stringa;
  l'accesso tipizzato è `run_branches()`); `envelope_contract.error` (:27) dichiara
  `Exception | None` mentre è `ErrorInfo | None` (envelope.py:69); il pattern `mcp_server`
  (:18) usa `MCP` che dal 0.8 vive nel package esterno `lazytoolkit` (import `lazytools`)
  senza indicarlo.
  Questo file esiste apposta per guidare gli assistenti LLM: contenuto stale = codice
  generato sbagliato.

**M-5 — Google e LiteLLM perdono messaggi SYSTEM (incoerenza con Anthropic/OpenAI)**
- File: `lazybridge/core/providers/google.py:375-381` — `_get_system_instruction` ritorna
  `request.system` OPPURE il primo messaggio `Role.SYSTEM`: i SYSTEM successivi sono persi
  in silenzio, e se `request.system` è settato vengono persi tutti.
  `lazybridge/core/providers/litellm.py:186-189` — i SYSTEM sono inoltrati solo
  `if not request.system`. Anthropic concatena esplicitamente tutte le fonti
  (anthropic.py:426-440, con commento che documenta il bug analogo già fissato) e OpenAI
  inoltra entrambi (openai.py:490-494): stessa richiesta, risultati diversi per provider.

**M-6 — DeepSeek: warning spurio sui modelli V4 con thinking + parametri OpenAI-only inviati a `deepseek-reasoner`**
- File: `lazybridge/core/providers/deepseek.py:260-261,285-305` e
  `lazybridge/core/providers/openai.py:633-654`.
- (a) Per i modelli V4 `_is_reasoning_model` → `False`, quindi il `_build_chat_params`
  ereditato emette "OpenAI model 'deepseek-v4-flash' is not a reasoning model —
  ThinkingConfig is ignored" (openai.py:648-654) **e poi** `_apply_thinking_params`
  abilita davvero il thinking via `extra_body`: warning falso (emesso una volta per
  call-site, dato il filtro warnings di default — comunque fuorviante).
- (b) Per `deepseek-reasoner` il ramo reasoning imposta `reasoning_effort` e
  `max_completion_tokens` (parametri OpenAI) che non vengono mai rimossi
  (early-return in deepseek.py:294) e arrivano ad `api.deepseek.com`, che non li prevede.

**M-7 — Gli stream HTTP dei provider non vengono chiusi su uscita anticipata del consumer (tutti tranne Anthropic), incluso il percorso stall-detection**
- File: `openai.py:1031,1040,1138,1272`; `deepseek.py:455,496`; `litellm.py:565,592`;
  `google.py:836,946` — iterazione diretta `for`/`async for` sullo stream SDK senza
  context-manager/`aclose()`: se il caller interrompe l'iterazione la connessione httpx
  resta aperta fino al GC. Anthropic invece usa `with ctx as s:` (anthropic.py:911,1076).
  Aggravante: su idle-timeout `_idle_guarded_stream` (`lazybridge/engines/llm.py:1058-1070`)
  solleva `StreamStallError`; la cancellazione di `wait_for` chiude sì il frame del
  generatore provider, ma senza `try/finally` nei provider lo stream SDK/httpx sottostante
  viene solo abbandonato al GC — proprio il percorso pensato per gli stream appesi non
  chiude deterministicamente la connessione (tranne che su Anthropic, dove il `with`
  interno chiude in unwind).

**M-8 — Un judge `verify=` async (plain callable) viene silenziosamente sempre rifiutato**
- File: `lazybridge/_verify.py:128-130` — per un judge senza `.run` il verdetto è
  `verify_agent(result.text())` senza await: se l'utente passa un `async def`, `verdict` è
  una coroutine mai awaited (RuntimeWarning), `_is_approved(str(coroutine))` → sempre
  `False` (fail-safe). L'agent consuma sempre `max_verify` tentativi (costo LLM x N) e
  restituisce l'ultimo risultato senza errore esplicito. `Agent.as_tool` documenta
  "plain callable" senza vietare gli async (agent.py:907-910).

**M-9 — Lo `Store` in-memory ignora `close()` (violazione del contratto documentato)**
- File: `lazybridge/store/__init__.py:104-108` (docstring: "After close() the Store raises
  RuntimeError on further reads / writes") vs percorsi in-memory di `write` (:176-178),
  `read` (:193-195), `delete`, `clear`, `keys`, `items` che non controllano mai
  `self._closed` (solo `compare_and_swap`:326 e il percorso SQLite via `_conn()`:65-66 lo
  fanno). Uno Store in-memory chiuso continua ad accettare letture/scritture in silenzio.

**M-10 — `EvalSuite.arun` esegue tutti i casi con concorrenza illimitata e richiede un contratto diverso da `run`**
- File: `lazybridge/ext/evals/__init__.py:96` —
  `asyncio.gather(*[_run_one(c) for c in self.cases])` senza semaforo: una suite grande
  martella l'API del provider (rate limit) / esaurisce connessioni. Inoltre `run` chiama
  `agent(case.input).text()` (:71) mentre `arun` chiama `await agent.run(case.input)`
  (:85): lo stesso parametro `agent` deve soddisfare due contratti diversi, non documentati.

**M-11 — Resume di un plan "già done" da checkpoint restituisce un envelope stub senza metadata/costi**
- File: `lazybridge/engines/plan/_plan.py:371-373` — il ramo `status == "done"` ritorna
  `Envelope(task=..., payload=kv)` grezzo, bypassando `_aggregate_nested_metadata` che ogni
  altro return terminale applica (:467,482,589,591): payload di forma diversa dal primo run
  e costi azzerati.

**M-12 — Tabella native-tools manuale contraddice le capability reali e la tabella auto-generata (COMPUTER_USE)**
- File: `docs/guides/basic/native-tools.md:31` dichiara `COMPUTER_USE` solo Anthropic, ma
  `OpenAIProvider.supported_native_tools` (openai.py:298-305) lo include
  (`computer_use_preview`, openai.py:67). La tabella auto-generata in
  `docs/reference/providers.md` (resa a build-time da mkdocstrings su
  `lazybridge.matrix.native_tool_support`, providers.md:62) dice il contrario della
  tabella manuale nella stessa documentazione.

**M-13 (ex A-1, declassata in revisione) — Resume di un Plan da `plan_state` esplicito ignora `status`: uno stato "done" riparte da step 0**
- File: `lazybridge/engines/plan/_plan.py:367-375`.
- La scala di selezione del punto di ripresa testa solo `plan_state.next_step` (:367); un
  `PlanState` con `status="done"` ha `next_step=None`, quindi il flusso cade su
  `elif checkpoint...` (che è `None` quando `plan_state` è passato, vedi :328) e infine su
  `elif self.steps: current_name = self.steps[0].name` (:374-375). Lo short-circuit "già
  finito" (:371-373) esiste solo per il percorso checkpoint; `PlanState.status`
  (`_types.py:178`) non viene mai consultato in `_plan.py`.
- Impatto (ridimensionato in revisione): ri-esecuzione integrale della pipeline (side
  effect sui Store, costi LLM ri-fatturati) — ma il caso è raggiungibile solo con un
  `PlanState` **costruito a mano**: la libreria non restituisce mai istanze di `PlanState`
  (`plan.run()` ritorna un `Envelope`; nessun helper checkpoint→`PlanState` esiste — grep
  su `PlanState(` nel package: zero costruzioni). `PlanState` è però esportato e
  documentato come API pubblica (`docs/guides/full/plan.md:36`), quindi il footgun è
  reale ma non "resume normale di un run concluso".
- Riproduzione corretta: costruire `PlanState(..., next_step=None, status="done",
  history=..., store=...)` e chiamare `plan.run(env, plan_state=stato)` → tutti gli step
  rieseguiti. (La riproduzione della prima stesura era errata: `plan.run()` non restituisce
  alcuno "stato finale".)

**M-14 (ex A-5, declassata in revisione) — Anthropic forza lo streaming sui modelli moderni: `raw=None` e `messages.parse()` mai usato, comportamento non documentato**
- File: `lazybridge/core/providers/anthropic.py:67`
  (`_FORCE_STREAM_MAX_TOKENS = 20_000`), :283-294 (`get_default_max_tokens` → 128k per
  opus-4-8/4-7/4-6, 64k per sonnet-4-6/haiku-4-5/…), :730-746 (`_should_force_streaming`
  confronta il max *effettivo* col threshold), :810-811 (`complete()` devia su
  `_collect_streamed_response`).
- Con i default, ogni chiamata non-streaming ai modelli moderni viene convertita in
  streaming raccolto: `CompletionResponse.raw` è sempre `None` (:767) e il percorso
  `messages.parse()` con idratazione Pydantic nativa (:834-846) non viene di fatto mai
  eseguito su questi modelli. Il force-streaming in sé è un workaround legittimo del
  limite API Anthropic (richieste lunghe richiedono streaming).
- Mitigazioni esistenti che la prima stesura ignorava (motivo del declassamento):
  (1) il percorso streaming invia comunque `output_config` sul wire (enforcement
  server-side dello schema, anthropic.py:888-896) e applica
  `apply_structured_validation` sul final chunk (:960-963), quindi `parsed`/`validated`
  sono popolati anche in force-stream — l'output strutturato NON degrada, cambia solo il
  meccanismo; (2) esiste il knob `force_stream_threshold` nel costruttore
  (anthropic.py:303) per alzare/disattivare la soglia — però non documentato in docs/README;
  (3) il force-stream È segnalato a runtime, ma solo a livello DEBUG (:740-744).
- Residuo reale dell'issue: `raw=None` silenzioso (rompe chi legge `resp.raw`),
  knob non documentato, nessuna documentazione utente del comportamento.
- Riproduzione: `AnthropicProvider.complete()` su `claude-sonnet-4-6` senza `max_tokens`
  esplicito → `resp.raw is None`.

### BASSA

**B-1 — README omette `IMAGE_GENERATION` dai native tool** — `README.md:160` elenca 6
valori; `NativeTool.IMAGE_GENERATION` esiste (`core/types.py:133`), è supportato da OpenAI
ed è documentato in `docs/guides/basic/native-tools.md:32`.

**B-2 — `eval()` su stringhe di annotazione** — `core/tool_schema.py:503`
(`ann = eval(ann, func_globals)`, fallback quando `get_type_hints` fallisce con NameError).
Le annotazioni sono dello sviluppatore, non input utente: non è sfruttabile oggi, ma è il
punto da blindare se i tool verranno mai costruiti da sorgenti esterne.

**B-3 — `providers/__init__.py` non esporta `LiteLLMProvider`** —
`core/providers/__init__.py:1-25` esporta tutti i provider tranne LiteLLM (registrato in
`_registry.py:38`): `from lazybridge.core.providers import LiteLLMProvider` fallisce.

**B-4 — OpenAI chat senza choices → errore silenzioso non-retryable** — `openai.py:857-876`
ritorna `stop_reason="error"` senza sollevare, mentre il percorso Responses solleva
(`openai.py:908-924`): un content-filter/quota sul percorso chat non attiva retry/fallback.

**B-5 — OpenAI: messaggio misto tool_result+tool_use perde i tool call** —
`openai.py:584-598`: il ramo `if tool_results_in_msg` non emette mai `tool_calls_in_msg`.

**B-6 — Gemini: dedup dei tool-call in streaming rotto per part senza `fc.id`** —
`google.py:861,971`: la chiave sintetica `name-{len}` cambia a ogni part ripetuta,
vanificando la dedup che il commento promette. Correlata: la "recovery" del nome
`re.sub(r"-\d+$", "", tool_use_id)` (`google.py:347`) mangla nomi legittimi tipo `foo-2`.

**B-7 — OpenAI Responses: fallback del nome funzione = call_id** — `openai.py:996`
(`state.fc_names.get(call_id, call_id)`): un nome mancante produce un `ToolCall` con nome
casuale che l'engine non risolverà mai.

**B-8 — Anthropic: costo cache-write sottostimato con TTL 1h** — `anthropic.py:219-224`
fattura sempre 1.25× il prezzo input, ma il TTL `"1h"` (onorato sul wire a :606-607) costa
2×: il cost tracking diverge dal billing reale.

**B-9 — Anthropic: thinking budget clampabile a negativo** — `anthropic.py:576-586`:
`max_budget = effective_max - 1024`; con `max_tokens` piccolo (es. 1000) passa un
`budget_tokens` negativo → 400 garantito senza guardia di lower-bound.

**B-10 — `_parse_data_uri` codifica male i data-URI non-base64** — `core/types.py:121-123`:
il body percent-encoded viene ri-encodato base64 così com'è invece di essere prima
URL-unquoted → bytes sbagliati.

**B-11 — Serializzazione Plan lossy / KeyError su `Any`** —
`engines/plan/_serialisation.py:88-110,261-264`: `output=Any` serializza in `"typing.Any"`
che `_type_from_name` non risolve (KeyError al round-trip); `_step_to_dict` perde `sources`
e `Plan.to_dict` (`_plan.py:1113-1117`) perde `store`/`checkpoint_key`/`resume`/
`on_concurrent`/`stream_buffer` in silenzio.

**B-12 — Tool senza parametri invocato con `input=value`** —
`engines/plan/_serialisation.py:39-45` (fallback `{"input": value}` usato da
`_plan.py:925` e `replan/_engine.py:434`): un tool davvero zero-arg riceve un kwarg
inatteso → `TypeError`.

**B-13 — `ReplanEngine.run` accetta `memory` e `output_type` ma li ignora** —
`engines/replan/_engine.py:150-160,383-391`: incoerenza non documentata rispetto al
protocollo `Engine` (`engines/base.py:74-84`) implementato da LLMEngine/Plan.

**B-14 — Compiler: il check di compatibilità input/output salta ogni generic** —
`engines/plan/_compiler.py:370-372` (`if origin is not None: pass`): step adiacenti
incompatibili con `input=list[X]`/`dict[...]`/`Union` compilano puliti.

**B-15 — Parametro morto `kv` in `_resolve_sentinel`** — `engines/plan/_resolve.py:45`:
threaded da tutti i call site (`_plan.py:828,854`) e mai letto.

**B-16 — HIL: thread worker persi su timeout** — `ext/hil/human.py:519-530` (il worker
`run_in_executor` resta bloccato su `Queue.get` fino a `timeout` e consuma/scarta una
submission tardiva) e `ext/hil/supervisor.py:199-204` (il thread `input()` resta appeso su
stdin). Timeouts ripetuti degradano l'executor di default.

**B-17 — Viz: errori inghiottiti** — `ext/viz/visualizer.py:187-190` (`_store_payload`
cattura tutto → store rotto appare come store vuoto) e `ext/viz/server.py:204-211`
(il GET non ha il try/except che il POST ha a :254-258 → connessione droppata invece di
errore JSON).

**B-18 — Planner: stato in closure non concurrency-safe** — `ext/planners/blackboard.py:67,111-119`
e `ext/planners/builder.py:324`: due run concorrenti dello stesso planner-agent si
corrompono lo stato a vicenda (nessun lock, nessun guard).

**B-19 — Contatori drop dell'EventLog aggiornati senza lock** — `session.py:517-523`
(`_dropped_count += 1` da più thread produttori): possibile under-count, solo telemetria.

**B-20 — `key in store` è O(n)** — `store/__init__.py:287-294`: `__contains__`/`__len__`
materializzano tutte le chiavi (full scan su SQLite) invece di una query indicizzata.

**B-21 — `pip install -e .[dev]` non esiste** — `pyproject.toml:31-68` definisce `test`,
`docs`, `all` ma nessun extra `dev` (pip lo segnala solo come warning): incoerente con la
convenzione più comune e con flussi di onboarding che assumono `[dev]`.

---

## 4. Punti di miglioramento

**Qualità del codice**
1. Spezzare le funzioni-monstre dei runtime: `engines/llm.py:684-992` (`_loop`, ~308 righe),
   `engines/plan/_plan.py:294-591` (`_run_impl`, ~297), `engines/plan/_compiler.py:84-503`
   (`validate`, ~419), `engines/replan/_engine.py:383-626` (~243). Ognuna mescola
   scheduling, checkpoint, classificazione errori e aggregazione.
2. Deduplicare la state-machine di checkpoint: `replan/_engine.py:224-347` è una copia
   quasi letterale di `plan/_checkpoint.py:73-373` (`CheckpointMixin`, ~120 righe: stesso
   CAS/claim/adopt, stessi messaggi) — alto rischio di drift (un fix su un lato non arriva
   all'altro).
3. Deduplicare helper tra provider: `_safe_json_loads` (openai.py:114-145 ≈
   litellm.py:84-108), mappa MIME→formato audio con **default divergenti** ("wav" vs "mp3":
   openai.py:154-164 vs litellm.py:59-70), conversione messaggi OpenAI-shape
   (openai.py:484-608 ≈ litellm.py:180-222).
4. Tipizzazione: i moduli provider dichiarati strict usano `Any` estensivamente ai confini
   SDK (es. `google.py:211` `-> list[Any]`, `google.py:511` `-> Any | None`) — la
   strictness formale c'è, la garanzia sostanziale ai boundary no (già riconosciuto in
   `IMPLEMENTATION.md:315` come deferred).

**Performance**
5. Semaforo di concorrenza in `EvalSuite.arun` (M-10) e chiusura deterministica degli
   stream provider (M-7) — entrambe riducono pressione su connessioni/rate limit.
6. `Store.__contains__` con `SELECT 1 WHERE key=?` (B-20).

**DX / API**
7. Aggiungere extra `dev` (= `test` + ruff + mypy + pre-commit) in `pyproject.toml` (B-21).
8. Validare in `Agent.__init__`/`as_tool` che un `verify=` callable non sia una coroutine
   function, o gestirla in `_verify.py` con `inspect.iscoroutinefunction` (M-8).
9. Un'unica firma di judge/agent per `EvalSuite.run`/`arun` (M-10).
10. Esportare `LiteLLMProvider` da `core/providers/__init__.py` (B-3).

**Docs**
11. Rigenerare `lazybridge/llms.json` e agganciarlo al gate `skill_docs._build --check`
    (oggi il gate copre `SKILL.md` ma non `llms.json` — è così che è rimasto a 0.7.9).
12. Allineare IMPLEMENTATION.md alla realtà 1.0.1 (o marcarlo esplicitamente storico) e
    sostituire la tabella manuale di `docs/guides/basic/native-tools.md` con quella
    generata da `lazybridge.matrix` (stessa fonte di `docs/reference/providers.md`).
13. Documentare su `docs/reference/providers.md` l'effetto collaterale del force-streaming
    Anthropic (`raw=None`, niente `messages.parse()`) e il knob `force_stream_threshold`
    finché M-14 non è risolta.

---

## 5. Piano di risoluzione dettagliato

### Fase 1 — Correttezza runtime (priorità massima, ~2-3 giorni)

**Step 1.1 — Fix resume `plan_state` (M-13) e stub checkpoint-done (M-11)** — effort **S**
- File: `lazybridge/engines/plan/_plan.py:367-375`.
- Cosa fare: prima del ladder, aggiungere il ramo
  `if plan_state is not None and (plan_state.status == "done" or plan_state.next_step is None): return <envelope aggregato>`
  riusando la logica del ramo checkpoint-done, ma passando per `_aggregate_nested_metadata`
  (risolve anche M-11 per il percorso checkpoint: sostituire il return :371-373 con lo
  stesso helper).
- Test: nuovo `tests/unit/test_plan_resume_done_state.py` — costruire un `PlanState`
  con `status="done"` (la libreria non ne emette: crearlo nel test) → resume → assert
  nessuna ri-esecuzione (contatore sul tool) e metadata aggregata.
- Completamento: test verde + `pytest tests/unit -q` verde.

**Step 1.2 — Classificare le `BaseException` nel tool-loop (A-2)** — effort **S**
- File: `lazybridge/engines/llm.py:913-947`.
- Cosa fare: dopo il re-raise di `ConcludeSignal`, ri-sollevare anche ogni
  `isinstance(tr, BaseException) and not isinstance(tr, Exception)` (CancelledError,
  KeyboardInterrupt, SystemExit); in alternativa mappare `PlanPaused` esplicitamente.
- Test: tool che solleva `asyncio.CancelledError` → il run propaga, non produce
  `ToolResultContent` ok.
- Completamento: regression test in `tests/unit/test_tool_dispatch_baseexception.py`.

**Step 1.3 — Chiusura stream provider + stall path (M-7)** — effort **M**
- File: `engines/llm.py:1058-1070` (aggiungere `finally: await agen.aclose()` sul percorso
  `StreamStallError`); `openai.py:1031,1040,1138,1272`, `deepseek.py:455,496`,
  `litellm.py:565,592`, `google.py:836,946` (wrappare l'iterazione in
  `try/finally` con `close()`/`aclose()` best-effort, sul modello di anthropic.py:911).
- Test: fake stream con contatore `aclose`; consumer che fa break dopo il primo chunk.
- Completamento: test per provider + stall-test esteso in
  `tests/unit/test_stream_idle_timeout_default.py`.

**Step 1.4 — Judge async in `_verify` (M-8)** — effort **S**
- File: `lazybridge/_verify.py:128-130`: `verdict = verify_agent(result.text())`; poi
  `if inspect.isawaitable(verdict): verdict = await verdict`.
- Test: judge `async def` che approva → approvazione al primo tentativo.

### Fase 2 — Sicurezza HIL (~1 giorno)

**Step 2.1 — Token di autenticazione sulla Web UI HIL (A-3)** — effort **M**
- File: `lazybridge/ext/hil/human.py:436-509`.
- Cosa fare: replicare il pattern del viz server (`ext/viz/server.py:117`): generare
  `secrets.token_urlsafe(24)` alla costruzione, includerlo nell'URL stampato/aperto
  (`?token=...`), verificarlo con `hmac.compare_digest` su ogni GET/POST (campo hidden nel
  form), 403 altrimenti. Aggiungere header `Cache-Control: no-store` e `X-Frame-Options`.
- Completamento: test con POST senza/con token errato → 403; con token → accettato.

**Step 2.2 — Cap e validazione del Content-Length (A-4)** — effort **S**
- File: `lazybridge/ext/hil/human.py:461-462`: `try/except ValueError` + rifiuto con 413
  oltre una soglia (es. 1 MiB), come `ext/viz/server.py:244-247`.
- Completamento: test unit del handler con header malformato e con length enorme.

**Step 2.3 — Aggiornare SECURITY.md** — effort **S** — documentare il threat model della
Web UI HIL (localhost multiutente) e il token.

### Fase 3 — CI/test (~1 giorno)

**Step 3.1 — Job CI con tutti gli extra (M-1)** — effort **S**
- File: `.github/workflows/test.yml`: aggiungere al job `unit` una entry di matrice (o un
  job `unit-all-extras`, solo 3.12) con `pip install -e ".[all,test]"`, così encryption,
  OTel e litellm smettono di skippare.
- Completamento: run CI in cui `test_store_encryption.py` risulta eseguito (non skipped);
  coverage `store/encryption.py` > 80%.

**Step 3.2 — Coverage streaming provider e replan** — effort **L**
- Backfill test FakeTransport per i percorsi streaming di `google.py:836-1020` e
  `openai.py:1089-1330` e per `replan/_engine.py` (51%). Obiettivo: nessun modulo provider
  sotto il 75%; alzare poi `fail_under` (pyproject.toml:221) a 78 seguendo la bump-rule
  già scritta nel commento.

### Fase 4 — Provider parity (~2 giorni)

**Step 4.1 — SYSTEM messages su Google/LiteLLM (M-5)** — effort **S**
- `google.py:375-381` e `litellm.py:186-189`: concatenare tutte le fonti come
  `anthropic.py:426-440`. Test di parità in `test_provider_google.py` /
  `test_litellm_provider.py`.

**Step 4.2 — DeepSeek thinking/reasoner (M-6)** — effort **M**
- `deepseek.py:260-305`: sopprimere il warning ereditato per i modelli V4
  thinking-capable e rimuovere `reasoning_effort`/`max_completion_tokens` dai param per
  `deepseek-reasoner` (sostituire con `max_tokens`).

**Step 4.3 — Force-streaming Anthropic (M-14)** — effort **M**
- Opzioni (in ordine di preferenza): (a) popolare `raw` con l'ultimo evento/response
  ricostruita e tentare comunque `messages.parse()` quando `structured_output` è una classe
  Pydantic e l'effective max consente il non-streaming; (b) alzare il log DEBUG esistente
  (anthropic.py:740-744) a INFO/warn documentato al primo force-stream; (c) alzare il
  threshold al limite reale dell'API. In ogni caso: documentare in
  `docs/reference/providers.md` anche il knob `force_stream_threshold` (anthropic.py:303).

**Step 4.4 — Fix minori provider (B-4…B-10)** — effort **M** — no-choices raise coerente
(openai.py:857-876), messaggio misto tool_use+result (openai.py:584-598), dedup/naming
Gemini (google.py:347,861,971), fallback nome Responses (openai.py:996), costo cache 1h
(anthropic.py:219-224), clamp negativo thinking (anthropic.py:576-586), data-URI
(types.py:121-123). Un commit ciascuno con regression test.

### Fase 5 — Contratti e igiene (~1-2 giorni)

**Step 5.1 — `Store.close()` in-memory (M-9)** — effort **S** — aggiungere
`self._ensure_open()` (raise se `_closed`) in testa ai percorsi in-memory di
`write/read/read_entry/delete/clear/keys/items` (`store/__init__.py:170-294`); estendere
`tests/unit/test_store.py`.

**Step 5.2 — `EvalSuite` (M-10)** — effort **S** — `arun(agent, *, concurrency: int = 8)`
con `asyncio.Semaphore`; accettare in entrambi i metodi sia callable sync sia oggetti con
`.run` (helper condiviso).

**Step 5.3 — Serialization/compiler (B-11…B-15)** — effort **M** — round-trip `Any`,
campi persi in `to_dict`, tool zero-arg, generics nel compiler, parametro `kv` morto.

**Step 5.4 — Docs e metadata (M-2, M-3, M-4, M-12, B-1, B-21)** — effort **S/M**
- Rigenerare `lazybridge/llms.json` (version/stability/pattern parallel_fanout/error
  contract/nota lazytoolkit) e aggiungerne la verifica a
  `lazybridge/skill_docs/_build.py --check` (version == `lazybridge.__version__`).
- Aggiornare IMPLEMENTATION.md (nota di chiusura: "superato dagli eventi: 1.0.1 stable
  rilasciata; restano aperte Phase 7/8").
- Riformulare i tre DeprecationWarning (`__init__.py:270-296`) con scadenza "2.0".
- Sostituire la tabella manuale in `docs/guides/basic/native-tools.md:25-35` con lo
  snippet generato; aggiungere IMAGE_GENERATION al README:160.
- Aggiungere extra `dev` a pyproject.

### Fase 6 — Roadmap dichiarata (successiva, effort L)
- Implementare Phase 7 (`agents.md` parser + `LLMEngine.for_agent` + resolver UNSET) e
  Phase 8 (`MediaRef` + tool di generazione) secondo i design doc già approvati, oppure
  retrocederle esplicitamente a "icebox" nei documenti — oggi sono promesse "Must remember
  to implement" (IMPLEMENTATION.md:452-453) senza owner né milestone.

**Ordine consigliato**: Fase 1 → Fase 2 → Fase 3.1 subito (un giorno complessivo di lavoro
per i primi 6 step, tutti S/M); Fasi 4-5 nel prossimo ciclo; 3.2 e 6 pianificate.

---

## 6. Esito dei test eseguiti

Ambiente: venv Python 3.11.15 (riusato da scratchpad), installazione:
`pip install -e '/home/user/LazyBridge[test]'` → pydantic 2.13.4, pytest 9.1.1,
pytest-asyncio 1.4.0, pytest-cov 7.1.0, PyYAML 6.0.3, + `pytest-timeout` 2.4.0.
(Nota: `[dev]` richiesto dalla procedura non esiste come extra — usato `[test]`, vedi B-21.)

```
$ cd /home/user/LazyBridge && python -m pytest -x -q --timeout=120
1937 passed, 22 skipped, 7 deselected, 11 warnings in 26.90s          # VERDE
```

- 7 deselected: marker `live` / `heavy_render` (esclusi da `addopts`, pyproject.toml:94).
- 22 skipped: quasi tutti per dipendenze opzionali assenti (cryptography ×4,
  opentelemetry ×12, litellm ×1, anthropic SDK ×1, lazytools ×1, marker package ×2) —
  elenco completo ottenuto con `pytest -rs`.

Coverage:

```
$ python -m pytest -q --timeout=120 --cov=lazybridge --cov-report=term-missing
1937 passed, 22 skipped, 7 deselected in 28.68s
TOTAL 9006 stmt, 1765 miss, 80.40%   (gate fail_under=73: superato)
```

Moduli meno coperti: `ext/otel/exporter.py` 0% · `store/encryption.py` 31% ·
`engines/replan/_engine.py` 51% · `core/providers/google.py` 58% ·
`ext/viz/visualizer.py` 58% · `skill_docs/_build.py` 62% · `core/providers/openai.py` 65% ·
`core/providers/{anthropic,deepseek}.py` 69% · `ext/planners/builder.py` 70% ·
`ext/viz/server.py` 70%.

Verifica extra opzionali (installati `cryptography`, `opentelemetry-api/sdk` nel venv):

```
$ python -m pytest -q tests/unit/test_store_encryption.py tests/unit/test_otel_exception_logging.py \
    tests/unit/test_audit_short_term.py tests/unit/test_resolution_fixes.py \
    tests/unit/test_audit_amend.py tests/unit/test_audit_deep.py
88 passed, 2 warnings in 5.81s                                        # VERDE
```

Conclusione test: **nessun test fallito**; l'unico rilievo è che gli skip per extra
opzionali sono permanenti anche in CI (issue M-1).

---

## Scartate in revisione

Nessuna issue è risultata *falsa* alla verifica sul codice: tutte le affermazioni
meccaniche (file:riga, comportamento del codice) sono state riscontrate. Due issue ALTA
erano però **esagerate** e sono state declassate a MEDIA invece che eliminate:

- **A-1 → M-13**: il difetto (`PlanState.status` mai consultato) è reale, ma la
  riproduzione dichiarata era errata (`plan.run()` ritorna un `Envelope`, non uno stato) e
  la libreria non produce mai istanze di `PlanState`: lo scenario richiede uno stato
  costruito a mano dall'utente. Impatto reale = footgun su API pubblica di nicchia, non
  "ri-esecuzione ri-sottomettendo lo stato di un run concluso".
- **A-5 → M-14**: la prima stesura ignorava tre mitigazioni presenti nel codice: l'output
  strutturato NON degrada (il percorso force-stream invia comunque `output_config`
  server-side e popola `parsed`/`validated` via `apply_structured_validation`,
  anthropic.py:888-896,960-963); esiste un knob `force_stream_threshold` nel costruttore
  (anthropic.py:303); il comportamento è loggato a DEBUG (:740-744). Residuo reale:
  `raw=None` silenzioso e assenza totale di documentazione utente.

---

## Nota di revisione (verifica adversariale)

Seconda passata (2026-07-08) di verifica contro il codice reale, con ricerca attiva di
controprove (grep di guardie/validazioni/chiamanti, lettura dei percorsi citati).

- **Issue verificate**: 38 su 38 — tutte le 5 ALTA, tutte le 12 MEDIE (oltre la metà
  richiesta), tutte le 21 BASSE (almeno a livello di riferimento e meccanismo).
- **Confermate senza modifiche sostanziali**: 32 (A-3, A-4, M-1–M-3, M-5, M-8–M-12,
  B-1–B-21).
- **Corrette/precisate**: 6 —
  - A-2: eliminati `KeyboardInterrupt`/`SystemExit` dallo scenario (fanno comunque
    crashare l'event loop); il caso reale è `CancelledError` interna / `PlanPaused`.
  - A-1 e A-5: declassate a M-13/M-14 (vedi "Scartate in revisione").
  - M-6: "warning falso a ogni chiamata" → una volta per call-site (filtro warnings).
  - M-7: precisata la semantica dell'aggravante stall-path (il generatore provider viene
    chiuso dalla cancellazione; è lo stream SDK/httpx a restare al GC).
  - M-4: riferimenti riga corretti (llms.json:3-4, :27, :18) + nome import `lazytools`.
- **Riferimenti file:riga corretti**: `test.yml:106-108`→`:108-110` (M-1),
  `05_mcp_allowlisted.py:12`→`:3` (§2), cross-reference errato in §1
  ("stream non chiusi (§3 M-11)" → M-7).
- **Eliminate**: 0. **Nuove issue gravi**: 0 — verifica mirata su XSS nella Web UI HIL
  (l'HTML escapa correttamente task/campi via `html.escape`, human.py:234-252), su
  produttori di `PlanState`, su chiusura stream e su TODO/FIXME residui: nessun nuovo
  rilievo ALTA/CRITICA.
- **Coerenza §6**: numeri test/coverage internamente coerenti (9006−1765 = 80.40%;
  1937+22+7 costanti tra i run; gli 88 test extra citati in M-1 tornano col run dedicato).
  Non rieseguiti.
- **Conteggi finali**: 0 CRITICA · 3 ALTA (A-2, A-3, A-4) · 14 MEDIA (M-1–M-14) ·
  21 BASSA. **Voti invariati** (il declassamento di A-1/A-5 è compensato dalla conferma
  integrale di A-2/A-3/A-4; Correttezza B+, Sicurezza B, Test B+, Docs A-,
  Manutenibilità A- restano motivati).
- **Piano §5**: verificato coerente ed eseguibile; aggiornati i riferimenti degli step
  1.1 (M-13/M-11) e 4.3 (M-14, con menzione del log DEBUG esistente e del knob
  `force_stream_threshold`); comandi e percorsi file controllati (pyproject.toml:94
  `addopts`, :221 `fail_under=73`, `ext/viz/server.py:113-117,244-247` come modello).
