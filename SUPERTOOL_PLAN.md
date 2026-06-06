# SuperTool / Skill — Piano di implementazione LazyBridge

> **Status.** Proposal / design doc. Composes with `agents.md`
> (`PROJECT_LAYOUT.md`) when both are present — see *Composition* and
> *Review notes* at the end of this file.

## Principi fissati (invarianti di design)

Questi non si rinegoziano in fase di implementazione. Tutto il resto del piano discende da qui.

1. **La SuperTool non è un primitivo core.** È una *factory dichiarativa* che compila verso i primitivi esistenti (`Tool`, `Agent`, `as_tool`, `Guards`). Nessuna nuova classe nel core, nessuna per-pattern abstraction. Resta sulla scala di complessità progressiva.

2. **Nessun fork del formato.** La `Skill` legge il formato aperto Agent Skills (`SKILL.md`: frontmatter `name`/`description` + corpo + reference esterne). La differenziazione di LazyBridge non è un formato proprietario: è *la classe che lega i pezzi che lo standard file-based lascia slegati*.

3. **La Skill è puramente dichiarativa: suggerisce, non impone mai.** Tutto ciò che arriva da una Skill è sovrascrivibile per costruzione. La Skill non possiede policy.

4. **Gerarchia: l'override esterno vince sempre, uniformemente, a ogni livello.** Chi è più esterno ha più autorità — il chiamante batte il tool, l'orchestratore batte il chiamante, la policy di fleet batte tutto. Regola unica e lineare, applicata identica a `disclosure`, `output` e ogni futuro campo:
   ```python
   effective = caller_value if caller_value is not UNSET else skill_value
   ```
   > **Corretto** rispetto al draft originale (`is not None`): vedi *Review notes* #2 — `None` è un valore esplicito legittimo, quindi il marcatore "non passato" è un sentinel `UNSET`, non `None`.
   >
   > **Raffinato** dalla composizione con `agents.md` (vedi *Review notes* #5): la regola "uno vince" sopra è la strategia **OVERRIDE**, valida per i campi a valore singolo (`output`, `model`, knob). Il **prompt** usa invece la strategia **COMPOSE** (concatena, non sceglie). Resta un solo resolver, ma parametrizzato dalla strategia per-campo — *non* "stessa regola identica per ogni campo".

5. **L'enforcement derivato da una Skill vive in alto, mai sulla foglia.** Una garanzia che deve reggere (PII, safety) si monta sull'engine, sulla policy di fleet, o come `Guard` sull'`Agent` esterno — mai *dentro la Skill*. La sicurezza *suggerita da una Skill* si sposta verso l'alto nella gerarchia.
   > **Riconciliato** col modello safety di LazyTools: vedi *Review notes* #1 — un **Tool** (codice) può comunque avere i suoi gate hard (es. `ConfirmationGate`, `Allowlist`). Ciò che non impone mai è la **Skill** (documento).

6. **`output` su Skill è un default, non una garanzia.** Va rinominato per dire la verità in firma: `suggested_output=` (o `default_output=`). Documentare a chiare lettere che `Skill(suggested_output=Invoice)` *non* garantisce un `Invoice` in uscita.

7. **`output` e `safeguard` viaggiano come un'unica unità di default.** Nessun override → erediti entrambi (schema standard + validazione standard, coerenti). Override dell'output → la safeguard standard **decade** (non si trasferisce, non fallisce su una forma che non produci più).

8. **Override dell'output senza safeguard → nessuna safeguard.** La coppia output+safeguard si rompe *interamente*: o entrambi i pezzi o nessuno, mai metà. Non si deriva nulla, non resta montato niente. Mantenere la safeguard vecchia non avrebbe senso (validerebbe una forma non più prodotta), e derivarne una d'ufficio sarebbe il framework che impone qualcosa dal basso — vietato dalla gerarchia "esterno vince". Chi sovrascrive l'output si prende la responsabilità della validazione: se non la fornisce, corre senza rete, ed è una scelta legittima.
   > **Aggiunta** (non imposizione): il decadimento emette un **segnale di osservabilità** opt-in sulla `Session` ("output overridden → standard safeguard no longer applies"). Informare non è imporre — vedi *Review notes* #4.

---

## Modello a tre livelli di disclosure

La SuperTool standardizza un parametro `disclosure`, ciascun valore mappato a un trade-off contesto/latenza diverso:

- **`inline`** — alla selezione del tool, il corpo della skill entra una volta nel contesto del chiamante. Zero round-trip, paghi i token di teoria nel ragionamento. Per tool con poche best-practice critiche.
- **`on_request`** — la SuperTool registra un *tool compagno* `{name}__guide` che restituisce la skill solo se invocato. Niente in contesto finché l'agente non "consulta il manuale". Più economico, più fedele alla progressive disclosure: la teoria diventa essa stessa una tool-call.
- **`subagent`** — la skill diventa il `system` di un subagente che incapsula il tool; il chiamante vede solo la firma. Da usare quando la competenza è *genuinamente distinta*. Più costoso (round-trip in più) → in questa modalità **cachare il priming** legando il subagente a una `Session`, altrimenti ripaghi l'innesco a ogni chiamata.

> **Da fissare in Fase 0** (vedi *Review notes* #3): `disclosure` cambia *come* il tool viene materializzato (registra un companion tool, compone un subagente), non solo *quale valore* passa. Decidere se si risolve a **costruzione** (binding Skill+callable) o a **call-time**, perché passare da `inline` a `subagent` a chiamata fatta implica ri-materializzare il tool.

---

## Fasi

### Fase 0 — Contratto API e naming (bloccare prima di scrivere codice)
- Definire la firma di `Skill(...)`: `path`/`source`, `disclosure`, `suggested_output`, `safeguard`, eventuale `tier` override.
- Decidere il nome definitivo del default di output (`suggested_output` raccomandato).
- Decidere il nome del tool compagno per `on_request` (`{name}__guide`).
- **Decidere il timing di risoluzione di `disclosure`** (costruzione vs call-time).
- Scrivere i docstring che dichiarano esplicitamente la natura "suggerisce, non garantisce".

### Fase 1 — Skill loader
- Parser del formato aperto `SKILL.md` (frontmatter YAML + corpo Markdown + reference esterne).
- Mappare i tre tier di progressive disclosure: metadati (sempre) / corpo (alla selezione) / reference (on-demand).
- Validare il vincolo di lunghezza consigliato (<500 righe nel corpo, il resto in reference).
- Test: una skill di esempio carica e i tre tier sono accessibili separatamente.

### Fase 2 — Factory + resolver di precedenza (CONDIVISO con agents.md)
- `SuperTool`/factory che, dato `Skill + callable/Agent`, compila verso i primitivi.
- Implementare il **resolver** (invariante #4) come unico punto di risoluzione, riusato per ogni campo — **lo stesso resolver e lo stesso sentinel `UNSET` del `LLMEngine.for_agent` di `agents.md`**. Costruirlo una volta sola, in un posto solo.
- Resolver **una funzione, parametrizzata dalla strategia di merge per-campo**: `OVERRIDE` (valore singolo → vince la scala, agent batte Skill) vs `COMPOSE` (prompt → concatena). *Non* "stessa regola identica per ogni campo": la strategia è un attributo del campo, non un ramo ad-hoc.
- Test: per i campi OVERRIDE, caller batte agent batte Skill; per il prompt, agent + Skill si compongono (nulla perso) e `system=` esplicito del chiamante resta override; `None` esplicito è preservato (non confuso con "non passato").

### Fase 3 — Modalità di disclosure
- `inline`: iniezione una-tantum del corpo nel contesto alla selezione.
- `on_request`: registrazione del tool compagno `{name}__guide`; verificare che nulla entri in contesto prima dell'invocazione; verificare che la description del tool principale annunci l'esistenza della guida.
- `subagent`: composizione `Agent(engine=LLMEngine(system=<corpo skill>), tools=[Tool(fn)]).as_tool(name, description)`; legare a `Session` per il cache del priming.
- Test per ognuna: misurare i token in contesto e i round-trip, confermare i trade-off attesi.

### Fase 4 — Safeguard: accoppiamento e decadimento
- Legare `suggested_output` e `safeguard` come **una sola unità** nella factory (invariante #7): non si può ereditare metà coppia.
- Regola secca, nessun knob:
  - nessun override → coppia standard (output + safeguard)
  - override con safeguard → coppia del chiamante
  - override senza safeguard → output del chiamante, **zero validazione**
- Quando l'output viene sovrascritto, la safeguard standard **decade** sempre e *non* viene rimpiazzata d'ufficio. Emette un **evento `Session`** di osservabilità (opt-in), che è informazione, non enforcement.
- Test: override-con-safeguard (→ coppia caller), override-senza-safeguard (→ output caller, nessuna validazione, evento emesso), no-override (→ coppia standard). Verificare che non resti mai montata una safeguard su forma obsoleta e che il framework non monti mai una safeguard non richiesta dal chiamante.

### Fase 5 — Threading dell'output multi-provider
- `suggested_output` (o l'override) viene semplicemente inoltrato all'engine; l'enforcement per-provider è già gestito da LazyBridge. **Nessuna logica nuova qui — solo threading.**
- La Skill non sa nulla di provider. Eredita l'enforcement gratis dal momento in cui passa lo schema all'engine.
- Test: stessa SuperTool su due provider con backend di structured-output diversi → output conforme in entrambi.

### Fase 6 — Test d'integrazione, docs, migrazione
- Test end-to-end sulle tre disclosure × i tre casi di safeguard.
- Test di **composizione con `agents.md`**: un agente definito in `agents.md` che usa una SuperTool con Skill; verificare la scala di autorità unica (`PROJECT_LAYOUT.md` §4).
- Doc del concetto: i due piani (default sovrascrivibile vs enforcement che vive in alto), la regola di precedenza, la semantica di `disclosure`, la posizione di `output` come default.
- **Avviso esplicito in doc:** `suggested_output` non è una garanzia; chi vuole un invariante hard lo monta in alto (engine/fleet/Guard sull'Agent esterno), non sulla Skill.
- Aggiungere agli anti-pattern: "trattare `suggested_output` come garanzia", "montare enforcement dentro la Skill aspettandosi che regga contro l'override".
- Nota di migrazione se `Skill`/SuperTool tocca firme pubbliche esistenti.

---

## Composition con agents.md

Quando in un progetto sono presenti **sia** `agents.md` **sia** una
SuperTool/Skill, i due non competono: si compongono su **una sola scala
di autorità**, definita in dettaglio in `PROJECT_LAYOUT.md` §4.

I campi si risolvono con **due strategie**, non una sola
(`PROJECT_LAYOUT.md` §4):

- **OVERRIDE** (valore singolo: `output`, `model`, knob, `disclosure`) —
  vince uno, scelto dalla scala. Qui **l'agent batte la Skill**:
  ```
  fleet policy  >  caller (explicit)  >  agents.md  >  Skill (suggested)  >  engine default
  ```
- **COMPOSE** (il prompt) — non si sceglie, si **concatena**: prompt
  `agents.md` + corpo Skill, nulla si perde (salvo `system=` esplicito del
  chiamante, che resta override in cima alla scala).

In sintesi:

- `agents.md` configura l'**Agent** (model, prompt, output di base).
- la Skill configura un **Tool** che l'agente chiama (disclosure, output
  suggerito, safeguard accoppiata) e, per il prompt, *aggiunge* la propria
  competenza al prompt dell'agente.
- **Meccanismo condiviso:** un solo resolver, parametrizzato dalla
  strategia per-campo (OVERRIDE | COMPOSE); un solo sentinel `UNSET`; una
  sola rappresentazione dell'output. Costruiti una volta in `agents.md`
  (`LLMEngine.for_agent`), riusati qui (Fase 2).

---

## Review notes (correzioni concordate, da risolvere)

1. **Invariante #5 vs LazyTools.** Il modello safety di LazyTools mette
   enforcement *sulla foglia* di proposito (`ConfirmationGate` one-shot
   task-bound, `Allowlist`). Quindi "nessun enforcement sul tool" è
   troppo assoluto. Riconciliazione: la **Skill** (documento) non impone
   mai; il **Tool** (codice) può avere i suoi gate hard. #5 va inteso
   come "le *suggestioni derivate da una Skill* non sono mai
   enforcement", non "nessun enforcement vive sul tool".
2. **Sentinel `UNSET` nell'invariante #4.** `is not None` rompe il caso
   `None` esplicito (`temperature=None` → default provider; `output=None`
   → "niente structured output"). Usare un `UNSET` di modulo. È lo stesso
   problema/soluzione del `for_agent` di `agents.md`.
3. **Timing di `disclosure`.** A differenza di `output` (un valore che
   passa), `disclosure` cambia *come* il tool è materializzato. Fissare
   in Fase 0 se si risolve a costruzione o a call-time; "niente rami
   per-campo" potrebbe non bastare per questo campo.
4. **Decadimento safeguard (#8) — segnale, non silenzio.** Tenere la
   regola di decadimento, ma emettere un evento `Session` opt-in. C'è
   differenza tra *imporre* una validazione (vietato) e *informare* che
   è stata persa (osservabilità, lecita).
5. **Strategia di merge per-campo (emenda #4).** La composizione con
   `agents.md` impone due strategie, non una: **OVERRIDE** (campi a valore
   singolo — `output`, `model`, knob, `disclosure` — dove l'agent batte la
   Skill) e **COMPOSE** (il prompt, che si concatena senza perdere nulla).
   Resta un solo resolver, ma la strategia è un attributo del campo.
   **Confermato:** (a) ordine di concatenazione del prompt = agent prima,
   Skill in coda; (b) sì, tutti gli scalari (`model`/`temperature`/
   `thinking`/`max_tokens`) seguono `output` → agent vince.

---

## Riepilogo del modello in una frase

La SuperTool è una factory dichiarativa che lega un documento-skill (formato standard, portabile) a una `Tool`/`Agent` e ne trasporta i default — disclosure, output suggerito, safeguard accoppiata — applicando ovunque la stessa regola: *l'esterno vince*. L'enforcement non sparisce: si colloca al livello giusto della gerarchia, che non è mai la foglia — e, quando c'è anche `agents.md`, i due condividono un'unica scala di autorità e un unico resolver.
