# Piano Master dell'Ecosistema Lazy* — Assessment conclusivo (Luglio 2026)

> **Documento conclusivo** dell'assessment completo di implementazione condotto su tutte e 8 le repository
> dell'ecosistema. Ogni repository contiene il proprio documento di dettaglio **`ASSESSMENT_2026-07.md`**
> (sul branch `claude/repo-assessment-plan-ebby86`) con issue puntuali (file:riga), punti di miglioramento
> e guida operativa step-by-step. Questo documento aggrega i risultati, individua i temi trasversali e
> definisce l'ordine di intervento complessivo.

---

> **Aggiornato dopo la revisione adversariale.** Ogni singolo `ASSESSMENT_2026-07.md` è stato passato a un
> secondo revisore indipendente con l'ordine esplicito di verificare ogni issue contro il codice reale,
> smontare le esagerazioni e cercare attivamente issue sfuggite. Risultato: 2 issue ALTE declassate a MEDIA
> (esagerate), 1 declassata a BASSA (irraggiungibile a runtime), decine di riferimenti file:riga corretti,
> e 6 issue nuove trovate (incluse 2 riprodotte a runtime nel venv). I numeri sotto sono quelli **post-revisione**.

## 1. Sintesi esecutiva

L'ecosistema è **in buona salute complessiva**: tutte le suite di test sono verdi (3.343 test eseguiti
realmente in totale, 0 falliti), gli audit precedenti risultano effettivamente applicati nel codice
(verificato riga per riga, due volte), e nessuna repo è in stato critico strutturale. Restano però **2 issue
CRITICHE** (entrambe in LazyHMM, riconfermate con repro end-to-end in sede di revisione), **17 issue ALTE**
distribuite su 7 repo, e alcuni **temi trasversali** (supply chain, error handling che maschera i
fallimenti, lock/concorrenza) che vanno chiusi in modo sistematico.

**Totale issue trovate (post-revisione): 165** — 2 CRITICHE · 17 ALTE · 59 MEDIE · 87 BASSE.

## 2. Quadro riassuntivo per repository

| Repo | Salute | Critiche | Alte | Medie | Basse | Test eseguiti | Coverage |
|---|---|---|---|---|---|---|---|
| **LazyTools** | A- | 0 | 0 | 5 | 9 | 362 passed | 90% |
| **LazyFin** | A- | 0 | 2 | 8 | 13 | 204 passed | 96% |
| **LazyPulse** | B+/A- | 0 | 1 | 7 | 8 | 269 passed | 93% |
| **LazyCrawler** | B+ (pre-1.0) | 0 | 1 | 6 | 11 | 221 passed | 73% |
| **LazyBridge** | B+ | 0 | 3 | 14 | 21 | 1937 passed | 80% |
| **lazybridgewebsite** | B+ (manutenz. C) | 0 | 2 | 4 | 5 | build strict OK | n/a |
| **market-data-hub** | B | 0 | 3 | 7 | 10 | 105 passed | n/d |
| **LazyHMM** | B- | **2** | 5 | 8 | 10 | 245 passed | 65% |

Voti di dettaglio (correttezza / sicurezza / test / docs / manutenibilità) nel §1 di ogni
`ASSESSMENT_2026-07.md`. Nota: in LazyTools il voto Sicurezza è stato abbassato da A- a B+ in sede di
revisione (nuove issue di concorrenza/permessi trovate, vedi §3).

## 3. Le issue più gravi dell'ecosistema (da risolvere per prime)

### CRITICHE (2 — entrambe LazyHMM, riconfermate in revisione con repro end-to-end)
1. **LazyHMM** — `load_ticker` rotto: `show_errors` rimosso da yfinance (verificato: `pip show` → yfinance 1.5.1, la firma non ha più il parametro) → TypeError garantito (`db.py:945`). Fix da una riga.
2. **LazyHMM** — round-trip SQLite perde il tipo di modello: scrive chiave `model`, rilegge `mode` → metadati persi dopo ogni restart (`db.py:360` vs `tools.py:1252`; repro rieseguito: `model=None, mode=''`). Fix da una riga.

### ALTE più impattanti (selezione cross-repo, post-revisione)
3. **lazybridgewebsite** — supply chain nella pipeline di deploy: `requirements.txt` non pinnato installa `mkdocs-redirects==1.2.3`, ripubblicata da terzi ("ProperDocs") — verificata su PyPI live: richiede `properdocs>=1.6.5`, un **fork completo di MkDocs 1.6.x** con hook di aliasing dormiente. Oggi solo un warning su stderr, ma è nella pipeline di deploy con token `contents: write`.
4. **LazyBridge** — Web UI human-in-the-loop **senza autenticazione**: qualunque processo locale può forgiare approvazioni (`ext/hil/human.py:505`, confermata in revisione) + DoS memoria via Content-Length illimitato (`:462`, confermata).
5. **LazyPulse** — `asyncio.Lock` condiviso tra due event loop nel `WebhookAdapter`: sotto contesa **congela l'intero tick loop** (hang riprodotto **due volte** in sede di revisione, >120 s, Python 3.11.15) (`adapters/webhook.py:77,83,145`).
6. **market-data-hub** — HTTP 429/5xx di Yahoo esauriscono i retry e finiscono loggati come "empty": **un outage del provider è invisibile** (confermato: nessun retry a livello runner, nessuna mitigazione a monte) (`sources/yahoo_direct.py:99-118`); aggravato da `run_daily.py` che ritorna exit 0 anche a download completamente fallito.
7. **LazyFin** — workflow rebalance senza freshness gate, senza ADV e senza prezzi nel record: il limite di liquidità non può mai bocciare (`workflows/rebalance.py:129-146`, confermata al 100% in revisione).
8. **market-data-hub** — `custom.store_series/delete_series` scrivono **senza writer lock**: il publish da LazyFin concorrente al run schedulato causa IO error DuckDB (`custom.py:106,115`, confermata: nessun lock lato LazyFin).
9. **LazyBridge** — Anthropic forza streaming su tutti i modelli moderni: `raw=None` sempre e comportamento non documentato quando `output_config` e la validazione restano attivi in force-stream (`core/providers/anthropic.py:67,730-746,810`, precisata in revisione).
10. **LazyFin** — segno di `Transaction.amount` mai validato: un BUY con importo positivo *aggiunge* cassa invece di sottrarla e azzera il cost basis, in silenzio — **riprodotta a runtime** nel venv durante la revisione (`kernel/portfolio.py:358-362,416`).

> Due issue della prima stesura sono state **smontate dalla revisione** e non compaiono più tra le ALTE:
> il presunto bypass SSRF via fallback PDF in LazyCrawler (`pdf.py:118-142`) è risultato **irraggiungibile**
> a guardia SSRF attiva (branch mutuamente esclusivo con la protezione) — declassato a BASSA; il resume
> di `plan_state` e la gestione di `BaseException` in LazyBridge sono stati ridimensionati a MEDIA dopo
> verifica puntuale dei percorsi di codice. Dettagli nelle rispettive sezioni "Nota di revisione" dei
> singoli assessment.

## 4. Temi trasversali (pattern ricorrenti da chiudere ovunque)

**T1 — Supply chain e pinning.** Website: dipendenza ripubblicata + actions non SHA-pinnate; LazyPulse:
4 dipendenze git senza SHA nel deploy Docker; LazyFin è l'esempio virtuoso già a posto (pin a SHA).
→ Standard unico: pin a SHA per dipendenze git, versioni esatte nei requirements di deploy, actions SHA-pinnate.

**T2 — Error handling che maschera i fallimenti.** mdh (429→"empty", exit 0), LazyBridge
(BaseException→successo), LazyCrawler (`render_js` fabbrica status 200 e cacheia errori come `done`).
→ Regola: mai convertire un errore in un successo silenzioso; exit code e status sempre veritieri.

**T3 — Concorrenza e risorse.** LazyPulse (lock cross-loop), mdh (scritture senza lock), LazyTools/LazyFin
(client HTTP mai chiusi), LazyBridge (stream non chiusi su early-exit).
→ Audit sistematico di lock e lifecycle dei client in ogni repo (checklist nei singoli piani).

**T4 — Drift delle dipendenze esterne.** LazyHMM rotto da yfinance e matplotlib correnti; LazyPulse
non installabile da PyPI (lazytoolkit non pubblicato); rischio MkDocs 2.0 sul sito.
→ Job CI periodico "latest deps" che installi le dipendenze correnti e fumi i percorsi critici.

**T5 — `pytest-timeout` mancante dalle dev-deps** in LazyHMM, LazyPulse e LazyTools (emerso in 3 repo
su 8 durante l'esecuzione reale dei test). Fix banale, da fare ovunque.

**T6 — Coverage disomogenea.** Eccellente in LazyFin/LazyPulse/LazyTools (90-96%), debole in LazyHMM
(65%, contract 15%) e LazyCrawler (73%, browser 34%). → Gate `--cov-fail-under` calibrati per repo.

## 5. Piano complessivo — ordine di intervento consigliato

> Il dettaglio operativo di ogni step (comandi, file da toccare, criteri di completamento) è nel §5
> ("Piano di risoluzione step-by-step") dell'`ASSESSMENT_2026-07.md` di ciascuna repo. Qui la sequenza globale.

### Fase 0 — Quick win, tutti effort S (1 giorno complessivo)
| # | Repo | Intervento |
|---|---|---|
| 0.1 | LazyHMM | Fix `show_errors` (yfinance) e chiave `model`/`mode` nel depot — le 2 CRITICHE, entrambe one-line |
| 0.2 | LazyFin | Fix crash resolver: `(entry.get("ticker") or "").upper()` (`resolve/_resolver.py:81`) |
| 0.3 | market-data-hub | Exit code veritiero in `run_daily.py`; status non-200 Yahoo trattati come errori |
| 0.4 | lazybridgewebsite | Pinnare requirements ed **eliminare properdocs** dalla pipeline |
| 0.5 | Tutte | Aggiungere `pytest-timeout` alle dev-deps (LazyHMM, LazyPulse, LazyTools) |

### Fase 1 — Sicurezza (settimana 1)
| # | Repo | Intervento |
|---|---|---|
| 1.1 | LazyBridge | Token auth + cap Content-Length sulla Web UI HIL (pattern già pronto nel viz server) |
| 1.2 | LazyCrawler | Migrare fallback PDF sul client HTTP condiviso (debito di consistenza, non più bypass sfruttabile: la revisione ha confermato che il branch è irraggiungibile a guardia attiva — comunque da chiudere per difesa in profondità) + allowlist schemi http/https |
| 1.3 | lazybridgewebsite | Hardening workflow: permessi minimi, SHA-pin actions, build di verifica su PR |
| 1.4 | market-data-hub | Scritture `custom_series` dentro `db_write_lock()` (chiude anche il gap col publish LazyFin) |
| 1.5 | market-data-hub | Correggere README: FRED API key solo via env, mai nel `settings.yaml` tracciato |

### Fase 2 — Correttezza runtime (settimane 2-3)
| # | Repo | Intervento |
|---|---|---|
| 2.1 | LazyPulse | `threading.Lock` nel WebhookAdapter + test cross-loop (chiude l'hang del tick loop) |
| 2.2 | LazyBridge | Fix resume `plan_state`, classificazione `BaseException`, `aclose()` stream, judge async |
| 2.3 | LazyCrawler | Propagare status reale del browser (no cache `done` su errori); validazione modalità nel sync |
| 2.4 | LazyFin | Freshness gate + ADV + prezzi-nel-record nel workflow rebalance; `frozen=True` sui modelli |
| 2.5 | LazyHMM | `regime_store_delete` completo (depot SQLite); validazione NaN/simboli negativi; fix `cm.get_cmap` |
| 2.6 | LazyPulse | Rate limit su identità verificata (no `sender_raw`); potatura marker `pulse:event:*` |
| 2.7 | LazyTools | `close()`/context-manager su EdgarClient e StooqAdapter; `ConfirmationGate` su `datahub_refresh_prices`; fix TOCTOU `save_report` |

### Fase 3 — Packaging e supply chain (settimane 3-4)
| # | Repo | Intervento |
|---|---|---|
| 3.1 | LazyPulse | Pubblicare lazytoolkit su PyPI (o rimuovere gli extras rotti); SHA-pin delle 4 dipendenze git del deploy; HEALTHCHECK e utente non-root nel Dockerfile |
| 3.2 | Tutte | Standard di pinning uniforme (T1); riallineare CHANGELOG/versioni (LazyTools 0.3.1, LazyPulse 0.3.1, LazyCrawler versione triplicata) |
| 3.3 | LazyBridge | Job CI con `.[all,test]` (oggi encryption/OTel/litellm non hanno gate di regressione) |

### Fase 4 — Test e coverage (mese 2)
| # | Repo | Intervento |
|---|---|---|
| 4.1 | LazyHMM | Coverage 65%→80%+: contract (15%!), db, tools, plotting; test per `load_ticker`/`merge_series` |
| 4.2 | LazyCrawler | Coverage llm/pdf/browser a ≥75/75/60% con gate `--cov-fail-under` in CI |
| 4.3 | market-data-hub | Test per runner e fetcher di rete (oggi 0); CI se assente |
| 4.4 | Tutte | Job CI "latest deps" settimanale (T4) per intercettare i drift tipo yfinance/matplotlib |

### Fase 5 — Roadmap e debito documentale (mese 2+)
| # | Repo | Intervento |
|---|---|---|
| 5.1 | LazyBridge | Decidere su Phase 7 (SuperTool/agents.md) e Phase 8 (media output): implementare o depennare dai piani |
| 5.2 | LazyCrawler | Allineare ROADMAP/ANALYSIS (voci mancanti, nomi test inesistenti); unificare fallback testo sync/async |
| 5.3 | market-data-hub | Deduplicare gli script top-level in entry point del package |
| 5.4 | lazybridgewebsite | Ripristinare social/PyPI nel footer; prepararsi a MkDocs 2.0; rimuovere logo Ollama o aggiungere l'adapter |
| 5.5 | LazyHMM | Rimuovere/dichiarare la dipendenza privata dei contract test (oggi skip totale anche in CI) |

## 6. Ordine di priorità tra le repo

1. **LazyHMM** — uniche 2 CRITICHE dell'ecosistema, feature rotte con le dipendenze attuali
2. **market-data-hub** — gira schedulato senza sorveglianza: gli errori mascherati sono i più pericolosi
3. **lazybridgewebsite** — la issue supply-chain tocca la pipeline di deploy pubblica
4. **LazyBridge** — 5 ALTE, ma prodotto maturo con la suite test più ampia; correzioni ben delimitate
5. **LazyPulse** — 1 ALTA seria (hang) + packaging rotto da PyPI
6. **LazyCrawler** — residui SSRF noti e coverage
7. **LazyFin** — già molto sana; chiudere il rebalance gap
8. **LazyTools** — la più sana: solo manutenzione ordinaria

## 7. Come usare questi documenti

- Ogni repo: **`ASSESSMENT_2026-07.md`** = guida operativa autonoma (issue con file:riga → piano §5 con comandi, criteri di completamento ed effort S/M/L). Si può seguire senza altro contesto.
- Questo documento = ordine di lavoro tra le repo e temi da chiudere in modo uniforme.
- Tutti i numeri di test/coverage citati provengono da **esecuzioni reali** effettuate durante l'assessment (venv dedicati, comandi e output nel §6 di ogni documento).

## 8. Nota sul processo di verifica (revisione adversariale)

Ogni documento è stato prodotto in due passate indipendenti:

1. **Prima stesura** — un auditor analizza la repo da zero (codice, test eseguiti realmente, CI, docs) e scrive l'assessment.
2. **Revisione adversariale** — un secondo agente, senza fidarsi della prima stesura, riapre ogni file:riga citato, cerca attivamente controprove (mitigazioni ignorate, guardie esistenti, percorsi irraggiungibili), rilancia repro a runtime dove possibile, e corregge il documento **in place**: elimina le issue che non reggono, le sposta in una sezione "Scartate in revisione" con motivazione, corregge i riferimenti imprecisi, e — quando trova qualcosa di grave sfuggito alla prima passata — lo aggiunge.

Risultato della revisione su tutte le 8 repo: **165 issue verificate** (partenza: 156), **6 issue nuove** trovate (2 riprodotte a runtime: LazyFin/portfolio.py e LazyPulse/adapters/webhook.py), **3 issue declassate** per esagerazione o irraggiungibilità (2 in LazyBridge da ALTA a MEDIA, 1 in LazyCrawler da ALTA a BASSA), **nessuna issue eliminata per infondatezza totale** — ogni claim della prima stesura si è rivelato quantomeno un problema reale, solo talvolta di severità inferiore a quanto scritto inizialmente. Ogni documento riporta la propria sezione "Nota di revisione (verifica adversariale)" con il dettaglio.

**In sintesi: i numeri e le priorità di questo documento sono quelli emersi dopo doppia verifica, non dalla prima stesura.**
