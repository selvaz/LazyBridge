# Piano Master dell'Ecosistema Lazy* — Assessment conclusivo (Luglio 2026)

> **Documento conclusivo** dell'assessment completo di implementazione condotto su tutte e 8 le repository
> dell'ecosistema. Ogni repository contiene il proprio documento di dettaglio **`ASSESSMENT_2026-07.md`**
> (sul branch `claude/repo-assessment-plan-ebby86`) con issue puntuali (file:riga), punti di miglioramento
> e guida operativa step-by-step. Questo documento aggrega i risultati, individua i temi trasversali e
> definisce l'ordine di intervento complessivo.

---

## 1. Sintesi esecutiva

L'ecosistema è **in buona salute complessiva**: tutte le suite di test sono verdi (3.343 test eseguiti
realmente in totale, 0 falliti), gli audit precedenti risultano effettivamente applicati nel codice
(verificato riga per riga), e nessuna repo è in stato critico strutturale. Restano però **2 issue
CRITICHE** (entrambe in LazyHMM), **20 issue ALTE** distribuite su 7 repo, e alcuni **temi trasversali**
(supply chain, error handling che maschera i fallimenti, lock/concorrenza) che vanno chiusi in modo
sistematico.

**Totale issue trovate: 156** — 2 CRITICHE · 20 ALTE · 51 MEDIE · 83 BASSE.

## 2. Quadro riassuntivo per repository

| Repo | Salute | Critiche | Alte | Medie | Basse | Test eseguiti | Coverage |
|---|---|---|---|---|---|---|---|
| **LazyTools** | A- | 0 | 0 | 3 | 8 | 362 passed | 90% |
| **LazyFin** | A- | 0 | 2 | 7 | 12 | 204 passed | 96% |
| **LazyPulse** | B+/A- | 0 | 1 | 6 | 8 | 269 passed | 93% |
| **LazyCrawler** | B+ (pre-1.0) | 0 | 2 | 6 | 10 | 221 passed | 73% |
| **LazyBridge** | B+ | 0 | 5 | 12 | 21 | 1937 passed | 80% |
| **lazybridgewebsite** | B+ (manutenz. C) | 0 | 2 | 3 | 5 | build strict OK | n/a |
| **market-data-hub** | B | 0 | 3 | 6 | 10 | 105 passed | n/d |
| **LazyHMM** | B- | **2** | 5 | 8 | 9 | 245 passed | 65% |

Voti di dettaglio (correttezza / sicurezza / test / docs / manutenibilità) nel §1 di ogni
`ASSESSMENT_2026-07.md`.

## 3. Le issue più gravi dell'ecosistema (da risolvere per prime)

### CRITICHE (2 — entrambe LazyHMM, rotture reali con le dipendenze correnti)
1. **LazyHMM** — `load_ticker` rotto: `show_errors` rimosso da yfinance → TypeError garantito (`db.py:945`). Fix da una riga.
2. **LazyHMM** — round-trip SQLite perde il tipo di modello: scrive chiave `model`, rilegge `mode` → metadati persi dopo ogni restart (`db.py:360` vs `tools.py:1252`). Fix da una riga.

### ALTE più impattanti (selezione cross-repo)
3. **lazybridgewebsite** — supply chain nella pipeline di deploy: `requirements.txt` non pinnato installa una `mkdocs-redirects` **ripubblicata da terzi** ("ProperDocs") che si esegue nella build con token `contents: write`. Oggi innocua, rischio di esecuzione arbitraria a ogni release futura.
4. **LazyBridge** — Web UI human-in-the-loop **senza autenticazione**: qualunque processo locale può forgiare approvazioni (`ext/hil/human.py:505`) + DoS memoria via Content-Length illimitato (`:462`).
5. **LazyPulse** — `asyncio.Lock` condiviso tra due event loop nel `WebhookAdapter`: sotto contesa **congela l'intero tick loop** (hang riprodotto, >120 s) (`adapters/webhook.py:77,83,145`).
6. **market-data-hub** — HTTP 429/5xx di Yahoo esauriscono i retry e finiscono loggati come "empty": **un outage del provider è invisibile** (`sources/yahoo_direct.py:99-118`); aggravato da `run_daily.py` che ritorna exit 0 anche a download completamente fallito.
7. **LazyCrawler** — fallback PDF via `urllib` **bypassa le protezioni SSRF**, i retry e il proxy (`pdf.py:118-142`).
8. **LazyBridge** — resume da `plan_state` completato riesegue l'intero Plan da step 0 (`engines/plan/_plan.py:367-375`); `BaseException` dei tool trasformate in output "riuscito" (`engines/llm.py:941-947`).
9. **LazyFin** — workflow rebalance senza freshness gate, senza ADV e senza prezzi nel record: il limite di liquidità non può mai bocciare (`workflows/rebalance.py:129-146`).
10. **market-data-hub** — `custom.store_series/delete_series` scrivono **senza writer lock**: il publish da LazyFin concorrente al run schedulato causa IO error DuckDB (`custom.py:106,115`).

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
| 1.2 | LazyCrawler | Migrare fallback PDF sul client HTTP condiviso (guardia SSRF) + allowlist schemi http/https |
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
