# Domande di gate per l'operatore — presenza pubblica di LazyBridge

> Stato: **APERTO — in attesa di risposta.** Nessuna azione pubblica o uso di
> credenziali reali viene eseguito finché entrambe le domande sotto non ricevono
> una risposta esplicita dell'operatore. Il materiale preparatorio già presente
> nel repository non costituisce autorizzazione a pubblicare.

Questo documento mantiene visibili e tracciabili le due decisioni che spettano
all'operatore. Quando arrivano le risposte, vanno registrate nel "Log risposte"
senza cancellare o riscrivere le domande originali.

## Domanda 1 — Chi possiede e controlla le credenziali del canale scelto?

Per ciascun canale che si intende usare (Show HN, X/Twitter, LinkedIn, Reddit o
altro), l'operatore deve indicare:

- il canale e l'account esatto: account personale esistente, account di
  un'organizzazione/progetto esistente, oppure nuovo account da creare e a nome
  di chi;
- il titolare dell'account, chi ne è amministratore e chi è autorizzato a
  pubblicare e a rispondere ai commenti; queste persone possono non coincidere;
- chi custodisce password, secondo fattore, codici di recupero ed eventuali
  API key o autorizzazioni OAuth, e con quale sistema approvato; nel log non
  vanno mai inseriti segreti;
- chi può concedere e revocare l'accesso e chi interviene se l'account viene
  bloccato, compromesso o richiede una verifica dell'identità.

La scelta va fatta nel rispetto del modello del canale: Show HN e altri canali
possono richiedere un account personale e attività umana; LinkedIn può separare
proprietario della pagina e amministratori; API, automazione o account di
progetto possono non essere disponibili o consentiti. Il possesso di un account
non autorizza automaticamente l'automazione.

La ricerca in `docs/marketing/launch-patterns-research.md` e la proposta in
`docs/marketing/positioning.md` indicano Show HN come primo candidato sulla base
dei precedenti osservati, ma non decidono né il canale né l'account.

**Perché è un gate:** senza titolare, accessi e responsabilità definiti non è
possibile usare credenziali in sicurezza, stabilire chi pubblica o pianificare
una finestra in cui una persona autorizzata possa seguire i commenti.

## Domanda 2 — Quale livello di approvazione serve prima di pubblicare contenuti?

L'operatore deve scegliere uno dei seguenti modelli, oppure descriverne una
combinazione con confini altrettanto espliciti:

- **Approvazione preventiva per tutto:** ogni post, risposta, modifica,
  correzione o rimozione viene mostrato all'operatore e richiede un "ok"
  esplicito prima dell'azione.
- **Autonomia limitata per categoria:** alcune azioni definite in anticipo
  richiedono sempre approvazione (per esempio annunci, claim, confronti con
  concorrenti o risposte controverse), mentre altre possono essere eseguite
  senza approvazione caso per caso entro regole concordate (per esempio
  risposte tecniche fattuali). L'operatore deve elencare categorie, canali e
  limiti autorizzati.
- **Fiducia dopo un collaudo:** si applica l'approvazione preventiva per un
  periodo o numero di pubblicazioni definito; al termine, l'autonomia scatta
  solo se l'operatore conferma che i criteri di esito concordati sono
  soddisfatti.

La risposta deve inoltre specificare chi approva, su quale canale arriva
l'approvazione, la durata o il numero di pubblicazioni del collaudo, i criteri
per superarlo, le azioni che restano sempre soggette ad approvazione e come
sospendere o revocare rapidamente l'autonomia. L'assenza di risposta o il
silenzio dopo una bozza non valgono come approvazione.

**Perché è un gate:** la scelta determina se l'eventuale tooling di annuncio
(oggi solo scaffolding mock/dry-run in `tools/marketing/`, inattivo) debba
fermarsi a una bozza, usare una coda di approvazione o possa compiere alcune
azioni live entro limiti verificabili. Non si assume che ogni canale consenta
pubblicazione automatica: le sue regole e capacità tecniche restano vincolanti.

## Cosa resta bloccato finché entrambe le domande non ricevono risposta

- Qualsiasi pubblicazione, risposta, modifica o rimozione reale su una
  piattaforma esterna per conto di LazyBridge.
- Creazione o modifica di account reali e assegnazione di ruoli o accessi.
- Richiesta, inserimento, memorizzazione o uso di password, token, API key,
  autorizzazioni OAuth, secondo fattore o codici di recupero reali.
- Passaggio del tooling in `tools/marketing/` da mock/dry-run a qualunque
  modalità live, inclusi test su account reali.

## Cosa rimane disponibile nel frattempo

- Ricerca, positioning, audit e pianificazione interna che non producano azioni
  pubbliche.
- Bozze chiaramente marcate come non approvate, purché non vengano inviate né
  caricate su piattaforme esterne.
- Scaffolding e test locali in mock/dry-run con credenziali e account fittizi.
- Manutenzione di questo documento e registrazione delle future risposte.

## Log risposte

| Data | Domanda | Risposta | Da chi |
|------|---------|----------|--------|
