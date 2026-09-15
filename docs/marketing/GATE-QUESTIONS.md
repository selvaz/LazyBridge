# Domande di gate per l'operatore — presenza pubblica di LazyBridge

> Stato: **APERTO — in attesa di risposta.** Nessuna azione pubblica o irreversibile
> (post reale, creazione di account, uso di credenziali reali su una piattaforma
> social) viene eseguita finche' entrambe le domande sotto non hanno una risposta
> esplicita dell'operatore. Questo vale a prescindere da quanto materiale di
> preparazione (ricerca, positioning, scaffolding tecnico dry-run) sia gia' pronto
> nel repository: essere pronti a pubblicare non e' un'autorizzazione a farlo.

Questo documento esiste per rendere le due domande di gate visibili e tracciabili
nel tempo, non per essere letto una volta sola. Va aggiornato (sezione
"Log risposte" in fondo) quando arriva una risposta, senza cancellare la
domanda originale.

## Domanda 1 — Chi possiede le credenziali del canale scelto?

Prima di qualunque annuncio pubblico (Show HN, X/Twitter, LinkedIn, Reddit, o
altro canale), serve sapere:

- Quale canale/i si usa per primo (la ricerca in
  `docs/marketing/launch-patterns-research.md` e la proposta in
  `docs/marketing/positioning.md` indicano Show HN come primo candidato, sulla
  base dei precedenti osservati — ma resta una proposta, non una decisione).
- Chi e' il titolare dell'account/delle credenziali su quel canale: un account
  esistente dell'operatore, un account nuovo da creare a nome di chi, o un
  account del progetto/organizzazione da creare.
- Chi detiene materialmente le credenziali una volta create (password,
  eventuali API key/app OAuth per un bot di pubblicazione) e come vengono
  conservate — nessuna credenziale reale esiste oggi in questo ambiente e
  nessuna verra' richiesta o generata da questo agente prima di questa
  risposta.

**Perche' serve prima di procedere:** senza sapere chi possiede l'account non e'
possibile nemmeno proporre una finestra di pubblicazione realistica (serve la
disponibilita' di chi risponde ai commenti nelle prime ore, come nota la
ricerca), ne' impostare in sicurezza un eventuale bot di pubblicazione.

## Domanda 2 — Supervisione: ogni contenuto va approvato prima di uscire, o fiducia piena dopo un collaudo?

Due modelli alternativi, da scegliere esplicitamente (anche una via di mezzo
va bene, ma va dichiarata):

- **Approvazione preventiva:** ogni contenuto (post, risposta, thread) viene
  mostrato all'operatore e pubblicato solo dopo un "ok" esplicito, per un
  periodo indefinito o fino a nuova decisione.
- **Fiducia dopo collaudo:** un primo periodo/numero di pubblicazioni con
  approvazione preventiva, dopo il quale — se l'esito e' soddisfacente — il
  bot/agente pubblica autonomamente entro regole concordate (es. solo risposte
  tecniche, mai contenuti promozionali non richiesti, mai sollecitare voti).

**Perche' serve prima di procedere:** determina se il bot di annuncio (oggi
scaffolding mock/dry-run in `tools/marketing/`, inattivo) puo' mai passare a
una modalita' "live" automatica, o se dovra' sempre presentare una bozza e
aspettare un click umano. Cambia anche il disegno tecnico (coda di
approvazione vs pubblicazione diretta), quindi la risposta va data prima di
costruire quella parte, non dopo.

## Cosa resta bloccato finche' non arriva risposta

- Qualsiasi pubblicazione reale su qualunque piattaforma.
- Creazione di un account reale su qualunque piattaforma.
- Uso di credenziali reali di qualunque tipo.
- Passaggio del bot di annuncio (`tools/marketing/`) da modalita' mock/dry-run
  a modalita' live, in qualunque forma.

## Cosa resta invece disponibile senza attendere risposta

- Ricerca, positioning, audit del repository, scaffolding tecnico dry-run:
  tutto cio' che non richiede credenziali reali o un'azione pubblica.
- Manutenzione di questo stesso documento.

## Log risposte

| Data | Domanda | Risposta | Da chi |
|------|---------|----------|--------|
| — | — | Nessuna risposta ricevuta finora | — |
