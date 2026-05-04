# Citations

The fragment-based reporting pipeline carries citations as first-class
data: every `Fragment` has a `citations: list[Citation]` field, agents
attach them as they write, and the renderer auto-resolves inline
references plus the bibliography.

The pipeline does **not** reimplement CSL — Pandoc citeproc (run via
Quarto) handles the actual formatting against 1000+ CSL styles.  We
just produce the Pandoc-shaped inputs.

## Inline reference syntax

Inside `body_md`, use Pandoc citation syntax:

```markdown
The market grew 18% YoY [@smith2024].

Several authors have argued the opposite [@brown2023; @lee2025, p. 12].

[@smith2024] reports the figure differently.

See @smith2024 for details.
```

Quarto + citeproc resolves these into formatted footnotes (default APA)
and a Sources section under whatever heading the CSL style names —
typically `## References` or `## Sources`.

## Building citations by hand

```python
from lazybridge.ext.report_builder import Citation

cite = Citation(
    key="smith2024",
    title="A study of widget adoption",
    authors=["Smith, J.", "Brown, K."],
    year=2024,
    doi="10.1234/widgets.2024.003",
    url="https://doi.org/10.1234/widgets.2024.003",
)
```

Required fields: just `key` and `title`.  Everything else is optional.
The `key` becomes the `[@key]` token agents use in prose.

Attach citations to a fragment:

```python
from lazybridge.ext.report_builder import Fragment

bus.append(Fragment(
    kind="text",
    body_md="Adoption climbed sharply [@smith2024].",
    citations=[cite],
))
```

The same citation can appear on multiple fragments — the assembler
de-duplicates by `key` (first occurrence wins).

## Auto-enrichment from URL / DOI

`enrich_from_url(url)` resolves a URL or DOI through Crossref (via
`habanero`) with an OpenAlex fallback (via `httpx`).  It returns a fully
populated `Citation` whose `csl` field carries the canonical CSL-JSON
record.

```python
from lazybridge.ext.report_builder.citations import enrich_from_url

cite = enrich_from_url("https://doi.org/10.1234/widgets.2024.003")
print(cite.title, cite.authors, cite.year)
```

The lookup pipeline:

1. **DOI extraction** — search the URL for a `10.xxxx/...` shape.
2. **Crossref** (if DOI found) — `habanero.Crossref().works(ids=doi)`
   returns a rich record we map to CSL-JSON.
3. **OpenAlex** (fallback) — `https://api.openalex.org/works/doi:{doi}`
   for DOIs, or a search query for arbitrary URLs / titles.
4. **Minimal** (last resort) — when both APIs miss, we synthesise a
   thin `Citation(title=url, url=url)` so the call site never has to
   handle `None`.  Pandoc renders this as "URL, n.d." which is honest
   about the missing metadata.

### Caching

Every successful lookup is cached in the same Store the bus uses, under
`__citations_cache__:{sha1(url_or_query)}`.  DOIs and full URLs are
immutable enough that aggressive caching is fine — re-running a
pipeline doesn't re-hit Crossref.

```python
from lazybridge.store import Store

store = Store(db="./run.sqlite")
cite = enrich_from_url("https://doi.org/...", store=store)   # first call: Crossref hit
cite = enrich_from_url("https://doi.org/...", store=store)   # second call: cache hit, ~0ms
```

### LLM-callable tool

`fragment_tools(bus)` includes `cite_url(url)` which wraps
`enrich_from_url` and returns a citation dict the agent can attach to
subsequent `append_text` / `append_chart` calls:

```python
# Inside an agent system prompt:
"""
For every claim, call cite_url(url) first to resolve it,
then pass the returned dict to append_text(citations=[...]).
"""
```

The citation dict is the JSON form of `Citation` — pass it through
verbatim.

## CSL-JSON output

```python
from lazybridge.ext.report_builder.citations import to_csl_json
from lazybridge.ext.report_builder import Citation

cites = [
    Citation(key="k1", title="T1", year=2024, authors=["A"]),
    Citation(key="k2", title="T2", url="https://e.com"),
]
csl = to_csl_json(cites)
# [
#   {"id": "k1", "type": "article", "title": "T1",
#    "author": [{"literal": "A"}], "issued": {"date-parts": [[2024]]}},
#   {"id": "k2", "type": "webpage", "title": "T2", "URL": "https://e.com"},
# ]
```

The Quarto exporter writes this list to `refs.json` in the output
directory and references it from `_quarto.yml`:

```yaml
bibliography: refs.json
```

Pandoc citeproc takes over from there: every `[@key]` becomes a
formatted reference and a Sources section appears at the end.

## CSL styles

Quarto defaults to APA.  To use a different style, pass a CSL filename
to the exporter via `_quarto.yml` (the project file the Quarto exporter
generates auto-includes `csl:` when `bus.export(..., csl_style="...")`
is supported in your Quarto install — currently you can edit the
generated `_quarto.yml` after `bus.export(...)` writes it, then re-render).

Common styles:

| Style                                  | File                       |
|----------------------------------------|----------------------------|
| APA 7                                  | `apa.csl`                  |
| Chicago Author-Date                    | `chicago-author-date.csl`  |
| Chicago Notes & Bibliography           | `chicago-note-bibliography.csl` |
| IEEE                                   | `ieee.csl`                 |
| Vancouver                              | `vancouver.csl`            |
| Nature                                 | `nature.csl`               |

Download styles from <https://www.zotero.org/styles> and drop them next
to `_quarto.yml`.

## Fallback rendering

The WeasyPrint exporter doesn't run citeproc.  It renders citations as
a plain Sources list at the end of the report — formatted as
`Title — Authors — Year — DOI/URL`.  Inline `[@key]` markers stay in the
prose verbatim.

This is honest second-best — accept it as the price of dropping the
Quarto / Pandoc dep, or set `backend="quarto"` and install the CLI for
real citations.

## Compliance and audit considerations

* Every `Citation` carries `accessed: datetime` (UTC) when enriched from
  a URL — useful for documenting "as-of" dates in regulatory contexts.
* The CSL-JSON record is preserved verbatim under `Citation.csl` so a
  later legal review can verify the metadata against the original
  source.
* Cached records survive process restarts.  In a long-running compliance
  pipeline this means citations don't drift even if Crossref updates the
  underlying record.

## See also

- [FragmentBus](fragment-bus.md)
- [Fragment schema](report-fragments.md) — Citation fields
- [Exporters](report-exporters.md) — how citations render per backend
- [CSL specification](https://citationstyles.org/)
- [Crossref REST API](https://api.crossref.org/swagger-ui/index.html)
- [OpenAlex API](https://docs.openalex.org/)
