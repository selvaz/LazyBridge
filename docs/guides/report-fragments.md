# Fragment schema

Every contribution to a parallel report is a `Fragment` — a typed,
JSON-round-trippable record carrying one of four kinds (text, chart,
table, callout) plus optional citations and a per-fragment audit stamp.

## Fragment

```python
from lazybridge.ext.report_builder import Fragment

Fragment(
    kind: Literal["text", "chart", "table", "callout"],
    section: str | None = None,        # routing key (alphabetical / dotted-path)
    heading: str | None = None,         # rendered as h3 above the fragment body
    order_hint: float = 0.0,            # within-section sort key (ascending)

    body_md: str | None = None,         # text + callout payload (Markdown)
    chart: ChartSpec | None = None,     # chart payload
    table: TableSpec | None = None,     # table payload
    callout_style: Literal["note","tip","important","warning","caution"] | None = None,

    citations: list[Citation] = [],     # bibliographic sources
    provenance: Provenance | None = None,  # who/what produced this

    id: str = uuid4().hex,              # auto-assigned; idempotency key
    created_at: datetime = utcnow(),    # auto-stamped
)
```

### `kind` and the discriminator rule

The `kind` field tells the assembler and exporter which payload field is
authoritative.  The pydantic validator enforces:

| `kind`        | Required field    | Optional fields            |
|---------------|-------------------|----------------------------|
| `"text"`      | `body_md`          | `heading`, `citations`     |
| `"callout"`   | `body_md`          | `callout_style` (default `"note"`), `heading` |
| `"chart"`     | `chart`            | `heading`, `citations`     |
| `"table"`     | `table`            | `heading`                  |

Other fields are tolerated but ignored.  Constructing `Fragment(kind="text")`
with no `body_md` raises `ValueError`.

### `section` — the routing key

`section` is the **only piece of routing the bus understands** — the
assembler decides what to do with it.

- For `BlackboardAssembler`: any string. Fragments group alphabetically
  by section; `None` lands in the unsectioned bucket at the top.
- For `OutlineAssembler`: a dotted path that matches the outline. `"1.2.intro"`
  becomes a leaf at depth 3; `"1"` becomes a parent of `"1.x"` siblings;
  unknown ids fall into a synthetic `__unrouted__` bucket appended at the
  end so nothing is silently dropped.

### `order_hint` and `created_at`

Within a section, fragments sort first by `order_hint` (ascending), then
by `created_at` for ties.  Use `order_hint` when an agent generates
fragments in a different order than they should appear (e.g., an LLM
writes the conclusion before the introduction).

### `id` and idempotency

`id` defaults to `uuid4().hex` and is the bus's idempotency key.
Re-appending the same id silently no-ops, which is what makes
`Plan(resume=True)` safe — replayed Steps won't double-emit.

If you want stable ids (e.g., to overwrite or tombstone a fragment),
supply your own.  `id` must be unique within the bus.

## Citation

```python
Citation(
    key: str,                           # BibTeX-shaped, e.g. "smith2024"
    title: str,
    url: str | None = None,
    authors: list[str] = [],
    year: int | None = None,
    doi: str | None = None,
    accessed: datetime | None = None,
    csl: dict | None = None,            # full CSL-JSON record (when enriched)
)
```

Citations attach to fragments via the `citations=[...]` field.  Inline
references inside `body_md` use Pandoc syntax: `[@smith2024]`, `[@smith2024, p. 5]`,
`[see @smith2024]`.

The Quarto exporter writes a CSL-JSON `refs.json` alongside the report
and Pandoc citeproc resolves both the inline markers and the Sources
section.  The WeasyPrint fallback renders citations as a plain list under
a `Sources` h2.

`csl` is filled automatically by `enrich_from_url()` (Crossref / OpenAlex
lookups). Hand-filled citations don't need `csl` — `to_csl_json()`
synthesises a minimal record from the structured fields.

→ See [Citations](report-citations.md) for the enrichment pipeline.

### Example

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

## Provenance

```python
Provenance(
    step_name: str | None = None,       # Plan Step that produced the fragment
    agent_name: str | None = None,      # Agent within that Step
    model: str | None = None,           # underlying model id
    provider: str | None = None,        # "anthropic" | "openai" | …
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_usd: float | None = None,
    latency_ms: float | None = None,
    timestamp: datetime = utcnow(),
)
```

Field names mirror `lazybridge.envelope.EnvelopeMetadata` so a tool-side
helper can copy the metadata over directly.  In practice, you populate
provenance two ways:

1. **Through `fragment_tools(bus, step_name=…)`** — every tool call
   stamps `step_name` automatically.  Other fields stay `None` because
   the tool callback doesn't have access to the running Envelope's
   metadata.
2. **Manually in Python-side appends** — set whichever fields you have:

   ```python
   bus.append(Fragment(
       kind="text",
       body_md="…",
       provenance=Provenance(
           step_name="research",
           agent_name="haiku",
           model="claude-haiku-4-5",
           tokens_in=env.metadata.input_tokens,
           tokens_out=env.metadata.output_tokens,
           cost_usd=env.metadata.cost_usd,
           latency_ms=env.metadata.latency_ms,
       ),
   ))
   ```

The assembler aggregates every fragment's provenance into
`AssembledReport.provenance_log`, which the exporter renders as a
per-fragment audit table at the end of the report.  This is what makes
multi-agent reports inspectable — you see which model said what, what it
cost, and how long it took.

## ChartSpec

```python
ChartSpec(
    engine: Literal["vega-lite", "plotly"] = "vega-lite",
    spec: dict,                         # raw engine-native JSON spec
    data: list[dict] | None = None,     # optional inline rows
    title: str = "",
)
```

`spec` is the engine's published JSON schema — Vega-Lite v5 or a Plotly
figure dict.  We never `exec` Python code for charts.  `data` is a
convenience: when present, it splices into `spec.data.values` (Vega-Lite)
or into the first trace's x/y arrays (Plotly).

→ See [Chart contract](report-charts.md) for full examples.

## TableSpec

```python
TableSpec(
    headers: list[str],
    rows: list[list[str]],
    caption: str = "",
)
```

Rows are pre-formatted strings — number formatting, currencies, units
all happen in the agent that produces the row, not at render time.
Pandoc emits this as a pipe table; the WeasyPrint fallback renders it as
HTML.  No sorting, filtering, or conditional formatting on the render
side — keep that logic in the agent or upstream tool.

## JSON shape

Every model is pydantic v2; round-tripping through `model_dump(mode="json")`
is lossless.  The bus stores fragments in this serialised form, so
`Store` (SQLite or in-memory) doesn't need any custom serialisation.

```python
f = Fragment(kind="text", body_md="x", citations=[Citation(key="k", title="T")])
d = f.model_dump(mode="json")        # plain dict
f2 = Fragment.model_validate(d)      # back to a Fragment
assert f2 == f
```

## See also

- [FragmentBus](fragment-bus.md)
- [Assemblers](report-assemblers.md)
- [Citations](report-citations.md)
