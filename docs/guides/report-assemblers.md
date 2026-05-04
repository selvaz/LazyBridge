# Assemblers

An `Assembler` reduces the flat fragment list collected in a
`FragmentBus` into a structured `AssembledReport` that an `Exporter` can
render.  Two strategies ship; you pick one when constructing the bus.

```python
from lazybridge.external_tools.report_builder import (
    BlackboardAssembler,
    OutlineAssembler,
    FragmentBus,
)

bus = FragmentBus("rep", assembler=OutlineAssembler({"1.intro": "Introduction"}))
```

The assembler runs late — at `bus.assemble()` / `bus.export()` time — so
you can swap it for a different one to render the same fragment set
differently:

```python
report_a = BlackboardAssembler().assemble(bus.fragments(), title="Free-form")
report_b = OutlineAssembler({"…": "…"}).assemble(bus.fragments(), title="Structured")
```

## BlackboardAssembler

**Use when** you don't know the report shape up front: news digests,
audit summaries, scratch reports where the structure emerges from what
agents write.

### Behavior

* Groups fragments by `section` string.
* Sections sort alphabetically; fragments with `section=None` group at
  the **top** in a single unsectioned bucket.
* Within a section, fragments sort by `order_hint` ascending, then by
  `created_at` for ties.
* Citations across all fragments are deduped by `Citation.key`
  (first-occurrence wins).

### Example

```python
from lazybridge.external_tools.report_builder import (
    BlackboardAssembler, FragmentBus, Fragment,
)

bus = FragmentBus("news")
bus.append(Fragment(kind="text", body_md="US stock movement", section="us"))
bus.append(Fragment(kind="text", body_md="China EV exports",  section="cn"))
bus.append(Fragment(kind="text", body_md="India services",    section="in"))
bus.append(Fragment(kind="text", body_md="Top-line takeaway"))   # no section

report = BlackboardAssembler().assemble(bus.fragments(), title="Daily")
# Sections in render order:
#   1. None      ("Top-line takeaway")
#   2. cn        (China EV exports)
#   3. in        (India services)
#   4. us        (US stock movement)
```

The "no section" fragments at the top are useful for an agent-written
intro paragraph that should appear before any region-specific content.

### When alphabetical isn't what you want

If "us / cn / in" should render in that order rather than alphabetically,
either:

* **Use string prefixes for ordering**: `"1.us"`, `"2.cn"`, `"3.in"`.
  Blackboard treats these as opaque strings and sorts alphabetically,
  so prefixes give you the order.  This is exactly what `OutlineAssembler`
  does, but without the tree shape.
* **Switch to `OutlineAssembler`**: explicitly declare the section order
  upfront.

## OutlineAssembler

**Use when** the report has a known shape: research reports, financial
briefings, regulatory filings.  Agents tag fragments with dotted-path
section ids; the assembler builds a tree.

### Behavior

* Construct with `outline = {section_id: heading, …}` in declaration
  order — Python ≥3.7 preserves dict order, which becomes the rendered
  order at each tree level.
* Fragments whose `section` matches an outline id land in that node.
* Fragments with no section (or with an unknown section) land in a
  synthetic `__unrouted__` bucket appended at the end — content is never
  silently dropped.
* Dotted-path nesting builds the tree: `"1"` parents `"1.a"`; `"1.a"`
  parents `"1.a.intro"`.  Heading levels (h2/h3/h4/…) follow depth.
* Outline nodes that received no fragments still emit a heading — the
  document structure survives partial writes.

### Example

```python
from lazybridge.external_tools.report_builder import (
    OutlineAssembler, FragmentBus, Fragment,
)

OUTLINE = {
    "1.exec":      "Executive Summary",
    "2.findings":  "Findings",
    "2.findings.market":  "Market context",
    "2.findings.risk":    "Risk profile",
    "3.outlook":   "Outlook",
}

bus = FragmentBus("research")
bus.append(Fragment(kind="text", body_md="…", section="1.exec"))
bus.append(Fragment(kind="text", body_md="…", section="2.findings.market"))
bus.append(Fragment(kind="text", body_md="…", section="2.findings.risk"))
bus.append(Fragment(kind="text", body_md="…", section="3.outlook"))

report = OutlineAssembler(OUTLINE).assemble(bus.fragments(), title="Q1 Brief")

# Rendered structure:
#   ## Executive Summary
#   ## Findings
#       ### Market context
#       ### Risk profile
#   ## Outlook
```

The `2.findings` heading still renders even though no fragment was
attached to it directly — the children inherit it.

### Unknown sections

```python
bus.append(Fragment(kind="text", body_md="stray", section="9.elsewhere"))
report = OutlineAssembler(OUTLINE).assemble(bus.fragments(), title="…")
# Adds a synthetic top-level "Other" section at the end with the stray fragment.
```

`AssembledReport.metadata["unrouted_fragments"]` exposes the count so
you can detect and surface this in CI:

```python
assert report.metadata["unrouted_fragments"] == 0, "agents drifted off-outline"
```

## AssembledReport

The shared output type produced by every assembler.  Exporters consume
it without caring which assembler ran.

```python
class AssembledReport(BaseModel):
    title: str
    sections: list[RenderedSection]   # tree (Outline) or flat (Blackboard)
    citations: list[Citation]         # deduped union from all fragments
    provenance_log: list[Provenance]  # append-only audit trail
    metadata: dict                    # generated_at, fragment_count, totals, …
```

### `RenderedSection`

```python
class RenderedSection(BaseModel):
    section_id: str | None       # the original Fragment.section value
    heading: str | None          # rendered heading text
    level: int = 2               # h2 by default; deeper for nested outline nodes
    fragments: list[Fragment]    # leaves attached directly to this section
    children: list[RenderedSection]   # nested sub-sections (Outline only)
```

### `metadata` keys

Every assembler stamps these:

| Key                | Type    | Meaning                               |
|--------------------|---------|---------------------------------------|
| `generated_at`     | str     | UTC ISO timestamp                     |
| `fragment_count`   | int     | total fragments in the report         |
| `tokens_in_total`  | int     | sum from every fragment's provenance  |
| `tokens_out_total` | int     | sum from every fragment's provenance  |
| `cost_usd_total`   | float   | sum from every fragment's provenance  |
| `assembler`        | str     | `"blackboard"` or `"outline"`         |

`OutlineAssembler` adds `outline_size` and `unrouted_fragments`.

`metadata["author"]` is filled by `bus.export(..., author="…")`.

## Choosing between them

| Question                                             | Use                  |
|------------------------------------------------------|----------------------|
| Do I know the section list before agents run?        | `OutlineAssembler`   |
| Do agents discover the structure as they go?         | `BlackboardAssembler`|
| Do I want headings in a specific (non-alphabetic) order? | `OutlineAssembler`   |
| Should missing sections render an empty heading?     | `OutlineAssembler` (Blackboard simply omits them) |
| Do I want nested h2 / h3 / h4 from a tree?           | `OutlineAssembler`   |

You can also compose: have an outlining Step write the `OutlineAssembler`
configuration into the Plan's KV store, then construct the bus from
that — the outline is itself an LLM artifact.

## Custom assemblers

`Assembler` is a `Protocol`:

```python
class Assembler(Protocol):
    def assemble(self, fragments: list[Fragment], *, title: str) -> AssembledReport: ...
```

Any callable / class satisfying that signature is acceptable.  Examples
of custom strategies you might write:

* **RankedAssembler** — sort all fragments globally by `order_hint`, ignore section.
* **ClusteringAssembler** — group fragments whose headings have high
  semantic similarity, run an LLM to draft the heading text per cluster.
* **AppendOnlyAssembler** — flatten everything into one section in
  insertion order (skip the time-sort).

Drop them into `FragmentBus(..., assembler=MyAssembler())` and the rest
of the pipeline doesn't change.

## See also

- [FragmentBus](fragment-bus.md)
- [Fragment schema](report-fragments.md)
- [Exporters](report-exporters.md)
