"""Citation enrichment + CSL-JSON I/O.

Two enrichment paths, in priority order:

1. **Crossref via habanero** — best when we have a DOI (or a DOI-shaped URL).
   Returns a full bibliographic record we mint into CSL-JSON.
2. **OpenAlex via httpx** — fallback for arbitrary URLs and for non-DOI works
   (preprints, blog posts that OpenAlex has indexed, government reports, …).

Both responses are normalised to CSL-JSON which is exactly the schema that
Pandoc citeproc consumes — so the Quarto exporter can write the result to
``refs.json`` and have the citations resolve out of the box.

Caching: every successful lookup is stored under
``__citations_cache__:{sha1(query)}`` in the same Store the FragmentBus uses
(or a private in-memory Store when none is supplied).  DOIs and full URLs
are immutable enough that aggressive caching is fine — re-running a pipeline
shouldn't re-hit Crossref every time.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from lazybridge.external_tools.report_builder.fragments import Citation
from lazybridge.store import Store

_CACHE_KEY_PREFIX = "__citations_cache__"
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)


def _cache_key(query: str) -> str:
    h = hashlib.sha1(query.strip().lower().encode("utf-8")).hexdigest()
    return f"{_CACHE_KEY_PREFIX}:{h}"


def _extract_doi(url_or_text: str) -> str | None:
    m = _DOI_RE.search(url_or_text)
    return m.group(0) if m else None


def _safe_key(title: str, year: int | None) -> str:
    """Build a BibTeX-shaped citation key from a title + year.

    Lower-case the first alphanumeric word of the title, strip non-
    alphanumerics, append the year.  Walks past leading punctuation-only
    tokens so titles like "@@@ Wow!" yield ``wow`` not ``ref``.
    """
    cleaned = "ref"
    for word in re.split(r"\s+", title.strip()):
        word_clean = re.sub(r"[^a-z0-9]", "", word.lower())[:24]
        if word_clean:
            cleaned = word_clean
            break
    return f"{cleaned}{year}" if year else cleaned


def _crossref_lookup(doi: str) -> dict[str, Any] | None:
    """Resolve a DOI via habanero.  Returns the message body or None on failure."""
    try:
        from lazybridge.external_tools.report_builder._deps import require_habanero

        Crossref = require_habanero()
    except ImportError:
        return None
    try:
        cr = Crossref(mailto="lazybridge-citations@users.noreply.github.com")
        result = cr.works(ids=doi)
        return result.get("message") if isinstance(result, dict) else None
    except Exception:
        return None


def _openalex_lookup(query: str) -> dict[str, Any] | None:
    """Resolve a URL or DOI via OpenAlex.  Free, no auth required."""
    try:
        from lazybridge.external_tools.report_builder._deps import require_httpx

        httpx = require_httpx()
    except ImportError:
        return None
    # OpenAlex accepts a DOI directly via /works/doi:10.xxx and a URL as a
    # filter on its search index.  We try the DOI path first, falling back
    # to a search.
    doi = _extract_doi(query)
    try:
        with httpx.Client(timeout=10.0) as client:
            if doi:
                r = client.get(f"https://api.openalex.org/works/doi:{doi}")
                if r.status_code == 200:
                    return r.json()
            r = client.get(
                "https://api.openalex.org/works",
                params={"search": query, "per-page": 1},
            )
            if r.status_code == 200:
                payload = r.json()
                results = payload.get("results", [])
                return results[0] if results else None
    except Exception:
        return None
    return None


def _crossref_to_csl(message: dict[str, Any]) -> dict[str, Any]:
    """Translate a Crossref ``message`` into a CSL-JSON record."""
    title_list = message.get("title") or []
    title = title_list[0] if title_list else "Untitled"
    issued = message.get("issued", {}).get("date-parts", [[None]])
    year = issued[0][0] if issued and issued[0] else None
    authors = []
    for a in message.get("author", []):
        name = " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
        if name:
            authors.append({"literal": name})
    return {
        "id": _safe_key(title, year),
        "type": message.get("type", "article-journal"),
        "title": title,
        "author": authors,
        "issued": {"date-parts": [[year]]} if year else None,
        "DOI": message.get("DOI"),
        "URL": message.get("URL"),
        "container-title": (message.get("container-title") or [None])[0],
    }


def _openalex_to_csl(work: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAlex ``work`` into a CSL-JSON record."""
    title = work.get("title") or work.get("display_name") or "Untitled"
    year = work.get("publication_year")
    authors = []
    for ship in work.get("authorships", []):
        author = ship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append({"literal": name})
    doi_url = work.get("doi") or ""
    doi = _extract_doi(doi_url) if doi_url else None
    return {
        "id": _safe_key(title, year),
        "type": work.get("type") or "article-journal",
        "title": title,
        "author": authors,
        "issued": {"date-parts": [[year]]} if year else None,
        "DOI": doi,
        "URL": work.get("doi") or work.get("id"),
        "container-title": (work.get("primary_location") or {}).get("source", {}).get("display_name"),
    }


def _from_csl(csl: dict[str, Any], fallback_url: str | None) -> Citation:
    title = csl.get("title", "Untitled")
    issued = (csl.get("issued") or {}).get("date-parts") or [[None]]
    year = issued[0][0] if issued and issued[0] else None
    authors = [
        a.get("literal") or " ".join(filter(None, [a.get("given"), a.get("family")])) for a in csl.get("author", [])
    ]
    authors = [a for a in authors if a]
    return Citation(
        key=csl.get("id") or _safe_key(title, year),
        title=title,
        url=csl.get("URL") or fallback_url,
        authors=authors,
        year=year if isinstance(year, int) else None,
        doi=csl.get("DOI"),
        accessed=datetime.now(UTC),
        csl=csl,
    )


def enrich_from_url(url: str, *, store: Store | None = None) -> Citation:
    """Resolve ``url`` (or a DOI hidden in it) into a populated :class:`Citation`.

    Tries Crossref first when a DOI is present, then OpenAlex.  If both miss,
    returns a minimal :class:`Citation` with just ``title=url`` so the call
    site never has to handle ``None`` — the LLM can attach it to a fragment
    and Pandoc citeproc will still produce a valid (if thin) bibliography
    entry.
    """
    cache = store if store is not None else Store(db=None)
    cached = cache.read(_cache_key(url))
    if cached:
        return Citation.model_validate(cached)

    csl: dict[str, Any] | None = None
    doi = _extract_doi(url)
    if doi:
        message = _crossref_lookup(doi)
        if message:
            csl = _crossref_to_csl(message)
    if csl is None:
        work = _openalex_lookup(url)
        if work:
            csl = _openalex_to_csl(work)

    if csl is None:
        # Best-effort: build a minimal citation so the caller still gets
        # something usable.  Pandoc will render this as "Untitled, n.d."
        # which is honest about the missing metadata.
        cit = Citation(
            key=_safe_key(url, None),
            title=url,
            url=url,
            accessed=datetime.now(UTC),
        )
    else:
        cit = _from_csl(csl, fallback_url=url)

    cache.write(_cache_key(url), cit.model_dump(mode="json"))
    return cit


def to_csl_json(citations: list[Citation]) -> list[dict[str, Any]]:
    """Render a list of :class:`Citation` to a CSL-JSON array.

    The output is suitable for writing to ``refs.json`` and pointing
    Quarto / Pandoc at via ``bibliography:`` in ``_quarto.yml``.
    """
    return [c.to_csl_json() for c in citations]
