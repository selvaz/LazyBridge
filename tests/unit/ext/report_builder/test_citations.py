"""Tests for citation enrichment + CSL-JSON I/O."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from lazybridge.ext.report_builder import Citation
from lazybridge.ext.report_builder.citations import (
    _crossref_to_csl,
    _extract_doi,
    _openalex_to_csl,
    _safe_key,
    enrich_from_url,
    to_csl_json,
)
from lazybridge.store import Store


class TestExtractDoi:
    def test_doi_in_url(self):
        assert _extract_doi("https://doi.org/10.1234/abc.5678") == "10.1234/abc.5678"

    def test_no_doi(self):
        assert _extract_doi("https://example.com/article") is None


class TestSafeKey:
    def test_first_word_plus_year(self):
        assert _safe_key("Hello World", 2024) == "hello2024"

    def test_special_chars_stripped(self):
        assert _safe_key("@@@ Wow!", 2026).startswith("wow")

    def test_no_year(self):
        # Non-empty key even when year missing.
        assert _safe_key("title", None) == "title"


class TestCrossrefToCsl:
    def test_basic(self):
        message = {
            "title": ["A great paper"],
            "issued": {"date-parts": [[2024]]},
            "author": [{"given": "Jane", "family": "Doe"}],
            "DOI": "10.1/abc",
            "URL": "https://doi.org/10.1/abc",
            "type": "journal-article",
        }
        csl = _crossref_to_csl(message)
        assert csl["title"] == "A great paper"
        assert csl["issued"] == {"date-parts": [[2024]]}
        assert csl["author"] == [{"literal": "Jane Doe"}]
        assert csl["DOI"] == "10.1/abc"


class TestOpenalexToCsl:
    def test_basic(self):
        # Use a 4-digit DOI prefix — OpenAlex's `doi` field carries the
        # canonical doi.org URL and our regex requires the standard
        # registrant prefix length (10.4-10.9 digits).
        work = {
            "title": "OpenAlex paper",
            "publication_year": 2025,
            "authors": [],
            "authorships": [{"author": {"display_name": "Pat Hacker"}}],
            "doi": "https://doi.org/10.5555/xyz",
            "type": "article-journal",
        }
        csl = _openalex_to_csl(work)
        assert csl["title"] == "OpenAlex paper"
        assert csl["author"] == [{"literal": "Pat Hacker"}]
        assert csl["DOI"] == "10.5555/xyz"


class TestEnrichFromUrl:
    def test_falls_back_to_minimal_when_lookups_fail(self):
        # Make both lookups return None so we exercise the fallback path.
        with patch("lazybridge.ext.report_builder.citations._crossref_lookup", return_value=None), \
             patch("lazybridge.ext.report_builder.citations._openalex_lookup", return_value=None):
            store = Store(db=None)
            cit = enrich_from_url("https://example.com/abc", store=store)
            assert isinstance(cit, Citation)
            assert cit.url == "https://example.com/abc"
            assert cit.title == "https://example.com/abc"

    def test_caches_result(self):
        store = Store(db=None)
        # Use a real-shaped DOI URL — our extractor requires the standard
        # 4-9 digit registrant prefix, so '10.1234/...' is the minimum.
        url = "https://doi.org/10.1234/cached.example"
        with patch(
            "lazybridge.ext.report_builder.citations._crossref_lookup",
            return_value={
                "title": ["Cached"],
                "issued": {"date-parts": [[2024]]},
                "author": [],
                "DOI": "10.1234/cached.example",
                "URL": url,
                "type": "article",
            },
        ):
            first = enrich_from_url(url, store=store)
        # Now wipe the lookup and confirm we still get the result back from cache.
        with patch(
            "lazybridge.ext.report_builder.citations._crossref_lookup",
            return_value=None,
        ), patch(
            "lazybridge.ext.report_builder.citations._openalex_lookup",
            return_value=None,
        ):
            second = enrich_from_url(url, store=store)
        assert second.title == first.title == "Cached"


class TestToCslJson:
    def test_renders_list(self):
        cits = [
            Citation(key="k1", title="T1", year=2024),
            Citation(key="k2", title="T2", url="https://e.com"),
        ]
        out = to_csl_json(cits)
        assert len(out) == 2
        assert out[0]["id"] == "k1"
        assert out[1]["id"] == "k2"
