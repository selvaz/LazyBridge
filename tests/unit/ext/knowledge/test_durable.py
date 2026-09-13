"""DurableKnowledgeBase — durable, Store-backed lessons with keyword search.

Ported from the project this was promoted from (LazyCEO's
``lazyceo.lessons``), adapted from module-level functions taking a bare
``store`` argument to methods on a ``DurableKnowledgeBase`` instance
(mirroring :class:`~lazybridge.ext.approval.ApprovalQueue`'s own
configurable-prefix design).
"""

from __future__ import annotations

import time
import unicodedata

import pytest

from lazybridge import Store
from lazybridge.ext.knowledge import (
    DEFAULT_PREFIX,
    MAX_SLUG_LENGTH,
    STALE_AFTER_DAYS,
    DurableKnowledgeBase,
    render_lesson_line,
    slugify,
)


def test_slugify_normalizes_case_and_punctuation() -> None:
    assert slugify("Windows Worktree Paths!") == "windows-worktree-paths"
    assert slugify("  extra   spaces  ") == "extra-spaces"


def test_slugify_rejects_a_topic_with_no_alphanumeric_content() -> None:
    with pytest.raises(ValueError, match="alphanumeric"):
        slugify("   ---   ")


def test_slugify_rejects_a_topic_made_only_of_unattached_combining_marks() -> None:
    """_tokenize treats a combining mark as part of a word (needed so a
    real word like Devanagari "कि" keeps its vowel mark attached), but
    that also means a topic that is nothing BUT a bare, unattached
    combining mark -- no base letter at all -- would tokenize to a
    non-empty slug that is nonetheless invisible/unusable as a key. A
    plain `not slug` emptiness check doesn't catch this; it takes an
    explicit alphanumeric-content check. Found by Codex review before
    this ever shipped."""
    with pytest.raises(ValueError, match="alphanumeric"):
        slugify("́")  # a bare combining acute accent, no base letter


def test_slugify_accepts_non_ascii_alphanumeric_topics() -> None:
    """An earlier version's ASCII-only regex ([a-z0-9]) rejected any topic
    with no ASCII letter/digit at all -- including a topic entirely in a
    non-Latin script, even though it genuinely contains alphanumeric
    characters in the Unicode sense this function's own error message
    promises to accept. Found by Codex review before this ever shipped."""
    assert slugify("東京") == "東京"
    assert slugify("café société") == "café-société"


def test_slugify_distinguishes_topics_that_differ_only_by_a_combining_mark() -> None:
    """A `\\W`-based regex treats a combining mark as a word BOUNDARY (see
    this module's own comment above _is_word_char), so two topics
    differing only in which vowel mark is attached to the same base
    letter -- e.g. Devanagari "कि" (ka + vowel sign i) vs "कु" (ka +
    vowel sign u), a script with no precomposed form for NFKC to collapse
    the way Latin accents have -- would both reduce to the SAME bare
    base-letter slug. That's not cosmetic: it means the second topic's
    save_lesson call gets wrongly rejected as a duplicate of the first.
    Found by Codex review before this ever shipped."""
    ka_with_i = "कि"
    ka_with_u = "कु"

    assert slugify(ka_with_i) != slugify(ka_with_u)

    kb = DurableKnowledgeBase(Store())
    first = kb.save_lesson(topic=ka_with_i, what_worked="first meaning")
    second = kb.save_lesson(topic=ka_with_u, what_worked="second, unrelated meaning")

    assert "saved" in first
    assert "saved" in second


def test_find_lessons_matches_accented_query_and_content() -> None:
    """A lesson saved with accented-Latin what_worked text must remain
    searchable -- an earlier version's ASCII-only tokenizer ([a-z0-9])
    excluded accented letters entirely, so a word like "café" contributed
    NO term at all, making it permanently unmatchable regardless of
    query. Found by Codex review before this ever shipped.

    (Purely non-Latin scripts with no word-separating whitespace, e.g.
    Japanese or Chinese, remain a narrower, separate limitation: this is
    a regex boundary tokenizer, not a language-aware segmenter, so a
    whole CJK phrase with no spaces still extracts as one long token
    rather than the individual words within it -- see this module's own
    comment above _is_word_char.)"""
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="café société notes", what_worked="the café closes early on Sundays")

    assert kb.find_lessons("café")


def test_find_lessons_matches_a_word_built_from_combining_marks() -> None:
    """`\\w`-based tokenization excludes Unicode combining marks (category
    Mn), so a word that represents its vowels as marks layered on a base
    letter -- e.g. fully vocalized Arabic, with no precomposed form for
    NFKC to collapse the way Latin accents have -- gets torn apart at
    every mark into single-character fragments, each discarded as
    too-short noise. A word that was already correctly space-delimited
    would then be permanently unsearchable by its own exact spelling.
    Found by Codex review before this ever shipped."""
    vocalized_word = "كَتَبَ"  # kataba, vocalized with fatha marks
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="arabic verb notes", what_worked=f"the word {vocalized_word} means 'he wrote'")

    assert kb.find_lessons(vocalized_word)


def test_save_lesson_requires_what_worked() -> None:
    kb = DurableKnowledgeBase(Store())
    assert kb.save_lesson(topic="x", what_worked="   ").startswith("REJECTED")


def test_save_lesson_then_search_finds_it() -> None:
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="telegram bot token leak", what_worked="never use basicConfig(level=INFO) at root")

    result = kb.search_lessons("telegram token")

    assert "telegram bot token leak" in result
    assert "never use basicConfig" in result


def test_search_lessons_with_no_query_terms() -> None:
    kb = DurableKnowledgeBase(Store())
    assert kb.search_lessons("   ") == "no query given"


def test_search_lessons_with_no_matches() -> None:
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="a", what_worked="something")
    assert kb.search_lessons("unrelated query") == "no matching lessons"


def test_find_lessons_rejects_a_nonpositive_limit() -> None:
    """A nonpositive limit silently returns an empty (or, for a negative
    limit, a Python-slice-surprising) result instead of raising -- the
    same class of footgun ApprovalQueue.list_pending_tickets' own
    ``limit`` already guards against."""
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="t", what_worked="something")
    with pytest.raises(ValueError, match="limit"):
        kb.find_lessons("something", limit=0)
    with pytest.raises(ValueError, match="limit"):
        kb.find_lessons("something", limit=-1)


def test_default_prefix_is_isolated_from_a_differently_prefixed_base() -> None:
    """Two DurableKnowledgeBase instances over the same Store, with
    different prefixes, must not see each other's lessons -- the whole
    point of a configurable prefix (unlike the single hardcoded
    LESSON_PREFIX the project this was promoted from used)."""
    store = Store()
    kb_a = DurableKnowledgeBase(store, prefix="app-a:")
    kb_b = DurableKnowledgeBase(store, prefix="app-b:")
    kb_a.save_lesson(topic="shared topic", what_worked="from app a")
    kb_b.save_lesson(topic="shared topic", what_worked="from app b")

    assert kb_a.find_lessons("shared topic")[0]["what_worked"] == "from app a"
    assert kb_b.find_lessons("shared topic")[0]["what_worked"] == "from app b"


def test_default_prefix_constant_matches_the_documented_value() -> None:
    assert DEFAULT_PREFIX == "knowledge:"


# ---------------------------------------------------------------------------
# save_lesson is create-only
# ---------------------------------------------------------------------------


def test_save_lesson_is_create_only_a_second_save_is_refused() -> None:
    store = Store()
    kb = DurableKnowledgeBase(store)
    first = kb.save_lesson(topic="Windows Paths", what_worked="v1")
    assert "saved" in first

    second = kb.save_lesson(topic="windows paths", what_worked="v2")

    assert second.startswith("REJECTED")
    assert "revise_lesson" in second
    # The original content is untouched -- no blind overwrite.
    stored = store.read(f"{DEFAULT_PREFIX}windows-paths")
    assert stored["what_worked"] == "v1"
    assert stored["revision"] == 1


def test_save_lesson_rejects_a_topic_whose_slug_exceeds_the_length_limit() -> None:
    """render_lesson_line truncates a slug over MAX_SLUG_LENGTH for display
    (with a trailing "..." it doesn't strip), so a slug read back from
    search_lessons would no longer match the real Store key -- any
    follow-up revise_lesson/retraction on it would spuriously fail with
    "no lesson exists". Rejecting the oversized topic at creation time
    instead guarantees every lesson made through this API has a slug that
    round-trips through search_lessons unmodified. Found by Codex review
    before this ever shipped."""
    store = Store()
    kb = DurableKnowledgeBase(store)
    long_topic = "a very long and needlessly specific topic name " * 3

    result = kb.save_lesson(topic=long_topic, what_worked="v1")

    assert result.startswith("REJECTED")
    assert store.items(prefix=DEFAULT_PREFIX) == []


def test_search_lessons_slug_round_trips_into_revise_lesson() -> None:
    """The practical scenario the length limit above exists for: a slug
    returned by search_lessons must be usable, unmodified, as the `slug=`
    argument to revise_lesson."""
    store = Store()
    kb = DurableKnowledgeBase(store)
    topic = "a fairly long but still acceptable topic about windows paths"
    assert len(slugify(topic)) <= MAX_SLUG_LENGTH
    kb.save_lesson(topic=topic, what_worked="quote the path")

    rendered = kb.search_lessons("windows paths")
    slug = rendered.split("[", 1)[1].split(" rev", 1)[0]
    assert "..." not in slug

    result = kb.revise_lesson(slug=slug, expected_revision=1, what_worked="quote the path always")

    assert result.startswith("updated")


def test_save_lesson_treats_casefold_equivalent_topics_as_the_same_slug() -> None:
    """.lower() alone leaves German "ß" untouched, so "Straße" and
    "STRASSE" -- text a human would consider the exact same topic --
    would slugify to two DIFFERENT slugs and silently create two
    unrelated lessons instead of the second being refused as a
    duplicate. casefold() (used via _normalize) correctly folds both to
    "strasse". Found by Codex review before this ever shipped."""
    kb = DurableKnowledgeBase(Store())
    first = kb.save_lesson(topic="Straße", what_worked="v1")
    assert "saved" in first

    second = kb.save_lesson(topic="STRASSE", what_worked="v2")

    assert second.startswith("REJECTED")


def test_find_lessons_matches_across_unicode_normalization_forms() -> None:
    """The same visible text can be represented as different Unicode code
    point sequences -- a precomposed "é" (U+00E9) vs. "e" + a combining
    acute accent (U+0065 U+0301) -- which compare UNEQUAL as plain
    strings despite looking identical. Without NFKC normalization first,
    a lesson saved with one form would be invisible to a query written in
    the other. Found by Codex review before this ever shipped."""
    precomposed = unicodedata.normalize("NFC", "café")
    decomposed = unicodedata.normalize("NFD", "café")
    assert precomposed != decomposed  # sanity: genuinely different code points

    kb = DurableKnowledgeBase(Store())
    # Deliberately no plain-ASCII "cafe" anywhere in the saved content --
    # that would let the query match through an unrelated word instead of
    # actually exercising normalization-form equivalence.
    kb.save_lesson(topic=f"{precomposed} notes", what_worked=f"the {precomposed} closes early on Sundays")

    assert kb.find_lessons(decomposed)


def test_save_lesson_treats_casefold_normalization_drift_as_the_same_slug() -> None:
    """str.casefold() is not guaranteed to preserve normalization form --
    Greek "ΐ" (U+0390) casefolds to a sequence that, compared against
    "ΐ".upper() run through the same casefold, wouldn't converge without
    re-normalizing AFTER casefolding too (the standard Unicode canonical
    caseless matching pitfall: normalize, casefold, normalize again).
    Without the second pass, a lowercase lesson and an uppercase query
    for the same word would fail to match, and save_lesson would accept
    both spellings as distinct slugs despite the documented
    case-insensitive create-only identity. Found by Codex review before
    this ever shipped."""
    lower = "ΐ"
    upper = lower.upper()
    assert slugify(lower) == slugify(upper)

    kb = DurableKnowledgeBase(Store())
    first = kb.save_lesson(topic=lower, what_worked="first meaning")
    second = kb.save_lesson(topic=upper, what_worked="second, should collide")

    assert "saved" in first
    assert second.startswith("REJECTED")


# ---------------------------------------------------------------------------
# revise_lesson
# ---------------------------------------------------------------------------


def test_revise_lesson_updates_content_and_bumps_revision() -> None:
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="windows paths", what_worked="v1")

    result = kb.revise_lesson(slug="windows-paths", expected_revision=1, what_worked="v2, also quote the path")

    assert "updated" in result
    stored = store.read(f"{DEFAULT_PREFIX}windows-paths")
    assert stored["what_worked"] == "v2, also quote the path"
    assert stored["revision"] == 2
    assert stored["status"] == "active"


def test_revise_lesson_preserves_gotchas_when_omitted() -> None:
    """gotchas defaults to None ("leave it as it is"), not "" -- a caller
    correcting only what_worked shouldn't have to also re-paste an
    unrelated gotchas just to avoid silently wiping it out. An earlier
    version defaulted to "" and clobbered any existing gotchas on every
    revision that didn't happen to repeat it. Found by Codex review
    before this ever shipped."""
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="windows paths", what_worked="v1", gotchas="requires admin access")

    kb.revise_lesson(slug="windows-paths", expected_revision=1, what_worked="v2, also quote the path")

    stored = store.read(f"{DEFAULT_PREFIX}windows-paths")
    assert stored["gotchas"] == "requires admin access"


def test_revise_lesson_clears_gotchas_when_explicitly_given_empty() -> None:
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="windows paths", what_worked="v1", gotchas="requires admin access")

    kb.revise_lesson(slug="windows-paths", expected_revision=1, what_worked="v2", gotchas="")

    stored = store.read(f"{DEFAULT_PREFIX}windows-paths")
    assert stored["gotchas"] == ""


def test_revise_lesson_rejects_a_stale_expected_revision() -> None:
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="windows paths", what_worked="v1")
    kb.revise_lesson(slug="windows-paths", expected_revision=1, what_worked="v2")

    refusal = kb.revise_lesson(slug="windows-paths", expected_revision=1, what_worked="v3, stale caller")

    assert refusal.startswith("REJECTED")
    assert store.read(f"{DEFAULT_PREFIX}windows-paths")["what_worked"] == "v2"


def test_revise_lesson_on_a_missing_slug_is_refused() -> None:
    kb = DurableKnowledgeBase(Store())
    assert kb.revise_lesson(slug="nope", expected_revision=1, what_worked="x").startswith("REJECTED")


def test_revise_lesson_requires_what_worked() -> None:
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="t", what_worked="v1")
    assert kb.revise_lesson(slug="t", expected_revision=1, what_worked="  ").startswith("REJECTED")


def test_revise_lesson_rejects_an_invalid_status() -> None:
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="t", what_worked="v1")
    refusal = kb.revise_lesson(slug="t", expected_revision=1, what_worked="v2", status="bogus")  # type: ignore[arg-type]
    assert refusal.startswith("REJECTED")


def test_revise_lesson_requires_a_correction_reason_to_retract() -> None:
    """The docstring promises a tombstone that explains why a lesson was
    withdrawn -- an unenforced default of "" would let a reasonless
    retraction through, leaving no audit trail for a later reader to
    judge whether it's safe to trust or reactivate. Found by Codex review
    before this ever shipped."""
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="t", what_worked="v1")

    refusal = kb.revise_lesson(slug="t", expected_revision=1, what_worked="v1", status="retracted")

    assert refusal.startswith("REJECTED")
    assert store.read(f"{DEFAULT_PREFIX}t")["status"] == "active"


def test_retracting_a_lesson_keeps_it_as_a_tombstone_excluded_from_search() -> None:
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="bad advice", what_worked="do the risky thing")

    result = kb.revise_lesson(
        slug="bad-advice",
        expected_revision=1,
        what_worked="do the risky thing",
        status="retracted",
        correction_reason="turned out to break production",
    )

    assert "retracted" in result
    stored = store.read(f"{DEFAULT_PREFIX}bad-advice")
    assert stored["status"] == "retracted"
    assert stored["correction_reason"] == "turned out to break production"
    # Excluded from default search...
    assert kb.search_lessons("risky thing") == "no matching lessons"
    # ...but still readable directly, and findable when explicitly asked for.
    assert kb.find_lessons("risky thing", include_retracted=True)


def test_retracting_does_not_bump_verified_at() -> None:
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="t", what_worked="v1")
    before = store.read(f"{DEFAULT_PREFIX}t")["verified_at"]

    kb.revise_lesson(slug="t", expected_revision=1, what_worked="v1", status="retracted", correction_reason="wrong")

    after = store.read(f"{DEFAULT_PREFIX}t")["verified_at"]
    assert after == before


# ---------------------------------------------------------------------------
# find_lessons ranking
# ---------------------------------------------------------------------------


def test_find_lessons_ranks_distinct_term_coverage_over_repeated_terms() -> None:
    """A lesson matching two different query words should outrank one that
    only repeats a single word many times -- the fix for the naive
    substring-count scoring an earlier version of this module used."""
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="telegram retry backoff", what_worked="exponential backoff on telegram send failures")
    kb.save_lesson(topic="repeats one word", what_worked="telegram telegram telegram telegram")

    matches = kb.find_lessons("telegram backoff", limit=2)

    assert matches[0]["slug"] == "telegram-retry-backoff"


def test_find_lessons_deduplicates_repeated_query_terms_for_ranking() -> None:
    """A query that repeats a word (e.g. a rambling natural-language task
    description) must not count as multiple distinct term hits -- a
    lesson matching only the repeated word must not outrank one matching
    two genuinely different words, which iterating the query's own
    (non-deduplicated) term list would produce. Found by Codex review
    before this ever shipped."""
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="repeats one word", what_worked="alpha alpha alpha alpha alpha")
    kb.save_lesson(topic="two distinct words", what_worked="beta and gamma both appear here")

    matches = kb.find_lessons("alpha alpha alpha beta gamma", limit=2)

    assert matches[0]["slug"] == "two-distinct-words"


def test_find_lessons_respects_limit() -> None:
    kb = DurableKnowledgeBase(Store())
    for i in range(5):
        kb.save_lesson(topic=f"topic {i}", what_worked="windows path handling")

    assert len(kb.find_lessons("windows", limit=2)) == 2


def test_find_lessons_ignores_stopwords_only_queries() -> None:
    kb = DurableKnowledgeBase(Store())
    kb.save_lesson(topic="t", what_worked="something about the")
    assert kb.find_lessons("the a an") == []


# ---------------------------------------------------------------------------
# render_lesson_line -- staleness / retraction flags
# ---------------------------------------------------------------------------


def test_render_lesson_line_flags_a_stale_lesson() -> None:
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="old lesson", what_worked="something from long ago")
    stale = dict(store.read(f"{DEFAULT_PREFIX}old-lesson"))
    stale["verified_at"] = time.time() - (STALE_AFTER_DAYS + 1) * 86400

    line = render_lesson_line(stale)

    assert "STALE" in line


def test_render_lesson_line_does_not_flag_a_fresh_lesson() -> None:
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="fresh lesson", what_worked="something recent")
    fresh = store.read(f"{DEFAULT_PREFIX}fresh-lesson")

    assert "STALE" not in render_lesson_line(fresh)


def test_render_lesson_line_is_a_single_bounded_line_regardless_of_input() -> None:
    """The whole point of this function is a SHORT, SINGLE line per
    lesson -- a caller injecting several of these into an agent's own
    context can't afford one lesson's own topic/what_worked to blow that
    budget or split across multiple lines. An earlier version only
    truncated what_worked (never topic) and never stripped embedded
    newlines from either field. Found by Codex review before this ever
    shipped."""
    lesson = {
        "slug": "s",
        "revision": 1,
        "topic": "line one\nline two\n" + ("x" * 200),
        "what_worked": "worked one\nworked two\n" + ("y" * 200),
        "verified_at": time.time(),
    }

    line = render_lesson_line(lesson)

    assert "\n" not in line
    assert len(line) < 260


def test_render_lesson_line_bounds_a_long_slug_too() -> None:
    """slugify() doesn't cap length either, so a real lesson saved under a
    very long topic ends up with an equally long slug -- an earlier
    version of this function truncated topic and what_worked but
    interpolated the slug uncapped, still letting one "line" blow the
    context-size bound the whole function exists to enforce. Found by
    Codex review before this ever shipped."""
    lesson = {
        "slug": "x" * 5000,
        "revision": 1,
        "topic": "t",
        "what_worked": "w",
        "verified_at": time.time(),
    }

    line = render_lesson_line(lesson)

    assert len(line) < 200


def test_render_lesson_line_collapses_whitespace_in_the_slug_too() -> None:
    """slugify()'s own output never contains whitespace, but this
    function can't assume every record it's asked to render came from
    it -- search_lessons renders whatever raw Store record matched, and
    a slug containing an embedded newline (externally seeded, migrated,
    or otherwise hand-written) would still split this "line" across
    several physical lines despite the length cap, the exact contract
    this function exists to enforce. Found by Codex review before this
    ever shipped."""
    lesson = {
        "slug": "safe\nINJECTED",
        "revision": 1,
        "topic": "t",
        "what_worked": "w",
        "verified_at": time.time(),
    }

    line = render_lesson_line(lesson)

    assert "\n" not in line


def test_render_lesson_line_falls_back_for_an_invalid_verified_at() -> None:
    """An externally seeded or migrated record's verified_at might be
    None, an ISO string, or anything else non-numeric -- float(...) on
    that raises, and since search_lessons calls this once PER MATCH, one
    malformed record would abort retrieval for every other, perfectly
    fine match in the same result set. Found by Codex review before this
    ever shipped."""
    lesson = {
        "slug": "s",
        "revision": 1,
        "topic": "t",
        "what_worked": "w",
        "verified_at": "2026-01-01T00:00:00Z",
    }

    line = render_lesson_line(lesson)  # must not raise

    assert "STALE" not in line  # falls back to "just verified", not stale


def test_render_lesson_line_sanitizes_a_malformed_revision() -> None:
    """revision is conceptually always an int, but an externally seeded
    or migrated record isn't guaranteed to have kept it that way --
    interpolating it unchecked would bypass every whitespace/length
    guard this function otherwise enforces on topic/gist/slug. Found by
    Codex review before this ever shipped."""
    lesson = {
        "slug": "s",
        "revision": "safe\nINJECTED" + ("x" * 200),
        "topic": "t",
        "what_worked": "w",
        "verified_at": time.time(),
    }

    line = render_lesson_line(lesson)

    assert "\n" not in line
    assert len(line) < 200


def test_render_lesson_line_falls_back_for_a_non_finite_verified_at() -> None:
    """A non-finite verified_at (e.g. float("-inf")) passes the float(...)
    conversion cleanly -- it's a valid float -- but then age_days becomes
    infinite and int(age_days) raises OverflowError, which the earlier
    TypeError/ValueError guard does not catch. Same "one malformed record
    aborts the whole result set" failure mode as the ISO-string case
    above, just a different input shape. Found by Codex review before
    this ever shipped."""
    lesson = {
        "slug": "s",
        "revision": 1,
        "topic": "t",
        "what_worked": "w",
        "verified_at": float("-inf"),
    }

    line = render_lesson_line(lesson)  # must not raise

    assert "STALE" not in line  # falls back to "just verified", not stale


def test_render_lesson_line_falls_back_for_a_verified_at_too_large_for_a_float() -> None:
    """A migrated record's verified_at could be a legitimately-parsed but
    astronomically large int (e.g. 10**1000) -- float(...) on that raises
    OverflowError, not TypeError/ValueError, so it slips past the earlier
    except clause and aborts search_lessons for every other, perfectly
    fine match in the same result set. Found by Codex review before this
    ever shipped."""
    lesson = {
        "slug": "s",
        "revision": 1,
        "topic": "t",
        "what_worked": "w",
        "verified_at": 10**1000,
    }

    line = render_lesson_line(lesson)  # must not raise

    assert "STALE" not in line


def test_render_lesson_line_sanitizes_a_non_finite_revision() -> None:
    """int(float("inf")) raises OverflowError, not TypeError/ValueError,
    so a non-finite revision slips past the earlier except clause and
    aborts rendering. Found by Codex review before this ever shipped."""
    lesson = {
        "slug": "s",
        "revision": float("inf"),
        "topic": "t",
        "what_worked": "w",
        "verified_at": time.time(),
    }

    line = render_lesson_line(lesson)  # must not raise

    assert "\n" not in line
    assert len(line) < 200


def test_render_lesson_line_bounds_a_huge_integer_revision_too() -> None:
    """A migrated record's revision could be a legitimately-parsed but
    enormous int (Python ints are arbitrary precision, so int(10**1000)
    succeeds) -- that sails past the except clause meant to catch
    malformed values, but interpolating it unchecked would still blow the
    "one bounded line" contract just as badly as an unbounded string
    would. Found by Codex review before this ever shipped."""
    lesson = {
        "slug": "s",
        "revision": 10**1000,
        "topic": "t",
        "what_worked": "w",
        "verified_at": time.time(),
    }

    line = render_lesson_line(lesson)

    assert len(line) < 200


def test_render_lesson_line_flags_retracted_lessons() -> None:
    store = Store()
    kb = DurableKnowledgeBase(store)
    kb.save_lesson(topic="t", what_worked="v1")
    kb.revise_lesson(slug="t", expected_revision=1, what_worked="v1", status="retracted", correction_reason="wrong")

    line = render_lesson_line(store.read(f"{DEFAULT_PREFIX}t"))

    assert "RETRACTED" in line
