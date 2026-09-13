"""Durable, Store-backed "lessons" a long-running agent records after
genuinely non-obvious success -- separate from a durable plan/blackboard
(:mod:`lazybridge.ext.planners.durable_blackboard`) so a lesson outlives
whichever plan or task it was learned on, and separate from
:class:`lazybridge.Memory` (turn-scoped conversational history with
compression) -- neither of those is the same abstraction as durable,
cross-session knowledge that should survive indefinitely and be searched
back into later, unrelated turns.

Promoted from LazyCEO's ``lazyceo.lessons`` (a sibling project built on
this package) after live production use.

Design rationale, preserved from the project this was promoted from: two
earlier versions were reshaped by Codex review, informed by three pieces
of prior art -- Voyager's execution-verified, embedding-retrieved skill
library; AutoGen Teachability's automatic pre-response memo injection
instead of a tool the main agent calls at its own discretion; LangMem's
separate background "memory manager" -- whose common thread is that
reliable recall does NOT come from trusting a busy task-agent to remember
to search on its own. Two concrete consequences here:

* :meth:`DurableKnowledgeBase.save_lesson` is CREATE-ONLY (CAS against
  ``None``) -- a second save under the same topic is rejected with a
  pointer to :meth:`~DurableKnowledgeBase.revise_lesson` instead of
  silently overwriting. ``revise_lesson`` itself is CAS'd against an
  ``expected_revision``, so a correction can't race another writer's
  concurrent edit, and every revision keeps a ``status`` so a lesson later
  found wrong can be retracted (kept as a tombstone) instead of only
  fixable by chance.
* Retrieval is not left purely opt-in in the caller's own integration --
  see :meth:`~DurableKnowledgeBase.find_lessons`'s own docstring: a
  caller wiring this into an agent's task-claim loop should call it
  directly and inject matches into what the agent sees, the same way
  LazyCEO's own ``claim_next()`` does, rather than relying on the agent
  remembering to search first. :meth:`~DurableKnowledgeBase.search_lessons`
  still exists as an explicit tool for ad-hoc checks (e.g. before deciding
  a topic is new).

Embeddings were deliberately NOT added: at the dozens-to-low-hundreds
scale this is meant to operate at, and with no embedding provider assumed
to be wired in, a small in-process lexical scorer (term counts weighted
by field, requiring real term matches) is good enough -- revisit only
once a real missed-recall case shows up, per the same review.
"""

from __future__ import annotations

import math
import time
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from lazybridge import Store

#: Default Store key prefix for every lesson record -- ``store.items(prefix=...)``
#: scans this namespace in :meth:`DurableKnowledgeBase.find_lessons`. Pass
#: ``prefix=`` to :class:`DurableKnowledgeBase` for a namespaced knowledge
#: base instead (e.g. LazyCEO's own facade uses ``"ceo:lesson:"`` to
#: preserve its own on-disk key format).
DEFAULT_PREFIX = "knowledge:"

#: A lesson not re-verified in this many days is flagged (not hidden) at
#: retrieval time -- an old lesson may still be correct, so ranking is by
#: relevance, not recency; staleness is a warning for the reader, not a
#: filter.
STALE_AFTER_DAYS = 180.0

#: Rejected by :meth:`DurableKnowledgeBase.save_lesson` at creation time if
#: ``slugify(topic)`` would exceed this length -- kept in sync with
#: :func:`render_lesson_line`'s own slug truncation bound so a slug
#: returned by :meth:`~DurableKnowledgeBase.search_lessons` can always be
#: passed straight back into :meth:`~DurableKnowledgeBase.revise_lesson`
#: unmodified, for any lesson created through this class's own API. Without
#: this check, a long-but-legitimate topic could silently produce a slug
#: that ``render_lesson_line`` truncates for display (with a trailing
#: "..." it doesn't strip), so the identifier a caller reads back from
#: search results would no longer match the real Store key -- any
#: follow-up ``revise_lesson``/retraction on it would spuriously fail with
#: "no lesson exists". Only an externally seeded or migrated record that
#: bypassed this check can still hit that truncation. Found by Codex
#: review before this ever shipped.
MAX_SLUG_LENGTH = 60

LessonStatus = Literal["active", "retracted"]

#: Word-splitting history for both `slugify` (joins word-runs with `-`)
#: and search tokenization (`_query_terms`, below): an earlier, ASCII-only
#: version (`[a-z0-9]+`/`[^a-z0-9]+`) rejected any topic without at least
#: one ASCII letter/digit -- including ones in non-Latin scripts (a topic
#: like "東京" has no ASCII alnum at all, so `slugify` raised even though
#: its own error message promises to accept "at least one alphanumeric
#: character") -- and made non-ASCII `what_worked`/`gotchas` text
#: completely unsearchable. A later, `\w`-based Unicode-aware regex fixed
#: that, but `\w` still excludes Unicode COMBINING MARKS (general category
#: Mn/Mc/Me): invisible for scripts where accents have precomposed forms
#: NFKC collapses to one code point (Latin "café"), but scripts that
#: represent vowels/tone purely as marks with no precomposed form
#: (Devanagari matras, Arabic harakat, Hebrew niqqud) would have those
#: marks read as word BOUNDARIES -- shredding a search term into
#: unsearchable single-character fragments, and, worse for `slugify`,
#: making two topics differing only in which mark is attached to a base
#: letter collapse onto the SAME slug (e.g. Hindi "कि" vs "कु" both
#: reducing to "क"), wrongly rejecting the second `save_lesson` as a
#: duplicate of the first. `_is_word_char`/`_tokenize` below fix both by
#: scanning character-by-character instead of via regex. Found by Codex
#: review before this ever shipped (three times: the ASCII-only gap, then
#: combining marks in search tokenization, then the same combining-mark
#: gap in `slugify` itself).
#:
#: Known accepted gap: this is still a word-BOUNDARY splitter, not a
#: language-aware segmenter -- it still assumes words are separated by
#: whitespace/punctuation, true for English/Italian/French/etc. but not
#: for scripts like Japanese or Chinese written with no spaces between
#: words. A whole CJK phrase with no spaces extracts as ONE long token
#: (every character in it is a Unicode "word" character, so nothing
#: splits it), not the individual words within it -- searchable only as
#: that exact whole phrase, not by any word inside it. Proper CJK support
#: needs a real segmenter (e.g. MeCab, Jieba), a real dependency this
#: module deliberately doesn't take on; revisit only if a real missed-
#: recall case for such a script shows up, the same standard this
#: module's own docstring already applies to embeddings.


def _is_word_char(ch: str) -> bool:
    """`\\w`'s own Unicode word-character class (letters, decimal digits,
    and underscore) excludes Unicode COMBINING MARKS (general category
    Mn/Mc/Me) -- so a plain `\\w`-based regex tears a word apart at every
    diacritic. That's invisible for scripts where accents have precomposed
    forms NFKC collapses to one code point (Latin "café"'s decomposed form
    recombines in `_normalize` before this ever runs), but scripts that
    represent vowels or tone purely as combining marks layered on a base
    letter -- vocalized Arabic harakat, Hebrew niqqud, Devanagari matras --
    have no such precomposed form: the marks stay separate code points,
    read as word BOUNDARIES by `\\w`, and the resulting one-character
    fragments are then discarded as too-short noise. A word that was
    already correctly space-delimited would silently fail to round-trip
    through save -> search. Treating any Unicode Mark category as part of
    a word closes that gap without a real Unicode-property regex engine --
    the stdlib `re` module has no `\\p{...}` support; that needs the
    third-party `regex` package, a dependency this module deliberately
    doesn't take on (same standard as the CJK segmenter gap above). Found
    by Codex review before this ever shipped.
    """
    return ch.isalnum() or unicodedata.category(ch)[0] == "M"


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into maximal runs of `_is_word_char` characters --
    the same shape of result a `\\w`-based regex would produce, but
    combining-mark-aware (see `_is_word_char`). A plain per-character scan
    is simple, correct, and fast enough at the dozens-to-low-hundreds-of-
    lessons scale this module targets.
    """
    tokens: list[str] = []
    current: list[str] = []
    for ch in text:
        if _is_word_char(ch):
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


#: Small, deliberately short stopword list (English + Italian, matching
#: the project this was promoted from) -- just enough to stop single
#: common words from dominating the match score; not a real linguistic
#: resource.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to", "is", "are",
        "was", "were", "with", "this", "that", "it", "its", "be", "as", "at", "by", "not",
        "il", "lo", "la", "i", "gli", "le", "di", "che", "un", "una", "e", "per", "con",
    }
)  # fmt: skip


def _normalize(text: str) -> str:
    """Canonical form for both slug identity and search-term matching --
    NFKC normalization first, so two Unicode representations of the same
    visible text (a precomposed "é" vs. "e" + a combining acute accent,
    e.g.) collapse to the same code points before anything else runs,
    THEN casefold rather than :meth:`str.lower` -- a more aggressive,
    correctly Unicode-aware case-insensitive comparison (German "Straße"
    casefolds to "strasse", matching the ASCII-ish spelling a caller
    might otherwise query with; plain ``.lower()`` leaves the "ß"
    untouched, so the two would silently diverge). Skipping either step
    let two calls a human would consider "the same topic" collide into
    DIFFERENT slugs (defeating ``save_lesson``'s whole create-only-CAS
    point) and let a differently-normalized or differently-cased query
    silently miss a match it should have found. Found by Codex review
    before this ever shipped.

    Normalized a SECOND time after casefolding, not just once before it --
    ``str.casefold()`` is not guaranteed to preserve normalization form
    (e.g. Greek "ΐ" (U+0390) casefolds to a sequence that, compared
    against ``"ΐ".upper()`` run through this same function, wouldn't
    converge without re-normalizing afterward too), which is exactly the
    standard "canonical caseless matching" pitfall Unicode's own
    case-folding guidance warns about (normalize, casefold, normalize
    again). Skipping the second pass let a lowercase lesson and an
    uppercase query for the same word fail to match, and let
    ``save_lesson`` accept both spellings as distinct slugs despite the
    documented case-insensitive create-only identity. Found by Codex
    review before this ever shipped.
    """
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).casefold())


def slugify(topic: str) -> str:
    """A stable, key-safe identity for a topic.

    Two calls with the same wording (modulo case, punctuation, or Unicode
    normalization form -- see :func:`_normalize`) collide on purpose --
    that collision is what makes a second ``save_lesson`` on the same
    topic get refused (and pointed at ``revise_lesson``) instead of
    silently landing as an unrelated new record.

    Built on the same combining-mark-aware :func:`_tokenize` used for
    search terms (word-runs joined by ``-``), not a plain ``\\W``-based
    regex -- a regex splitter would treat a combining mark as a
    separator, which is invisible for scripts NFKC precomposes (Latin
    accents) but for scripts that represent vowels/tone purely as marks
    with no precomposed form (Devanagari matras, Arabic harakat, Hebrew
    niqqud) would make two topics differing ONLY in which mark is
    attached to a base letter collapse onto the SAME slug -- e.g. Hindi
    "कि" (ka + vowel sign i) and "कु" (ka + vowel sign u) both
    reducing to slug "क". That's a real, not cosmetic, harm: the second
    ``save_lesson`` call would be wrongly rejected as a duplicate of the
    first via the create-only CAS check, even though the two topics are
    genuinely different. Found by Codex review before this ever shipped.
    """
    slug = "-".join(_tokenize(_normalize(topic.strip())))
    # `_tokenize` treats a combining mark as part of a word (see its own
    # docstring), which is exactly what makes the fix above work -- but it
    # also means a topic made of nothing BUT unattached combining marks
    # (e.g. a bare "́", with no base letter at all) tokenizes to a
    # non-empty slug that is nonetheless invisible/unusable as a key: `not
    # slug` alone doesn't catch it. Checking for at least one genuinely
    # alphanumeric character (not just "non-empty") closes that gap. Found
    # by Codex review before this ever shipped.
    if not any(ch.isalnum() for ch in slug):
        raise ValueError("topic must contain at least one alphanumeric character")
    return slug


def _query_terms(text: str) -> list[str]:
    return [t for t in _tokenize(_normalize(text)) if t not in _STOPWORDS and len(t) > 1]


def _field_terms(text: str) -> Counter[str]:
    return Counter(_query_terms(text))


def render_lesson_line(lesson: dict[str, Any]) -> str:
    """One line: slug/revision/staleness flag, topic, and a short gist --
    never the full text, so a caller injecting several of these (e.g. into
    an agent's own task-claim result) can't flood context the way dumping
    every lesson's full content on every tick would.

    ``topic``, ``what_worked``, AND ``slug`` are all collapsed to
    single-line text (embedded newlines replaced with spaces) and
    length-capped -- an earlier version truncated only ``what_worked``,
    so a caller free to write an arbitrarily long or multiline ``topic``
    (nothing validates it beyond :func:`slugify` needing SOME alphanumeric
    content) could still make one "line" span many lines or dwarf the
    gist entirely; a second version fixed ``topic`` but missed that
    :func:`slugify` doesn't cap length either, so the SLUG itself (derived
    directly from an uncapped ``topic``) could still blow the bound this
    function exists to enforce. Found by Codex review before this ever
    shipped (twice: once for topic/gist, again for the slug).
    """
    topic = " ".join(str(lesson.get("topic", "")).split())
    if len(topic) > 80:
        topic = topic[:77] + "..."
    gist = " ".join(str(lesson.get("what_worked", "")).split())
    if len(gist) > 150:
        gist = gist[:147] + "..."
    # Collapsed the same way as topic/gist above, not just length-capped:
    # a Store record isn't guaranteed to have gone through slugify() at
    # all (externally seeded, migrated, or otherwise hand-written) --
    # slugify()'s OWN output never contains whitespace, but this function
    # can't assume every record it's asked to render came from it. A
    # slug containing an embedded newline would otherwise still split
    # this "line" across several physical lines despite the length cap,
    # the exact contract this function exists to enforce -- found
    # entirely correctly by Codex review, since search_lessons renders
    # arbitrary matching Store records through this same function. Found
    # by Codex review before this ever shipped.
    slug = " ".join(str(lesson.get("slug", "")).split())
    if len(slug) > MAX_SLUG_LENGTH:
        slug = slug[: MAX_SLUG_LENGTH - 3] + "..."
    # Same defensive posture as the slug above: an externally seeded or
    # migrated record's verified_at might be missing, None, an ISO
    # string, or anything else non-numeric. `float(...)` on that would
    # raise and abort this ONE call -- but search_lessons calls this once
    # PER MATCH, so a single malformed record would abort retrieval for
    # every OTHER, perfectly fine match in the same result set too.
    # Falling back to "just verified" (not stale) is the safe default:
    # under-warning about staleness is far less harmful than search
    # breaking entirely. Found by Codex review before this ever shipped.
    # A non-finite value (inf/-inf/nan) passes float() but would blow up
    # int(age_days) below with OverflowError -- reject it here too, since
    # it's the same "malformed record" failure mode, just a different
    # shape of malformed. Found by Codex review before this ever shipped.
    try:
        # float(10**1000) raises OverflowError, not ValueError -- a
        # migrated record's timestamp being a legitimately-parsed but
        # astronomically large int is malformed the same way a
        # non-numeric one is, so it must fall into the same fallback.
        verified_at = float(lesson.get("verified_at", lesson.get("updated_at", time.time())))
    except (TypeError, ValueError, OverflowError):
        verified_at = time.time()
    if not math.isfinite(verified_at):
        verified_at = time.time()
    age_days = (time.time() - verified_at) / 86400.0
    flags = ""
    if lesson.get("status") == "retracted":
        flags += " RETRACTED"
    elif age_days > STALE_AFTER_DAYS:
        flags += f" STALE(verified {int(age_days)}d ago)"
    # Sanitized the same way as topic/gist/slug: revision is conceptually
    # always an int, but an externally seeded or migrated record isn't
    # guaranteed to have kept it that way -- interpolating it unchecked
    # would bypass every whitespace/length guard this function otherwise
    # enforces. Found by Codex review before this ever shipped.
    raw_revision = lesson.get("revision", 1)
    try:
        # int(float("inf")) raises OverflowError, not ValueError -- a
        # non-finite revision is malformed the same way a non-numeric one
        # is, so it must fall into the same string-sanitizing branch.
        revision = str(int(raw_revision))
    except (TypeError, ValueError, OverflowError):
        revision = " ".join(str(raw_revision).split())
    # A successfully-parsed int still isn't guaranteed to be SHORT -- a
    # migrated record with revision=10**1000 sails through `int(...)`
    # (Python ints are arbitrary precision) and would otherwise bypass
    # every other length guard in this function, blowing the "one bounded
    # line" contract just as badly as an unbounded string would. Applying
    # the same cap to both branches uniformly closes that gap. Found by
    # Codex review before this ever shipped.
    if len(revision) > 20:
        revision = revision[:17] + "..."
    return f"- [{slug} rev{revision}]{flags} {topic}: {gist}"


class DurableKnowledgeBase:
    """A durable, Store-backed collection of "lessons" -- reusable notes
    an agent records after genuinely non-obvious success, retrievable by
    keyword search across every future session.

    Keyed under a configurable ``prefix`` so multiple independent
    knowledge bases (e.g. one per application, or one shared fleet-wide
    base) can live in one ``Store`` without key collisions -- construct
    one ``DurableKnowledgeBase`` per prefix a caller needs; the object
    itself holds no connection of its own beyond the ``Store`` it's
    given, so constructing one is cheap.

    ``prefix`` matching is a literal string prefix (same semantics as
    :meth:`Store.items`'s own ``prefix=``), NOT a namespace boundary: a
    base whose prefix is itself a prefix of another base's prefix (e.g.
    ``"app:"`` and ``"app:team:"``) is NOT isolated from it -- listing the
    outer base's lessons will include the inner base's too, since every
    inner key literally starts with the outer prefix. Choose prefixes that
    are NOT string-prefixes of each other if isolation between two bases
    matters.
    """

    def __init__(self, store: Store, *, prefix: str = DEFAULT_PREFIX) -> None:
        self._store = store
        self._prefix = prefix

    def _key(self, slug: str) -> str:
        return f"{self._prefix}{slug}"

    def save_lesson(self, *, topic: str, what_worked: str, gotchas: str = "") -> str:
        """Create the lesson for ``topic``. Rejected (not overwritten) if
        one already exists under that slug -- the caller is pointed at
        :meth:`revise_lesson` with the current revision, so an update is
        always an explicit, CAS'd action, never an accidental blind
        overwrite.

        Also rejected if ``topic`` would produce a slug longer than
        :data:`MAX_SLUG_LENGTH` -- see that constant's docstring for why.

        Known accepted gap: create-only relies on
        :meth:`Store.compare_and_swap`'s own ``expected=None`` contract,
        which -- by that method's own documented semantics -- cannot
        distinguish "key absent" from "key present holding a literal JSON
        ``null``". A record ever written as bare ``null`` (nothing in this
        codebase does that) would be silently treated as absent and
        overwritten rather than rejected. Not fixed here: it would need a
        new Store-level primitive to atomically distinguish the two cases,
        and every other create-only-via-CAS caller in this codebase leans
        on the identical contract. Found by Codex review before this ever
        shipped.
        """
        if not what_worked.strip():
            return "REJECTED: what_worked is required."
        slug = slugify(topic)
        if len(slug) > MAX_SLUG_LENGTH:
            return (
                f"REJECTED: topic produces a {len(slug)}-character slug, over the "
                f"{MAX_SLUG_LENGTH}-character limit -- use a shorter, more specific topic."
            )
        key = self._key(slug)
        existing = self._store.read(key)
        if isinstance(existing, dict):
            revision = existing.get("revision", 1)
            return (
                f"REJECTED: lesson {slug!r} already exists at revision {revision} -- call "
                f"revise_lesson(slug={slug!r}, expected_revision={revision}, ...) to update it, "
                "or search_lessons to read it first."
            )
        now = time.time()
        record = {
            "topic": topic.strip(),
            "slug": slug,
            "what_worked": what_worked.strip(),
            "gotchas": gotchas.strip(),
            "status": "active",
            "revision": 1,
            "correction_reason": "",
            "created_at": now,
            "updated_at": now,
            "verified_at": now,
        }
        if not self._store.compare_and_swap(key, None, record):
            # Another writer created the same slug between our read and
            # our write -- a real (if rare) race, not something to
            # silently clobber.
            return (
                f"REJECTED: lesson {slug!r} was just created by another writer -- "
                "call search_lessons to see it, or revise_lesson to update it."
            )
        return f"saved lesson {slug!r} (revision 1)"

    def revise_lesson(
        self,
        *,
        slug: str,
        expected_revision: int,
        what_worked: str,
        gotchas: str | None = None,
        status: LessonStatus = "active",
        correction_reason: str = "",
    ) -> str:
        """Update (or retract) an existing lesson, CAS'd against
        ``expected_revision`` so a correction can't silently clobber a
        concurrent edit -- a mismatch is rejected with instructions to
        re-read the current version first, the same shape as
        :class:`~lazybridge.ext.planners.DurableBlackboard`'s stale-claim
        rejections.

        ``status="retracted"`` keeps the record (a tombstone with
        ``correction_reason`` explaining why) instead of deleting it --
        retracted lessons are excluded from
        :meth:`search_lessons`/:meth:`find_lessons` by default, but the
        history of "this used to be believed" survives.

        ``gotchas`` defaults to ``None``, meaning "leave it as it is" --
        unlike ``what_worked`` (always required, since a revision without
        new content isn't really a revision), a caller correcting
        ``what_worked`` alone shouldn't have to also re-paste an unrelated
        ``gotchas`` just to avoid silently wiping it. Pass ``gotchas=""``
        explicitly to actually clear it. Found by Codex review before this
        ever shipped, after an earlier version defaulted to ``""`` and
        clobbered any existing ``gotchas`` on every revision that didn't
        happen to repeat it.
        """
        if status not in ("active", "retracted"):
            return "REJECTED: status must be 'active' or 'retracted'."
        if not what_worked.strip():
            return "REJECTED: what_worked is required."
        # The docstring above promises a tombstone that explains why a
        # lesson was withdrawn -- an unenforced default of "" would let a
        # reasonless retraction through, leaving a later reader no way to
        # tell whether it's safe to reactivate or trust again. Found by
        # Codex review before this ever shipped.
        if status == "retracted" and not correction_reason.strip():
            return "REJECTED: correction_reason is required when retracting a lesson."
        key = self._key(slug)
        existing = self._store.read(key)
        if not isinstance(existing, dict):
            return f"REJECTED: no lesson {slug!r} exists -- call save_lesson to create it."
        current_revision = existing.get("revision", 1)
        if current_revision != expected_revision:
            return (
                f"REJECTED: lesson {slug!r} is at revision {current_revision}, not {expected_revision} -- "
                "call search_lessons to see the current version before revising."
            )
        now = time.time()
        updated = {
            **existing,
            "what_worked": what_worked.strip(),
            "gotchas": existing.get("gotchas", "") if gotchas is None else gotchas.strip(),
            "status": status,
            "revision": current_revision + 1,
            "correction_reason": correction_reason.strip(),
            "updated_at": now,
            # A retraction isn't re-confirming the content as correct, so
            # it does not bump verified_at -- only an active revision does.
            "verified_at": now if status == "active" else existing.get("verified_at", now),
        }
        if not self._store.compare_and_swap(key, existing, updated):
            return (
                f"REJECTED: lesson {slug!r} was changed by another writer while you were revising it -- "
                "call search_lessons to see the current version and try again."
            )
        verb = "retracted" if status == "retracted" else "updated"
        return f"{verb} lesson {slug!r} (revision {updated['revision']})"

    def find_lessons(self, query: str, *, limit: int = 5, include_retracted: bool = False) -> list[dict[str, Any]]:
        """The lessons best matching ``query``, most relevant first.

        A small lexical scorer, not semantic search: query terms
        (stopwords and single characters dropped) are matched against each
        lesson's topic (weight 3), what_worked (weight 2), and gotchas
        (weight 1) token counts. Ranked first by how many DISTINCT query
        terms hit at all, then by total weighted count -- a lesson
        matching two different words beats one matching the same word
        three times, which plain substring-count scoring (an earlier
        version of this module) could get backwards.

        Returns raw records (not rendered text) so both
        :meth:`search_lessons` and a caller's own forced-retrieval hook
        (e.g. wired into an agent's task-claim loop, the way LazyCEO's own
        ``claim_next()`` does -- see this module's own docstring) share
        one scoring implementation instead of drifting apart. Call this
        directly from such a hook rather than relying on the agent to
        remember to call :meth:`search_lessons` itself.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        # Deduplicated: `terms` is a list, so a query that repeats a word
        # (e.g. "alpha alpha alpha beta gamma") would otherwise count
        # `distinct` once PER OCCURRENCE of a hit, not once per unique
        # matching word -- a lesson matching only the repeated "alpha"
        # would outrank one matching both "beta" and "gamma", exactly
        # backwards from the distinct-term-coverage ranking this method's
        # own docstring promises. Found by Codex review before this ever
        # shipped.
        terms = set(_query_terms(query))
        if not terms:
            return []
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for _key, raw in self._store.items(prefix=self._prefix):
            if not isinstance(raw, dict):
                continue
            if not include_retracted and raw.get("status") == "retracted":
                continue
            topic_terms = _field_terms(str(raw.get("topic", "")))
            worked_terms = _field_terms(str(raw.get("what_worked", "")))
            gotcha_terms = _field_terms(str(raw.get("gotchas", "")))
            distinct = 0
            total = 0
            for term in terms:
                hits = topic_terms[term] * 3 + worked_terms[term] * 2 + gotcha_terms[term] * 1
                if hits:
                    distinct += 1
                    total += hits
            if distinct:
                scored.append((distinct, total, raw))
        scored.sort(key=lambda triple: (triple[0], triple[1]), reverse=True)
        return [lesson for _distinct, _total, lesson in scored[:limit]]

    def search_lessons(self, query: str, *, limit: int = 5) -> str:
        """Keyword-match ``query`` against saved lessons; a short ranked
        list (never full text). See :meth:`find_lessons` for the scoring
        rule.
        """
        if not query.strip():
            return "no query given"
        matches = self.find_lessons(query, limit=limit)
        if not matches:
            return "no matching lessons"
        return "\n".join(render_lesson_line(m) for m in matches)


__all__ = [
    "DEFAULT_PREFIX",
    "MAX_SLUG_LENGTH",
    "STALE_AFTER_DAYS",
    "DurableKnowledgeBase",
    "LessonStatus",
    "render_lesson_line",
    "slugify",
]
