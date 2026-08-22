"""Parsing Claude Code's own ``/usage`` prose.

This is a screen-scrape by construction — see the module docstring for why
there is no typed alternative. So these tests lean on the exact text captured
live (2026-08-21), plus every shape of wording drift that would make a naive
parser invent a number instead of admitting it does not know one.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from lazybridge.engines.claude_code.usage import UsageWindow, parse_usage_report

# Captured live against a real account, 2026-08-21.
REAL_REPORT = """You are currently using your subscription to power your Claude Code usage

Current session: 10% used · resets Aug 21, 10:50pm (America/Los_Angeles)
Current week (all models): 74% used · resets Aug 25, 12pm (America/Los_Angeles)
Current week (Fable): 78% used · resets Aug 25, 12pm (America/Los_Angeles)

What's contributing to your limits usage?
"""

_NOW = datetime(2026, 8, 21, 20, 0, tzinfo=ZoneInfo("America/Los_Angeles"))


def test_the_real_report_parses_session_and_every_weekly_window():
    snapshot = parse_usage_report(REAL_REPORT, now=_NOW)

    assert snapshot.parsed
    assert snapshot.session == UsageWindow(
        10,
        "Aug 21, 10:50pm (America/Los_Angeles)",
        datetime(2026, 8, 21, 22, 50, tzinfo=ZoneInfo("America/Los_Angeles")),
    )
    assert set(snapshot.weekly) == {"all models", "Fable"}
    assert snapshot.weekly["all models"].used_percent == 74
    assert snapshot.weekly["Fable"].used_percent == 78


def test_most_used_picks_the_window_closest_to_its_limit():
    snapshot = parse_usage_report(REAL_REPORT, now=_NOW)
    label, window = snapshot.most_used()
    assert label == "Fable"
    assert window.used_percent == 78


def test_a_reset_date_before_now_rolls_forward_a_year():
    """No year is printed. A December reset read in January is next year's,
    not one that already passed -- otherwise every window near a new year
    would silently report as already expired."""
    text = "Current session: 5% used · resets Dec 31, 11pm (America/Los_Angeles)"
    now = datetime(2027, 1, 5, tzinfo=ZoneInfo("America/Los_Angeles"))
    snapshot = parse_usage_report(text, now=now)
    assert snapshot.session.resets_at.year == 2027


def test_a_reset_date_still_ahead_this_year_keeps_the_current_year():
    text = "Current session: 5% used · resets Dec 31, 11pm (America/Los_Angeles)"
    now = datetime(2026, 6, 1, tzinfo=ZoneInfo("America/Los_Angeles"))
    snapshot = parse_usage_report(text, now=now)
    assert snapshot.session.resets_at.year == 2026


def test_noon_and_midnight_convert_correctly():
    """12pm/12am is the one place naive hour+12 arithmetic gets it backwards."""
    text = "Current session: 1% used · resets Aug 25, 12am (America/Los_Angeles)"
    snapshot = parse_usage_report(text, now=_NOW)
    assert snapshot.session.resets_at.hour == 0

    text = "Current session: 1% used · resets Aug 25, 12pm (America/Los_Angeles)"
    snapshot = parse_usage_report(text, now=_NOW)
    assert snapshot.session.resets_at.hour == 12


def test_a_reset_missing_minutes_defaults_to_the_top_of_the_hour():
    text = "Current session: 1% used · resets Aug 25, 3pm (America/Los_Angeles)"
    snapshot = parse_usage_report(text, now=_NOW)
    assert snapshot.session.resets_at.minute == 0


def test_a_reset_with_no_timezone_is_left_unparsed_not_guessed():
    """Assuming UTC or the machine's zone would silently attach an hour that
    the CLI never printed. The raw text still survives."""
    text = "Current session: 40% used · resets Aug 25, 3pm"
    snapshot = parse_usage_report(text, now=_NOW)
    assert snapshot.session.used_percent == 40
    assert snapshot.session.resets_at is None
    assert snapshot.session.resets_raw == "Aug 25, 3pm"


def test_an_unrecognised_timezone_name_is_left_unparsed():
    text = "Current session: 40% used · resets Aug 25, 3pm (Mars/Colony_One)"
    snapshot = parse_usage_report(text, now=_NOW)
    assert snapshot.session.resets_at is None
    assert "Mars/Colony_One" in snapshot.session.resets_raw


def test_text_with_no_recognisable_window_reports_itself_as_unparsed():
    """`parsed=False` is the signal to fall back to raw_text, not an empty
    weekly dict silently standing in for 'nothing is being used'."""
    snapshot = parse_usage_report("The CLI printed something else entirely.", now=_NOW)
    assert not snapshot.parsed
    assert snapshot.session is None
    assert snapshot.weekly == {}
    assert snapshot.most_used() is None
    assert "something else" in snapshot.raw_text


def test_raw_text_is_preserved_even_when_the_report_parses_cleanly():
    snapshot = parse_usage_report(REAL_REPORT, now=_NOW)
    assert snapshot.raw_text == REAL_REPORT


def test_now_defaults_to_the_current_moment_when_not_supplied():
    """A caller who does not pin `now` should still get a real timestamp,
    not a crash -- this exercises the default path the other tests bypass."""
    snapshot = parse_usage_report(REAL_REPORT)
    assert isinstance(snapshot.fetched_at, datetime)
    assert snapshot.fetched_at.tzinfo is not None


def test_a_third_or_further_weekly_window_is_picked_up_without_a_schema_change():
    """The label set is not fixed by this module -- proven by adding one
    this test invents, not one observed live."""
    text = (
        "Current session: 1% used · resets Aug 25, 3pm (America/Los_Angeles)\n"
        "Current week (all models): 10% used · resets Aug 25, 3pm (America/Los_Angeles)\n"
        "Current week (Fable): 20% used · resets Aug 25, 3pm (America/Los_Angeles)\n"
        "Current week (Opus): 30% used · resets Aug 25, 3pm (America/Los_Angeles)\n"
    )
    snapshot = parse_usage_report(text, now=_NOW)
    assert set(snapshot.weekly) == {"all models", "Fable", "Opus"}
    assert snapshot.most_used()[0] == "Opus"


# --------------------------------------------------------------------------- #
# fetch_claude_usage / ClaudeCodeEngine.usage() -- no network, injected client
# --------------------------------------------------------------------------- #
class _FakeUsageClient:
    """Stands in for ``AgentSdkClient``: same ``ClaudeSdkClient`` shape."""

    def __init__(self, text: str = REAL_REPORT, *, error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    async def run(self, prompt, *, options, attachments=()):
        from lazybridge.engines.claude_code.protocol import ClaudeSdkResult

        self.calls.append({"prompt": prompt, "options": options})
        if self.error is not None:
            raise self.error
        return ClaudeSdkResult(text=self.text, session_id="fake-usage-session")

    def stream(self, prompt, *, options, attachments=()):  # pragma: no cover - unused here
        raise NotImplementedError


def test_fetch_claude_usage_sends_the_slash_command_and_parses_the_reply():
    import anyio

    from lazybridge.engines.claude_code.usage import fetch_claude_usage

    fake = _FakeUsageClient()
    snapshot = anyio.run(lambda: fetch_claude_usage(client=fake))

    assert fake.calls[0]["prompt"] == "/usage"
    assert snapshot.session.used_percent == 10
    assert snapshot.weekly["Fable"].used_percent == 78


def test_fetch_claude_usage_asks_for_one_turn_and_no_settings_sources():
    import anyio

    from lazybridge.engines.claude_code.usage import fetch_claude_usage

    fake = _FakeUsageClient()
    anyio.run(lambda: fetch_claude_usage(client=fake))

    options = fake.calls[0]["options"]
    assert options.max_turns == 1
    assert options.setting_sources == ()


def test_fetch_claude_usage_raises_when_the_client_itself_fails():
    """Distinguishable from a parse failure: this is the CLI refusing the
    turn (auth, no session, etc.), not the report having unfamiliar wording."""
    import anyio
    import pytest

    from lazybridge.engines.claude_code.usage import fetch_claude_usage

    fake = _FakeUsageClient(error=RuntimeError("Claude Agent SDK failed: not authenticated"))
    with pytest.raises(RuntimeError, match="not authenticated"):
        anyio.run(lambda: fetch_claude_usage(client=fake))


def test_fetch_claude_usage_times_out_rather_than_hanging():
    import anyio
    import pytest

    from lazybridge.engines.claude_code.usage import fetch_claude_usage

    class _Hangs:
        async def run(self, prompt, *, options, attachments=()):
            await anyio.sleep(10)

        def stream(self, prompt, *, options, attachments=()):
            raise NotImplementedError

    with pytest.raises(TimeoutError):
        anyio.run(lambda: fetch_claude_usage(client=_Hangs(), timeout=0.05))


def test_engine_usage_reuses_its_own_model_cwd_and_injected_client():
    """The one thing that makes this a convenience and not a duplicate: it
    must route through the SAME client the engine was built with, so a test
    engine never reaches the real SDK through this path either."""
    import anyio
    import pytest

    pytest.importorskip("claude_agent_sdk", reason="needs lazybridge[claude-code]")
    from lazybridge.engines.claude_code import ClaudeCodeEngine

    fake = _FakeUsageClient()
    engine = ClaudeCodeEngine(model="opus", cwd="/work", client=fake)
    snapshot = anyio.run(engine.usage)

    assert fake.calls[0]["options"].model == "opus"
    assert fake.calls[0]["options"].cwd == "/work"
    assert snapshot.weekly["all models"].used_percent == 74


# --------------------------------------------------------------------------- #
# Cases the second review round caught
# --------------------------------------------------------------------------- #
def test_the_year_comes_from_the_reports_own_zone_not_nows():
    """A near-midnight-UTC `now` sits on a different calendar date than the
    same instant in America/Los_Angeles. Using now.year (UTC's) turned a
    reset half an hour away into one reported a year out."""
    from zoneinfo import ZoneInfo

    now = datetime(2027, 1, 1, 0, 30, tzinfo=ZoneInfo("UTC"))  # Dec 31 2026, 16:30 in LA
    text = "Current session: 90% used · resets Dec 31, 5pm (America/Los_Angeles)"
    snapshot = parse_usage_report(text, now=now)

    resets_at = snapshot.session.resets_at
    assert resets_at.year == 2026
    assert (resets_at - now).total_seconds() < 3600  # imminent, not a year away


def test_an_ambiguous_fall_back_hour_is_left_unparsed_not_guessed():
    """1:30am America/Los_Angeles occurs twice on the fall-back night. The
    report gives no UTC offset to tell the occurrences apart, so guessing
    fold=0 would be exactly the invented precision this module refuses
    elsewhere -- the reset must come back unparsed instead."""
    from zoneinfo import ZoneInfo

    now = datetime(2026, 10, 1, tzinfo=ZoneInfo("America/Los_Angeles"))
    text = "Current session: 5% used · resets Nov 1, 1:30am (America/Los_Angeles)"
    snapshot = parse_usage_report(text, now=now)
    assert snapshot.session.used_percent == 5
    assert snapshot.session.resets_at is None
    assert "Nov 1" in snapshot.session.resets_raw


def test_a_naive_now_is_rejected_rather_than_compared_against_an_aware_reset():
    import pytest

    with pytest.raises(ValueError, match="timezone-aware"):
        parse_usage_report(REAL_REPORT, now=datetime(2026, 8, 21, 20, 0))


def test_zoneinfo_resolves_the_zone_the_real_report_uses():
    """Guards the environment, not the logic: on Windows this needs the
    conditional `tzdata` dependency declared for the `claude-code` extra, or
    every reset silently comes back unparsed instead of erroring loudly."""
    from zoneinfo import ZoneInfo

    ZoneInfo("America/Los_Angeles")  # raises ZoneInfoNotFoundError if absent
