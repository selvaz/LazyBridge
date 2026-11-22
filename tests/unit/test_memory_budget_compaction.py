"""D140 regressions for text-sized memory compaction."""

from lazybridge.memory import Memory


def _fill(memory: Memory, *, turns: int = 11, tokens: int = 1) -> None:
    for index in range(turns):
        memory.add(
            f"User turn {index}: " + ("context " * 12),
            f"Assistant turn {index}: " + ("response " * 12),
            tokens=tokens,
        )


def test_summary_compacts_when_remembered_text_exceeds_budget() -> None:
    memory = Memory(strategy="summary", max_tokens=100)
    _fill(memory)

    assert memory._summary
    assert 2 <= len(memory._turns) <= 10


def test_auto_compacts_when_remembered_text_exceeds_budget() -> None:
    memory = Memory(strategy="auto", max_tokens=100)
    _fill(memory)

    assert memory._summary
    assert 2 <= len(memory._turns) <= 10


def test_summarizer_is_called_only_when_budget_compaction_happens() -> None:
    prompts: list[str] = []
    memory = Memory(strategy="summary", max_tokens=1000, summarizer=lambda prompt: prompts.append(prompt) or "summary")

    for index in range(11):
        memory.add(f"short {index}", "reply", tokens=9999)
    assert prompts == []

    for index in range(200):
        memory.add(f"more short context {index}", "reply")
        if prompts:
            break
    assert len(prompts) == 1


def test_recent_window_preserves_facts_verbatim() -> None:
    memory = Memory(strategy="summary", max_tokens=100)
    memory.add("Old background sentence.", "Old answer sentence.")
    for index in range(9):
        memory.add(f"Recent filler {index}.", f"Recent answer {index}.")
    memory.add("The launch code is ORCHID.", "The fallback operation is TEMPEST.")

    assert "The launch code is ORCHID." in memory.text()
    assert "The fallback operation is TEMPEST." in memory.text()
    assert 2 <= len(memory._turns) <= 10


def test_text_estimate_does_not_depend_on_provider_cache_tokens() -> None:
    memory = Memory(strategy="auto", max_tokens=100)
    _fill(memory, tokens=1)

    assert memory._summary
    assert all(turn.token_estimate > 1 for turn in memory._turns)


def test_extractive_fallback_keeps_complete_sentences_and_named_entities() -> None:
    memory = Memory(strategy="summary", max_tokens=100)
    memory.add("Project ORCHID launches on Tuesday.", "TEMPEST remains the rollback plan.")
    for index in range(10):
        memory.add(f"Filler sentence number {index}.", "Acknowledged.")

    assert "Project ORCHID launches on Tuesday." in memory._summary
    assert "TEMPEST remains the rollback plan." in memory._summary


def test_budget_compaction_uses_hysteresis_and_calls_summarizer_once_per_add() -> None:
    calls: list[str] = []
    memory = Memory(
        strategy="summary",
        max_tokens=120,
        summarizer=lambda prompt: calls.append(prompt) or ("A bounded summary. " * 100),
    )

    previous_calls = 0
    for index in range(12):
        memory.add(f"request {index} " + ("detail " * 12), "response " + ("answer " * 8))
        assert len(calls) - previous_calls <= 1
        previous_calls = len(calls)

    assert len(memory._turns) == 2
    assert len(memory._summary) <= int(120 * memory._SUMMARY_SHARE * 4)
    # Oversized mandatory tails use the bounded local fallback instead of
    # paying for an LLM summary on every add.
    assert calls == []


def test_rule_summary_has_a_hard_budget_cap_for_unpunctuated_text() -> None:
    memory = Memory(strategy="summary", max_tokens=100)
    for index in range(3):
        memory.add("unbroken " * 200 + str(index), "reply")

    assert len(memory._summary) <= 120


def test_load_recomputes_stale_persisted_token_estimates() -> None:
    class StubStore:
        def read_memory(self, key: str, *, session_key: str) -> dict[str, object]:
            return {
                "turns": [{"user": "long " * 40, "assistant": "answer", "token_estimate": 1}],
                "summary": "",
            }

        def write_memory(self, *args: object, **kwargs: object) -> None:
            pass

    memory = Memory(store=StubStore(), key="test")
    assert memory._turns[0].token_estimate > 1


def test_tokens_argument_warns_and_is_ignored() -> None:
    import pytest

    memory = Memory(strategy="none")
    with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
        memory.add("a long stored request", "a stored response", tokens=1)

    assert memory._turns[0].token_estimate > 1
