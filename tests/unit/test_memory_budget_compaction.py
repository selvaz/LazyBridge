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
    assert len(memory._turns) == 10


def test_auto_compacts_when_remembered_text_exceeds_budget() -> None:
    memory = Memory(strategy="auto", max_tokens=100)
    _fill(memory)

    assert memory._summary
    assert len(memory._turns) == 10


def test_summarizer_is_called_only_when_budget_compaction_happens() -> None:
    prompts: list[str] = []
    memory = Memory(strategy="summary", max_tokens=1000, summarizer=lambda prompt: prompts.append(prompt) or "summary")

    for index in range(11):
        memory.add(f"short {index}", "reply", tokens=9999)
    assert prompts == []

    memory.add("large request " + ("detail " * 600), "large response")
    assert len(prompts) == 1


def test_recent_window_preserves_facts_verbatim() -> None:
    memory = Memory(strategy="summary", max_tokens=100)
    memory.add("Old background sentence.", "Old answer sentence.")
    for index in range(9):
        memory.add(f"Recent filler {index}.", f"Recent answer {index}.")
    memory.add("The launch code is ORCHID.", "The fallback operation is TEMPEST.")

    assert "The launch code is ORCHID." in memory.text()
    assert "The fallback operation is TEMPEST." in memory.text()
    assert len(memory._turns) == 10


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
