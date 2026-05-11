"""Shared fixtures for live tests (require real LLM API keys)."""

from __future__ import annotations

import os

import pytest

from lazybridge import Session

# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------

#: Cheapest model for simple single-step rungs (no tool-use needed).
MODEL_CHEAP = os.getenv("LB_LIVE_MODEL", "claude-haiku-4-5")

#: More capable model for tool-use, parallel, and Plan rungs.
MODEL_CAPABLE = os.getenv("LB_LIVE_MODEL_CAPABLE", "claude-haiku-4-5")


@pytest.fixture
def model() -> str:
    return MODEL_CHEAP


@pytest.fixture
def model_capable() -> str:
    return MODEL_CAPABLE


@pytest.fixture
def sess(tmp_path) -> Session:
    """Fresh Session per test: console output + ephemeral SQLite DB."""
    s = Session(db=str(tmp_path / "live.db"), console=True)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# --viz flag: pytest --viz opens the browser during Rung 7
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--viz",
        action="store_true",
        default=False,
        help="Open browser during Rung 7 visualizer test.",
    )


@pytest.fixture
def viz_open(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--viz")
