"""Wave 2.2 — beta_overrides validation at provider construction.

Pre-W2.2 ``AnthropicProvider(beta_overrides=...)`` accepted any dict
silently.  A user typing ``"web-search"`` instead of ``"web_search"``
got no feedback — the override was silently ignored, the default
header sent, and the misconfiguration only surfaced (if at all) as
an opaque API failure under specific request shapes.

W2.2 validates at construction:
* Unknown keys → UserWarning + dropped from effective dict.
* Non-string values → UserWarning + dropped.
* Malformed values (don't match ``<feature>-<YYYY>-<MM>-<DD>``) →
  UserWarning, kept (the API is source of truth — don't block a
  freshly-released variant LazyBridge hasn't learned).
"""

from __future__ import annotations

import warnings

import pytest

from lazybridge.core.providers.anthropic import AnthropicProvider


# ---------------------------------------------------------------------------
# _validate_beta_overrides — pure function tests
# ---------------------------------------------------------------------------


def test_unknown_key_warns_and_is_dropped():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = AnthropicProvider._validate_beta_overrides(
            {"web-search": "web-search-2025-03-05"}  # typo: hyphen instead of underscore
        )
    assert out == {}
    msgs = [str(x.message) for x in w]
    assert any("unknown beta_overrides key" in m and "web-search" in m for m in msgs), msgs


def test_known_key_with_well_formed_value_passes_silently():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = AnthropicProvider._validate_beta_overrides(
            {"web_search": "web-search-2025-03-05"}
        )
    assert out == {"web_search": "web-search-2025-03-05"}
    # No warnings on the happy path.
    msgs = [str(x.message) for x in w]
    assert not any("beta_overrides" in m for m in msgs), msgs


def test_known_key_with_malformed_value_warns_but_keeps():
    """Malformed value (e.g. wrong date format) is kept — the API is
    the source of truth for what header strings it accepts.  We just
    surface the suspicion to the user."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = AnthropicProvider._validate_beta_overrides(
            {"web_search": "garbage-no-date"}
        )
    assert out == {"web_search": "garbage-no-date"}  # kept
    msgs = [str(x.message) for x in w]
    assert any("doesn't match the expected pattern" in m for m in msgs), msgs


def test_non_string_value_warns_and_is_dropped():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = AnthropicProvider._validate_beta_overrides(
            {"web_search": 42}  # type: ignore[dict-item]
        )
    assert out == {}
    msgs = [str(x.message) for x in w]
    assert any("must be a string" in m for m in msgs), msgs


def test_non_dict_input_warns_and_returns_empty():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = AnthropicProvider._validate_beta_overrides("not a dict")  # type: ignore[arg-type]
    assert out == {}
    msgs = [str(x.message) for x in w]
    assert any("must be a dict" in m for m in msgs), msgs


@pytest.mark.parametrize(
    "key",
    ["web_search", "code_execution", "computer_use", "skills", "files"],
)
def test_all_documented_keys_are_recognised(key):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = AnthropicProvider._validate_beta_overrides(
            {key: f"{key.replace('_', '-')}-2025-03-05"}
        )
    assert key in out, f"{key} should be a recognised override key"
    msgs = [str(x.message) for x in w]
    assert not any("unknown beta_overrides key" in m for m in msgs)


def test_mixed_valid_and_invalid_keys():
    """Valid keys survive even when invalid keys trigger warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = AnthropicProvider._validate_beta_overrides(
            {
                "web_search": "web-search-2025-03-05",
                "Computer-Use": "computer-use-2025-01-24",  # wrong key form
                "skills": "skills-2025-10-02",
            }
        )
    assert out == {
        "web_search": "web-search-2025-03-05",
        "skills": "skills-2025-10-02",
    }
    msgs = [str(x.message) for x in w]
    assert any("Computer-Use" in m for m in msgs)


# ---------------------------------------------------------------------------
# Integration: __init__ wires the validation
# ---------------------------------------------------------------------------


def test_init_validates_beta_overrides(monkeypatch):
    """Provider construction surfaces typo'd override keys as warnings."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Bypass actual SDK construction — we only need __init__'s
        # override validation to fire.  We catch the SDK error by
        # using kwargs that won't actually instantiate anthropic.
        try:
            AnthropicProvider(
                model="claude-opus-4-7",
                beta_overrides={"web-search": "web-search-2025-03-05"},  # typo
            )
        except Exception:
            # SDK may not be installed in test env; we only care that
            # the validation warning fired before any SDK call.
            pass
    msgs = [str(x.message) for x in w]
    assert any("unknown beta_overrides key" in m and "web-search" in m for m in msgs), msgs


def test_init_drops_unknown_keys_from_effective_overrides(monkeypatch):
    """Even if construction proceeds, the unknown key is GONE from
    the instance — it can't influence ``_build_betas``."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p = AnthropicProvider(
                model="claude-opus-4-7",
                beta_overrides={
                    "web-search": "wrong-key-form",
                    "web_search": "web-search-2025-03-05",
                },
            )
        except Exception:
            pytest.skip("anthropic SDK not installed in test env")
        # Only the recognised key survives.
        assert "web-search" not in p._beta_overrides
        assert p._beta_overrides.get("web_search") == "web-search-2025-03-05"
