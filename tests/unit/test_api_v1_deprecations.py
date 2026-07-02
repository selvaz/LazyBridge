"""v1 API pass — deprecated top-level aliases behave as documented.

``Task`` / ``CacheConfig`` / ``PROVIDER_ALIASES`` were removed from
``lazybridge.__all__`` in the v1 pass; they stay reachable through the
module-level ``__getattr__`` with a ``DeprecationWarning`` for the 0.10
series and are scheduled for removal in 1.0.
"""

from __future__ import annotations

import warnings

import pytest

import lazybridge


def test_replan_task_is_the_canonical_name():
    from lazybridge import ReplanTask
    from lazybridge.engines.replan import ReplanTask as SubmoduleReplanTask
    from lazybridge.engines.replan import Task as SubmoduleTask

    assert ReplanTask is SubmoduleReplanTask
    # The submodule alias stays a plain (non-warning) alias until 1.0.
    assert SubmoduleTask is SubmoduleReplanTask


def test_top_level_task_warns_and_resolves():
    with pytest.warns(DeprecationWarning, match="ReplanTask"):
        task_cls = lazybridge.Task
    assert task_cls is lazybridge.ReplanTask


def test_top_level_cacheconfig_warns_and_resolves():
    from lazybridge.core.types import CacheConfig

    with pytest.warns(DeprecationWarning, match="core.types"):
        cls = lazybridge.CacheConfig
    assert cls is CacheConfig


def test_top_level_provider_aliases_warns_and_resolves():
    with pytest.warns(DeprecationWarning, match="provider_aliases"):
        aliases = lazybridge.PROVIDER_ALIASES
    assert aliases == lazybridge.LLMEngine.provider_aliases()


def test_deprecated_names_not_in_all():
    for name in ("Task", "CacheConfig", "PROVIDER_ALIASES"):
        assert name not in lazybridge.__all__
    assert "ReplanTask" in lazybridge.__all__


def test_star_import_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        namespace: dict = {}
        exec("from lazybridge import *", namespace)
    assert "Agent" in namespace
    assert "Task" not in namespace


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError, match="no attribute"):
        _ = lazybridge.definitely_not_a_symbol
