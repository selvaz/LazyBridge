"""Tests for LazyTool wrappers — schema generation, no heavy deps needed."""

import pytest

from lazybridge.lazy_tool import LazyTool
from lazybridge.stat_runtime.tools import stat_tools


class TestStatToolsFactory:
    def test_returns_list_of_lazy_tools(self):
        class MockRuntime:
            pass
        tools = stat_tools(MockRuntime())
        assert isinstance(tools, list)
        assert all(isinstance(t, LazyTool) for t in tools)
        assert len(tools) == 15  # 4 high-level + 11 low-level

    def test_tool_names(self):
        class MockRuntime:
            pass
        tools = stat_tools(MockRuntime())
        names = {t.name for t in tools}
        expected = {
            # High-level
            "discover_data", "discover_analyses", "analyze", "register_dataset",
            # Low-level
            "list_datasets", "profile_dataset",
            "query_data", "fit_model", "forecast_model", "run_diagnostics",
            "get_run", "list_runs", "compare_models", "list_artifacts", "get_plot",
        }
        assert names == expected

    def test_tool_schemas_generated(self):
        class MockRuntime:
            pass
        tools = stat_tools(MockRuntime())
        for tool in tools:
            defn = tool.definition()
            assert defn.name
            assert defn.description
            assert "properties" in defn.parameters or defn.parameters == {}


class TestToolErrorHandling:
    """Tools bound to a broken runtime should return error dicts, not raise."""

    def test_get_run_returns_error_on_exception(self):
        class BrokenRuntime:
            def get_run(self, run_id):
                raise RuntimeError("DB connection failed")
            meta_store = None
            catalog = type("C", (), {"list_datasets": lambda self: []})()
            def list_runs(self, **kwargs):
                return []

        tools = stat_tools(BrokenRuntime())
        get_run_tool = next(t for t in tools if t.name == "get_run")
        result = get_run_tool.run({"run_id": "abc"})
        assert result.get("error") is True
        assert "DB connection failed" in result["message"]
