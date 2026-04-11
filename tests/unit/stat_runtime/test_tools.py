"""Tests for LazyTool wrappers — schema generation, no heavy deps needed."""

import pytest

from lazybridge.lazy_tool import LazyTool
from lazybridge.stat_runtime.tools import (
    compare_models,
    fit_model,
    get_plot,
    get_run,
    list_artifacts,
    list_datasets,
    list_runs,
    profile_dataset,
    query_data,
    register_dataset,
    run_diagnostics,
    stat_tools,
)


class TestToolFunctionsWithoutRuntime:
    """Tools should return error dicts when no runtime is initialized."""

    def test_register_dataset_no_runtime(self):
        result = register_dataset("test", "/test.parquet")
        assert result.get("error") is True
        assert "not initialized" in result["message"]

    def test_list_datasets_no_runtime(self):
        result = list_datasets()
        assert isinstance(result, list)
        assert result[0].get("error") is True

    def test_fit_model_no_runtime(self):
        result = fit_model("ols", "y")
        assert result.get("error") is True

    def test_query_data_no_runtime(self):
        result = query_data("SELECT 1")
        assert result.get("error") is True

    def test_get_run_no_runtime(self):
        result = get_run("abc")
        assert result.get("error") is True


class TestStatToolsFactory:
    def test_returns_list_of_lazy_tools(self):
        # Use a mock runtime
        class MockRuntime:
            pass
        tools = stat_tools(MockRuntime())
        assert isinstance(tools, list)
        assert all(isinstance(t, LazyTool) for t in tools)
        assert len(tools) == 12

    def test_tool_names(self):
        class MockRuntime:
            pass
        tools = stat_tools(MockRuntime())
        names = {t.name for t in tools}
        expected = {
            "register_dataset", "list_datasets", "profile_dataset",
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
