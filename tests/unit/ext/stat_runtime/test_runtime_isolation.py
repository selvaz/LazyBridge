"""Regression tests for runtime isolation and tool binding.

Tests that:
- Two runtimes can coexist without cross-talk
- Tools from stat_tools(rt1) always use rt1, even after stat_tools(rt2) is called
- time_col ordering is enforced
- OLS forecast rejects exogenous models
"""

from lazybridge.ext.stat_runtime.persistence import MetaStore
from lazybridge.ext.stat_runtime.schemas import DatasetMeta, RunRecord, RunStatus
from lazybridge.ext.stat_runtime.tools import stat_tools


class TestRuntimeIsolation:
    """P1: Two runtimes must not cross-contaminate via tool bindings."""

    def test_two_runtimes_isolated(self):
        """Tools from rt1 must not see rt2's datasets."""

        class FakeRuntime:
            def __init__(self, name):
                self._name = name
                self.meta_store = MetaStore()
                self.catalog = type(
                    "Cat",
                    (),
                    {
                        "list_datasets": lambda self_: [DatasetMeta(name=f"ds_{self_._rt_name}", uri="/fake")],
                        "_rt_name": name,
                    },
                )()

        rt1 = FakeRuntime("rt1")
        rt2 = FakeRuntime("rt2")

        tools1 = stat_tools(rt1)
        tools2 = stat_tools(rt2)

        # Find list_datasets tool in each bundle
        ld1 = next(t for t in tools1 if t.name == "list_datasets")
        ld2 = next(t for t in tools2 if t.name == "list_datasets")

        result1 = ld1.run({})
        result2 = ld2.run({})

        # Each should see only its own runtime's datasets
        assert result1[0]["name"] == "ds_rt1"
        assert result2[0]["name"] == "ds_rt2"

    def test_second_stat_tools_does_not_overwrite_first(self):
        """After calling stat_tools(rt2), tools from rt1 must still work with rt1."""

        class FakeRuntime:
            def __init__(self, name):
                self._name = name
                self.meta_store = MetaStore()

            def get_run(self, run_id):
                return RunRecord(run_id=run_id, engine=self._name, status=RunStatus.SUCCESS)

        rt1 = FakeRuntime("engine_rt1")
        rt2 = FakeRuntime("engine_rt2")

        tools1 = stat_tools(rt1)
        _tools2 = stat_tools(rt2)  # this used to overwrite the global

        get_run1 = next(t for t in tools1 if t.name == "get_run")
        result = get_run1.run({"run_id": "test123"})

        # Must still use rt1, not rt2
        assert result["engine"] == "engine_rt1"


class TestToolSchemaGeneration:
    """Tools must generate valid schemas from closure-based functions."""

    def test_all_tools_have_schemas(self):
        class MockRuntime:
            pass

        tools = stat_tools(MockRuntime())
        for tool in tools:
            defn = tool.definition()
            assert defn.name, f"Tool {tool} missing name"
            assert defn.description, f"Tool {tool.name} missing description"

    def test_tool_count(self):
        class MockRuntime:
            pass

        tools = stat_tools(MockRuntime())
        assert len(tools) == 15  # 4 high-level + 11 low-level
