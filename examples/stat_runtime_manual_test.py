"""Manual test script for stat_runtime high-level tools.

Run this to exercise the exact same functions an LLM agent would call.
Creates synthetic data, registers it, then walks through the full workflow.

Usage:
    python examples/stat_runtime_manual_test.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl


def pprint(label: str, obj):
    """Pretty-print a result dict."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    if isinstance(obj, dict):
        print(json.dumps(obj, indent=2, default=str)[:3000])
    elif isinstance(obj, list):
        print(json.dumps(obj, indent=2, default=str)[:3000])
    else:
        print(obj)
    print()


def create_test_data(tmpdir: Path) -> dict[str, str]:
    """Create synthetic parquet files for testing."""
    np.random.seed(42)

    # --- Dataset 1: Daily equity returns (panel data) ---
    n_days = 500
    symbols = ["SPY", "AAPL", "MSFT"]
    rows = []
    for sym in symbols:
        dates = pl.date_range(
            pl.date(2022, 1, 1),
            pl.date(2022, 1, 1) + pl.duration(days=n_days - 1),
            eager=True,
        )
        rets = np.random.normal(0.0005, 0.015, n_days)
        # Add volatility clustering for GARCH
        vol = np.zeros(n_days)
        vol[0] = 0.015
        for i in range(1, n_days):
            vol[i] = np.sqrt(0.00001 + 0.08 * rets[i-1]**2 + 0.90 * vol[i-1]**2)
            rets[i] = np.random.normal(0.0005, vol[i])
        rows.append(pl.DataFrame({
            "date": dates,
            "symbol": [sym] * n_days,
            "ret": rets,
            "volume": np.random.randint(1_000_000, 50_000_000, n_days),
        }))
    equities = pl.concat(rows)
    eq_path = str(tmpdir / "equities.parquet")
    equities.write_parquet(eq_path)

    # --- Dataset 2: Monthly macro data (single series, no entity key) ---
    dates = pl.date_range(
        pl.date(2014, 1, 1),
        pl.date(2023, 12, 1),
        interval="1mo",
        eager=True,
    )
    n_months = len(dates)
    macro = pl.DataFrame({
        "month": dates,
        "gdp_growth": np.random.normal(2.0, 0.8, n_months),
        "inflation": np.random.normal(2.5, 0.5, n_months),
        "unemployment": np.clip(np.random.normal(5.0, 1.0, n_months), 2, 12),
    })
    macro_path = str(tmpdir / "macro.parquet")
    macro.write_parquet(macro_path)

    # --- Dataset 3: Ambiguous targets (multiple numeric columns, no obvious target) ---
    n = 200
    ambig = pl.DataFrame({
        "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=n-1), eager=True),
        "price": np.cumsum(np.random.normal(0, 1, n)) + 100,
        "value": np.cumsum(np.random.normal(0, 0.5, n)) + 50,
        "yield_pct": np.random.normal(3.0, 0.5, n),
    })
    ambig_path = str(tmpdir / "ambiguous.parquet")
    ambig.write_parquet(ambig_path)

    return {
        "equities": eq_path,
        "macro": macro_path,
        "ambiguous": ambig_path,
    }


def main():
    from lazybridge.ext.stat_runtime.runner import StatRuntime
    from lazybridge.ext.stat_runtime.tools import stat_tools

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        artifacts_dir = str(tmpdir / "artifacts")
        paths = create_test_data(tmpdir)

        print("Created test data:")
        for name, path in paths.items():
            print(f"  {name}: {path}")

        # ============================================================
        # SETUP: Create runtime and get high-level tools
        # ============================================================
        rt = StatRuntime(artifacts_dir=artifacts_dir)
        tools = stat_tools(rt, level="high")

        # Get tool references (same as what an LLM would call)
        register = next(t for t in tools if t.name == "register_dataset")
        discover = next(t for t in tools if t.name == "discover_data")
        analyses = next(t for t in tools if t.name == "discover_analyses")
        analyze = next(t for t in tools if t.name == "analyze")

        # ============================================================
        # TEST 1: Register datasets with semantic metadata
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 1: Register datasets")
        print("#"*70)

        result = register.run({
            "name": "equities",
            "uri": paths["equities"],
            "time_column": "date",
            "frequency": "daily",
            "entity_keys": ["symbol"],
            "business_description": "Daily equity returns for SPY, AAPL, MSFT",
            "canonical_target": "ret",
            "identifiers_to_ignore": [],
        })
        pprint("register equities", result)
        assert not result.get("error"), f"Registration failed: {result}"

        result = register.run({
            "name": "macro",
            "uri": paths["macro"],
            "time_column": "month",
            "frequency": "monthly",
            "business_description": "Monthly macroeconomic indicators",
            "canonical_target": "gdp_growth",
        })
        pprint("register macro", result)
        assert not result.get("error"), f"Registration failed: {result}"

        result = register.run({
            "name": "ambiguous",
            "uri": paths["ambiguous"],
            "time_column": "date",
            "frequency": "daily",
        })
        pprint("register ambiguous (no canonical target)", result)
        assert not result.get("error"), f"Registration failed: {result}"

        # ============================================================
        # TEST 2: Discover data — see what's available
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 2: discover_data()")
        print("#"*70)

        result = discover.run({})
        pprint("discover_data", result)
        assert result["total_datasets"] == 3
        # Check that semantic fields came through
        eq = next(d for d in result["datasets"] if d["name"] == "equities")
        assert eq["business_description"] == "Daily equity returns for SPY, AAPL, MSFT"
        assert eq["canonical_target"] == "ret"
        assert eq["summary"] != ""
        print(f"  Equities summary: {eq['summary']}")
        print(f"  Column roles: {[r['column'] + '=' + r['inferred_role'] for r in eq['column_roles']]}")

        # ============================================================
        # TEST 3: analyze() with mode=recommend (auto target from canonical)
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 3: analyze(mode='recommend') — auto target from canonical_target")
        print("#"*70)

        result = analyze.run({
            "dataset_name": "equities",
            "mode": "recommend",
            "group_col": "symbol",
            "group_value": "SPY",
        })
        pprint("analyze equities (recommend, no explicit target)", result)
        if not result.get("error"):
            print(f"  Mode rationale: {result.get('mode_rationale', 'N/A')}")
            print(f"  Target: {result.get('target_col', 'N/A')}")
            print(f"  Engine: {result.get('engine', 'N/A')}")
            print(f"  Model adequate: {result.get('model_adequate', 'N/A')}")
            print(f"  Assumptions: {result.get('assumptions', [])[:2]}")
            print(f"  Interpretation: {result.get('interpretation', [])[:3]}")
            print(f"  Plots: {[p['name'] for p in result.get('plots', [])]}")
            print(f"  Next steps: {result.get('next_steps', [])[:2]}")
        else:
            print(f"  Error (expected if mock): {result.get('message', 'N/A')}")

        # ============================================================
        # TEST 4: analyze() with mode=volatility (explicit)
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 4: analyze(mode='volatility', target='ret', group SPY)")
        print("#"*70)

        result = analyze.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "mode": "volatility",
            "group_col": "symbol",
            "group_value": "SPY",
        })
        pprint("analyze volatility", result)
        if not result.get("error"):
            print(f"  Engine: {result['engine']}")
            print(f"  AIC: {result['metrics'].get('aic', 'N/A')}")
            print(f"  Model adequate: {result['model_adequate']}")
            print(f"  Plots: {[p['name'] for p in result.get('plots', [])]}")

        # ============================================================
        # TEST 5: analyze() with mode=forecast on macro data
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 5: analyze(mode='forecast') on macro data — auto target")
        print("#"*70)

        result = analyze.run({
            "dataset_name": "macro",
            "mode": "forecast",
            # target_col omitted — should use canonical_target "gdp_growth"
        })
        pprint("analyze forecast (macro)", result)
        if not result.get("error"):
            print(f"  Target: {result['target_col']}")
            print(f"  Engine: {result['engine']}")
            print(f"  Forecast: {result.get('forecast') is not None}")

        # ============================================================
        # TEST 6: analyze() with mode=describe — no target needed
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 6: analyze(mode='describe') on macro — no target")
        print("#"*70)

        result = analyze.run({
            "dataset_name": "macro",
            "mode": "describe",
        })
        pprint("analyze describe (macro)", result)
        if not result.get("error"):
            print(f"  Target used: {result['target_col']}")
            print(f"  Interpretation: {result.get('interpretation', [])[:3]}")

        # ============================================================
        # TEST 7: analyze() with ambiguous target — should return candidates
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 7: analyze(mode='recommend') on ambiguous data — no target")
        print("#"*70)

        result = analyze.run({
            "dataset_name": "ambiguous",
            "mode": "recommend",
        })
        pprint("analyze ambiguous (should show candidates)", result)
        if result.get("type") == "AmbiguousTarget":
            print(f"  Correctly returned ambiguity!")
            print(f"  Candidates: {result['candidates']}")
        elif result.get("error"):
            print(f"  Error: {result.get('message')}")

        # ============================================================
        # TEST 8: Group col injection attempt — should be blocked
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 8: SQL injection attempts via group_col/group_value")
        print("#"*70)

        # Bad column name
        result = analyze.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "group_col": "symbol; DROP TABLE--",
            "group_value": "SPY",
        })
        print(f"  Injection via group_col: error={result.get('error')}, msg={result.get('message', '')[:80]}")
        assert result.get("error"), "Should have blocked injection"

        # Bad value
        result = analyze.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "group_col": "symbol",
            "group_value": "' OR 1=1 --",
        })
        print(f"  Injection via group_value: error={result.get('error')}, msg={result.get('message', '')[:80]}")

        # Missing pair
        result = analyze.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "group_col": "symbol",
        })
        print(f"  group_col without value: error={result.get('error')}, msg={result.get('message', '')[:80]}")
        assert result.get("error"), "Should require both group_col and group_value"

        # ============================================================
        # TEST 9: discover_analyses() — review what was done
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 9: discover_analyses()")
        print("#"*70)

        result = analyses.run({})
        pprint("discover_analyses", result)
        print(f"  Total runs: {result['total_runs']}")
        print(f"  Datasets analyzed: {result['datasets_analyzed']}")
        if result["best_by_aic"]:
            print(f"  Best by AIC: {result['best_by_aic']}")
        for run in result.get("runs", [])[:3]:
            print(f"  Run {run['run_id'][:8]}: {run['engine']} on {run['dataset_name']} "
                  f"({run['status']}, AIC={run.get('aic', 'N/A')})")

        # ============================================================
        # TEST 10: Profile then re-discover (column signals)
        # ============================================================
        print("\n" + "#"*70)
        print("# TEST 10: Profile dataset then discover (column signals)")
        print("#"*70)

        # Use low-level tool for profiling
        all_tools = stat_tools(rt, level="all")
        profile = next(t for t in all_tools if t.name == "profile_dataset")
        profile.run({"name": "equities"})
        print("  Profiled equities dataset")

        result = discover.run({})
        eq = next(d for d in result["datasets"] if d["name"] == "equities")
        signals = eq.get("column_signals", {})
        print(f"  has_profile: {eq['has_profile']}")
        for col, sig in signals.items():
            print(f"  {col}: null_pct={sig.get('null_pct')}, unique={sig.get('unique_count')}, mean={sig.get('mean')}")

        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "="*70)
        print("  ALL MANUAL TESTS COMPLETED")
        print("="*70)
        print("""
Next step: Connect a real LLM agent:

    from lazybridge.ext.stat_runtime.tools import stat_agent

    agent, rt = stat_agent("anthropic")

    # Register your data
    rt.catalog.register_parquet(
        "my_data", "/path/to/data.parquet",
        time_column="date",
        canonical_target="returns",
    )

    # Let the agent work
    resp = agent.loop("What can you tell me about the data?")
    resp = agent.loop("Analyze the volatility of returns")
    resp = agent.loop("Compare with a regime-switching model")
""")


if __name__ == "__main__":
    main()
