"""Full LLM-driven stock analysis pipeline.

The agent downloads real market data via yfinance, registers it in the
stat_runtime, runs analysis, and reports findings — all autonomously.

Usage:
    # Set your API key first
    set ANTHROPIC_API_KEY=sk-ant-...

    python examples/stock_analysis_pipeline.py
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf

from lazybridge import LazyAgent, LazyTool
from lazybridge.ext.stat_runtime.runner import StatRuntime
from lazybridge.ext.stat_runtime.tools import stat_tools, build_stat_skills, stat_skill_tools


# ======================================================================
# Custom tool: download real stock data via yfinance
# ======================================================================

def download_stock_data(
    tickers: Annotated[list[str], "List of ticker symbols (e.g. ['SPY', 'AAPL', 'MSFT'])"],
    days: Annotated[int, "Number of trading days of history"] = 500,
    save_dir: Annotated[str, "Directory to save the parquet file"] = ".",
) -> dict[str, Any]:
    """Download historical daily OHLCV + returns for one or more tickers from Yahoo Finance.

    Saves a single parquet file with columns: date, symbol, open, high, low, close,
    volume, ret (daily log returns). Returns the file path and summary stats.
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=int(days * 1.5))  # buffer for non-trading days

        # Download all tickers at once (more reliable than one-by-one)
        raw = yf.download(tickers, start=start, end=end, progress=False)
        if raw.empty:
            return {"error": True, "message": f"No data returned for tickers: {tickers}"}

        frames = []
        for ticker in tickers:
            try:
                # Handle multi-ticker (multi-level columns) vs single-ticker
                if len(tickers) > 1:
                    df = raw.xs(ticker, level="Ticker", axis=1).copy()
                else:
                    df = raw.copy()
                    if hasattr(df.columns, 'levels'):
                        df.columns = df.columns.get_level_values(0)

                df = df.tail(days).reset_index()
                df["symbol"] = ticker
                df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
                df = df.dropna(subset=["ret"])
                frames.append(df[["Date", "symbol", "Open", "High", "Low", "Close", "Volume", "ret"]])
            except Exception as exc:
                print(f"  Warning: skipping {ticker}: {exc}")
                continue

        if not frames:
            return {"error": True, "message": f"Could not process data for tickers: {tickers}"}

        combined = pd.concat(frames, ignore_index=True)
        combined.columns = ["date", "symbol", "open", "high", "low", "close", "volume", "ret"]

        path = str(Path(save_dir) / "stock_data.parquet")
        combined.to_parquet(path, index=False)

        summary = {
            "file_path": path,
            "tickers": tickers,
            "rows": len(combined),
            "date_range": f"{combined['date'].min()} to {combined['date'].max()}",
            "columns": list(combined.columns),
            "per_ticker_rows": {t: int((combined["symbol"] == t).sum()) for t in tickers},
        }
        return summary

    except Exception as exc:
        return {"error": True, "type": type(exc).__name__, "message": str(exc)}


# ======================================================================
# Main pipeline
# ======================================================================

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        artifacts_dir = str(tmpdir / "artifacts")

        # --- 1. Create runtime and tools ---
        rt = StatRuntime(artifacts_dir=artifacts_dir)
        high_tools = stat_tools(rt, level="high")
        all_stat = stat_tools(rt, level="all")

        # Custom download tool (saves to tmpdir)
        def download_wrapper(
            tickers: Annotated[list[str], "Ticker symbols (e.g. ['SPY', 'AAPL'])"],
            days: Annotated[int, "Trading days of history"] = 500,
        ) -> dict[str, Any]:
            """Download real stock data from Yahoo Finance. Returns file path + summary."""
            return download_stock_data(tickers, days, save_dir=str(tmpdir))

        download_tool = LazyTool.from_function(
            download_wrapper,
            name="download_stock_data",
            description="Download historical daily stock data (OHLCV + returns) from Yahoo Finance",
            guidance="Call this first to get real market data. Provide ticker symbols and "
                     "number of trading days. The data is saved as a parquet file. "
                     "After downloading, register it with register_dataset().",
        )

        # Build skills for doc retrieval
        skill_tools = []
        try:
            skill_dirs = build_stat_skills(output_root=str(tmpdir / "skills"))
            skill_tools = stat_skill_tools(skill_dirs)
            print(f"Loaded {len(skill_tools)} skill tools")
        except Exception as e:
            print(f"Skills not loaded (non-fatal): {e}")

        # Combine: high-level stat tools + download + skills
        agent_tools = high_tools + [download_tool] + skill_tools

        # --- 2. Create the agent ---
        system_prompt = """You are a quantitative financial analyst with access to statistical analysis tools.

WORKFLOW — follow these steps for every analysis request:

1. DOWNLOAD DATA: Use download_stock_data() to fetch real market data
2. REGISTER: Use register_dataset() to register the parquet file
   - Set time_column="date", entity_keys=["symbol"], canonical_target="ret"
   - Set business_description to describe what you downloaded
3. DISCOVER: Use discover_data() to verify registration and see column roles
4. ANALYZE: Use analyze() with an appropriate mode:
   - mode="volatility" for risk/volatility analysis
   - mode="forecast" for price/return forecasting
   - mode="regime" for bull/bear market detection
   - mode="recommend" to let the runtime choose
   - Use group_col="symbol" and group_value="TICKER" to analyze individual stocks
5. REVIEW: Use discover_analyses() to see all completed analyses
6. REPORT: Summarize findings with key metrics, model adequacy, and recommendations

IMPORTANT:
- Always filter panel data by symbol using group_col/group_value in analyze()
- Check model_adequate in the result — if False, suggest alternative approaches
- Reference specific plot artifact paths so the user can view them
- Use the skill tools to look up methodology when unsure about model choice
"""

        agent = LazyAgent(
            "anthropic",
            name="stock_analyst",
            system=system_prompt,
            tools=agent_tools,
            model="claude-sonnet-4-20250514",
        )

        # --- 3. Run analysis ---
        print("=" * 70)
        print("  STARTING AGENT PIPELINE")
        print("=" * 70)

        # First request: download and analyze
        print("\n--- Request 1: Download data and run volatility analysis ---\n")
        resp = agent.loop(
            "Download 2 years of daily data for SPY, AAPL, and MSFT. "
            "Register it and run a volatility analysis on SPY returns. "
            "Tell me about volatility persistence and whether the model is adequate.",
            max_steps=12,
        )
        print("\n--- AGENT RESPONSE ---")
        print(resp.content)

        # Second request: compare across tickers
        print("\n\n--- Request 2: Compare volatility across tickers ---\n")
        messages = [
            {"role": "user", "content": (
                "Now run volatility analysis on AAPL and MSFT too (same dataset, "
                "just filter by symbol). Then tell me which stock has the highest "
                "volatility persistence and which one has the most adequate model."
            )}
        ]
        resp2 = agent.loop(messages, max_steps=12)
        print("\n--- AGENT RESPONSE ---")
        print(resp2.content)

        # Third request: regime detection
        print("\n\n--- Request 3: Regime detection ---\n")
        messages3 = [
            {"role": "user", "content": (
                "Run a regime detection analysis on SPY returns. "
                "Are there distinct bull/bear market regimes? "
                "How persistent are they?"
            )}
        ]
        resp3 = agent.loop(messages3, max_steps=10)
        print("\n--- AGENT RESPONSE ---")
        print(resp3.content)

        # Show all analyses
        print("\n\n--- Final: All analyses performed ---\n")
        discover_tool = next(t for t in agent_tools if t.name == "discover_analyses")
        all_analyses = discover_tool.run({})
        print(f"Total runs: {all_analyses['total_runs']}")
        print(f"Datasets analyzed: {all_analyses['datasets_analyzed']}")
        if all_analyses.get("best_by_aic"):
            print(f"Best model by AIC: {all_analyses['best_by_aic']}")
        for run in all_analyses.get("runs", []):
            status = "PASS" if run["status"] == "success" else "FAIL"
            aic = run.get("aic", "N/A")
            print(f"  [{status}] {run['engine']:8s} on {run['dataset_name']:15s} "
                  f"target={run.get('target_col', '?'):8s} AIC={aic}")

        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nArtifacts saved in: {artifacts_dir}")
        print("Check the plots/ subdirectories for generated visualizations.")

        rt.close()


if __name__ == "__main__":
    main()
