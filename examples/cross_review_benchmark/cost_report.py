"""Report paired benchmark outcomes and API-equivalent costs from a JSONL file."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from examples.cross_review_benchmark.harness import ARMS, load_rows, wilson
from lazybridge.core.providers.openai import _PRICE_TABLE


def call_cost(call):
    """Return (USD or None, upper_bound); missing data is never free."""
    meta = call.get("metadata") or {}
    if call.get("provider") == "claude_code":
        cost = meta.get("cost_usd")
        return (cost, False) if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0 else (None, False)
    if call.get("provider") != "codex":
        return None, False
    model = meta.get("model")
    # Exact IDs only: an unfamiliar model must never inherit a guessed price.
    price = _PRICE_TABLE.get(model.lower()) if isinstance(model, str) else None
    inputs, outputs = meta.get("input_tokens"), meta.get("output_tokens")
    if price is None or any(type(x) is not int or x < 0 for x in (inputs, outputs)):
        return None, False
    cached = meta.get("cached_input_tokens")
    upper = cached is None
    if upper:
        cached = 0
    if type(cached) is not int or not 0 <= cached <= inputs:
        return None, False
    input_price, cache_price, output_price = price
    return ((inputs - cached) * input_price + cached * (cache_price if cache_price is not None else input_price)
            + outputs * output_price) / 1_000_000, upper


def row_cost(row):
    calls = row.get("writer_calls", []) + row.get("reviewer_calls", [])
    if not calls:
        return None, False
    costs = [call_cost(call) for call in calls]
    if any(value is None for value, _ in costs):
        return None, False
    return sum(value for value, _ in costs), any(upper for _, upper in costs)


def report(rows):
    lines = ["| Arm | Passes / n | Pass rate (95% Wilson) | USD / row | USD / pass |",
             "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        n, passes = len(group), sum(r["passed"] for r in group)
        costs = [row_cost(row) for row in group]
        if not n or any(cost is None for cost, _ in costs):
            per_row = per_pass = "unknown"
        else:
            total = sum(cost for cost, _ in costs)
            suffix = " (upper bound)" if any(upper for _, upper in costs) else ""
            per_row = f"${total / n:.6f}{suffix}"
            per_pass = f"${total / passes:.6f}{suffix}" if passes else "undefined (0 passes)"
        lo, hi = wilson(passes, n)
        rate = f"{passes / n:.1%} ({lo:.1%} to {hi:.1%})" if n else "pending"
        lines.append(f"| {arm} | {passes}/{n} | {rate} | {per_row} | {per_pass} |")

    by_arm = {arm: {r["task_id"]: r["passed"] for r in rows if r["arm"] == arm} for arm in ARMS}
    shared = by_arm["A"].keys() & by_arm["B"].keys()
    lines += ["", "| Comparison | Passes / paired n | Rate |", "|---|---:|---:|"]
    def add(label, values):
        n = len(values)
        count = sum(values)
        lines.append(f"| {label} | {count}/{n} | {count / n:.1%} |" if n else f"| {label} | 0/0 | pending |")

    a, b = by_arm["A"], by_arm["B"]
    add("max(A,B) on A/B pairs", [True] * max(sum(a[t] for t in shared), sum(b[t] for t in shared))
        + [False] * (len(shared) - max(sum(a[t] for t in shared), sum(b[t] for t in shared))))
    add("Oracle (A or B)", [a[t] or b[t] for t in shared])
    for arm in "CDEF":
        paired = shared & by_arm[arm].keys()
        add(f"{arm} (paired with A and B)", [by_arm[arm][t] for t in paired])
    complete = set.intersection(*(set(by_arm[arm]) for arm in ARMS))
    lines += ["", "Primary comparisons below use the same tasks completed in all six arms.", "",
              "| Comparison | Passes / complete n | Rate |", "|---|---:|---:|"]
    best = max(sum(a[t] for t in complete), sum(b[t] for t in complete))
    add("max(A,B), complete cohort", [True] * best + [False] * (len(complete) - best))
    for arm in "CDEF":
        add(f"{arm}, complete cohort", [by_arm[arm][t] for t in complete])
    lines += ["", f"A/B failure overlap on {len(shared)} paired tasks: "
              f"both fail {sum(not a[t] and not b[t] for t in shared)}; "
              f"only A fails {sum(not a[t] and b[t] for t in shared)}; "
              f"only B fails {sum(a[t] and not b[t] for t in shared)}.",
              "Costs are API-equivalent estimates; subscription users pay a flat fee. "
              "Unknown call costs make the entire arm cost unknown."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(report(load_rows(args.results)))


if __name__ == "__main__":
    main()

