
"""Latency-budget sensitivity analysis for AODMA BQRPS."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from bqrps.baselines import Budget, select_candidate


METHODS = ["bqrps", "best_fid", "fastest", "lowest_nfe"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_sensitivity(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    sensitivity = config["sensitivity"]
    memory_budget = float(sensitivity["max_memory_mb"])
    nfe_budget = int(sensitivity["max_nfe"])
    quality_margin = float(
        config.get("selection", {}).get("quality_safety_margin", 0.0)
    )

    results = []

    for latency_budget in sensitivity["latency_budgets_ms"]:
        budget = Budget(
            max_latency_ms=float(latency_budget),
            max_memory_mb=memory_budget,
            max_nfe=nfe_budget,
        )

        for method in METHODS:
            selection = select_candidate(
                candidates,
                budget,
                method=method,
                use_latency_uncertainty=True,
                quality_safety_margin=quality_margin,
            )

            selected = selection["selected"]

            row = {
                "latency_budget_ms": float(latency_budget),
                "method": method,
                "selected_candidate": (
                    selected["config_id"] if selected else None
                ),
                "sampler": selected["sampler"] if selected else None,
                "selected_nfe": selected["nfe"] if selected else None,
                "fid": selected["fid"] if selected else None,
                "latency_mean_ms": (
                    selected["latency_mean_ms"] if selected else None
                ),
                "latency_upper_risk_ms": (
                    selected["latency_upper_risk_ms"]
                    if selected else None
                ),
                "memory_mb": (
                    selected["memory_mb"] if selected else None
                ),
                "feasible_candidates": selection["feasible_count"],
                "total_candidates": len(candidates),
                "status": (
                    "selected" if selected else "no_feasible_candidate"
                ),
            }
            results.append(row)

    return results


def print_results(results: list[dict[str, Any]]) -> None:
    print("\nAODMA — LATENCY BUDGET SENSITIVITY")
    print("=" * 120)
    print(
        f"{'Budget(ms)':<13}"
        f"{'Method':<15}"
        f"{'Selected candidate':<36}"
        f"{'FID':>10}"
        f"{'Latency':>12}"
        f"{'NFE':>7}"
        f"{'Feasible':>11}"
    )
    print("-" * 120)

    for row in results:
        candidate = row["selected_candidate"] or "NONE"
        fid = f"{row['fid']:.3f}" if row["fid"] is not None else "-"
        latency = (
            f"{row['latency_mean_ms']:.2f}"
            if row["latency_mean_ms"] is not None else "-"
        )
        nfe = (
            str(row["selected_nfe"])
            if row["selected_nfe"] is not None else "-"
        )

        print(
            f"{row['latency_budget_ms']:<13.0f}"
            f"{row['method']:<15}"
            f"{candidate:<36}"
            f"{fid:>10}"
            f"{latency:>12}"
            f"{nfe:>7}"
            f"{row['feasible_candidates']:>6}/"
            f"{row['total_candidates']:<5}"
        )

    print("=" * 120)


def save_csv(results: list[dict[str, Any]], path: Path) -> None:
    if not results:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BQRPS latency-budget sensitivity analysis."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/bqrps_budget_sensitivity.yaml",
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    portfolio_path = Path(config["portfolio"]["path"])
    portfolio = load_json(portfolio_path)
    candidates = portfolio["candidates"]

    if not candidates:
        raise ValueError("Portfolio contains no candidates.")

    expected_count = portfolio.get("candidate_count")
    if expected_count is not None and expected_count != len(candidates):
        raise ValueError(
            f"Portfolio count mismatch: expected {expected_count}, "
            f"found {len(candidates)}."
        )

    results = run_sensitivity(candidates, config)
    print_results(results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("results/cifar10/bqrps/sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"budget_sensitivity_{timestamp}.json"
    csv_path = output_dir / f"budget_sensitivity_{timestamp}.csv"

    report = {
        "experiment": config["experiment"]["name"],
        "dataset": portfolio.get("dataset", "CIFAR-10"),
        "quality_metric": portfolio.get("quality_metric", "FID"),
        "portfolio_path": str(portfolio_path),
        "candidate_count": len(candidates),
        "latency_budgets_ms": config["sensitivity"]["latency_budgets_ms"],
        "fixed_memory_budget_mb": config["sensitivity"]["max_memory_mb"],
        "fixed_nfe_budget": config["sensitivity"]["max_nfe"],
        "latency_risk_method": (
            "One-sided Student-t upper confidence bound on mean latency, "
            "estimated from three repetitions."
        ),
        "quality_uncertainty_note": (
            "One FID observation per candidate; quality uncertainty "
            "is not empirically estimated."
        ),
        "results": results,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    save_csv(results, csv_path)

    print(f"\nJSON saved: {json_path}")
    print(f"CSV saved:  {csv_path}")


if __name__ == "__main__":
    main()
