
"""Run BQRPS baselines and resource-constraint ablations."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from bqrps.baselines import Budget, select_candidate


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_method(
    candidates: list[dict[str, Any]],
    *,
    method: str,
    budget: Budget,
    use_latency_uncertainty: bool,
    quality_safety_margin: float,
) -> dict[str, Any]:
    result = select_candidate(
        candidates,
        budget,
        method=method,
        use_latency_uncertainty=use_latency_uncertainty,
        quality_safety_margin=quality_safety_margin,
    )

    selected = result["selected"]

    return {
        "experiment_type": "baseline",
        "name": method,
        "method": method,
        "selected_candidate": (
            selected["config_id"] if selected else None
        ),
        "sampler": selected["sampler"] if selected else None,
        "nfe": selected["nfe"] if selected else None,
        "fid": selected["fid"] if selected else None,
        "latency_mean_ms": (
            selected["latency_mean_ms"] if selected else None
        ),
        "latency_upper_risk_ms": (
            selected["latency_upper_risk_ms"] if selected else None
        ),
        "memory_mb": selected["memory_mb"] if selected else None,
        "feasible_candidates": result["feasible_count"],
        "total_candidates": len(candidates),
        "status": "selected" if selected else "no_feasible_candidate",
    }


def run_ablation(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    ablation: dict[str, Any],
) -> dict[str, Any]:
    budget = Budget(
        max_latency_ms=float(ablation["max_latency_ms"]),
        max_memory_mb=float(ablation["max_memory_mb"]),
        max_nfe=int(ablation["max_nfe"]),
    )

    result = select_candidate(
        candidates,
        budget,
        method="bqrps",
        use_latency_uncertainty=bool(
            ablation["use_latency_uncertainty"]
        ),
        quality_safety_margin=float(
            ablation.get(
                "quality_safety_margin",
                config["selection"].get("quality_safety_margin", 0.0),
            )
        ),
    )

    selected = result["selected"]

    return {
        "experiment_type": "ablation",
        "name": ablation["name"],
        "method": "bqrps",
        "selected_candidate": (
            selected["config_id"] if selected else None
        ),
        "sampler": selected["sampler"] if selected else None,
        "nfe": selected["nfe"] if selected else None,
        "fid": selected["fid"] if selected else None,
        "latency_mean_ms": (
            selected["latency_mean_ms"] if selected else None
        ),
        "latency_upper_risk_ms": (
            selected["latency_upper_risk_ms"] if selected else None
        ),
        "memory_mb": selected["memory_mb"] if selected else None,
        "feasible_candidates": result["feasible_count"],
        "total_candidates": len(candidates),
        "status": "selected" if selected else "no_feasible_candidate",
        "settings": {
            "use_latency_uncertainty": bool(
                ablation["use_latency_uncertainty"]
            ),
            "max_latency_ms": budget.max_latency_ms,
            "max_memory_mb": budget.max_memory_mb,
            "max_nfe": budget.max_nfe,
        },
    }


def print_results(results: list[dict[str, Any]]) -> None:
    print("\nAODMA — BQRPS BASELINES AND ABLATIONS")
    print("=" * 112)

    header = (
        f"{'Experiment':<27}"
        f"{'Selected candidate':<36}"
        f"{'FID':>10}"
        f"{'Latency':>12}"
        f"{'NFE':>8}"
        f"{'Feasible':>11}"
    )
    print(header)
    print("-" * 112)

    for row in results:
        candidate = row["selected_candidate"] or "NONE"
        fid = (
            f"{row['fid']:.3f}"
            if row["fid"] is not None else "-"
        )
        latency = (
            f"{row['latency_mean_ms']:.2f}"
            if row["latency_mean_ms"] is not None else "-"
        )
        nfe = str(row["nfe"]) if row["nfe"] is not None else "-"

        print(
            f"{row['name']:<27}"
            f"{candidate:<36}"
            f"{fid:>10}"
            f"{latency:>12}"
            f"{nfe:>8}"
            f"{row['feasible_candidates']:>6}/"
            f"{row['total_candidates']:<5}"
        )

    print("=" * 112)


def save_csv(results: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "experiment_type",
        "name",
        "method",
        "selected_candidate",
        "sampler",
        "nfe",
        "fid",
        "latency_mean_ms",
        "latency_upper_risk_ms",
        "memory_mb",
        "feasible_candidates",
        "total_candidates",
        "status",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AODMA BQRPS baseline and ablation experiments."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/bqrps_ablation.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)

    portfolio_path = Path(config["portfolio"]["path"])
    portfolio = load_json(portfolio_path)
    candidates = portfolio["candidates"]

    if not candidates:
        raise ValueError("The unified portfolio contains no candidates.")

    expected_count = portfolio.get("candidate_count")
    if expected_count is not None and expected_count != len(candidates):
        raise ValueError(
            f"Portfolio count mismatch: expected {expected_count}, "
            f"found {len(candidates)}."
        )

    base_budget = Budget(
        max_latency_ms=float(config["budget"]["max_latency_ms"]),
        max_memory_mb=float(config["budget"]["max_memory_mb"]),
        max_nfe=int(config["budget"]["max_nfe"]),
    )

    quality_margin = float(
        config["selection"].get("quality_safety_margin", 0.0)
    )

    results: list[dict[str, Any]] = []

    for method in config["methods"]:
        results.append(
            run_method(
                candidates,
                method=method,
                budget=base_budget,
                use_latency_uncertainty=True,
                quality_safety_margin=quality_margin,
            )
        )

    for ablation in config["ablations"]:
        results.append(
            run_ablation(candidates, config, ablation)
        )

    print_results(results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("results/cifar10/bqrps/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"bqrps_ablation_{timestamp}.json"
    output_csv = output_dir / f"bqrps_ablation_{timestamp}.csv"

    report = {
        "experiment": config["experiment"]["name"],
        "dataset": portfolio.get("dataset", "CIFAR-10"),
        "quality_metric": portfolio.get("quality_metric", "FID"),
        "portfolio_path": str(portfolio_path),
        "portfolio_candidate_count": len(candidates),
        "base_budget": {
            "max_latency_ms": base_budget.max_latency_ms,
            "max_memory_mb": base_budget.max_memory_mb,
            "max_nfe": base_budget.max_nfe,
        },
        "quality_uncertainty_note": (
            "Only one FID observation is available per candidate. "
            "FID uncertainty is not empirically estimated."
        ),
        "latency_uncertainty_note": (
            "Latency risk uses a one-sided Student-t estimate from "
            "three benchmark repetitions."
        ),
        "results": results,
    }

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    save_csv(results, output_csv)

    print(f"\nJSON saved: {output_json}")
    print(f"CSV saved:  {output_csv}")


if __name__ == "__main__":
    main()
