
"""Baseline selection methods for the AODMA BQRPS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bqrps.uncertainty import (
    estimate_latency_risk,
    estimate_quality_risk,
)


@dataclass(frozen=True)
class Budget:
    max_latency_ms: float
    max_memory_mb: float
    max_nfe: int


def assess_candidate(
    candidate: dict[str, Any],
    budget: Budget,
    *,
    use_latency_uncertainty: bool = True,
    quality_safety_margin: float = 0.0,
) -> dict[str, Any]:
    latency_values = candidate["latency_ms"]
    fid = float(candidate["fid"])
    memory = float(candidate["peak_memory_mb"])
    nfe = int(candidate["actual_nfe"])

    latency_risk = estimate_latency_risk(latency_values)

    if use_latency_uncertainty:
        latency_for_budget = latency_risk.upper_risk
    else:
        latency_for_budget = float(candidate["latency_mean_ms"])

    quality_risk = estimate_quality_risk(
        [fid],
        safety_margin=quality_safety_margin,
    )

    reasons = []

    if latency_for_budget > budget.max_latency_ms:
        reasons.append("latency_budget_exceeded")

    if memory > budget.max_memory_mb:
        reasons.append("memory_budget_exceeded")

    if nfe > budget.max_nfe:
        reasons.append("nfe_budget_exceeded")

    return {
        "config_id": candidate["config_id"],
        "sampler": candidate["sampler"],
        "nfe": nfe,
        "fid": fid,
        "quality_risk": quality_risk.upper_risk,
        "quality_uncertainty_empirical": (
            quality_risk.empirically_estimated
        ),
        "latency_mean_ms": float(candidate["latency_mean_ms"]),
        "latency_upper_risk_ms": latency_risk.upper_risk,
        "latency_used_for_budget_ms": latency_for_budget,
        "latency_uncertainty_source": latency_risk.uncertainty_source,
        "memory_mb": memory,
        "feasible": len(reasons) == 0,
        "reasons": reasons,
    }


def select_candidate(
    candidates: list[dict[str, Any]],
    budget: Budget,
    *,
    method: str = "bqrps",
    use_latency_uncertainty: bool = True,
    quality_safety_margin: float = 0.0,
) -> dict[str, Any]:
    """Select one candidate using a baseline or BQRPS policy."""

    valid_methods = {"bqrps", "best_fid", "fastest", "lowest_nfe"}

    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {sorted(valid_methods)}."
        )

    assessments = [
        assess_candidate(
            candidate,
            budget,
            use_latency_uncertainty=use_latency_uncertainty,
            quality_safety_margin=quality_safety_margin,
        )
        for candidate in candidates
    ]

    feasible = [item for item in assessments if item["feasible"]]

    if not feasible:
        return {
            "method": method,
            "selected": None,
            "feasible_count": 0,
            "assessments": assessments,
        }

    if method in {"bqrps", "best_fid"}:
        key = lambda item: (
            item["quality_risk"] if method == "bqrps" else item["fid"],
            item["latency_upper_risk_ms"],
            item["nfe"],
        )
    elif method == "fastest":
        key = lambda item: (
            item["latency_mean_ms"],
            item["fid"],
            item["nfe"],
        )
    else:
        key = lambda item: (
            item["nfe"],
            item["fid"],
            item["latency_mean_ms"],
        )

    selected = min(feasible, key=key)

    return {
        "method": method,
        "selected": selected,
        "feasible_count": len(feasible),
        "assessments": assessments,
    }
