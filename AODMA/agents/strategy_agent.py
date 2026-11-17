
from bqrps.baselines import Budget, select_candidate


def load_candidate_profiles(portfolio: dict) -> list[dict]:
    """
    Adapt the flat unified_portfolio.json schema
    to the dictionary schema expected by BQRPS.
    """
    raw_candidates = portfolio.get("candidates", [])

    if not raw_candidates:
        raise ValueError("Portfolio contains no candidates.")

    candidates = []

    required_fields = {
        "config_id",
        "sampler",
        "fid",
        "latency_ms",
        "latency_mean_ms",
        "peak_memory_mb",
        "actual_nfe",
    }

    for index, item in enumerate(raw_candidates):
        missing = required_fields - item.keys()

        if missing:
            raise ValueError(
                f"Candidate {index} is missing fields: {sorted(missing)}"
            )

        candidates.append({
            "config_id": str(item["config_id"]),
            "sampler": str(item["sampler"]),
            "schedule": str(item["schedule"]),
            "nfe": int(item["nfe"]),
            "precision": str(item["precision"]),
            "fid": float(item["fid"]),
            "kid_mean": float(item["kid_mean"]),
            "kid_std": float(item["kid_std"]),
            "latency_ms": [float(x) for x in item["latency_ms"]],
            "latency_mean_ms": float(item["latency_mean_ms"]),
            "peak_memory_mb": float(item["peak_memory_mb"]),
            "actual_nfe": int(item["actual_nfe"]),
        })

    return candidates


def strategy_agent(state: dict) -> dict:
    """Select a diffusion configuration using the BQRPS policy."""

    portfolio = state["portfolio"]
    candidates = load_candidate_profiles(portfolio)

    budget_data = state.get("budget", {})

    budget = Budget(
        max_latency_ms=float(budget_data.get("max_latency_ms", 350.0)),
        max_memory_mb=float(budget_data.get("max_memory_mb", 100.0)),
        max_nfe=int(budget_data.get("max_nfe", 50)),
    )

    selection = select_candidate(
        candidates,
        budget,
        method="bqrps",
        use_latency_uncertainty=True,
    )

    selected = selection.get("selected")

    if selected is None:
        raise RuntimeError(
            "BQRPS could not find a feasible candidate. "
            f"Budget: {budget}. "
            f"Assessments: {selection['assessments']}"
        )

    selected_candidate = {

        "config_id": selected["config_id"],
        "candidate_id": selected["config_id"],
        "sampler": selected["sampler"],
        "schedule": next(
            c["schedule"] for c in candidates
            if c["config_id"] == selected["config_id"]
        ),
        "nfe": selected["nfe"],
        "precision": next(
            c["precision"] for c in candidates
            if c["config_id"] == selected["config_id"]
        ),
        "fid": selected["fid"],
        "latency_mean_ms": selected["latency_mean_ms"],
        "latency_upper_risk_ms": selected["latency_upper_risk_ms"],
        "peak_memory_mb": selected["memory_mb"],
    }

    messages = list(state.get("messages", []))
    messages.append(
        f"BQRPS selected {selected_candidate['config_id']} "
        f"from {selection['feasible_count']} feasible candidates "
        f"out of {len(candidates)}."
    )

    return {
        "selected_candidate": selected_candidate,
        "selection_result": selection,
        "messages": messages,
        "status": "strategy_completed",
    }
