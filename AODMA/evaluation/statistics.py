
import numpy as np


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        raise ValueError("Cannot summarize an empty collection.")

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def bootstrap_confidence_interval(
    values,
    confidence=0.95,
    bootstrap_iterations=5000,
    seed=42,
):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty collection.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    if bootstrap_iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(
        bootstrap_iterations,
        dtype=np.float64,
    )

    for index in range(bootstrap_iterations):
        sample = rng.choice(
            values,
            size=values.size,
            replace=True,
        )
        bootstrap_means[index] = np.mean(sample)

    alpha = 1.0 - confidence

    lower = np.quantile(bootstrap_means, alpha / 2)
    upper = np.quantile(bootstrap_means, 1 - alpha / 2)

    return {
        "confidence_level": confidence,
        "lower": float(lower),
        "upper": float(upper),
        "method": "percentile_bootstrap",
        "bootstrap_iterations": bootstrap_iterations,
    }


def summarize_latency(latencies_ms, seed=42):
    summary = summarize_values(latencies_ms)

    summary["unit"] = "milliseconds"

    summary["confidence_interval"] = (
        bootstrap_confidence_interval(
            latencies_ms,
            confidence=0.95,
            seed=seed,
        )
    )

    return summary
