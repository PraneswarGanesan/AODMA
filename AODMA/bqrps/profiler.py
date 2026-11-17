
from dataclasses import dataclass
import math
import statistics

from bqrps.portfolio import CandidateConfig, CandidateProfile


@dataclass
class MeasurementSummary:
    mean: float
    std: float
    minimum: float
    maximum: float
    count: int


def summarize(values: list[float]) -> MeasurementSummary:
    if not values:
        raise ValueError("Cannot summarize empty measurements.")

    if not all(math.isfinite(x) for x in values):
        raise ValueError("Measurements must be finite.")

    return MeasurementSummary(
        mean=statistics.mean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
        count=len(values),
    )


def build_profile(
    config: CandidateConfig,
    quality_scores: list[float],
    latency_ms: list[float],
    peak_memory_mb: float,
    actual_nfe: int,
    metadata: dict | None = None,
) -> CandidateProfile:
    """
    Construct a candidate profile from real measurements.
    """

    summarize(quality_scores)
    summarize(latency_ms)

    return CandidateProfile(
        config=config,
        quality_scores=quality_scores,
        latency_ms=latency_ms,
        peak_memory_mb=peak_memory_mb,
        actual_nfe=actual_nfe,
        metadata=metadata or {},
    )
