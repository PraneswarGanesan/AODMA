
"""Uncertainty-aware risk estimation for BQRPS.

Lower values are better for both FID and latency.
A single observation is not treated as an empirical uncertainty estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Iterable


# One-sided Student-t critical values for 95% confidence.
# Indexed by degrees of freedom (df).
_T_95_ONE_SIDED = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    11: 1.796,
    12: 1.782,
    13: 1.771,
    14: 1.761,
    15: 1.753,
    16: 1.746,
    17: 1.740,
    18: 1.734,
    19: 1.729,
    20: 1.725,
    25: 1.708,
    30: 1.697,
}


@dataclass(frozen=True)
class RiskEstimate:
    mean: float
    standard_deviation: float
    upper_risk: float
    sample_count: int
    confidence_level: float
    calibration_offset: float
    uncertainty_source: str
    empirically_estimated: bool


def _validate_values(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]

    if not result:
        raise ValueError("At least one observation is required.")

    if any(not math.isfinite(value) for value in result):
        raise ValueError("Observations must be finite numbers.")

    if any(value < 0 for value in result):
        raise ValueError("Risk observations must be non-negative.")

    return result


def _t_critical_95(df: int) -> float:
    if df in _T_95_ONE_SIDED:
        return _T_95_ONE_SIDED[df]

    if df > 30:
        return 1.645

    # Conservative fallback for degrees of freedom not in the table.
    nearest_df = min(_T_95_ONE_SIDED, key=lambda key: abs(key - df))
    return _T_95_ONE_SIDED[nearest_df]


def estimate_upper_risk(
    values: Iterable[float],
    *,
    confidence_level: float = 0.95,
    calibration_offset: float = 0.0,
    fallback_margin: float = 0.0,
) -> RiskEstimate:
    """Estimate a conservative upper risk value.

    For n >= 2:
        upper = mean + t_critical * standard_error + calibration_offset

    For n == 1:
        upper = observed_value + fallback_margin + calibration_offset

    The n == 1 result is a configured safety-margin estimate, not a
    statistically estimated confidence bound.
    """

    observations = _validate_values(values)

    if not 0.50 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.50 and 1.0.")

    if calibration_offset < 0:
        raise ValueError("calibration_offset must be non-negative.")

    if fallback_margin < 0:
        raise ValueError("fallback_margin must be non-negative.")

    sample_mean = mean(observations)
    count = len(observations)

    if count == 1:
        std_dev = 0.0
        upper = sample_mean + fallback_margin + calibration_offset
        source = "single_observation_plus_configured_margin"
        empirical = False
    else:
        std_dev = stdev(observations)

        if confidence_level == 0.95:
            critical_value = _t_critical_95(count - 1)
        else:
            # Until other confidence levels are explicitly implemented,
            # use the conservative 95% one-sided critical value.
            critical_value = _t_critical_95(count - 1)

        standard_error = std_dev / math.sqrt(count)
        upper = (
            sample_mean
            + critical_value * standard_error
            + calibration_offset
        )
        source = "one_sided_student_t_upper_bound"
        empirical = True

    return RiskEstimate(
        mean=sample_mean,
        standard_deviation=std_dev,
        upper_risk=upper,
        sample_count=count,
        confidence_level=confidence_level,
        calibration_offset=calibration_offset,
        uncertainty_source=source,
        empirically_estimated=empirical,
    )


def estimate_quality_risk(
    fid_values: Iterable[float],
    *,
    safety_margin: float = 0.0,
    calibration_offset: float = 0.0,
) -> RiskEstimate:
    """Estimate FID risk.

    FID is a distribution-level metric. One FID result cannot establish
    empirical quality variance. With one observation, use an explicit
    safety margin and mark the estimate as non-empirical.
    """

    return estimate_upper_risk(
        fid_values,
        confidence_level=0.95,
        calibration_offset=calibration_offset,
        fallback_margin=safety_margin,
    )


def estimate_latency_risk(
    latency_values_ms: Iterable[float],
    *,
    calibration_offset: float = 0.0,
) -> RiskEstimate:
    """Estimate latency risk from repeated latency measurements."""

    return estimate_upper_risk(
        latency_values_ms,
        confidence_level=0.95,
        calibration_offset=calibration_offset,
        fallback_margin=0.0,
    )
