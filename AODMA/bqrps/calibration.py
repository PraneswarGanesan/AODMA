
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    quality_offset: float
    latency_offset: float
    confidence_level: float
    calibration_samples: int


class ResidualCalibrator:
    """
    Split-conformal-style upper residual calibration.

    Inputs:
        predicted_quality: predicted quality metric values
        observed_quality: actual quality metric values
        predicted_latency: predicted latency values
        observed_latency: actual latency values

    Assumption:
        Quality and latency metrics are lower-is-better.

    Calibration data must be held out from model fitting.
    """

    def __init__(self, confidence_level: float = 0.90):
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")

        self.confidence_level = confidence_level
        self.result: CalibrationResult | None = None

    @staticmethod
    def _upper_residual_quantile(
        predicted: list[float],
        observed: list[float],
        confidence_level: float,
    ) -> float:

        if len(predicted) != len(observed):
            raise ValueError("Prediction and observation lengths differ.")

        if not predicted:
            raise ValueError("Calibration data cannot be empty.")

        residuals = np.asarray(observed) - np.asarray(predicted)

        if not np.isfinite(residuals).all():
            raise ValueError("Calibration data must be finite.")

        residuals = np.sort(residuals)

        n = len(residuals)

        # Finite-sample conformal quantile index.
        k = math.ceil((n + 1) * confidence_level)
        k = min(max(k, 1), n)

        # Non-negative correction avoids reducing the risk bound.
        return float(max(0.0, residuals[k - 1]))

    def fit(
        self,
        predicted_quality: list[float],
        observed_quality: list[float],
        predicted_latency: list[float],
        observed_latency: list[float],
    ) -> CalibrationResult:

        if len(predicted_quality) != len(predicted_latency):
            raise ValueError("Quality and latency calibration sizes differ.")

        quality_offset = self._upper_residual_quantile(
            predicted_quality,
            observed_quality,
            self.confidence_level,
        )

        latency_offset = self._upper_residual_quantile(
            predicted_latency,
            observed_latency,
            self.confidence_level,
        )

        self.result = CalibrationResult(
            quality_offset=quality_offset,
            latency_offset=latency_offset,
            confidence_level=self.confidence_level,
            calibration_samples=len(predicted_quality),
        )

        return self.result

    def get_offsets(self) -> tuple[float, float]:
        if self.result is None:
            raise RuntimeError("Calibrator has not been fitted.")

        return (
            self.result.quality_offset,
            self.result.latency_offset,
        )
# Research note: This calibration is only meaningful if the calibration observations are representative of future inference conditions. We will later evaluate empirical coverage across held-out experiments and datasets. It is not a guarantee of universal coverage under distribution shift.