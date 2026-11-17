
"""Budget-aware candidate selection for BQRPS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bqrps.uncertainty import (
    RiskEstimate,
    estimate_latency_risk,
    estimate_quality_risk,
)


@dataclass(frozen=True)
class SelectionConstraints:
    max_latency_ms: float
    max_memory_mb: float
    max_nfe: int

    def __post_init__(self) -> None:
        if self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive.")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive.")
        if self.max_nfe <= 0:
            raise ValueError("max_nfe must be positive.")


@dataclass
class CandidateAssessment:
    candidate_id: str
    feasible: bool
    quality_risk: RiskEstimate
    latency_risk: RiskEstimate
    memory_mb: float
    actual_nfe: int
    reasons: list[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    selected_candidate: Any
    selected_assessment: CandidateAssessment
    assessments: list[CandidateAssessment]
    constraints: SelectionConstraints


class NoFeasibleCandidateError(RuntimeError):
    pass


class BQRPSSelector:
    def __init__(
        self,
        constraints: SelectionConstraints,
        *,
        quality_safety_margin: float = 0.0,
        quality_calibration_offset: float = 0.0,
        latency_calibration_offset: float = 0.0,
    ) -> None:
        if quality_safety_margin < 0:
            raise ValueError("quality_safety_margin must be non-negative.")

        self.constraints = constraints
        self.quality_safety_margin = quality_safety_margin
        self.quality_calibration_offset = quality_calibration_offset
        self.latency_calibration_offset = latency_calibration_offset

    def _assess(self, profile: Any) -> CandidateAssessment:
        config = profile.config
        candidate_id = getattr(profile, "config_id", None)

        if not candidate_id:
            candidate_id = (
                f"{config.sampler}_{config.schedule}_"
                f"nfe{config.nfe}_{config.precision}"
            )

        quality_values = list(profile.quality_scores)
        latency_values = list(profile.latency_ms)

        quality_risk = estimate_quality_risk(
            quality_values,
            safety_margin=self.quality_safety_margin,
            calibration_offset=self.quality_calibration_offset,
        )

        latency_risk = estimate_latency_risk(
            latency_values,
            calibration_offset=self.latency_calibration_offset,
        )

        memory_mb = float(profile.peak_memory_mb)
        actual_nfe = int(profile.actual_nfe)

        reasons: list[str] = []

        if latency_risk.upper_risk > self.constraints.max_latency_ms:
            reasons.append("Latency budget exceeded.")

        if memory_mb > self.constraints.max_memory_mb:
            reasons.append("Memory budget exceeded.")

        if actual_nfe > self.constraints.max_nfe:
            reasons.append("NFE budget exceeded.")

        return CandidateAssessment(
            candidate_id=candidate_id,
            feasible=len(reasons) == 0,
            quality_risk=quality_risk,
            latency_risk=latency_risk,
            memory_mb=memory_mb,
            actual_nfe=actual_nfe,
            reasons=reasons,
        )

    def select(self, portfolio: Any) -> SelectionResult:
        profiles = portfolio.all()

        if not profiles:
            raise ValueError("Cannot select from an empty portfolio.")

        assessments: list[CandidateAssessment] = []
        profile_by_id: dict[str, Any] = {}

        for profile in profiles:
            assessment = self._assess(profile)
            assessments.append(assessment)
            profile_by_id[assessment.candidate_id] = profile

        feasible = [item for item in assessments if item.feasible]

        if not feasible:
            details = "; ".join(
                f"{item.candidate_id}: {', '.join(item.reasons)}"
                for item in assessments
            )
            raise NoFeasibleCandidateError(
                "No candidate satisfies all constraints. " + details
            )

        # Primary objective: minimum conservative FID risk.
        # Tie-breakers: lower latency risk, then lower NFE.
        selected_assessment = min(
            feasible,
            key=lambda item: (
                item.quality_risk.upper_risk,
                item.latency_risk.upper_risk,
                item.actual_nfe,
            ),
        )

        return SelectionResult(
            selected_candidate=profile_by_id[
                selected_assessment.candidate_id
            ],
            selected_assessment=selected_assessment,
            assessments=assessments,
            constraints=self.constraints,
        )
