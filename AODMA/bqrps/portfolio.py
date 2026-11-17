
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateConfig:
    """
    One diffusion inference configuration.
    """

    sampler: str
    schedule: str
    nfe: int
    precision: str = "fp32"

    def __post_init__(self):
        if not self.sampler:
            raise ValueError("Sampler cannot be empty.")

        if not self.schedule:
            raise ValueError("Schedule cannot be empty.")

        if self.nfe <= 0:
            raise ValueError("NFE must be positive.")

        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Unsupported precision.")

    @property
    def config_id(self) -> str:
        return (
            f"{self.sampler}_{self.schedule}_"
            f"nfe{self.nfe}_{self.precision}"
        )


@dataclass
class CandidateProfile:
    """
    Actual measurements collected for one configuration.

    Quality scores should all use the same metric and evaluation
    protocol, such as FID. Lower is better.
    """

    config: CandidateConfig
    quality_scores: list[float]
    latency_ms: list[float]
    peak_memory_mb: float
    actual_nfe: int

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.quality_scores:
            raise ValueError("At least one quality measurement is required.")

        if not self.latency_ms:
            raise ValueError("At least one latency measurement is required.")

        if any(x < 0 for x in self.quality_scores):
            raise ValueError("Quality scores cannot be negative.")

        if any(x <= 0 for x in self.latency_ms):
            raise ValueError("Latency measurements must be positive.")

        if self.peak_memory_mb <= 0:
            raise ValueError("Memory measurement must be positive.")

        if self.actual_nfe <= 0:
            raise ValueError("Actual NFE must be positive.")

    @property
    def config_id(self):
        return self.config.config_id


class CandidatePortfolio:
    """
    Registry of experimentally profiled configurations.
    """

    def __init__(self):
        self._candidates: dict[str, CandidateProfile] = {}

    def add(self, profile: CandidateProfile):
        key = profile.config_id

        if key in self._candidates:
            raise ValueError(f"Duplicate candidate: {key}")

        self._candidates[key] = profile

    def get(self, config_id: str) -> CandidateProfile:
        return self._candidates[config_id]

    def all(self) -> list[CandidateProfile]:
        return list(self._candidates.values())

    def __len__(self):
        return len(self._candidates)

    def __iter__(self):
        return iter(self._candidates.values())
