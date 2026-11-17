
from bqrps import (
    CandidateConfig,
    CandidatePortfolio,
    build_profile,
    BQRPSSelector,
    SelectionConstraints,
    ResidualCalibrator,
)


def main():

    portfolio = CandidatePortfolio()

    # Illustrative candidate measurements.
    candidates = [
        {
            "sampler": "ddim",
            "schedule": "linear",
            "nfe": 20,
            "quality": [18.5, 18.8, 18.4],
            "latency": [420, 430, 415],
            "memory": 1200,
        },
        {
            "sampler": "ddim",
            "schedule": "linear",
            "nfe": 30,
            "quality": [16.2, 16.5, 16.1],
            "latency": [590, 600, 580],
            "memory": 1400,
        },
        {
            "sampler": "dpm_solver",
            "schedule": "linear",
            "nfe": 20,
            "quality": [15.9, 16.1, 16.0],
            "latency": [460, 470, 455],
            "memory": 1300,
        },
    ]

    for item in candidates:

        config = CandidateConfig(
            sampler=item["sampler"],
            schedule=item["schedule"],
            nfe=item["nfe"],
            precision="fp32",
        )

        profile = build_profile(
            config=config,
            quality_scores=item["quality"],
            latency_ms=item["latency"],
            peak_memory_mb=item["memory"],
            actual_nfe=item["nfe"],
        )

        portfolio.add(profile)

    # Held-out calibration observations.
    calibrator = ResidualCalibrator(
        confidence_level=0.90
    )

    calibrator.fit(
        predicted_quality=[16.0, 17.0, 18.0, 19.0],
        observed_quality=[16.4, 17.2, 18.5, 19.3],
        predicted_latency=[400, 450, 500, 550],
        observed_latency=[410, 465, 510, 565],
    )

    constraints = SelectionConstraints(
        max_latency_ms=550,
        max_memory_mb=2000,
        max_nfe=30,
    )

    selector = BQRPSSelector(
        beta_quality=1.0,
        beta_latency=1.0,
    )

    result = selector.select(
        portfolio=portfolio,
        constraints=constraints,
        calibrator=calibrator,
    )

    print("\n========== AODMA / BQRPS ==========")

    print("Selected configuration:")
    print(result.selected.config_id)

    print(
        "Quality risk:",
        round(result.assessment.quality_risk.upper_bound, 4),
    )

    print(
        "Latency risk:",
        round(result.assessment.latency_risk.upper_bound, 2),
        "ms",
    )

    print("Calibrated:", result.calibrated)

    print("\nCandidate assessments:")

    for assessment in result.assessments:
        print(
            assessment.config_id,
            "| Feasible:", assessment.feasible,
            "| Quality risk:",
            round(assessment.quality_risk.upper_bound, 3),
            "| Reasons:",
            assessment.rejection_reasons,
        )


if __name__ == "__main__":
    main()
