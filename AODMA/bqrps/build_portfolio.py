
import argparse
import json
from pathlib import Path

from bqrps.portfolio import (
    CandidateConfig,
    CandidateProfile,
    CandidatePortfolio,
)


def read_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def candidate_id(sampler, nfe):
    return f"{sampler.lower()}_{int(nfe)}"


def load_quality_records(report):
    records = {}

    for item in report["results"]:
        candidate = item["candidate"]
        sampler = candidate["sampler"].lower()
        requested_steps = int(candidate["steps"])

        # The current DDPM implementation always uses 1000 steps.
        if sampler == "ddpm":
            if requested_steps != 1000:
                continue
            nfe = 1000
        else:
            nfe = requested_steps

        key = candidate_id(sampler, nfe)

        if key in records:
            raise ValueError(f"Duplicate quality candidate: {key}")

        records[key] = {
            "fid": float(item["fid"]),
            "kid_mean": float(item["kid"]["mean"]),
            "kid_std": float(item["kid"]["std"]),
            "sample_count": int(item["sample_count"]),
            "source": item.get("real_reference"),
        }

    return records


def load_benchmark_records(report):
    records = {}

    for item in report["results"]:
        sampler = item["sampler"].lower()

        nfe = round(float(item["nfe_per_sample"]["mean"]))
        key = candidate_id(sampler, nfe)

        if key in records:
            raise ValueError(f"Duplicate benchmark candidate: {key}")

        latency = item["latency"]
        memory = item["memory"]

        raw_latency = item.get("raw_latency_ms", [])

        if not raw_latency:
            raise ValueError(
                f"Missing raw latency measurements for {key}"
            )

        records[key] = {
            "latency_ms": [float(x) for x in raw_latency],
            "latency_mean_ms": float(latency["mean"]),
            "latency_ci95": latency.get("confidence_interval"),
            "throughput": float(
                item["throughput_images_per_second"]
            ),
            "peak_allocated_mb": float(
                memory["peak_allocated_mb"]
            ),
            "peak_reserved_mb": float(
                memory["peak_reserved_mb"]
            ),
            "actual_nfe": nfe,
            "repetitions": int(item["repetitions"]),
            "batch_size": int(item["batch_size"]),
        }

    return records


def build_portfolio(quality_path, benchmark_path):
    quality_report = read_json(quality_path)
    benchmark_report = read_json(benchmark_path)

    quality_metadata = quality_report.get("metadata", {})
    benchmark_metadata = benchmark_report.get("metadata", {})

    # Prevent accidental cross-dataset/model merges.
    quality_checkpoint = quality_report.get("checkpoint")
    benchmark_checkpoint = benchmark_metadata.get("checkpoint")

    if quality_checkpoint and benchmark_checkpoint:
        quality_name = Path(quality_checkpoint).name
        benchmark_name = Path(benchmark_checkpoint).name

        if quality_name != benchmark_name:
            raise ValueError(
                "Checkpoint mismatch between quality and benchmark reports."
            )

    quality_records = load_quality_records(quality_report)
    benchmark_records = load_benchmark_records(benchmark_report)

    quality_ids = set(quality_records)
    benchmark_ids = set(benchmark_records)

    missing_benchmark = quality_ids - benchmark_ids
    missing_quality = benchmark_ids - quality_ids

    if missing_benchmark or missing_quality:
        raise ValueError(
            "Candidate mismatch.\n"
            f"Missing benchmark results: {sorted(missing_benchmark)}\n"
            f"Missing quality results: {sorted(missing_quality)}"
        )

    portfolio = CandidatePortfolio()

    for key in sorted(quality_ids):
        quality = quality_records[key]
        performance = benchmark_records[key]

        sampler, nfe_text = key.rsplit("_", 1)
        nfe = int(nfe_text)

        config = CandidateConfig(
            sampler=sampler,
            schedule="linear",
            nfe=nfe,
            precision="fp32",
        )

        profile = CandidateProfile(
            config=config,
            quality_scores=[quality["fid"]],
            latency_ms=performance["latency_ms"],
            peak_memory_mb=performance["peak_allocated_mb"],
            actual_nfe=performance["actual_nfe"],
            metadata={
                "dataset": "CIFAR-10",
                "quality_metric": "FID",
                "fid": quality["fid"],
                "kid_mean": quality["kid_mean"],
                "kid_std": quality["kid_std"],
                "quality_sample_count": quality["sample_count"],
                "real_reference": quality["source"],
                "latency_mean_ms": performance["latency_mean_ms"],
                "latency_ci95": performance["latency_ci95"],
                "throughput_images_per_second": performance["throughput"],
                "peak_allocated_mb": performance["peak_allocated_mb"],
                "peak_reserved_mb": performance["peak_reserved_mb"],
                "benchmark_repetitions": performance["repetitions"],
                "benchmark_batch_size": performance["batch_size"],
                "quality_source": str(quality_path),
                "benchmark_source": str(benchmark_path),
                "quality_observations": 1,
            },
        )

        portfolio.add(profile)

    return portfolio


def print_portfolio(portfolio):
    print("\nAODMA — UNIFIED BQRPS PORTFOLIO")
    print("=" * 95)

    print(
        f"{'Candidate':<22}"
        f"{'NFE':<8}"
        f"{'FID':<14}"
        f"{'Latency(ms)':<16}"
        f"{'Memory(MB)':<14}"
        f"{'Throughput':<14}"
    )

    print("-" * 95)

    for profile in portfolio:
        metadata = profile.metadata

        print(
            f"{profile.config_id:<22}"
            f"{profile.actual_nfe:<8}"
            f"{metadata['fid']:<14.4f}"
            f"{metadata['latency_mean_ms']:<16.3f}"
            f"{profile.peak_memory_mb:<14.2f}"
            f"{metadata['throughput_images_per_second']:<14.3f}"
        )

    print("-" * 95)
    print(f"Total candidates: {len(portfolio)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build the AODMA unified BQRPS portfolio."
    )

    parser.add_argument("--quality", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--output",
        default="results/cifar10/bqrps/unified_portfolio.json",
    )

    args = parser.parse_args()

    portfolio = build_portfolio(
        args.quality,
        args.benchmark,
    )

    print_portfolio(portfolio)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "aodma_bqrps_unified_portfolio_v1",
        "dataset": "CIFAR-10",
        "quality_metric": "FID",
        "candidate_count": len(portfolio),
        "uncertainty_note": (
            "One FID observation per candidate. "
            "Quality uncertainty is not yet statistically estimated."
        ),
        "candidates": [],
    }

    for profile in portfolio:
        output["candidates"].append({
            "config_id": profile.config_id,
            "sampler": profile.config.sampler,
            "schedule": profile.config.schedule,
            "nfe": profile.actual_nfe,
            "precision": profile.config.precision,
            "fid": profile.metadata["fid"],
            "kid_mean": profile.metadata["kid_mean"],
            "kid_std": profile.metadata["kid_std"],
            "latency_ms": profile.latency_ms,
            "latency_mean_ms": profile.metadata["latency_mean_ms"],
            "latency_ci95": profile.metadata["latency_ci95"],
            "peak_memory_mb": profile.peak_memory_mb,
            "peak_reserved_mb": profile.metadata["peak_reserved_mb"],
            "throughput": profile.metadata[
                "throughput_images_per_second"
            ],
            "actual_nfe": profile.actual_nfe,
            "benchmark_repetitions": profile.metadata[
                "benchmark_repetitions"
            ],
            "quality_sample_count": profile.metadata[
                "quality_sample_count"
            ],
        })

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\nPortfolio saved:", output_path)


if __name__ == "__main__":
    main()
