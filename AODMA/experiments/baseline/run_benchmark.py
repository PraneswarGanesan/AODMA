
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from diffusion.models import UNet
from diffusion.noise_schedule import NoiseSchedule

from evaluation.benchmark import BenchmarkEngine
from evaluation.reproducibility import (
    set_global_seed,
    collect_environment_metadata,
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(checkpoint_path, model_config, device):
    model = UNet(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        base_channels=model_config["base_channels"],
        time_dim=model_config["time_dim"],
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "ema" not in checkpoint:
        raise KeyError("Checkpoint does not contain EMA weights.")

    model.load_state_dict(checkpoint["ema"])
    model.eval()

    return model


def flatten_result(result):
    latency = result["latency"]
    memory = result["memory"]

    return {
        "candidate_name": result["candidate"].get("name"),
        "sampler": result["sampler"],
        "steps_requested": result["steps_requested"],
        "eta": result["eta"],
        "batch_size": result["batch_size"],
        "repetitions": result["repetitions"],
        "nfe_mean": result["nfe_per_sample"]["mean"],
        "latency_mean_ms": latency["mean"],
        "latency_median_ms": latency["median"],
        "latency_std_ms": latency["std"],
        "latency_p95_ms": latency["p95"],
        "latency_ci95_lower_ms": (
            latency["confidence_interval"]["lower"]
        ),
        "latency_ci95_upper_ms": (
            latency["confidence_interval"]["upper"]
        ),
        "throughput_images_per_second": (
            result["throughput_images_per_second"]
        ),
        "peak_allocated_mb": memory["peak_allocated_mb"],
        "peak_reserved_mb": memory["peak_reserved_mb"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="AODMA reproducible sampler benchmark."
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ddpm_mnist/latest.pt",
    )

    parser.add_argument(
        "--model-config",
        default="experiments/configs/ddpm_mnist.yaml",
    )

    parser.add_argument(
        "--benchmark-config",
        default="experiments/configs/benchmark.yaml",
    )

    args = parser.parse_args()

    model_config_file = load_yaml(args.model_config)
    benchmark_config_file = load_yaml(args.benchmark_config)

    experiment = benchmark_config_file["experiment"]
    benchmark_config = benchmark_config_file["benchmark"]

    seed = int(experiment["seed"])

    set_global_seed(
        seed,
        deterministic=bool(experiment.get("deterministic", False)),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("AODMA REPRODUCIBLE BENCHMARK")
    print("=" * 60)
    print("Experiment:", experiment["name"])
    print("Device:", device)

    model_settings = model_config_file["model"]
    diffusion_settings = model_config_file["diffusion"]
    dataset_settings = model_config_file["dataset"]

    model = load_model(
        args.checkpoint,
        model_settings,
        device,
    )

    schedule = NoiseSchedule(
        timesteps=diffusion_settings["timesteps"],
        schedule_type=diffusion_settings["schedule_type"],
        beta_start=diffusion_settings["beta_start"],
        beta_end=diffusion_settings["beta_end"],
        device=device,
    )

    engine = BenchmarkEngine(
        model=model,
        schedule=schedule,
        device=device,
        benchmark_config=benchmark_config,
    )

    batch_size = int(benchmark_config["batch_size"])

    shape = (
        batch_size,
        model_settings["out_channels"],
        dataset_settings["image_size"],
        dataset_settings["image_size"],
    )

    output_root = Path(benchmark_config["output_dir"])

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = output_root / f"{experiment['name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for candidate in benchmark_config["candidates"]:
        candidate_name = candidate.get(
            "name",
            candidate["sampler"],
        )

        print("\nRunning:", candidate_name)

        sample_path = (
            run_dir / "samples" / f"{candidate_name}.png"
        )

        result = engine.run_candidate(
            candidate=candidate,
            shape=shape,
            seed=seed,
            save_samples=bool(
                benchmark_config.get("save_samples", False)
            ),
            sample_output_path=sample_path,
        )

        result["candidate"]["name"] = candidate_name

        all_results.append(result)

        print(
            f"Latency: {result['latency']['mean']:.3f} ms | "
            f"NFE: {result['nfe_per_sample']['mean']:.0f} | "
            f"Throughput: "
            f"{result['throughput_images_per_second']:.3f} images/s"
        )

    metadata = collect_environment_metadata()

    metadata["experiment_name"] = experiment["name"]
    metadata["seed"] = seed
    metadata["checkpoint"] = str(args.checkpoint)
    metadata["model_config"] = str(args.model_config)
    metadata["benchmark_config"] = str(args.benchmark_config)
    metadata["image_shape"] = list(shape)

    complete_results = {
        "metadata": metadata,
        "configuration": benchmark_config_file,
        "results": all_results,
    }

    json_path = run_dir / "benchmark_results.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            complete_results,
            file,
            indent=4,
        )

    csv_path = run_dir / "benchmark_results.csv"

    rows = [
        flatten_result(result)
        for result in all_results
    ]

    if rows:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)
    print("JSON:", json_path)
    print("CSV:", csv_path)
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()
