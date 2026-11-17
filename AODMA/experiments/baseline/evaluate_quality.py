
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusion.models import UNet
from diffusion.noise_schedule import NoiseSchedule

from samplers.ddpm import DDPMSampler
from samplers.ddim import DDIMSampler
from samplers.dpm_solver import DPMSolverSampler
from samplers.unipc import UniPCSampler

from evaluation.quality_metrics import QualityEvaluator
from evaluation.reproducibility import set_global_seed


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(checkpoint_path, config, device):
    settings = config["model"]

    model = UNet(
        in_channels=settings["in_channels"],
        out_channels=settings["out_channels"],
        base_channels=settings["base_channels"],
        time_dim=settings["time_dim"],
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["ema"])
    model.eval()

    return model


def create_real_loader(config, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    dataset = datasets.CIFAR10(
        root=config["dataset"]["data_dir"],
        train=False,
        download=config["dataset"].get("download", True),
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def create_schedule(config, device):
    settings = config["diffusion"]

    return NoiseSchedule(
        timesteps=settings["timesteps"],
        schedule_type=settings["schedule_type"],
        beta_start=settings["beta_start"],
        beta_end=settings["beta_end"],
        device=device,
    )


def create_sampler(name, model, schedule):
    samplers = {
        "ddpm": DDPMSampler,
        "ddim": DDIMSampler,
        "dpm_solver": DPMSolverSampler,
        "unipc": UniPCSampler,
    }

    if name not in samplers:
        raise ValueError(f"Unsupported sampler: {name}")

    return samplers[name](model, schedule)


def build_candidates(sampler_names, step_counts):
    candidates = []

    for sampler_name in sampler_names:
        if sampler_name == "ddpm":
            candidates.append({
                "name": "ddpm_1000",
                "sampler": "ddpm",
                "steps": 1000,
            })
            continue

        for step_count in step_counts:
            candidates.append({
                "name": f"{sampler_name}_{step_count}",
                "sampler": sampler_name,
                "steps": step_count,
            })

    names = [candidate["name"] for candidate in candidates]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate candidate configurations detected.")

    return candidates


def generate_batch(
    sampler,
    candidate,
    batch_size,
    image_size,
    channels,
    device,
    seed,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    shape = (
        batch_size,
        channels,
        image_size,
        image_size,
    )

    if candidate["sampler"] == "ddpm":
        return sampler.sample(
            shape=shape,
            device=device,
            progress=False,
            generator=generator,
        )

    return sampler.sample(
        shape=shape,
        device=device,
        inference_steps=int(candidate["steps"]),
        progress=False,
        generator=generator,
    )


def evaluate_candidate(
    candidate,
    model,
    schedule,
    loader,
    args,
    config,
    device,
    seed,
):
    evaluator = QualityEvaluator(
        device=device,
        kid_subset_size=1000,
    )

    real_count = 0

    # Fixed CIFAR-10 test reference across all evaluation seeds.
    for images, _ in loader:
        if real_count >= args.samples:
            break

        remaining = args.samples - real_count
        images = images[:remaining].to(device)

        evaluator.update_real(images)
        real_count += images.shape[0]

    if real_count != args.samples:
        raise RuntimeError(
            "Not enough real images for the requested evaluation."
        )

    sampler = create_sampler(
        candidate["sampler"],
        model,
        schedule,
    )

    generated_count = 0
    batch_index = 0

    while generated_count < args.samples:
        current_batch = min(
            args.batch_size,
            args.samples - generated_count,
        )

        batch_seed = seed + batch_index

        generated = generate_batch(
            sampler=sampler,
            candidate=candidate,
            batch_size=current_batch,
            image_size=config["dataset"]["image_size"],
            channels=config["model"]["out_channels"],
            device=device,
            seed=batch_seed,
        )

        evaluator.update_fake(generated)

        generated_count += current_batch
        batch_index += 1

        print(
            f"\rGenerated {generated_count}/{args.samples}",
            end="",
            flush=True,
        )

    print()

    scores = evaluator.compute()

    result = {
        "candidate": candidate,
        "seed": seed,
        "sample_count": args.samples,
        "real_reference": "CIFAR-10 test split",
        "fid": float(scores["fid"]),
        "kid_mean": float(scores["kid"]["mean"]),
        "kid_std": (
            float(scores["kid"]["std"])
            if scores["kid"]["std"] is not None
            else None
        ),
    }

    del evaluator
    del sampler

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Reproducible AODMA CIFAR-10 quality evaluation."
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ddpm_cifar10/latest.pt",
    )

    parser.add_argument(
        "--config",
        default="experiments/configs/ddpm_cifar10.yaml",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["ddim", "dpm_solver", "unipc"],
        choices=["ddpm", "ddim", "dpm_solver", "unipc"],
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[10, 20, 50],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Evaluation seed. Defaults to the experiment config seed.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/cifar10/quality",
    )

    args = parser.parse_args()

    if args.samples < 1000:
        raise ValueError(
            "Use at least 1000 samples for this KID configuration."
        )

    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")

    if any(step < 1 for step in args.steps):
        raise ValueError("Inference steps must be positive.")

    config = load_yaml(args.config)

    seed = (
        int(args.seed)
        if args.seed is not None
        else int(config["experiment"]["seed"])
    )

    set_global_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("AODMA CIFAR-10 QUALITY EVALUATION")
    print("=" * 60)
    print("Device:", device)
    print("Seed:", seed)
    print("Evaluation samples:", args.samples)
    print("Batch size:", args.batch_size)

    model = load_model(
        args.checkpoint,
        config,
        device,
    )

    schedule = create_schedule(config, device)

    loader = create_real_loader(
        config,
        args.batch_size,
    )

    candidates = build_candidates(
        args.samplers,
        args.steps,
    )

    results = []

    for candidate in candidates:
        print("\nEvaluating:", candidate["name"])

        result = evaluate_candidate(
            candidate=candidate,
            model=model,
            schedule=schedule,
            loader=loader,
            args=args,
            config=config,
            device=device,
            seed=seed,
        )

        results.append(result)

        print("FID:", result["fid"])
        print("KID mean:", result["kid_mean"])
        print("KID std:", result["kid_std"])

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"quality_seed{seed}_{timestamp}.json"
    )

    report = {
        "experiment": "aodma_cifar10_quality_repeated_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "sample_count": args.samples,
        "batch_size": args.batch_size,
        "real_reference": "CIFAR-10 test split",
        "metrics": {
            "fid": "Frechet Inception Distance",
            "kid": "Kernel Inception Distance",
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("QUALITY EVALUATION COMPLETED")
    print("Seed:", seed)
    print("Candidates:", len(results))
    print("Report:", output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
