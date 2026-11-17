
import argparse
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image

from diffusion.models import UNet
from diffusion.noise_schedule import NoiseSchedule
from samplers.ddpm import DDPMSampler
from samplers.ddim import DDIMSampler


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
        raise KeyError("EMA weights were not found in the checkpoint.")

    model.load_state_dict(checkpoint["ema"])
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples using AODMA diffusion samplers."
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ddpm_mnist/latest.pt",
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ddpm_mnist.yaml",
    )
    parser.add_argument(
        "--sampler",
        choices=["ddpm", "ddim"],
        default="ddim",
    )
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/ddpm_mnist/generated.png",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also save every generated image separately.",
    )

    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1.")

    if args.steps < 2:
        parser.error("--steps must be at least 2.")

    if args.eta < 0:
        parser.error("--eta cannot be negative.")

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Sampler:", args.sampler)

    model_config = config["model"]
    diffusion_config = config["diffusion"]
    dataset_config = config["dataset"]

    model = load_model(
        args.checkpoint,
        model_config,
        device,
    )

    schedule = NoiseSchedule(
        timesteps=diffusion_config["timesteps"],
        schedule_type=diffusion_config["schedule_type"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
        device=device,
    )

    if args.sampler == "ddpm":
        sampler = DDPMSampler(model, schedule)
        print(f"DDPM uses all {schedule.timesteps} training steps.")

    else:
        sampler = DDIMSampler(model, schedule)
        print("DDIM inference steps:", args.steps)
        print("DDIM eta:", args.eta)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    sample_shape = (
        args.count,
        model_config["out_channels"],
        dataset_config["image_size"],
        dataset_config["image_size"],
    )

    if args.sampler == "ddpm":
        samples = sampler.sample(
            shape=sample_shape,
            device=device,
            progress=True,
            generator=generator,
        )
    else:
        samples = sampler.sample(
            shape=sample_shape,
            device=device,
            inference_steps=args.steps,
            eta=args.eta,
            progress=True,
            generator=generator,
        )

    # Convert from [-1, 1] to [0, 1].
    samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save combined grid.
    grid_columns = max(1, int(args.count ** 0.5))

    save_image(
        samples,
        output_path,
        nrow=grid_columns,
    )

    print("Sample grid saved to:", output_path)

    # Optionally save each sample separately.
    if args.individual:
        individual_dir = output_path.parent / "individual_samples"
        individual_dir.mkdir(parents=True, exist_ok=True)

        for index, sample in enumerate(samples):
            sample_path = individual_dir / f"sample_{index + 1:03d}.png"
            save_image(sample, sample_path)

        print("Individual samples saved to:", individual_dir)

    print(f"Successfully generated {args.count} samples.")


if __name__ == "__main__":
    main()
