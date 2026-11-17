
import argparse
from pathlib import Path

import torch
import yaml

from diffusion.models import UNet
from diffusion.noise_schedule import NoiseSchedule
from diffusion.datasets import create_dataloader
from diffusion.training import DiffusionTrainer, set_seed


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/configs/ddpm_mnist.yaml"
    )
    parser.add_argument(
        "--resume",
        default=None
    )

    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config["experiment"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("AODMA - DIFFUSION TRAINING ENGINE")
    print("=" * 55)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2
            ),
            "GB"
        )

    dataset_config = config["dataset"]
    training_config = config["training"]
    model_config = config["model"]
    diffusion_config = config["diffusion"]

    dataloader, channels = create_dataloader(
        dataset_name=dataset_config["name"],
        data_dir=dataset_config["data_dir"],
        batch_size=training_config["batch_size"],
        image_size=dataset_config["image_size"],
        num_workers=training_config["num_workers"],
        download=dataset_config["download"],
    )

    model = UNet(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        base_channels=model_config["base_channels"],
        time_dim=model_config["time_dim"],
    )

    if channels != model_config["in_channels"]:
        raise ValueError(
            "Dataset channels do not match model configuration."
        )

    schedule = NoiseSchedule(
        timesteps=diffusion_config["timesteps"],
        schedule_type=diffusion_config["schedule_type"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
        device=device,
    )

    runtime_config = {
        **training_config,
        "checkpoint_dir": config["storage"]["checkpoint_dir"],
        "log_path": config["storage"]["log_path"],
    }

    trainer = DiffusionTrainer(
        model=model,
        schedule=schedule,
        config=runtime_config,
        device=device,
    )

    if args.resume:
        print("Resuming checkpoint:", args.resume)
        trainer.resume(args.resume)

    print("Dataset:", dataset_config["name"])
    print("Training images:", len(dataloader.dataset))
    print("Batch size:", training_config["batch_size"])
    print("Diffusion steps:", diffusion_config["timesteps"])
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    trainer.fit(dataloader)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()
