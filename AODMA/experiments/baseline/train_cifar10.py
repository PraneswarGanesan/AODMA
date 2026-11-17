
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusion.models import UNet
from diffusion.noise_schedule import NoiseSchedule


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()

        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        target = self.shadow.state_dict()

        for key, target_value in target.items():
            source_value = source[key].detach()

            if target_value.is_floating_point():
                target_value.mul_(self.decay).add_(
                    source_value,
                    alpha=1.0 - self.decay,
                )
            else:
                target_value.copy_(source_value)

    def state_dict(self):
        return self.shadow.state_dict()


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_dataset(config):
    dataset_config = config["dataset"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    return datasets.CIFAR10(
        root=dataset_config["data_dir"],
        train=True,
        download=dataset_config.get("download", True),
        transform=transform,
    )


def create_noise_schedule(config, device):
    settings = config["diffusion"]

    return NoiseSchedule(
        timesteps=settings["timesteps"],
        schedule_type=settings["schedule_type"],
        beta_start=settings["beta_start"],
        beta_end=settings["beta_end"],
        device=device,
    )


def forward_diffusion(clean_images, timesteps, schedule):
    noise = torch.randn_like(clean_images)

    alpha_bar = schedule.alpha_bars.gather(
        0, timesteps
    ).view(-1, 1, 1, 1)

    noisy_images = (
        torch.sqrt(alpha_bar) * clean_images
        + torch.sqrt(1.0 - alpha_bar) * noise
    )

    return noisy_images, noise


def save_checkpoint(
    path,
    model,
    ema,
    optimizer,
    epoch,
    global_step,
    config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = (
            torch.cuda.get_rng_state_all()
        )

    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path,
    model,
    ema,
    optimizer,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    ema.shadow.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    if (
        torch.cuda.is_available()
        and "cuda_rng_state" in checkpoint
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng_state"]
        )

    return (
        int(checkpoint["epoch"]),
        int(checkpoint["global_step"]),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="experiments/configs/ddpm_cifar10.yaml",
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("AODMA CIFAR-10 DDPM TRAINING")
    print("=" * 60)
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = create_dataset(config)

    training_config = config["training"]

    loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model_config = config["model"]

    model = UNet(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        base_channels=model_config["base_channels"],
        time_dim=model_config["time_dim"],
    ).to(device)

    schedule = create_noise_schedule(config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    ema = EMA(
        model,
        decay=training_config["ema_decay"],
    )

    use_amp = (
        bool(training_config.get("mixed_precision", False))
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_dir = Path(
        config["storage"]["checkpoint_dir"]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(config["storage"]["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume,
            model,
            ema,
            optimizer,
            device,
        )

        print("Resuming from epoch:", start_epoch)

    epochs = int(training_config["epochs"])

    for epoch in range(start_epoch, epochs):
        model.train()

        epoch_loss = 0.0
        batches = 0
        epoch_start = time.perf_counter()

        for images, _ in loader:
            images = images.to(
                device,
                non_blocking=(device.type == "cuda"),
            )

            timesteps = torch.randint(
                0,
                schedule.timesteps,
                (images.shape[0],),
                device=device,
                dtype=torch.long,
            )

            noisy_images, noise = forward_diffusion(
                images,
                timesteps,
                schedule,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                predicted_noise = model(
                    noisy_images,
                    timesteps,
                )

                loss = F.mse_loss(
                    predicted_noise,
                    noise,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_config["gradient_clip"],
            )

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            epoch_loss += loss.item()
            batches += 1
            global_step += 1

            if (
                global_step
                % int(training_config["log_interval"])
                == 0
            ):
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Step {global_step} | "
                    f"Loss {loss.item():.6f}"
                )

        mean_loss = epoch_loss / max(1, batches)

        memory_allocated = None
        memory_reserved = None

        if device.type == "cuda":
            memory_allocated = round(
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2),
                2,
            )

            memory_reserved = round(
                torch.cuda.max_memory_reserved(device)
                / (1024 ** 2),
                2,
            )

        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "training_loss": mean_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": (
                time.perf_counter() - epoch_start
            ),
            "gpu_memory_allocated_mb": memory_allocated,
            "gpu_memory_reserved_mb": memory_reserved,
        }

        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        print(json.dumps(record))

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            ema,
            optimizer,
            epoch + 1,
            global_step,
            config,
        )

    print("=" * 60)
    print("CIFAR-10 TRAINING COMPLETED")
    print("Checkpoint:", checkpoint_dir / "latest.pt")
    print("Training log:", log_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
