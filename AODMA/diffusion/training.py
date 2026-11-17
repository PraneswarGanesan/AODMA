
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from diffusion.forward_process import q_sample


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:

    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):

        source = model.state_dict()
        target = self.model.state_dict()

        for key in target:
            if target[key].is_floating_point():
                target[key].mul_(self.decay).add_(
                    source[key].detach(),
                    alpha=1.0 - self.decay
                )
            else:
                target[key].copy_(source[key])


class DiffusionTrainer:

    def __init__(
        self,
        model,
        schedule,
        config,
        device,
    ):
        self.model = model.to(device)
        self.schedule = schedule.to(device)
        self.config = config
        self.device = device

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4),
        )

        self.ema = EMA(
            self.model,
            decay=config.get("ema_decay", 0.999)
        )

        self.use_amp = (
            config.get("mixed_precision", False)
            and device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp
        )

        self.global_step = 0
        self.start_epoch = 0

        self.checkpoint_dir = Path(
            config["checkpoint_dir"]
        )
        self.checkpoint_dir.mkdir(
            parents=True, exist_ok=True
        )

        self.log_path = Path(config["log_path"])
        self.log_path.parent.mkdir(
            parents=True, exist_ok=True
        )

    def train_epoch(self, dataloader, epoch):

        self.model.train()

        running_loss = 0.0
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}"
        )

        for images, _ in progress:

            images = images.to(
                self.device,
                non_blocking=True
            )

            batch_size = images.shape[0]

            timesteps = torch.randint(
                0,
                self.schedule.timesteps,
                (batch_size,),
                device=self.device,
            )

            noisy_images, noise = q_sample(
                images,
                timesteps,
                self.schedule,
            )

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                predicted_noise = self.model(
                    noisy_images,
                    timesteps
                )

                loss = F.mse_loss(
                    predicted_noise,
                    noise
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss encountered."
                )

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get(
                    "gradient_clip", 1.0
                )
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update(self.model)

            self.global_step += 1
            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.5f}"
            )

        average_loss = running_loss / len(dataloader)

        record = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "training_loss": average_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        if self.device.type == "cuda":
            record["gpu_memory_allocated_mb"] = (
                torch.cuda.memory_allocated(self.device)
                / 1024**2
            )

            record["gpu_memory_reserved_mb"] = (
                torch.cuda.memory_reserved(self.device)
                / 1024**2
            )

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        return average_loss

    def save_checkpoint(self, epoch):

        checkpoint = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "ema": self.ema.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
        }

        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = (
                torch.cuda.get_rng_state_all()
            )

        destination = self.checkpoint_dir / "latest.pt"
        temporary = self.checkpoint_dir / "latest.tmp"

        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)

    def resume(self, checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model"])
        self.ema.model.load_state_dict(checkpoint["ema"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])

        if (
            torch.cuda.is_available()
            and "cuda_rng_state" in checkpoint
        ):
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng_state"]
            )

    def fit(self, dataloader):

        epochs = self.config["epochs"]

        for epoch in range(self.start_epoch, epochs):

            loss = self.train_epoch(
                dataloader,
                epoch
            )

            self.save_checkpoint(epoch)

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Average loss: {loss:.6f}"
            )
