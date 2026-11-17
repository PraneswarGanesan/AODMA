
from pathlib import Path
import time

import torch
import yaml

from diffusion.models import UNet
from diffusion.noise_schedule import NoiseSchedule

from samplers.ddpm import DDPMSampler
from samplers.ddim import DDIMSampler
from samplers.dpm_solver import DPMSolverSampler
from samplers.unipc import UniPCSampler


SAMPLER_REGISTRY = {
    "ddpm": DDPMSampler,
    "ddim": DDIMSampler,
    "dpm_solver": DPMSolverSampler,
    "unipc": UniPCSampler,
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_diffusion_runtime(checkpoint_path, config_path):
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = load_yaml(config_path)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    weights = checkpoint.get("ema")

    if weights is None:
        weights = checkpoint.get("model")

    if weights is None:
        raise KeyError(
            "Checkpoint has neither 'ema' nor 'model' weights."
        )

    model.load_state_dict(weights)
    model.eval()

    diffusion = config["diffusion"]

    schedule = NoiseSchedule(
        timesteps=diffusion["timesteps"],
        schedule_type=diffusion["schedule_type"],
        beta_start=diffusion["beta_start"],
        beta_end=diffusion["beta_end"],
        device=device,
    )

    return {
        "model": model,
        "schedule": schedule,
        "config": config,
        "device": device,
    }


def create_sampler(name, model, schedule):
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unsupported sampler: {name}")

    return SAMPLER_REGISTRY[name](model, schedule)


def generate_samples(
    sampler,
    candidate,
    count,
    batch_size,
    image_size,
    channels,
    device,
    seed,
):
    generated_batches = []
    batch_latencies_ms = []
    generated_count = 0
    batch_index = 0

    while generated_count < count:
        current_batch = min(
            batch_size,
            count - generated_count,
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(seed + batch_index)

        shape = (
            current_batch,
            channels,
            image_size,
            image_size,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            if candidate["sampler"] == "ddpm":
                samples = sampler.sample(
                    shape=shape,
                    device=device,
                    progress=False,
                    generator=generator,
                )
            else:
                samples = sampler.sample(
                    shape=shape,
                    device=device,
                    inference_steps=int(candidate["nfe"]),
                    progress=False,
                    generator=generator,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_latency = (
            time.perf_counter() - start
        ) * 1000.0

        batch_latencies_ms.append(batch_latency)
        generated_batches.append(samples.detach().cpu())

        generated_count += current_batch
        batch_index += 1

        print(
            f"\rGenerated {generated_count}/{count}",
            end="",
            flush=True,
        )

    print()

    return (
        torch.cat(generated_batches, dim=0),
        batch_latencies_ms,
    )
