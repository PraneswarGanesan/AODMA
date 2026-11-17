
import torch
from diffusion.noise_schedule import NoiseSchedule


def q_sample(
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: NoiseSchedule,
    noise: torch.Tensor = None,
):
    """
    Directly sample x_t from x_0 using the DDPM
    forward-process closed-form equation.
    """

    if x_start.ndim != 4:
        raise ValueError("Expected [B, C, H, W] input.")

    if timesteps.shape != (x_start.shape[0],):
        raise ValueError("Expected one timestep per image.")

    timesteps = timesteps.to(
        device=x_start.device,
        dtype=torch.long
    )

    if torch.any(timesteps < 0) or torch.any(
        timesteps >= schedule.timesteps
    ):
        raise ValueError("Timestep out of range.")

    if noise is None:
        noise = torch.randn_like(x_start)

    if noise.shape != x_start.shape:
        raise ValueError("Noise shape mismatch.")

    alpha = schedule.extract(
        schedule.sqrt_alpha_bars,
        timesteps,
        x_start.shape
    ).to(x_start.device)

    sigma = schedule.extract(
        schedule.sqrt_one_minus_alpha_bars,
        timesteps,
        x_start.shape
    ).to(x_start.device)

    return alpha * x_start + sigma * noise, noise
