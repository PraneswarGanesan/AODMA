
import math
import torch


class NoiseSchedule:
    """Precomputed DDPM noise schedule."""

    def __init__(
        self,
        timesteps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        if timesteps < 2:
            raise ValueError("At least two timesteps are required.")

        self.timesteps = timesteps
        self.device = torch.device(device)

        if schedule_type == "linear":
            betas = torch.linspace(
                beta_start, beta_end, timesteps
            )

        elif schedule_type == "cosine":
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            t = steps / timesteps

            alpha_bar = torch.cos(
                ((t + 0.008) / 1.008) * math.pi / 2
            ) ** 2

            alpha_bar = alpha_bar / alpha_bar[0]

            betas = 1 - (
                alpha_bar[1:] / alpha_bar[:-1]
            )

            betas = betas.clamp(1e-8, 0.999).float()

        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        if not torch.all((betas > 0) & (betas < 1)):
            raise ValueError("Invalid beta values.")

        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas, dim=0
        )

        self.alpha_bars_prev = torch.cat([
            torch.ones(1, device=self.device),
            self.alpha_bars[:-1]
        ])

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0 - self.alpha_bars
        )

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alpha_bars_prev)
            / (1.0 - self.alpha_bars)
        )

        self.posterior_variance = self.posterior_variance.clamp(
            min=1e-20
        )

    def extract(self, values, timesteps, x_shape):
        values = values.to(timesteps.device)

        result = values.gather(0, timesteps)

        return result.reshape(
            timesteps.shape[0],
            *([1] * (len(x_shape) - 1))
        )

    def to(self, device):
        self.device = torch.device(device)

        for name in (
            "betas",
            "alphas",
            "alpha_bars",
            "alpha_bars_prev",
            "sqrt_alpha_bars",
            "sqrt_one_minus_alpha_bars",
            "posterior_variance",
        ):
            setattr(
                self,
                name,
                getattr(self, name).to(self.device)
            )

        return self
