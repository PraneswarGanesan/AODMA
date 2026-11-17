
import torch
from tqdm import tqdm


class DDIMSampler:
    """
    Denoising Diffusion Implicit Models (DDIM) sampler.

    Supports:
      - Configurable inference steps
      - Deterministic sampling with eta=0
      - Stochastic sampling with eta>0
      - Optional predicted x0 clipping
    """

    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    @torch.no_grad()
    def sample(
        self,
        shape,
        device,
        inference_steps=50,
        eta=0.0,
        clip_denoised=True,
        progress=True,
        generator=None,
    ):
        device = torch.device(device)
        self.model.eval()

        total_steps = self.schedule.timesteps

        if not 2 <= inference_steps <= total_steps:
            raise ValueError(
                f"inference_steps must be between 2 and {total_steps}."
            )

        if eta < 0:
            raise ValueError("eta must be non-negative.")

        # Select a subset of the training timesteps.
        timestep_sequence = torch.linspace(
            0,
            total_steps - 1,
            inference_steps,
            dtype=torch.float64,
        ).round().long().unique()

        if len(timestep_sequence) != inference_steps:
            raise ValueError("Could not construct unique inference timesteps.")

        timestep_sequence = timestep_sequence.flip(0).tolist()

        # Initial Gaussian noise.
        x = torch.randn(
            shape,
            device=device,
            generator=generator,
        )

        steps = list(enumerate(timestep_sequence))

        if progress:
            steps = tqdm(steps, desc=f"DDIM Sampling ({inference_steps} steps)")

        for index, timestep in steps:
            previous_timestep = (
                timestep_sequence[index + 1]
                if index + 1 < len(timestep_sequence)
                else -1
            )

            t = torch.full(
                (shape[0],),
                timestep,
                device=device,
                dtype=torch.long,
            )

            predicted_noise = self.model(x, t)

            alpha_bar_t = self.schedule.alpha_bars[timestep].to(device)

            if previous_timestep >= 0:
                alpha_bar_previous = (
                    self.schedule.alpha_bars[previous_timestep].to(device)
                )
            else:
                alpha_bar_previous = torch.ones(
                    (),
                    device=device,
                    dtype=alpha_bar_t.dtype,
                )

            # Predict the original clean sample x_0.
            predicted_x0 = (
                x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)

            if clip_denoised:
                predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

            # DDIM variance parameter.
            sigma_squared = (
                (1.0 - alpha_bar_previous)
                / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_previous)
            )

            sigma = eta * torch.sqrt(sigma_squared.clamp(min=0.0))

            # Direction pointing toward the next noisy state.
            direction = torch.sqrt(
                (1.0 - alpha_bar_previous - sigma.square()).clamp(min=0.0)
            ) * predicted_noise

            x = (
                torch.sqrt(alpha_bar_previous) * predicted_x0
                + direction
            )

            # Add stochastic noise only when eta > 0 and not at the final step.
            if eta > 0.0 and previous_timestep >= 0:
                noise = torch.randn(
                    x.shape,
                    device=device,
                    dtype=x.dtype,
                    generator=generator,
                )
                x = x + sigma * noise

        return x.clamp(-1.0, 1.0)
