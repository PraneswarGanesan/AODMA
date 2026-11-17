
import torch
from tqdm import tqdm


class DDPMSampler:
    """Ancestral DDPM reverse-process sampler."""

    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    @torch.no_grad()
    def sample(
        self,
        shape,
        device,
        progress=True,
        generator=None,
    ):

        device = torch.device(device)
        self.model.eval()

        x = torch.randn(
            shape,
            device=device,
            generator=generator,
        )

        steps = range(self.schedule.timesteps - 1, -1, -1)

        if progress:
            steps = tqdm(steps, desc="DDPM Sampling")

        for timestep in steps:

            t = torch.full(
                (shape[0],),
                timestep,
                device=device,
                dtype=torch.long,
            )

            predicted_noise = self.model(x, t)

            beta = self.schedule.betas[timestep]
            alpha = self.schedule.alphas[timestep]
            alpha_bar = self.schedule.alpha_bars[timestep]

            mean = (
                x - beta / torch.sqrt(1.0 - alpha_bar)
                * predicted_noise
            ) / torch.sqrt(alpha)

            if timestep > 0:

                variance = self.schedule.posterior_variance[timestep]

                noise = torch.randn(
                    x.shape,
                    device=device,
                    dtype=x.dtype,
                    generator=generator,
                )

                x = mean + torch.sqrt(variance) * noise

            else:
                x = mean

        return x.clamp(-1.0, 1.0)
