
import torch
from tqdm import tqdm
from diffusers import DPMSolverMultistepScheduler


class DPMSolverSampler:
    """DPM-Solver++ second-order multistep sampler."""

    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    def _create_scheduler(self):
        return DPMSolverMultistepScheduler(
            num_train_timesteps=self.schedule.timesteps,
            trained_betas=(
                self.schedule.betas.detach().cpu().numpy()
            ),
            prediction_type="epsilon",
            algorithm_type="dpmsolver++",
            solver_order=2,
            solver_type="midpoint",
            lower_order_final=True,
            timestep_spacing="linspace",
            final_sigmas_type="zero",
        )

    @torch.no_grad()
    def sample(
        self,
        shape,
        device,
        inference_steps=20,
        progress=True,
        generator=None,
    ):
        device = torch.device(device)
        self.model.eval()

        if not 2 <= inference_steps <= self.schedule.timesteps:
            raise ValueError("Invalid inference_steps.")

        scheduler = self._create_scheduler()
        scheduler.set_timesteps(inference_steps, device=device)

        x = torch.randn(
            shape,
            device=device,
            generator=generator,
        )

        steps = scheduler.timesteps

        if progress:
            steps = tqdm(steps, desc=f"DPM-Solver++ ({inference_steps})")

        for timestep in steps:
            t = timestep.expand(shape[0]).long()

            model_input = scheduler.scale_model_input(x, timestep)
            predicted_noise = self.model(model_input, t)

            x = scheduler.step(
                predicted_noise,
                timestep,
                x,
            ).prev_sample

        return x.clamp(-1.0, 1.0)
