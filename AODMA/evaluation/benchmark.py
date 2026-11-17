
import torch

from samplers.ddpm import DDPMSampler
from samplers.ddim import DDIMSampler
from samplers.dpm_solver import DPMSolverSampler
from samplers.unipc import UniPCSampler

from evaluation.metrics import (
    timed_inference,
    reset_gpu_peak_memory,
    get_gpu_memory,
    summarize_performance,
)

from evaluation.reproducibility import create_generator


class CountingModel:
    """Counts actual neural network forward evaluations."""

    def __init__(self, model):
        self.model = model
        self.forward_calls = 0

    def eval(self):
        self.model.eval()
        return self

    def reset_counter(self):
        self.forward_calls = 0

    def __call__(self, *args, **kwargs):
        self.forward_calls += 1
        return self.model(*args, **kwargs)


class BenchmarkEngine:

    def __init__(
        self,
        model,
        schedule,
        device,
        benchmark_config,
    ):
        self.device = torch.device(device)
        self.schedule = schedule
        self.config = benchmark_config
        self.counting_model = CountingModel(model)

    def build_sampler(self, candidate):
        sampler_name = candidate["sampler"].lower()

        sampler_classes = {
            "ddpm": DDPMSampler,
            "ddim": DDIMSampler,
            "dpm_solver": DPMSolverSampler,
            "unipc": UniPCSampler,
        }

        if sampler_name not in sampler_classes:
            raise ValueError(
                f"Unsupported sampler: {sampler_name}"
            )

        return sampler_classes[sampler_name](
            self.counting_model,
            self.schedule,
        )

    def run_sampler(
        self,
        sampler,
        candidate,
        shape,
        seed,
    ):
        generator = create_generator(
            self.device,
            seed,
        )

        sampler_name = candidate["sampler"].lower()

        if sampler_name == "ddpm":
            return sampler.sample(
                shape=shape,
                device=self.device,
                progress=False,
                generator=generator,
            )

        return sampler.sample(
            shape=shape,
            device=self.device,
            inference_steps=int(candidate["steps"]),
            progress=False,
            generator=generator,
        )

    def run_candidate(
        self,
        candidate,
        shape,
        seed,
        save_samples=False,
        sample_output_path=None,
    ):
        sampler = self.build_sampler(candidate)

        warmup_trials = int(self.config["warmup_trials"])
        repetitions = int(self.config["repetitions"])
        batch_size = shape[0]

        # Warm-up executions are excluded from reported timings.
        for warmup_index in range(warmup_trials):
            self.counting_model.reset_counter()

            self.run_sampler(
                sampler,
                candidate,
                shape,
                seed + warmup_index,
            )

        latencies = []
        memory_allocated = []
        memory_reserved = []
        nfe_values = []

        saved_samples = None

        for trial_index in range(repetitions):
            trial_seed = seed + 1000 + trial_index

            self.counting_model.reset_counter()
            reset_gpu_peak_memory(self.device)

            samples, elapsed_ms = timed_inference(
                lambda: self.run_sampler(
                    sampler,
                    candidate,
                    shape,
                    trial_seed,
                ),
                self.device,
            )

            actual_nfe = self.counting_model.forward_calls
            memory = get_gpu_memory(self.device)

            latencies.append(elapsed_ms)
            nfe_values.append(actual_nfe)

            if memory["peak_allocated_mb"] is not None:
                memory_allocated.append(
                    memory["peak_allocated_mb"]
                )
                memory_reserved.append(
                    memory["peak_reserved_mb"]
                )

            if trial_index == 0:
                saved_samples = samples.detach().cpu()

        performance = summarize_performance(
            latencies,
            sample_count=batch_size,
            seed=seed,
        )

        result = {
            "candidate": candidate,
            "sampler": candidate["sampler"],
            "steps_requested": candidate.get("steps"),
            "eta": candidate.get("eta", 0.0),
            "batch_size": batch_size,
            "repetitions": repetitions,
            "warmup_trials": warmup_trials,
            "nfe_per_sample": {
                "mean": float(
                    sum(nfe_values) / len(nfe_values)
                ),
                "values": nfe_values,
            },
            "latency": performance["latency"],
            "throughput_images_per_second": (
                performance["throughput_images_per_second"]
            ),
            "memory": {
                "peak_allocated_mb": (
                    max(memory_allocated)
                    if memory_allocated else None
                ),
                "peak_reserved_mb": (
                    max(memory_reserved)
                    if memory_reserved else None
                ),
            },
            "raw_latency_ms": latencies,
        }

        if save_samples and sample_output_path:
            from pathlib import Path
            from torchvision.utils import save_image

            output = Path(sample_output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            image_tensor = (
                saved_samples.clamp(-1.0, 1.0) + 1.0
            ) / 2.0

            save_image(
                image_tensor,
                output,
                nrow=max(1, int(batch_size ** 0.5)),
            )

            result["sample_grid"] = str(output)

        return result
