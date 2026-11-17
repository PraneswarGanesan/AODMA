
from pathlib import Path
import statistics

import torch

from agents.diffusion_runtime import (
    create_sampler,
    generate_samples,
)


def execution_agent(state: dict) -> dict:
    candidate = state.get("selected_candidate")
    runtime = state.get("runtime")

    if not candidate:
        raise ValueError("No candidate selected.")

    if not runtime:
        raise RuntimeError("Diffusion runtime is missing.")

    model = runtime["model"]
    schedule = runtime["schedule"]
    config = runtime["config"]
    device = runtime["device"]

    execution_config = state.get("execution_config", {})

    sample_count = int(execution_config.get("samples", 1000))
    batch_size = int(execution_config.get("batch_size", 16))
    seed = int(execution_config.get("seed", 42))

    if sample_count < 1 or batch_size < 1:
        raise ValueError("Invalid sample count or batch size.")

    sampler = create_sampler(
        candidate["sampler"],
        model,
        schedule,
    )

    image_size = int(config["dataset"]["image_size"])
    channels = int(config["model"]["out_channels"])

    baseline_allocated_mb = None
    peak_allocated_mb = None
    peak_reserved_mb = None

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        baseline_allocated_mb = (
            torch.cuda.memory_allocated(device) / (1024 ** 2)
        )

    samples, batch_latencies = generate_samples(
        sampler=sampler,
        candidate=candidate,
        count=sample_count,
        batch_size=batch_size,
        image_size=image_size,
        channels=channels,
        device=device,
        seed=seed,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )

    total_latency_ms = sum(batch_latencies)
    mean_batch_latency_ms = statistics.mean(batch_latencies)

    output_dir = Path(
        execution_config.get(
            "output_dir",
            "results/agentic/generated",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / (
        f"{candidate['candidate_id']}_seed{seed}.pt"
    )

    torch.save(samples, sample_path)

    memory = {
        "baseline_allocated_mb": baseline_allocated_mb,
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "incremental_peak_allocated_mb": (
            max(0.0, peak_allocated_mb - baseline_allocated_mb)
            if peak_allocated_mb is not None
            and baseline_allocated_mb is not None
            else None
        ),
    }

    result = {
        "candidate_id": candidate["candidate_id"],
        "sampler": candidate["sampler"],
        "nfe": int(candidate["nfe"]),
        "sample_count": sample_count,
        "batch_size": batch_size,
        "number_of_batches": len(batch_latencies),
        "seed": seed,
        "latency_ms": mean_batch_latency_ms,
        "total_generation_latency_ms": total_latency_ms,
        "min_batch_latency_ms": min(batch_latencies),
        "max_batch_latency_ms": max(batch_latencies),
        "batch_latency_samples_ms": batch_latencies,
        "throughput_images_per_second": (
            sample_count / (total_latency_ms / 1000.0)
        ),
        "memory": memory,
        "generated_samples_path": str(sample_path),
        "device": str(device),
    }

    messages = list(state.get("messages", []))
    messages.append(
        f"Execution completed: {candidate['candidate_id']} "
        f"with mean batch latency {mean_batch_latency_ms:.2f} ms."
    )

    del samples
    del sampler

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "execution_result": result,
        "messages": messages,
        "status": "execution_completed",
    }
