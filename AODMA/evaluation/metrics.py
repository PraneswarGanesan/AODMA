
import time

import torch

from evaluation.statistics import summarize_latency


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def reset_gpu_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_gpu_memory(device):
    if torch.device(device).type != "cuda":
        return {
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }

    return {
        "peak_allocated_mb": round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            3,
        ),
        "peak_reserved_mb": round(
            torch.cuda.max_memory_reserved(device) / (1024 ** 2),
            3,
        ),
    }


def timed_inference(function, device):
    synchronize(device)

    start = time.perf_counter()

    result = function()

    synchronize(device)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return result, elapsed_ms


def summarize_performance(
    latencies_ms,
    sample_count,
    seed=42,
):
    latency = summarize_latency(latencies_ms, seed=seed)

    mean_latency = latency["mean"]

    throughput = (
        sample_count / (mean_latency / 1000.0)
        if mean_latency > 0
        else 0.0
    )

    return {
        "latency": latency,
        "throughput_images_per_second": throughput,
        "samples_per_trial": sample_count,
    }
