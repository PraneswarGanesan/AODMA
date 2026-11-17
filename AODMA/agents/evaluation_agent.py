
# from pathlib import Path

# import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# from evaluation.quality_metrics import QualityEvaluator


# def evaluation_agent(state: dict) -> dict:
#     execution = state.get("execution_result")

#     if not execution:
#         raise ValueError(
#             "Evaluation Agent received no execution result."
#         )

#     runtime = state["runtime"]
#     config = runtime["config"]
#     device = runtime["device"]

#     generated_path = Path(
#         execution["generated_samples_path"]
#     )

#     if not generated_path.exists():
#         raise FileNotFoundError(generated_path)

#     generated = torch.load(
#         generated_path,
#         map_location="cpu",
#         weights_only=True,
#     )

#     sample_count = int(execution["sample_count"])
#     batch_size = int(execution["batch_size"])

#     if generated.shape[0] != sample_count:
#         raise ValueError(
#             "Generated sample count does not match execution metadata."
#         )

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=(0.5, 0.5, 0.5),
#             std=(0.5, 0.5, 0.5),
#         ),
#     ])

#     dataset = datasets.CIFAR10(
#         root=config["dataset"]["data_dir"],
#         train=False,
#         download=config["dataset"].get("download", True),
#         transform=transform,
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=(device.type == "cuda"),
#     )

#     evaluator = QualityEvaluator(
#         device=device,
#         kid_subset_size=1000,
#     )

#     real_count = 0

#     for images, _ in loader:
#         if real_count >= sample_count:
#             break

#         remaining = sample_count - real_count
#         images = images[:remaining].to(device)

#         evaluator.update_real(images)
#         real_count += images.shape[0]

#     if real_count != sample_count:
#         raise RuntimeError(
#             "Insufficient real CIFAR-10 reference images."
#         )

#     generated_count = 0

#     for start in range(0, sample_count, batch_size):
#         batch = generated[start:start + batch_size].to(device)
#         evaluator.update_fake(batch)
#         generated_count += batch.shape[0]

#     if generated_count != sample_count:
#         raise RuntimeError(
#             "Not all generated images were evaluated."
#         )

#     scores = evaluator.compute()

#     fid = float(scores["fid"])
#     kid = scores["kid"]

#     result = {
#         "candidate_id": execution["candidate_id"],
#         "sample_count": sample_count,
#         "reference_dataset": "CIFAR-10 test split",
#         "fid": fid,
#         "kid_mean": float(kid["mean"]),
#         "kid_std": float(kid["std"]),
#         "latency_ms": float(execution["latency_ms"]),
#         "throughput_images_per_second": float(
#             execution["throughput_images_per_second"]
#         ),
#         "memory": execution["memory"],
#         "nfe": int(execution["nfe"]),
#         "device": execution["device"],
#     }

#     messages = list(state.get("messages", []))
#     messages.append(
#         f"Evaluation completed: FID={fid:.4f}, "
#         f"KID={result['kid_mean']:.6f}."
#     )

#     del evaluator
#     del generated

#     if device.type == "cuda":
#         torch.cuda.empty_cache()

#     return {
#         "evaluation_result": result,
#         "messages": messages,
#         "status": "evaluation_completed",
#     }

from pathlib import Path
import torch


def evaluation_agent(state: dict) -> dict:
    execution = state.get("execution_result")

    if not execution:
        raise ValueError("No execution result available.")

    generated_path = Path(execution["generated_samples_path"])

    if not generated_path.exists():
        raise FileNotFoundError(generated_path)

    samples = torch.load(
        generated_path,
        map_location="cpu",
        weights_only=True,
    )

    expected_count = int(execution["sample_count"])

    if samples.shape[0] != expected_count:
        raise ValueError("Generated sample count mismatch.")

    if not torch.isfinite(samples).all():
        raise ValueError("Generated samples contain NaN or Inf.")

    sample_count = samples.shape[0]

    result = {
        "candidate_id": execution["candidate_id"],
        "sample_count": sample_count,
        "sample_shape": list(samples.shape),
        "samples_finite": True,
        "sample_min": float(samples.min()),
        "sample_max": float(samples.max()),
        "latency_ms": execution["latency_ms"],
        "throughput_images_per_second": (
            execution["throughput_images_per_second"]
        ),
        "memory": execution["memory"],
        "nfe": execution["nfe"],
        "fid": None,
        "kid_mean": None,
        "kid_std": None,
        "evaluation_mode": "smoke_test",
        "note": (
            "FID and KID skipped. This run validates "
            "the execution pipeline, not image quality."
        ),
    }

    messages = list(state.get("messages", []))
    messages.append(
        f"Smoke evaluation passed for {sample_count} generated images."
    )

    del samples

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "evaluation_result": result,
        "messages": messages,
        "status": "evaluation_completed",
    }
