
import os
import random
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def create_generator(device, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_environment_metadata():
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_commit": get_git_commit(),
        "cpu_count": os.cpu_count(),
    }

    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_count"] = torch.cuda.device_count()

        properties = torch.cuda.get_device_properties(0)

        metadata["gpu_total_memory_mb"] = round(
            properties.total_memory / (1024 ** 2), 2
        )

    return metadata
