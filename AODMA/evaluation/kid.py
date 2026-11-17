
import torch

from torchmetrics.image.kid import KernelInceptionDistance

from evaluation.feature_extractor import (
    images_to_uint8,
    validate_image_batch,
)


class KIDEvaluator:

    def __init__(
        self,
        device,
        subset_size=1000,
        subsets=100,
    ):
        self.metric = KernelInceptionDistance(
            feature=2048,
            subset_size=subset_size,
            subsets=subsets,
            normalize=False,
        ).to(device)

    @torch.no_grad()
    def update_real(self, images):
        validate_image_batch(images)
        self.metric.update(
            images_to_uint8(images).to(self.metric.device),
            real=True,
        )

    @torch.no_grad()
    def update_fake(self, images):
        validate_image_batch(images)
        self.metric.update(
            images_to_uint8(images).to(self.metric.device),
            real=False,
        )

    def compute(self):
        mean, std = self.metric.compute()

        return {
            "mean": float(mean.item()),
            "std": float(std.item()),
        }

    def reset(self):
        self.metric.reset()
