
import torch

from torchmetrics.image.fid import FrechetInceptionDistance

from evaluation.feature_extractor import (
    images_to_uint8,
    validate_image_batch,
)


class FIDEvaluator:

    def __init__(self, device):
        self.metric = FrechetInceptionDistance(
            feature=2048,
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
        return float(self.metric.compute().item())

    def reset(self):
        self.metric.reset()
