

import torch


def images_to_uint8(images):
    """
    Convert images in [-1, 1] or [0, 1] to uint8 [0, 255].

    Expected shape: N x 3 x H x W
    """
    if images.ndim != 4:
        raise ValueError("Expected a 4D image tensor.")

    if images.shape[1] != 3:
        raise ValueError(
            "CIFAR-10 quality evaluation requires RGB images."
        )

    images = images.detach().float()

    images = ((images.clamp(-1, 1) + 1.0) * 127.5)

    return images.round().clamp(0, 255).to(torch.uint8)


def validate_image_batch(images):
    if images.ndim != 4:
        raise ValueError("Images must have shape NCHW.")

    if images.shape[1] != 3:
        raise ValueError("Expected three RGB channels.")

    if images.shape[2] != 32 or images.shape[3] != 32:
        raise ValueError("Expected 32x32 CIFAR-10 images.")
