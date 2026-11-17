
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloader(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    image_size: int = 32,
    num_workers: int = 0,
    download: bool = True,
):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        channels = 1

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = datasets.MNIST(
            root=Path(data_dir),
            train=True,
            transform=transform,
            download=download,
        )

    elif dataset_name == "cifar10":
        channels = 3

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

        dataset = datasets.CIFAR10(
            root=Path(data_dir),
            train=True,
            transform=transform,
            download=download,
        )

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    return loader, channels
