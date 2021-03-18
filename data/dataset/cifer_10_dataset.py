import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler


class CIFAR10Dataset:

    def __init__(self, root: str, transform, validation_size: float) -> None:
        self.set_train_and_validation_data(
            train=True,
            download=True,
            root=root,
            transform=transform,
            validation_size=validation_size,
        )

        self.set_test_data(
            train=False, download=True, root=root, transform=transform
        )

    def set_train_and_validation_data(
        self,
        download: bool,
        root: str,
        train: bool,
        transform,
        validation_size: float,
    ) -> None:
        train_dataset = CIFAR10(download=download, root=root, train=train, transform=transform)
        train_num = len(train_dataset)
        indices = list(range(train_num))
        split = int(np.floor(validation_size * train_num))
        train_indices, validation_indices = indices[split:], indices[:split]
        self.train_data_dict = {
            'dataset': train_dataset,
            'train_sampler': SubsetRandomSampler(train_indices),
            'validation_sampler': SubsetRandomSampler(validation_indices),
        }

    def set_test_data(
        self,
        download: bool,
        root: str,
        train: bool,
        transform,
    ) -> None:
        self.test_data_dict = {
            'dataset': CIFAR10(
                download=download, root=root, train=train, transform=transform
            )
        }
