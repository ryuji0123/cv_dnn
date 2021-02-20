import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler


class CIFAR10Dataset:

    def __init__(self, root, transform, val_size):
        self.setTrainAndValData(
                train=True, download=True, root=root, transform=transform, val_size=val_size
                )

        self.setTestData(
                train=False, download=True, root=root, transform=transform
                )

    def setTrainAndValData(self, download, root, train, transform, val_size):
        train_dataset = CIFAR10(download=download, root=root, train=train, transform=transform)
        train_num = len(train_dataset)
        indices = list(range(train_num))
        split = int(np.floor(val_size * train_num))
        train_indices, val_indices = indices[split:], indices[:split]
        self.train_data = {
                'dataset': train_dataset,
                'train_sampler': SubsetRandomSampler(train_indices),
                'val_sampler': SubsetRandomSampler(val_indices),
                }

    def setTestData(self, download, root, train, transform):
        self.test_data = {
                'dataset': CIFAR10(
                    download=download, root=root, train=train, transform=transform
                    )
                }
