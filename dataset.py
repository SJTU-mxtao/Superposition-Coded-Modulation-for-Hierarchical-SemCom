import torch
from torch.utils.data import Dataset
import numpy as np


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                              3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                              6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                              0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                              5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                              10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                              2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

    return coarse_labels[targets]


class Cifar100(Dataset):
    def __init__(self, cifar100, transform):
        self.cifar100 = cifar100
        self.transform = transform
        self.coarse_labels = sparse2coarse(self.cifar100.targets)
        # self.data = self.cifar100.data
        self.fine_labels = self.cifar100.targets

    def __len__(self):
        return len(self.fine_labels)

    def __getitem__(self, index):
        # data = torch.tensor(self.data[index]).float()
        data = self.cifar100[index][0]
        data = self.transform(data)
        coarse_label = torch.tensor(self.coarse_labels[index])
        fine_label = torch.tensor(self.fine_labels[index])
        return data, coarse_label, fine_label
