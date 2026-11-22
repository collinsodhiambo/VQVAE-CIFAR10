import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as Compose
from torchvision.utils import make_grid, save_image

transforms = Compose.Compose([Compose.ToImage(),
                              Compose.RandomHorizontalFlip(p=0.5),
                             Compose.ToDtype(torch.float32, scale=True)
                              ])
train_dataset = torchvision.datasets.CIFAR10(root="data",
                                       train=True,
                                       download=True,
                                       transform=transforms)
test_dataset = torchvision.datasets.CIFAR10(root="data",
                                            train=False,
                                            download=True,
                                            transform=transforms)
