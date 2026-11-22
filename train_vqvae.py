import torch
from torch import nn
from vq_vae import VQVAE
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as Compose
from torchvision.utils import make_grid, save_image

BATCH_SIZE = 64
LR         = 2e-4

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

print(len(train_dataset), len(test_dataset))

train_dataloader = DataLoader(dataset=train_dataset + test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              )
test_dataloader   = DataLoader(dataset=test_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=False)

#images = next(iter(train_dataloader))[0]
#grid = make_grid(images,
#                 padding=2,
#                 )
#save_image(grid, "sample_images.png")


