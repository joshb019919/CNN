import torchvision as tv
import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class CNN(nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        super().__init__()
        # CIFAR-10 inputs: 32x32
        # Size out: ((size in - kernel size + 2(padding)) / stride) + 1
        # max pooling output size: ((size in - kernel size) / stride) + 1

        # Layer 1 shapes:
        # Size out = ((32 - 3 + 2) / 1) + 1 = 32x32
        # After pooling = ((32 - 2) / 2) + 1 = 16
        # 16 * 16 * 16 = 4096

        # Layer 2 shapes:
        # Size out = ((16 - 3 + 2) / 1) + 1 = 16x16
        # After pooling = ((16 - 2) / 2) + 1 = 8
        # 32 * 8 * 8 = 2048

        # Layer 3 shapes:
        # Size out = ((8 - 3 + 2) / 1) + 1 = 8x8
        # After pooling = ((8 - 2) / 2) + 1 = 4x4
        # 64 * 4 * 4 = 1024
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=16, 
                               kernel_size=3, 
                               padding=1,
                               device=self.device)  # Output dim 4096
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=3,
                               padding=1,
                               device=self.device)  # Output dim 2048
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1,
                               device=self.device)  # Output dim 1024
        self.output = nn.Linear(in_features=1024,
                                out_features=10,
                                device=self.device)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(self.conv1,
                                   self.act,
                                   self.pool,

                                   self.conv2,
                                   self.act,
                                   self.pool,

                                   self.conv3,
                                   self.act,
                                   self.pool,

                                   nn.Flatten(),
                                   self.output)
        
        self.loss = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = 20

        self.dataset = tv.datasets.CIFAR10(root="/data/data/cifar10", transform=tv.transforms.ToTensor())
        self.dataloader = DataLoader(self.dataset, 64, 
                                     shuffle=True, 
                                     num_workers=4, 
                                     pin_memory=True, 
                                     prefetch_factor=64,
                                     persistent_workers=True)

    def forward(self, x):
        return self.model(x)
