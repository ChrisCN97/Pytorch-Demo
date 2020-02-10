import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST('./data/mnistd',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.MNIST('./data/mnistd',
    download=True,
    train=False,
    transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=0)
