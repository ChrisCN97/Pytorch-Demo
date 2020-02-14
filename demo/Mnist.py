import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor()])
batch_size = 64

trainset = torchvision.datasets.MNIST('data/mnistd',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.MNIST('data/mnistd',
    download=True,
    train=False,
    transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

writer = SummaryWriter('runs/mnist')
"""
# 查看图片
dataiter = iter(trainloader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
writer.add_image('four_mnist_images', img_grid)
writer.close()
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(40, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = x.view(in_size, -1)
        x = self.fc(x)
        x = F.log_softmax(x)
        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    trainLen = len(trainloader)
    for e_idx in range(epoch):
        for batch_idx, (data, target) in enumerate(trainloader):
            output = net(data)
            loss = F.nll_loss(output, target)
            running_loss += loss.item()
            if batch_idx % 100 == 99:
                writer.add_scalar('training loss',
                                  running_loss / 100,
                                  e_idx * trainLen + batch_idx)
                running_loss = 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test():
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            output = net(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    print('Accuracy of the network: %d %%' % (100 * correct / len(testset)))

train(3)
test()

# Accuracy of the network: 96 %
