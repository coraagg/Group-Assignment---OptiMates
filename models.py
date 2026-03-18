import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, hidden_sizes=[256, 256]):   # B 可自行修改参数
        super().__init__()
        layers = []
        prev = 28*28
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 256)
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28->26->13
        x = self.pool(F.relu(self.conv2(x)))   # 13->11->5
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class OptimizedCNN(nn.Module):
    # 由组长实现
    def __init__(self):
        super().__init__()
        # TODO: 添加 BatchNorm, Dropout, 更深的层等
        pass

    def forward(self, x):
        pass