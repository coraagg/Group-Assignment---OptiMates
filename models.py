import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    """线性模型：输入 3072，输出 100"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 100)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MLP(nn.Module):
    """
    可配置的多层感知机：
    - hidden_sizes: 例如 [512, 512] / [1024, 1024]
    - activation: 'relu' or 'tanh'
    - dropout: dropout rate
    - use_batchnorm: 是否使用 BatchNorm1d
    """
    def __init__(
        self,
        hidden_sizes=(512, 512),
        activation='relu',
        dropout=0.0,
        use_batchnorm=False,
        num_classes=100
    ):
        super().__init__()

        if activation == 'relu':
            act_layer = nn.ReLU
        elif activation == 'tanh':
            act_layer = nn.Tanh
        else:
            raise ValueError("activation must be 'relu' or 'tanh'")

        layers = []
        in_dim = 3 * 32 * 32

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

        self._init_weights(activation)

    def _init_weights(self, activation):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation == 'relu':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class BasicCNN(nn.Module):
    """基础 CNN：2 个卷积层 + 2 个全连接层"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 100)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class OptimizedCNN(nn.Module):
    """优化版 CNN：更深卷积 + 简单残差连接"""
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32 -> 16

        identity = self.shortcut(out)

        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(F.relu(self.bn3(self.conv3(out))))  # 16 -> 8

        out += identity

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out