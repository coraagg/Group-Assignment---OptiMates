import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    """线性模型：输入 3072，输出 100"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3*32*32, 100)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        return self.fc(x)


class MLP(nn.Module):
    """多层感知机，可指定隐藏层大小"""
    def __init__(self, hidden_sizes=[512, 512], dropout=0.0):
        super().__init__()
        layers = []
        prev = 3*32*32
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 100))
        self.net = nn.Sequential(*layers)

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
        # 经过两次池化后，特征图尺寸 32 -> 16 -> 8
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
    """优化版 CNN：使用更深的卷积结构和残差连接（类似简单 ResNet）"""
    def __init__(self):
        super().__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # 第二个卷积块（带残差连接）
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # 跳跃连接用的 1x1 卷积（调整通道数）
        self.shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2)  # 因为池化会降维
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 块1
        out = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32 -> 16
        # 块2（带残差）
        identity = self.shortcut(out)                     # 调整维度
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(F.relu(self.bn3(self.conv3(out)))) # 16 -> 8
        out += identity                                    # 残差连接
        # 展平
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
