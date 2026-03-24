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


class ResidualBlock(nn.Module):
    """
    残差块：包含两个卷积层，如果输入输出通道数不同，则使用1x1卷积调整跳跃连接
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        # 主路径：两个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接：如果通道数变化或步长不为1，则用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        identity = self.shortcut(x)  # 跳跃连接部分

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 残差相加
        out = self.relu(out)
        return out

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
    """
    为CIFAR-100设计的优化版CNN，使用残差块和批量归一化。
    结构：初始卷积 -> 3个残差块组 -> 全局平均池化 -> 全连接输出
    """
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(OptimizedCNN, self).__init__()

        # 初始卷积层：将3通道输入映射到32个特征图
        self.conv_initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 残差块组，逐渐增加通道数，降低特征图尺寸
        # 第一组：输入32通道，输出32通道，步长1，尺寸不变
        self.block1 = self._make_layer(32, 32, 2, stride=1, dropout_rate=dropout_rate)
        # 第二组：输入32通道，输出64通道，步长2，尺寸减半
        self.block2 = self._make_layer(32, 64, 2, stride=2, dropout_rate=dropout_rate)
        # 第三组：输入64通道，输出128通道，步长2，尺寸再减半
        self.block3 = self._make_layer(64, 128, 2, stride=2, dropout_rate=dropout_rate)

        # 全局平均池化，代替全连接层，减少参数量
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 最终的分类层
        self.fc = nn.Linear(128, num_classes)

        # 可选：在最终输出前再加一个Dropout
        self.dropout_final = nn.Dropout(dropout_rate)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        """构建一组残差块"""
        layers = []
        # 第一个残差块可能改变通道数和尺寸
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))
        # 剩余的残差块，通道数不变，步长为1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_initial(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)  # 展平
        out = self.dropout_final(out)
        out = self.fc(out)
        return out
