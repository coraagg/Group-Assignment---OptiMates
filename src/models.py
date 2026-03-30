import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    """Linear model: input 3072, output 100"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 100)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron:
    - hidden_sizes: e.g., [512, 512] / [1024, 1024]
    - activation: 'relu' or 'tanh'
    - dropout: dropout rate
    - use_batchnorm: whether to use BatchNorm1d
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
    Residual block: two convolutional layers. If input and output channels differ
    or stride != 1, a 1x1 convolution is used for the skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        # Main path: two convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection: adjust dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class BasicCNN(nn.Module):
    """Basic CNN: 2 convolutional layers + 2 fully connected layers"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        # After two poolings, feature map size: 32 -> 16 -> 8
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
    Optimized CNN for CIFAR-100 using residual blocks and batch normalization.
    Structure: initial conv -> 3 residual groups -> global average pooling -> fully connected
    """
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(OptimizedCNN, self).__init__()

        # Initial convolution: map 3 channels to 32 feature maps
        self.conv_initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Residual groups, gradually increase channels and reduce spatial size
        # Group 1: input 32 channels, output 32 channels, stride 1, size unchanged
        self.block1 = self._make_layer(32, 32, 2, stride=1, dropout_rate=dropout_rate)
        # Group 2: input 32 -> output 64, stride 2, size halved
        self.block2 = self._make_layer(32, 64, 2, stride=2, dropout_rate=dropout_rate)
        # Group 3: input 64 -> output 128, stride 2, size halved again
        self.block3 = self._make_layer(64, 128, 2, stride=2, dropout_rate=dropout_rate)

        # Global average pooling replaces fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final classification layer
        self.fc = nn.Linear(128, num_classes)

        # Optional dropout before final layer
        self.dropout_final = nn.Dropout(dropout_rate)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        """Build a group of residual blocks"""
        layers = []
        # First block may change channels and size
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))
        # Remaining blocks keep channels and size unchanged
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_initial(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout_final(out)
        out = self.fc(out)
        return out
