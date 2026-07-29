import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU block helper."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, act=True):
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

class XResNetBlock(nn.Module):
    """
    XResNet Bottleneck Block with ResNet-D tweaks:
    1. Stride=2 placed in 3x3 conv instead of initial 1x1 conv.
    2. Skip connection uses AvgPool2d(2, stride=2) + 1x1 conv for downsampling.
    3. Zero-initialized gamma in final BatchNorm layer.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(XResNetBlock, self).__init__()
        
        # 1x1 Conv
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=1, stride=1)
        
        # 3x3 Conv with downsampling stride if applicable
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=stride)
        
        # 1x1 Expansion Conv (act=False for BN before residual addition)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        # Zero-initialize the last BN in each bottleneck block
        nn.init.constant_(self.bn3.weight, 0.0)

        # Downsample path (ResNet-D tweak)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            shortcut_layers = []
            if stride != 1:
                shortcut_layers.append(nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True))
            shortcut_layers.append(nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=1, bias=False))
            shortcut_layers.append(nn.BatchNorm2d(planes * self.expansion))
            self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class XResNet50(nn.Module):
    """
    XResNet50 Architecture:
    - 3x3 Conv stem (3 consecutive 3x3 convolutions replacing 7x7 conv)
    - 4 stages with bottleneck blocks [3, 4, 6, 3]
    - Global Average Pooling & linear classifier head
    """
    def __init__(self, num_classes=2, in_channels=3):
        super(XResNet50, self).__init__()
        self.in_planes = 64

        # Stem: 3 consecutive 3x3 convolutions (Bag of Tricks for Image Classification)
        self.stem = nn.Sequential(
            ConvLayer(in_channels, 32, kernel_size=3, stride=2),
            ConvLayer(32, 32, kernel_size=3, stride=1),
            ConvLayer(32, 64, kernel_size=3, stride=1)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages: [3, 4, 6, 3] blocks for ResNet50 depth
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512 * XResNetBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(XResNetBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * XResNetBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def get_xresnet50(num_classes=2, pretrained_weights=None):
    model = XResNet50(num_classes=num_classes)
    if pretrained_weights and torch.cuda.is_available():
        model.load_state_dict(torch.load(pretrained_weights))
    elif pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights, map_location='cpu'))
    return model
