"""
3D ResNet Implementation
Based on the original ResNet paper and adapted for 3D medical imaging.
author:px
date:2022-01-07
version:2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Optional, Union, Tuple
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SELayer3D(nn.Module):
    """Squeeze-and-Excitation block for 3D inputs."""
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class Conv3DBlock(nn.Module):
    """Enhanced 3D convolution block with optional SE and dropout."""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 use_se: bool = False,
                 dropout_rate: float = 0.0):
        super(Conv3DBlock, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        if isinstance(padding, int):
            padding = (padding,) * 3
            
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer3D(out_channels) if use_se else None
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.se is not None:
            x = self.se(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    """Enhanced residual block with SE attention and dropout."""
    expansion = 1

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1, 
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 use_se: bool = False,
                 dropout_rate: float = 0.0):
        super(ResidualBlock, self).__init__()
        
        width = int(out_channels * (base_width / 64.)) * groups
        
        self.conv1 = Conv3DBlock(
            in_channels, width, 3, stride, 1, groups=groups,
            use_se=use_se, dropout_rate=dropout_rate
        )
        self.conv2 = Conv3DBlock(
            width, out_channels * self.expansion, 3, 1, 1, groups=groups,
            use_se=use_se, dropout_rate=dropout_rate
        )
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """Enhanced bottleneck block with SE attention and dropout."""
    expansion = 4

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1, 
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 use_se: bool = False,
                 dropout_rate: float = 0.0):
        super(Bottleneck, self).__init__()
        
        width = int(out_channels * (base_width / 64.)) * groups
        
        self.conv1 = Conv3DBlock(
            in_channels, width, 1, 1, 0, groups=1,
            use_se=False, dropout_rate=dropout_rate
        )
        self.conv2 = Conv3DBlock(
            width, width, 3, stride, 1, groups=groups,
            use_se=use_se, dropout_rate=dropout_rate
        )
        self.conv3 = Conv3DBlock(
            width, out_channels * self.expansion, 1, 1, 0, groups=1,
            use_se=False, dropout_rate=dropout_rate
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    """Enhanced ResNet3D with advanced features and configurations."""
    
    def __init__(self, 
                 block: Type[Union[ResidualBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 2,
                 input_channels: int = 1,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 use_se: bool = False,
                 dropout_rate: float = 0.0):
        super(ResNet3D, self).__init__()

        self._validate_params(layers, num_classes, input_channels)
        
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial layers
        self.conv1 = Conv3DBlock(
            input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
            use_se=use_se, dropout_rate=dropout_rate
        )
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], use_se=use_se, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_se=use_se, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_se=use_se, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_se=use_se, dropout_rate=dropout_rate)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._initialize_weights(zero_init_residual)
        self.print_model_stats()

    def _validate_params(self, layers, num_classes, input_channels):
        """Validate input parameters."""
        if not isinstance(layers, list) or len(layers) != 4:
            raise ValueError("layers should be a list of 4 integers")
        if input_channels < 1:
            raise ValueError("input_channels must be at least 1")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

    def _make_layer(self, 
                   block: Type[Union[ResidualBlock, Bottleneck]],
                   planes: int,
                   blocks: int,
                   stride: int = 1,
                   use_se: bool = False,
                   dropout_rate: float = 0.0) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                          self.groups, self.base_width, use_se, dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, use_se=use_se,
                              dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool = False) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.conv3.bn.weight, 0)
                elif isinstance(m, ResidualBlock):
                    nn.init.constant_(m.conv2.bn.weight, 0)

    def print_model_stats(self):
        """Print model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model Statistics:")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet3d18(**kwargs):
    return ResNet3D(ResidualBlock, [2, 2, 2, 2], **kwargs)

def resnet3d34(**kwargs):
    return ResNet3D(ResidualBlock, [3, 4, 6, 3], **kwargs)

def resnet3d50(**kwargs):
    return ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet3d101(**kwargs):
    return ResNet3D(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet3d152(**kwargs):
    return ResNet3D(Bottleneck, [3, 8, 36, 3], **kwargs)

if __name__ == "__main__":
    # Test different configurations
    model_configs = [
        {'model_fn': resnet3d18, 'name': 'ResNet18'},
        {'model_fn': resnet3d50, 'name': 'ResNet50'},
    ]
    
    test_shapes = [
        (1, 1, 64, 64, 64),
        (1, 1, 128, 128, 128),
    ]
    
    for config in model_configs:
        logger.info(f"\nTesting {config['name']}:")
        model = config['model_fn'](
            num_classes=2,
            input_channels=1,
            use_se=True,
            dropout_rate=0.2
        )
        
        for shape in test_shapes:
            x = torch.randn(*shape)
            with torch.no_grad():
                output = model(x)
            logger.info(f"Input shape: {x.shape} → Output shape: {output.shape}")