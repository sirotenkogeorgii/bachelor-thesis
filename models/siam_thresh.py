import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Optional, Callable


class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        
        return F.relu(x, inplace=True)


class UpSamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels_down: int,
        in_channels_path: int,
        reduce_dim_to: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if conv_block is None:
            conv_block = ConvBlock

        self.conv1 = conv_block(in_channels_path, reduce_dim_to, 1)
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels_down, out_channels=reduce_dim_to, kernel_size=2, stride=2)
        self.conv2 = conv_block(reduce_dim_to * 2, reduce_dim_to, 3, padding="same")
        self.conv3 = conv_block(reduce_dim_to, reduce_dim_to, 3, padding="same")

    def forward(self, contract_path: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        contract_path = self.conv1(contract_path)
        x = self.upsample(x)
        concatenated = torch.cat([contract_path, x], dim=1)
        concatenated = self.conv2(concatenated)
        concatenated = self.conv3(concatenated)

        return concatenated
        

class SiamThresh(nn.Module):
    def __init__(
        self, 
        out_channels: int,
        pretrained: bool = False,
        thresholds: dict[str, float] = None
    ) -> None:
        super().__init__()

        return_nodes = {
            "x": "layer0",
            "features.1.1.add": "layer1",
            "features.2.3.add": "layer2",
            "features.4.0.block.0": "layer3",
            "features.6.0.block.0": "layer4",
            "features.7": "layer5"
        }

        self.thresholds = thresholds
        if thresholds is None:
            self.thresholds = {
                "threshold0": 0.4,
                "threshold1": 0.6,
                "threshold2": 0.8,
                "threshold3": 1.0,
                "threshold4": 1.2,
                "threshold5": 1.4,
            }

        weights = None
        if pretrained:
            weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
            
        model = torchvision.models.efficientnet_v2_s(weights=weights)
        self._backbone = create_feature_extractor(model, return_nodes=return_nodes)

        self.upsample1 = UpSamplingBlock(1280, 960, 512)
        self.upsample2 = UpSamplingBlock(512, 256, 256)
        self.upsample3 = UpSamplingBlock(256, 48, 128)
        self.upsample4 = UpSamplingBlock(128, 24, 64)
        self.upsample5 = UpSamplingBlock(64, 3, 16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1)

        self._mid_level_features = None
    
    def set_thresholds(self, thresholds: dict) -> None:
        self.thresholds = thresholds

    def freeze_backbone(self) -> None:
        for param in self._backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self) -> None:
        for param in self._backbone.parameters():
            param.requires_grad = True
            
    def forward(self, x: torch.Tensor, second_prop: bool = False) -> torch.Tensor:
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float()
        if x.shape[1] == 1: x = x.expand(x.shape[0], 3, *x.shape[2:])

        feature_maps = self._backbone(x)
        if not second_prop:
            self._mid_level_features = feature_maps.copy()

        if second_prop:
            for layer_num in range(len(self.thresholds)):
                current_layer = f"layer{layer_num}"
                feature_maps[current_layer] = torch.where(abs(feature_maps[current_layer] - self._mid_level_features[current_layer]) <= self.thresholds[f"threshold{layer_num}"], \
                                                    0, feature_maps[current_layer])

        upsampled1 = self.upsample1(feature_maps["layer4"], feature_maps["layer5"])
        upsampled2 = self.upsample2(feature_maps["layer3"], upsampled1)
        upsampled3 = self.upsample3(feature_maps["layer2"], upsampled2)
        upsampled4 = self.upsample4(feature_maps["layer1"], upsampled3)
        upsampled5 = self.upsample5(feature_maps["layer0"], upsampled4)

        prediction = self.conv3(upsampled5)
        return F.sigmoid(prediction)