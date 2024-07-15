import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Optional, Callable

class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d
        self.upsample_mode = 'bilinear'

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.operation_function = self._concatenation


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f
    

class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock2D, self).__init__(in_channels=in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x  


class Respath(torch.nn.Module):
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
		
	'''

	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()

		self.respath_length = respath_length
		self.shortcuts = torch.nn.ModuleList([])
		self.convs = torch.nn.ModuleList([])

		for i in range(self.respath_length):
			if(i==0):
				self.shortcuts.append(BasicConv(num_in_filters, num_out_filters, kernel_size = 1, relu=False))
				self.convs.append(BasicConv(num_in_filters, num_out_filters, kernel_size = 3, relu=True, padding=1))
			else:
				self.shortcuts.append(BasicConv(num_out_filters, num_out_filters, kernel_size = 1, relu=False))
				self.convs.append(BasicConv(num_out_filters, num_out_filters, kernel_size = 3, relu=True, padding=1))
		
	
	def forward(self,x):
		for i in range(self.respath_length):
			shortcut = self.shortcuts[i](x)
			direct = self.convs[i](x)
			output = direct + shortcut
			output = torch.nn.functional.relu(output)
		return output


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


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
        conv_block: Optional[Callable[..., nn.Module]] = None,
        make_upsample: bool = True
    ) -> None:
        super().__init__()

        if conv_block is None:
            conv_block = ConvBlock

        self.conv1 = conv_block(in_channels_path, reduce_dim_to, 1)
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels_down, out_channels=reduce_dim_to, kernel_size=2, stride=2) if make_upsample else conv_block(in_channels_down, reduce_dim_to, 1)
        # self.conv2 = conv_block(reduce_dim_to * 2 if make_upsample else reduce_dim_to + in_channels_down, reduce_dim_to, 3, padding="same")
        self.conv2 = conv_block(reduce_dim_to * 2, reduce_dim_to, 3, padding="same")
        self.conv3 = conv_block(reduce_dim_to, reduce_dim_to, 3, padding="same")

    def forward(self, contract_path: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        contract_path = self.conv1(contract_path)
        x = self.upsample(x)
        concatenated = torch.cat([contract_path, x], dim=1)
        concatenated = self.conv2(concatenated)
        concatenated = self.conv3(concatenated)

        return concatenated



class CDUnetResPath(nn.Module):
    def __init__(
        self, 
        out_channels: int,
        pretrained: bool = False,
        thresholds: dict[str, float] = None
    ) -> None:
        super().__init__()

        self.return_nodes = {
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
        self._backbone = create_feature_extractor(model, return_nodes=self.return_nodes)

        self.respaths = torch.nn.ModuleList([
             Respath(3, 3, 6),
             Respath(24, 24, 5),
             Respath(48, 48, 4),
             Respath(256, 256, 3),
             Respath(960, 960, 2)
        ])

        self.attention_gate1 = GridAttentionBlock2D(in_channels=960, gating_channels=1280)
        self.upsample1 = UpSamplingBlock(1280, 960, 512)
        self.attention_gate2 = GridAttentionBlock2D(in_channels=256, gating_channels=512)
        self.upsample2 = UpSamplingBlock(512, 256, 256)
        self.attention_gate3 = GridAttentionBlock2D(in_channels=48, gating_channels=256)
        self.upsample3 = UpSamplingBlock(256, 48, 128)
        self.attention_gate4 = GridAttentionBlock2D(in_channels=24, gating_channels=128)
        self.upsample4 = UpSamplingBlock(128, 24, 64)
        self.attention_gate5 = GridAttentionBlock2D(in_channels=3, gating_channels=64)
        self.upsample5 = UpSamplingBlock(64, 3, 16)


        self.conv3 = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1)
        
        self._mid_level_features = None
        self.activated_attention_gates = False
    
    def set_thresholds(self, thresholds: dict) -> None:
        self.thresholds = thresholds

    def freeze_backbone(self) -> None:
        for param in self._backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self) -> None:
        for param in self._backbone.parameters():
            param.requires_grad = True
    
    def activate_attention_gates(self):  self.activated_attention_gates = True
    def deactivate_attention_gates(self):  self.activated_attention_gates = False
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray): y = torch.from_numpy(y).float()
        if x.shape[1] == 1: x = x.expand(x.shape[0], 3, *x.shape[2:])
        if y.shape[1] == 1: y = y.expand(y.shape[0], 3, *y.shape[2:])

        feature_maps_x = self._backbone(x)
        feature_maps_y = self._backbone(y)

        for layer_num in range(len(self.return_nodes) - 1): # -1!
            current_layer = f"layer{layer_num}"

            diff = torch.abs(self.respaths[layer_num](feature_maps_y[current_layer]) - self.respaths[layer_num](feature_maps_x[current_layer]))
            feature_maps_y[current_layer] = diff
        
        if self.activated_attention_gates:
            upsampled1 = self.upsample1(self.attention_gate1(feature_maps_y["layer4"], feature_maps_y["layer5"])[0], feature_maps_y["layer5"])
            upsampled2 = self.upsample2(self.attention_gate2(feature_maps_y["layer3"], upsampled1)[0], upsampled1)
            upsampled3 = self.upsample3(self.attention_gate3(feature_maps_y["layer2"], upsampled2)[0], upsampled2)
            upsampled4 = self.upsample4(self.attention_gate4(feature_maps_y["layer1"], upsampled3)[0], upsampled3)
            upsampled5 = self.upsample5(self.attention_gate5(feature_maps_y["layer0"], upsampled4)[0], upsampled4)
        else:
            upsampled1 = self.upsample1(feature_maps_y["layer4"], feature_maps_y["layer5"])
            upsampled2 = self.upsample2(feature_maps_y["layer3"], upsampled1)
            upsampled3 = self.upsample3(feature_maps_y["layer2"], upsampled2)
            upsampled4 = self.upsample4(feature_maps_y["layer1"], upsampled3)
            upsampled5 = self.upsample5(feature_maps_y["layer0"], upsampled4)

        prediction = self.conv3(upsampled5)
        return F.sigmoid(prediction)