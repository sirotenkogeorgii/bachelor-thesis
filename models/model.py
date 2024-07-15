# from .change_vit import *
# from change_vit import *
# from models.change_vit import *
from models.change_vit import Trainer, Encoder, Decoder, DinoVisionTransformer, PatchEmbed, Block, MemEffAttention, Mlp, BasicBlock, FeatureInjector, BlockInjector, CrossAttention, MlpDecoder, ResNet
from models.efficientunet import CDUnet, UpSamplingBlock, ConvBlock
from models.siamconc import SiamUnet_conc
from models.siamdiff import SiamUnet_diff
from models.stackunet import Unet
from models.efficientunet_respath_attn import CDUnetResPath, Respath, BasicConv, GridAttentionBlock2D

import torch
import torchvision
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = Path(__file__).parent.absolute()

def preprocess(img1):
    img1_norm = TF.normalize(TF.to_tensor(img1), mean=(0.485), std=(0.229))
    return img1_norm


class WrapperPredDiffModel(torch.nn.Module):
    def __init__(self, original_model):
        super(WrapperPredDiffModel, self).__init__()
        self.original_model = original_model
        self.original_model.deactivate_threshold_mode()

    def forward(self, input1, input2):
        output1 = (self.original_model(None, input1, one_pass=True) > 0.5).float()
        output2 = (self.original_model(None, input2, one_pass=True) > 0.5).float()
        return torch.clamp(abs(output1 - output2), 0.0, 1.0)
    

class WrapperFeaturesDiffModel(torch.nn.Module):
    def __init__(self, original_model):
        super(WrapperFeaturesDiffModel, self).__init__()
        self.original_model = original_model
        self.original_model.activate_threshold_mode()

    def forward(self, input1, input2):
        output = self.original_model(input1, input2)
        return output


class SemiSupervisedModel(torch.nn.Module):
    def __init__(self, model_name, thresholds=None, output_size=[500, 500]):
        super(SemiSupervisedModel, self).__init__()
        model = torch.load(f"{CURRENT_DIR}/../weights/semi_supervised_cracks/crack_segmentation.pth", map_location=device).to(device)
        self.input_resizer = torchvision.transforms.Resize([512, 512])
        self.output_resizer = torchvision.transforms.Resize(output_size)
        if thresholds is not None: model.set_thresholds(thresholds)
        else:
            if model_name.lower() == "siamthreshevolv": model.set_thresholds(
                {'threshold0': 0.05, 'threshold1': 0.1, 'threshold2': 0.2, 'threshold3': 0.3, 'threshold4': 0.4, 'threshold5': 0.5}
                )
        self._wrap_and_set_model(model_name, model)
        
    def _wrap_and_set_model(self, model_name, model):
        model_name = model_name.lower()
        if model_name == "siamthreshevolv" or model_name == "siamthresh": self.model = WrapperFeaturesDiffModel(model)
        elif model_name == "pred_diff": self.model = WrapperPredDiffModel(model)
        else: raise Exception("Unknown model name. Available model: ['siamthresh', 'siamthreshevolv', 'pred_diff']")

    def forward(self, sample_before, sample_after):
        self.model.train()
        sample_before = self.input_resizer(sample_before).to(device)
        sample_after = self.input_resizer(sample_after).to(device)

        with torch.no_grad():
            result = self.model(sample_before, sample_after)
        return self.output_resizer(result)


class SupervisedModel(torch.nn.Module):
    def __init__(self, weights_path, input_size, output_size=[500, 500]):
        super(SupervisedModel, self).__init__()
        self.model = torch.load(weights_path, map_location=device)
        self.input_resizer = torchvision.transforms.Resize(input_size)
        self.output_resizer = torchvision.transforms.Resize(output_size)
        self._model_preparation(weights_path)

    def _model_preparation(self, weights_path):
        if ("efficient" in weights_path.lower() or "eff" in weights_path.lower()):
            if ("attention" in weights_path.lower() or "attn" in weights_path.lower()): self.model.activate_attention_gates() 
            else: 
                try: self.model.deactivate_attention_gates() 
                except: self.model.deactivate_threshold_mode()

    
    def forward(self, sample_before, sample_after):
        sample_before = self.input_resizer(sample_before).to(device)
        sample_after = self.input_resizer(sample_after).to(device)
        result = self.model(sample_before, sample_after)
        return self.output_resizer(result)
