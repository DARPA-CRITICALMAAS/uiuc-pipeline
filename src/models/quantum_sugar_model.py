import logging
from types import SimpleNamespace

from .pipeline_pytorch_model import pipeline_pytorch_model
from submodules.models.quantum_sugar.model import DARPA_DET

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class quantum_sugar_model(pipeline_pytorch_model):
    
    def __init__(self):
        self.name = 'quantum sugar'
        self.checkpoint = 'submodules/models/quantum_sugar/model/saved_model/saved_poly_model.hdf5'

        self.args = SimpleNamespace(model='unet_cat')

    #@override
    def load_model(self):
        self.model = DARPA_DET.load_from_checkpoint(self.checkpoint, args=self.args)
        self.model.eval()

        return self.model

# import lightning.pytorch as pl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)
    
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

# class inf_MAP_UNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         n_channels = 6
#         n_classes = 1

#         self.unet = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
#         self.unet.inc = (DoubleConv(n_channels, 64))
#         self.unet.outc = (OutConv(64, n_classes))
    
#     def forward(self, batch):
#         x = self.unet(batch)
#         x = F.interpolate(x,size=256,mode="bilinear")
#         x = torch.sigmoid(x)
#         return x

# class DARPA_DET(pl.LightningModule):
#     def __init__(self) -> None:
#         super().__init__()
#         self.model = inf_MAP_UNet()        

#     def forward(self,batch):
#         output = self.model(batch)
#         return output
      