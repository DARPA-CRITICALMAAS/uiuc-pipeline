import os
import gc
import logging
import numpy as np
from time import time
from types import SimpleNamespace
from patchify import patchify, unpatchify

import torch
from torchvision import transforms

from src.patching import unpatch_img
from src.pipeline_manager import pipeline_manager
from .pipeline_pytorch_model import pipeline_pytorch_model
from cmaas_utils.types import MapUnitType

import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class happy_facet_model(pipeline_pytorch_model):
    def __init__(self):
        self.name = 'happy-facet'
        self.version = '0.1'
        self.feature_type = MapUnitType.POLYGON
        self._checkpoint = 'happy-facet-0.1.pt'
        self.est_patches_per_sec = 280 # Only used for estimating inference time
        self.device = torch.device("cuda")

        # Modifiable parameters
        self.batch_size = 64
        self.patch_size = 256
        self.patch_overlap = 64
        self.unpatch_mode = 'discard'


    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, self._checkpoint) 
        model_ckpt = torch.load(model_path)
        self.model = DeepLabV3Plus()
        self.model.load_state_dict(model_ckpt['model_state_dict'])
        self.model.eval()

        return self.model
    

    def my_norm(self, data):
        data = data / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)[None, :, None, None].expand(*data.shape)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)[None, :, None, None].expand(*data.shape)
        data = (data - mean)/std

        return data
    
    
    def inference(self, image, legend_images, data_id=-1):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""         

        # Get the size of the map
        map_channels, map_height, map_width = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1:   
            image = np.concatenate([image,image,image], axis=0)

        # Generate patches
        right_pad = self.patch_size - (map_width % self.patch_size)
        bottom_pad = self.patch_size - (map_height % self.patch_size)
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)
        map_patches = patchify(padded_image, (3, self.patch_size, self.patch_size), step=self.patch_size-self.patch_overlap)

        cols = map_patches.shape[1]
        rows = map_patches.shape[2]

        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        map_patches = torch.Tensor(map_patches).to(self.device)
        map_patches = self.my_norm(map_patches)

        map_prediction = np.zeros((1, map_height, map_width), dtype=np.float32)
        map_confidence = np.zeros((1, map_height, map_width), dtype=np.float32)

        legend_index = 1
        for legend_img in legend_images:
            lgd_stime = time()

            # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
            if legend_img.shape[0] == 1:
                legend_img = np.concatenate([legend_img,legend_img,legend_img], axis=0)

            legend_tensor = torch.Tensor(legend_img).to(self.device)
            resize_legend = transforms.Resize((self.patch_size, self.patch_size), antialias=None)
            legend_tensor = resize_legend(legend_tensor)

            # Create legend array to merge with patches
            legend_patches = torch.stack([legend_tensor for i in range(self.batch_size)], dim=0)
            legend_patches = self.my_norm(legend_patches)
            
            # Perform Inference in batches
            prediction_patches = []
            with torch.no_grad():
                for i in range(0, len(map_patches), self.batch_size):
                    prediction = self.model(map_patches[i:i+self.batch_size], legend_patches[:len(map_patches[i:i+self.batch_size])])
                    prediction = torch.softmax(prediction, dim=1)[:,-1].cpu().numpy().astype(np.float32)
                    prediction_patches.append(prediction)

            # unpatch
            prediction_patches = np.concatenate(prediction_patches, axis=0)
            prediction_patches = prediction_patches.reshape([1, cols, rows, 1, self.patch_size, self.patch_size])
            unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=self.patch_overlap, mode=self.unpatch_mode)
            prediction_image = unpatch_image[:,:map_height,:map_width]

            # Add legend to prediction mask
            map_prediction[prediction_image >= map_confidence] = legend_index
            map_confidence = np.maximum(map_confidence, prediction_image)
            
            gc.collect() # This is needed otherwise gpu memory is not freed up on each loop

            legend_index += 1
            lgd_time = time() - lgd_stime
        # Minimum confidence threshold for a prediction
        map_prediction[map_confidence <= 0.333] = 0

        return map_prediction
    


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# UNet Encoder to extract low-level and high-level features
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=6):
        super(UNetEncoder, self).__init__()
        self.in_channels = in_channels
        if in_channels == 6:
            self.inc = DoubleConv(in_channels, 64)  # Example input channels = 6
        else:
            self.inc_2 = DoubleConv(in_channels, 64)  # Example input channels = 12
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)


    def init_weights(self):
        checkpoint_path = "/u/dkwark/bcxi/shared/models/icy-resin-0.2.ckpt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
        self.load_state_dict(state_dict, strict=False)


    def forward(self, x):
        low_level_features = self.inc_2(x) if self.in_channels == 12 else self.inc(x)
        x = self.down1(low_level_features)
        x = self.down2(x)
        x = self.down3(x)
        high_level_features = self.down4(x)
        return low_level_features, high_level_features
    
# Atrous Spatial Pyramid Pooling (ASPP) Module
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.concat_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        out1 = self.conv1(x)
        out2 = self.conv3_1(x)
        out3 = self.conv3_2(x)
        out4 = self.conv3_3(x)
        out5 = F.interpolate(self.conv_pool(self.pool(x)), size=size, mode='bilinear', align_corners=False)
        x = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return self.concat_conv(x)


# DeepLabV3+ Model
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLabV3Plus, self).__init__()
        self.encoder = UNetEncoder()
        self.encoder.init_weights()
        self.aspp = ASPP(1024, 256)  # Adjust in_channels (1024) as per your encoder's last layer
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x, legend):

        x = torch.cat((x, legend), axis=1)
        # Encoder
        low_level_features, high_level_features = self.encoder(x)

        # ASPP
        aspp_output = self.aspp(high_level_features)

        # Low-level feature projection
        low_level_features = self.low_level_proj(low_level_features)

        # Upsample ASPP output and concatenate with low-level features
        aspp_upsampled = F.interpolate(aspp_output, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        decoder_input = torch.cat([aspp_upsampled, low_level_features], dim=1)

        # Decoder
        output = self.decoder(decoder_input)

        # Final upsample to match input size
        return F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
