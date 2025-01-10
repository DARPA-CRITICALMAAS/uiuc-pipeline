import os
import gc
import logging
import numpy as np
from time import time
from types import SimpleNamespace
from patchify import patchify, unpatchify

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.patching import unpatch_img
from src.pipeline_manager import pipeline_manager
from .pipeline_pytorch_model import pipeline_pytorch_model
from cmaas_utils.types import MapUnitType

import cv2

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class adventurous_ampere_model(pipeline_pytorch_model):
    def __init__(self):
        self.name = 'adventurous_ampere'
        self.version = '0.1'
        self.feature_type = MapUnitType.POLYGON
        self._checkpoint = 'adventurous-ampere-0.1.pt'
        self.est_patches_per_sec = 560 # Only used for estimating inference time

        self.device = torch.device("cuda")

        # Modifiable parameters
        self.batch_size = 64
        self.patch_size = 256
        self.patch_overlap = 64
        self.unpatch_mode = 'discard'

    #@override
    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, self._checkpoint) 
        self.model = Segmentor()

        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt['model_state_dict'])
        self.model.eval()

        return self.model
    
    def convert_to_lab_cpu(self, rgb_np):
        """
        rgb_np shape: (N, 3, H, W), dtype=uint8 or float32 in [0,255].
        Convert on CPU using OpenCV, return float32 array in Lab space.
        """

        nhwc = np.transpose(rgb_np, (0, 2, 3, 1)).astype(np.uint8)
        lab = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32) for img in nhwc])
        # Rescale L from [0..255] to [0..100]
        lab[..., 0] = lab[..., 0] / 255.0 * 100.0
        # Shift a,b from [0..255] to [-128..127]
        lab[..., 1:] -= 128.0
        return np.transpose(lab, (0, 3, 1, 2))

    def convert_to_hsv_cpu(self, rgb_np):
        nhwc = np.transpose(rgb_np, (0, 2, 3, 1)).astype(np.uint8)
        hsv = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) for img in nhwc])
        # Rescale:
        # H in [0..179], scale to [0..360]
        hsv[..., 0] = hsv[..., 0] / 179.0 * 360.0
        # S, V in [0..255], scale to [0..1]
        hsv[..., 1:] /= 255.0
        return np.transpose(hsv, (0, 3, 1, 2))

    def convert_to_yuv_cpu(self, rgb_np):
        nhwc = np.transpose(rgb_np, (0, 2, 3, 1)).astype(np.uint8)
        yuv = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float32) for img in nhwc])
        return np.transpose(yuv, (0, 3, 1, 2))

    def convert_to_hed_cpu(self, rgb_np):
        nhwc = np.transpose(rgb_np, (0, 2, 3, 1)).astype(np.float32) / 255.0  # Scale to [0, 1]
        eps = 1e-8
        od = -np.log((nhwc + eps) / (1.0 + eps))  # Optical density

        # Fixed conversion matrix
        conversion_matrix = np.array([
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78]
        ], dtype=np.float32)

        # Apply matrix multiplication across the batch
        hed = np.tensordot(od, conversion_matrix, axes=([3], [0]))
        return np.transpose(hed, (0, 3, 1, 2))

    def convert_to_rgb_cpu(self, rgb_np):

        return rgb_np.astype(np.float32)

    def resize_image(self, image, target_size=(256, 256)):
        """
        Resize image to target_size.
        Args:
            image (np.array): Image data in CHW format.
            target_size (tuple): Target size (H, W).
        Returns:
            np.array: Resized image in CHW format.
        """
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image, (2, 0, 1))
        return image

    # -----------
    # Normalization (CPU)
    # -----------
    def lab_norm_cpu(self, lab_np):
        # lab_np shape: (N, 3, H, W)
        mean = np.array([39.101, -128.178, -126.484], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.array([25.0, 7.736, 12.594], dtype=np.float32).reshape(1, -1, 1, 1)
        return (lab_np - mean) / std

    def hsv_norm_cpu(self, hsv_np):
        mean = np.array([230.569/360.0, 0.001, 0.816], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.array([212.114/360.0, 0.001, 0.215], dtype=np.float32).reshape(1, -1, 1, 1)
        return (hsv_np - mean) / std

    def rgb_norm_cpu(self, rgb_np):
        rgb_np = rgb_np / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, -1, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, -1, 1, 1)
        return (rgb_np - mean) / std

    def yuv_norm_cpu(self, yuv_np):
        mean = np.array([187.340, -11.024, 7.415], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.array([56.760, 21.319, 24.885], dtype=np.float32).reshape(1, -1, 1, 1)
        return (yuv_np - mean) / std

    def hed_norm_cpu(self, hed_np):
        mean = np.array([0.569, 1.313, 0.882], dtype=np.float32).reshape(1, -1, 1, 1)
        std  = np.array([1.381, 3.029, 2.244], dtype=np.float32).reshape(1, -1, 1, 1)
        return (hed_np - mean) / std

    # @override
    def inference(self, image, legend_images, data_id=-1):
        """
        Args:
            image (np.array): Image data in CHW format.
            legend_images (list): List of map_unit swatch images in CHW format.
            data_id (int): Data id for logging purposes.
        
        Returns:
            np.array: Prediction mask in CHW format.
        """          
        # For profiling memory usage 
        #torch.cuda.memory._record_memory_history()

        # Get the size of the map
        map_channels, map_height, map_width = image.shape

        # Generate patches
        # Pad image so we get a size that can be evenly divided into patches.
        right_pad = self.patch_size - (map_width % self.patch_size)
        bottom_pad = self.patch_size - (map_height % self.patch_size)
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)
        map_patches = patchify(padded_image, (3, self.patch_size, self.patch_size), step=self.patch_size-self.patch_overlap)

        cols = map_patches.shape[1]
        rows = map_patches.shape[2]

        # Flatten row col dims and normalize map patches to [0,1]
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)

        map_lab = self.lab_norm_cpu(self.convert_to_lab_cpu(map_patches))
        map_hsv = self.hsv_norm_cpu(self.convert_to_hsv_cpu(map_patches))
        map_rgb = self.rgb_norm_cpu(self.convert_to_rgb_cpu(map_patches))
        map_yuv = self.yuv_norm_cpu(self.convert_to_yuv_cpu(map_patches))
        map_hed = self.hed_norm_cpu(self.convert_to_hed_cpu(map_patches))

        # pipeline_manager.log(logging.DEBUG, f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
        map_prediction = np.zeros((1, map_height, map_width), dtype=np.float32)
        map_confidence = np.zeros((1, map_height, map_width), dtype=np.float32)

        legend_index = 1
        for legend_img in legend_images:
            lgd_stime = time()
            # Determine crop margins
            h, w = legend_img.shape[1], legend_img.shape[2]
            crop_h, crop_w = int(h * 0.1), int(w * 0.1)

            # Crop 10% from each side
            legend_img = legend_img[:, crop_h:h - crop_h, crop_w:w - crop_w]

            # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
            if legend_img.shape[0] == 1:
                legend_img = np.concatenate([legend_img,legend_img,legend_img], axis=0)
            
            legend_img = self.resize_image(legend_img, target_size=(self.patch_size, self.patch_size))
            legend_img = np.expand_dims(legend_img, axis=0)

            legend_lab = self.lab_norm_cpu(self.convert_to_lab_cpu(legend_img))
            legend_hsv = self.hsv_norm_cpu(self.convert_to_hsv_cpu(legend_img))
            legend_rgb = self.rgb_norm_cpu(self.convert_to_rgb_cpu(legend_img))
            legend_yuv = self.yuv_norm_cpu(self.convert_to_yuv_cpu(legend_img))
            legend_hed = self.hed_norm_cpu(self.convert_to_hed_cpu(legend_img))

            legend_lab_t = torch.from_numpy(legend_lab).to(self.device)
            legend_hsv_t = torch.from_numpy(legend_hsv).to(self.device)
            legend_rgb_t = torch.from_numpy(legend_rgb).to(self.device)
            legend_yuv_t = torch.from_numpy(legend_yuv).to(self.device)
            legend_hed_t = torch.from_numpy(legend_hed).to(self.device)


            # Perform Inference in batches
            prediction_patches = []

            N = map_patches.shape[0]
            with torch.no_grad():
                for start_idx in range(0, N, self.batch_size):
                    end_idx = start_idx + self.batch_size

                    # Convert sub-batch to torch (already in color spaces, just slice)
                    lab_sub_t = torch.from_numpy(map_lab[start_idx:end_idx]).to(self.device)
                    hsv_sub_t = torch.from_numpy(map_hsv[start_idx:end_idx]).to(self.device)
                    rgb_sub_t = torch.from_numpy(map_rgb[start_idx:end_idx]).to(self.device)
                    yuv_sub_t = torch.from_numpy(map_yuv[start_idx:end_idx]).to(self.device)
                    hed_sub_t = torch.from_numpy(map_hed[start_idx:end_idx]).to(self.device)

                    subB = lab_sub_t.shape[0]
                    # Repeat the legend for sub-batch
                    legend_lab_batch = legend_lab_t.repeat(subB, 1, 1, 1)
                    legend_hsv_batch = legend_hsv_t.repeat(subB, 1, 1, 1)
                    legend_rgb_batch = legend_rgb_t.repeat(subB, 1, 1, 1)
                    legend_yuv_batch = legend_yuv_t.repeat(subB, 1, 1, 1)
                    legend_hed_batch = legend_hed_t.repeat(subB, 1, 1, 1)

                    # concat each sub-batch with the legend
                    lab_sub_t = torch.cat([lab_sub_t, legend_lab_batch], dim=1)
                    hsv_sub_t = torch.cat([hsv_sub_t, legend_hsv_batch], dim=1)
                    rgb_sub_t = torch.cat([rgb_sub_t, legend_rgb_batch], dim=1)
                    yuv_sub_t = torch.cat([yuv_sub_t, legend_yuv_batch], dim=1)
                    hed_sub_t = torch.cat([hed_sub_t, legend_hed_batch], dim=1)

                    # Forward
                    prediction = self.model(
                        lab_sub_t, hsv_sub_t, hed_sub_t, yuv_sub_t, rgb_sub_t
                    )[0]  # shape (subB,2,ph,pw)
                    prediction = torch.softmax(prediction, dim=1)[:,-1].cpu().numpy().astype(np.float32)
                    prediction_patches.append(prediction)

                    # clean up
                    del lab_sub_t, hsv_sub_t, rgb_sub_t, yuv_sub_t, hed_sub_t
                    del legend_lab_batch, legend_hsv_batch, legend_rgb_batch, legend_yuv_batch, legend_hed_batch
                    
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
        map_prediction[map_confidence < 0.333] = 0
        
        return map_prediction
    


class LayerNorm(nn.Module):
    """ LayerNorm supporting channels_last and channels_first data formats. """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    """ ConvNeXt Block. """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    """ ConvNeXt backbone. """
    def __init__(
            self, 
            in_chans=3, 
            depths=[3, 3, 27, 3], 
            dims=[96, 192, 384, 768], 
            drop_path_rate=0.2, 
            layer_scale_init_value=1e-6, 
            out_indices=[0, 1, 2, 3]
        ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices
        for i_layer in range(4):
            layer = LayerNorm(dims[i_layer], eps=1e-6, data_format="channels_first")
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs

    def forward(self, x):
        return self.forward_features(x)


class PPM(nn.ModuleList):
    """ Pooling Pyramid Module used in PSPNet. """
    def __init__(self, pool_scales, in_channels, channels, conv_cfg=None, norm_cfg=None, act_cfg=None, align_corners=False):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels) if norm_cfg else nn.Identity(),
                    nn.ReLU(inplace=True) if act_cfg else nn.Identity()
                )
            )

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs
    
class UPerHead(nn.Module):
    """
    A close reproduction of MMSeg's UPerHead multi-scale logic:
    1) Lateral convs -> standard FPN top-down to get [p0, p1, p2, p3].
    2) PPM on p3 => p3_ppm.
    3) 3×3 conv on each p_i (including p3_ppm).
    4) Upsample each p_i to p0 size and concatenate.
    5) A final 3×3 conv and 1×1 classifier produce the final segmentation.
    """
    def __init__(self, 
                 in_channels,    # e.g. [96, 192, 384, 768]
                 channels=256,   # FPN and final feature channels
                 num_classes=2,
                 pool_scales=(1, 2, 3, 6),
                 align_corners=False):
        super().__init__()
        assert len(in_channels) == 4, "UPerNet expects 4 scales."
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners

        # --------------------------
        # 1) FPN: Lateral and output convs
        # --------------------------
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ic in in_channels:
            # Lateral projection to 'channels'
            self.lateral_convs.append(
                nn.Conv2d(ic, channels, kernel_size=1)
            )
            # FPN output conv
            self.fpn_convs.append(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            )

        # --------------------------
        # 2) PPM on the top feature (p3)
        # --------------------------
        self.ppm = PPM(pool_scales, in_channels=channels, channels=channels // 4,
                       align_corners=align_corners)
        self.ppm_bottleneck = nn.Conv2d(
            channels + len(pool_scales)*(channels // 4),
            channels,
            kernel_size=3,
            padding=1
        )

        # --------------------------
        # 3) Final "merge" conv after upsampling
        # We'll gather p0, p1, p2, p3 (post-PPM) at p0's resolution
        # Then do a final conv
        # --------------------------
        self.fpn_bottleneck = nn.Conv2d(
            channels * 4,   # concat p0, p1, p2, p3
            channels,
            kernel_size=3,
            padding=1
        )

        # --------------------------
        # 4) Classifier
        # --------------------------
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def fpn_forward(self, laterals):
        """
        Standard single-pass FPN top-down.
        laterals: [lat0, lat1, lat2, lat3]
        Returns: [p0, p1, p2, p3]
        """
        # We'll store the final FPN outputs in p_outs
        p_outs = [None] * 4
        # The top scale
        p_outs[3] = self.fpn_convs[3](laterals[3])

        # Move from top (3) down to scale 0
        for i in range(2, -1, -1):
            # upsample p_(i+1) to p_i size
            size = laterals[i].shape[2:]
            top = F.interpolate(
                p_outs[i+1],
                size=size,
                mode='bilinear',
                align_corners=self.align_corners
            )
            # add
            merged = laterals[i] + top
            p_outs[i] = self.fpn_convs[i](merged)

        return p_outs  # [p0, p1, p2, p3]

    def forward(self, inputs):
        """
        inputs = [f0, f1, f2, f3]
        Steps:
          1) Lateral conv => laterals
          2) FPN => [p0, p1, p2, p3]
          3) PPM on p3
          4) 3×3 conv on each p_i (post-PPM on p3)
          5) Upsample p1,p2,p3 to p0 size, concat => final
          6) self.cls_seg final classifier
        """
        # 1) Build lateral features
        laterals = []
        for i in range(4):
            lat = self.lateral_convs[i](inputs[i])  # NxC x H_i x W_i
            laterals.append(lat)

        # 2) FPN pass
        fpn_outs = self.fpn_forward(laterals)  # [p0, p1, p2, p3]

        # 3) PPM on the top feature p3
        p3 = fpn_outs[3]
        ppm_outs = self.ppm(p3)  # list of Nx(C//4) x ...
        p3_ppm = torch.cat([p3] + ppm_outs, dim=1)
        p3_ppm = self.ppm_bottleneck(p3_ppm)
        fpn_outs[3] = p3_ppm

        out_feats = fpn_outs 

        # 5) Upsample all scales to p0’s size and concatenate
        p0 = out_feats[0]
        out_size = p0.shape[2:]  # H0, W0

        p1_up = F.interpolate(out_feats[1], size=out_size, mode='bilinear', align_corners=self.align_corners)
        p2_up = F.interpolate(out_feats[2], size=out_size, mode='bilinear', align_corners=self.align_corners)
        p3_up = F.interpolate(out_feats[3], size=out_size, mode='bilinear', align_corners=self.align_corners)

        merged = torch.cat([p0, p1_up, p2_up, p3_up], dim=1)  # Nx(4C) x H0 x W0
        merged = self.fpn_bottleneck(merged)                 # NxC x H0 x W0

        # 6) Final classifier
        seg_out = self.cls_seg(merged)                       # Nx(num_classes) x H0 x W0
        return seg_out

class AuxiliaryHead(nn.Module):
    """ Auxiliary head for intermediate supervision. """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x, input_size):
        x = F.relu(self.conv(x))
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x
    

class FiveEncoderMoEDecoder(nn.Module):
    """
    Merges 5 encoders' multi-scale feats (0..3) with SoftGatingAggregator,
    then does a UPerHead on the fused feats.
    Optionally, we output the fused feature at scale=2 for auxiliary supervision.
    """
    def __init__(self, 
                 in_channels,  # [96,192,384,768]
                 decoder_channels=256,
                 num_classes=2,
                 with_moe_loss=True,
                 moe_loss_weight=1e-2,
                 pool_scales=(1,2,3,6),
                 align_corners=False,
                 aux_index=2):  # which scale to use for aux
        super().__init__()
        
        self.aggregators = nn.ModuleList()
        for c in in_channels:
            self.aggregators.append(
                SoftGatingAggregator(
                    in_channels=c, 
                    with_moe_loss=with_moe_loss, 
                    moe_loss_weight=moe_loss_weight
                )
            )

        # The main decode head (UPerHead) uses the 4 fused scales
        self.uper = UPerHead(
            in_channels=in_channels,  
            channels=decoder_channels,
            num_classes=num_classes,
            pool_scales=pool_scales,
            align_corners=align_corners
        )

        self.aux_index = aux_index  # e.g. 2 => scale=2

    def forward(self, feats1, feats2, feats3, feats4, feats5):
        """
        featsX = [fX_0, fX_1, fX_2, fX_3], each fX_s => [B, C_s, H_s, W_s]
        Returns:
          seg_logits: final segmentation 
          fused_feats: list [fused_0, fused_1, fused_2, fused_3]
          total_moe_loss
        """
        fused_feats = []
        total_moe_loss = torch.tensor(0.0, device=feats1[0].device)
        for s in range(4):
            # aggregator merges the 5 features at scale s
            fused_s, moe_loss_s = self.aggregators[s](
                feats1[s], feats2[s], feats3[s], feats4[s], feats5[s]
            )
            fused_feats.append(fused_s)
            total_moe_loss += moe_loss_s

        # UPerHead forward
        seg_logits = self.uper(fused_feats)  # => [B, num_classes, H0, W0]

        return seg_logits, fused_feats, total_moe_loss

class ConvFusion(nn.Module):
    """
    Learnable fusion of two feature maps via a simple 3x3 Conv + BN + ReLU.
    """
    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        # We concatenate along channel dimension, then reduce to out_channels
        self.conv = nn.Conv2d(
            in_channels1 + in_channels2, 
            out_channels, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f1, f2):
        """
        f1: NxC1xHxW
        f2: NxC2xHxW
        Output: Nx(out_channels)xHxW
        """
        x = torch.cat([f1, f2], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# ----------------------------------------------------
# (A) Soft Gating Aggregator
# ----------------------------------------------------
class SoftGatingAggregator(nn.Module):
    """
    Soft-Gating aggregator for 5 encoder features at a single scale.
    Each 'expert' is just the feature from one encoder.
    
    Steps:
      1) Stack or cat features => shape [B, 5*C, H, W].
      2) gating_logits => shape [B, 5, H, W].
      3) gates = softmax(gating_logits, dim=1).
      4) Weighted sum => fused feature [B, C, H, W].
      5) Optional load-balancing loss => encourages each 'expert' usage ~ 1/5.
    """
    def __init__(self, in_channels, with_moe_loss=True, moe_loss_weight=1e-2):
        super().__init__()
        self.in_channels = in_channels
        self.with_moe_loss = with_moe_loss
        self.moe_loss_weight = moe_loss_weight

        # we first cat => [B, 5*C, H, W], then reduce to 5 via conv.
        self.gate_conv = nn.Conv2d(5 * in_channels, 5, kernel_size=1, bias=True)

    def forward(self, f1, f2, f3, f4, f5):
        """
        f1..f5 each: [B, C, H, W]
        Returns: fused: [B, C, H, W], moe_loss (scalar)
        """
        B, C, H, W = f1.shape
        
        # 1) Concatenate features along channel => [B, 5*C, H, W]
        x = torch.cat([f1, f2, f3, f4, f5], dim=1)
        
        # 2) gating logits => [B, 5, H, W]
        gating_logits = self.gate_conv(x)
        
        # 3) softmax across "expert" dimension (dim=1 => 5 experts)
        gating_weights = F.softmax(gating_logits, dim=1)  # [B, 5, H, W]
        
        # 4) Weighted sum. We'll do it in a small loop or a more vectorized approach.
        #   Let's do: fused = sum_i( gating_weights[:, i] * f_i ).
        #   Each gating_weights[:, i] => shape [B, H, W], broadcast over channels.

        fused = torch.zeros_like(f1)  # [B, C, H, W]
        for i, fi in enumerate([f1, f2, f3, f4, f5]):
            # gating_weights[:, i, :, :] => shape [B, H, W], unsqueeze(1) => [B, 1, H, W]
            gi = gating_weights[:, i, :, :].unsqueeze(1)  # [B, 1, H, W]
            fused = fused + fi * gi  # broadcast multiply => [B, C, H, W]

        # 5) Optional load-balancing loss
        moe_loss = torch.tensor(0.0, device=f1.device)
        if self.with_moe_loss:
            # gating_weights shape [B, 5, H, W]
            # Temperature scaling to stabilize softmax probabilities
            temperature = 1.0  # Hinton et al., 2015
            gating_weights = F.softmax(gating_logits / temperature, dim=1)

            # Calculate average expert usage across batch and spatial dims
            alpha = gating_weights.mean(dim=(0, 2, 3))  # shape [num_experts]

            # Target distribution: uniform usage of experts
            num_experts = gating_weights.size(1)  # Dynamic number of experts
            target = torch.full_like(alpha, 1.0 / num_experts)

            # Use KL Divergence for better probabilistic comparison
            # Goodfellow et al., 2016, Chapter 3
            moe_loss = F.kl_div(alpha.log(), target, reduction='batchmean') * self.moe_loss_weight

            # Optional: Sparsity regularization for expert specialization
            # Fedus et al., Switch Transformers, 2021
            sparsity_loss = torch.norm(gating_weights, p=1, dim=1).mean()  # L1 norm
            moe_loss += 0.01 * sparsity_loss  # Scale the sparsity penalty
            
        return fused, moe_loss
    

class Segmentor(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        depths = [3, 3, 27, 3]
        dims = [96, 192, 384, 768]

        # 5 ConvNeXt encoders
        self.encoder1 = ConvNeXt(
            in_chans=6, 
            depths=depths,
            dims=dims, 
            drop_path_rate=0.2,
            out_indices=[0,1,2,3]
        )
        self.encoder2 = ConvNeXt(
            in_chans=6, 
            depths=depths, 
            dims=dims, 
            drop_path_rate=0.2,
            out_indices=[0,1,2,3]
        )
        self.encoder3 = ConvNeXt(
            in_chans=6, 
            depths=depths, 
            dims=dims, 
            drop_path_rate=0.2,
            out_indices=[0,1,2,3]
        )
        self.encoder4 = ConvNeXt(
            in_chans=6, 
            depths=depths, 
            dims=dims, 
            drop_path_rate=0.2,
            out_indices=[0,1,2,3]
        )
        self.encoder5 = ConvNeXt(
            in_chans=6, 
            depths=depths, 
            dims=dims, 
            drop_path_rate=0.2,
            out_indices=[0,1,2,3]
        )

        # MoE-based decoder
        self.decoder = FiveEncoderMoEDecoder(
            in_channels=dims,
            decoder_channels=256,
            num_classes=num_classes,
            with_moe_loss=True,
            moe_loss_weight=1e-2,
            aux_index=2  # for aux loss
        )

        # Auxiliary head that expects dims[2] = 384 channels
        # Because after the aggregator, scale=2 has shape [B, 384, H/16, W/16]
        self.aux_head = AuxiliaryHead(in_channels=dims[2], num_classes=num_classes)

    def forward(self, x1, x2, x3, x4, x5):
        B, _, H, W = x1.shape

        # 1) Get multi-scale feats from each encoder
        f1 = self.encoder1(x1)  # 4-scale list
        f2 = self.encoder2(x2)
        f3 = self.encoder3(x3)
        f4 = self.encoder4(x4)
        f5 = self.encoder5(x5)

        # 2) Decoder -> main seg + fused feats + moe_loss
        seg_logits, fused_feats, moe_loss = self.decoder(f1, f2, f3, f4, f5)

        # 3) Upsample the final seg map to original size
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)

        # 4) Aux head from the fused feature at scale=2
        #    fused_feats[2] => shape [B, dims[2], H/16, W/16] if out_indices=[0,1,2,3]
        aux_out = self.aux_head(fused_feats[2], (H, W))  # Nx(num_classes)xHxW

        return seg_logits, aux_out, moe_loss
