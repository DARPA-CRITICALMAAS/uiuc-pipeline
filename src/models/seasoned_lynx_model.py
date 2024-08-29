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

import sys
submodule_path = os.path.abspath('/u/dkwark/uiuc-pipeline/submodules/models/seasoned_lynx')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)
submodule_path = os.path.abspath('/u/dkwark/uiuc-pipeline/submodules/models/seasoned_lynx/segment-anything')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)
from submodules.models.seasoned_lynx.model.VRP_encoder import VRP_encoder
from submodules.models.seasoned_lynx.SAM2pred import SAM_pred
# sys.path.pop(0)

from transformers import AutoImageProcessor, CLIPProcessor

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class seasoned_lynx_model(pipeline_pytorch_model):
    def __init__(self):
        self.name = 'seasoned_lynx'
        self.version = '0.1'
        self.feature_type = MapUnitType.POLYGON
        self._checkpoint = 'seasoned-lynx-0.1.pt'
        self._args = SimpleNamespace(arch_testing=False,
                                     dino_lora=True,
                                     clip_applied=True,
                                     pattern_text=False,
                                     lora_r=128,
                                     prompt_backbone_trainable=False,
                                     num_query=100,
                                     lora=True,
                                     if_encoder_lora_layer=True,
                                     encoder_lora_layer=[],
                                     if_decoder_lora_layer=True,
                                     mask_decoder_trainable=False,
                                     backbone='dino_v2')


        self.est_patches_per_sec = 280 # Only used for estimating inference time -- need to modify this

        self.device = torch.device("cuda")

        self.model = VRP_encoder(self._args, self._args.backbone, use_original_imgsize=False, pretrained=False)
        self.sam_model = SAM_pred(self._args.mask_decoder_trainable, self._args.lora, self._args)

        self.model.to(self.device)
        self.sam_model.to(self.device)

        # Modifiable parameters
        self.batch_size = 10
        self.patch_size = 1024 # need to modify this patch size
        patch_overlap_dict = {'1024': 256,
                              '512': 128,
                              '256': 64, }
        self.patch_overlap = patch_overlap_dict[str(self.patch_size)]
        self.unpatch_mode = 'discard'

        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    #@override
    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, self._checkpoint)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.sam_model.load_state_dict(checkpoint['sam_model_state_dict'])
        self.model.eval()
        self.sam_model.eval()

        return self.model, self.sam_model
    
    def my_norm(self, data):
        data = data / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)[None, :, None, None].expand(*data.shape)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)[None, :, None, None].expand(*data.shape)
        data = (data - mean)/std
        return data

    # @override
    def inference(self, image, legend_images, data_id=-1):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""         
        # For profiling memory usage 
        #torch.cuda.memory._record_memory_history()

        # Get the size of the map
        map_channels, map_height, map_width = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!    
            image = np.concatenate([image,image,image], axis=0)

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

        # we use the DINOv2 processing pipeline to normalize the patches
        with torch.no_grad():
            map_img_DINO_patches = self.dino_processor(map_patches, return_tensors="pt")['pixel_values'].to(self.device)       

        map_patches = torch.Tensor(map_patches).to(self.device)
        map_patches = self.my_norm(map_patches)

        # pipeline_manager.log(logging.DEBUG, f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
        map_prediction = np.zeros((1, map_height, map_width), dtype=np.float32)
        map_confidence = np.zeros((1, map_height, map_width), dtype=np.float32)
        legend_index = 1
        for label, legend_img in legend_images.items():
            lgd_stime = time()
            
            # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
            if legend_img.shape[0] == 1:
                legend_img = np.concatenate([legend_img,legend_img,legend_img], axis=0)

            with torch.no_grad():
                legend_img_DINO = self.dino_processor(legend_img, return_tensors="pt")['pixel_values'][0].to(self.device)
                legend_img_CLIP = self.clip_processor(text=None, images=legend_img, return_tensors="pt")['pixel_values'][0].to(self.device)
                legend_mask = torch.ones_like(legend_img_DINO)[0].to(self.device)
                legend_mask[0:14, :] = 0
                legend_mask[-14:, :] = 0
                legend_mask[:, 0:14] = 0
                legend_mask[:, -14:] = 0
            

            legend_img_DINO_patches = torch.stack([legend_img_DINO for i in range(self.batch_size)], dim=0)
            legend_img_CLIP_patches = torch.stack([legend_img_CLIP for i in range(self.batch_size)], dim=0)
            legend_mask_patches = torch.stack([legend_mask for i in range(self.batch_size)], dim=0)         
            
            # Perform Inference in batches
            prediction_patches = []
            with torch.no_grad():
                for i in range(0, len(map_patches), self.batch_size):
                    protos, _ = self.model('mask', 
                                    map_img_DINO_patches[i:i+self.batch_size], 
                                    legend_img_DINO_patches[:len(map_patches[i:i+self.batch_size])],
                                    legend_mask_patches[:len(map_patches[i:i+self.batch_size])],
                                    False,
                                    legend_img_CLIP_patches[:len(map_patches[i:i+self.batch_size])])
                    
                    _, prediction = self.sam_model(map_patches[i:i+self.batch_size], None, protos)

                    # print(torch.cuda.memory_summary(device=self.device, abbreviated=True))

                    # prediction = self.model.model(map_patches[i:i+self.batch_size], legend_patches[:len(map_patches[i:i+self.batch_size])])
                    # prediction = torch.softmax(prediction.float(), dim=1)[:,-1].cpu().numpy().astype(np.float32)
                    prediction = torch.sigmoid(prediction).cpu().numpy().astype(np.float32)
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
            # pipeline_manager.log(logging.DEBUG, "\t\tExecution time for {} legend: {:.2f} seconds. {:.2f} patches per second".format(label, lgd_time, (rows*cols)/lgd_time))

        # Minimum confidence threshold for a prediction
        map_prediction[map_confidence < 0.333] = 0

        # For profiling memory usage 
        # torch.cuda.memory._dump_snapshot(f'gpu_snapshots/{data_id}_inference.pickle')
        # torch.cuda.reset_max_memory_allocated(0)
        # pipeline_manager.log_to_monitor(data_id, {'GPU Mem (Alloc/Reserve/Avail)' : f'-'})
        
        return map_prediction
    

