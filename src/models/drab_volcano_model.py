import os
import gc
import json
import logging
import numpy as np
from time import time
from patchify import patchify

import torch
from torchvision import transforms

from cmaas_utils.types import MapUnitType
from src.patching import unpatch_img
from src.pipeline_manager import pipeline_manager
from src.models.pipeline_pytorch_model import pipeline_pytorch_model
from submodules.models.drab_volcano.inference_multi_class import MulticlassYOLO

log = logging.getLogger('DARPA_CMAAS_PIPELINE')


class drab_volcano_model(pipeline_pytorch_model):
    def __init__(self):
        super().__init__()
        self.name = 'drab volcano'
        self.version = '0.1'
        self.feature_type = MapUnitType.POINT
        self.checkpoint = 'drab-volcano/best.pt'
        self.label_name_path = 'submodules/models/drab_volcano/config/legend_name2class.json'
        self.est_patches_per_sec = 4500 # Ignore the number is not actually patches per sec for this model as its multi-class

        # Modifiable parameters
        self.device = torch.device("cuda")
        self.batch_size = 256
        self.patch_size = 256
        self.patch_overlap = 32
        self.unpatch_mode = 'discard'

    # @override
    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, self._checkpoint) 
        self.model = MulticlassYOLO()
        self.model.load(model_path)
        self.model.eval()

        label_names = json.load(open(self.label_name_path))
        self.name_to_class = {name: int(label_names[i]["id"]) for i in range(len(label_names)) for name in label_names[i]["names"]}

    # @override
    def inference(self, image, legend_images, data_id=-1):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""         

        # Get the size of the map
        map_channels, map_height, map_width = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!
            image = np.concatenate([image,image,image], axis=0)        

        # Generate patches
        # Pad image so we get a size that can be evenly divided into patches.
        right_pad = self.patch_size - (map_width % self.patch_size)
        bottom_pad = self.patch_size - (map_height % self.patch_size)
   
        # pad the H, W, 3 image, pad at the right and bottom
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)
        map_patches = patchify(padded_image, (3, self.patch_size, self.patch_size), step=self.patch_size-self.patch_overlap)

        cols = map_patches.shape[1]
        rows = map_patches.shape[2]

        # Flatten row col dims
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        
        # transpose BCHW to BHWC
        map_patches = np.transpose(map_patches, (0, 2, 3, 1))

        # pipeline_manager.log(logging.DEBUG, f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")

        # Perform Inference in batches
        prediction_patches = []
        with torch.no_grad():
            for i in range(0, len(map_patches), self.batch_size):
                prediction,_ = self.model(map_patches[i:i+self.batch_size])
                prediction_patches += prediction
        
        # Merge patches back into single image and remove padding
        prediction_patches = np.stack(prediction_patches, axis=0)
        prediction_patches = prediction_patches.reshape([1, cols, rows, 1, self.patch_size, self.patch_size])
        unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=self.patch_overlap, mode=self.unpatch_mode)
        prediction_image = unpatch_image[:,:map_height,:map_width]
            
        gc.collect() # This is needed otherwise gpu memory is not freed up on each loop
        
        return prediction_image