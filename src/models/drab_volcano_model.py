import gc
import cv2
#import nvtx
import logging
import numpy as np

import torch
from torchvision import transforms

from time import time
from src.patching import unpatch_img
from patchify import patchify
from .pipeline_pytorch_model import pipeline_pytorch_model
from submodules.models.drab_volcano.inference_multi_class import MulticlassYOLO
import json

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class drab_volcano_model(pipeline_pytorch_model):
    def __init__(self):
        super().__init__()
        self.name = 'drab volcano'
        self.checkpoint = '/projects/bbym/shared/models/drab-volcano/best.pt'
        self.label_name_path = '/projects/bbym/shared/models/drab-volcano/legend_name2class.json'
        # Modifiable parameters
        self.device = torch.device("cuda")
        self.batch_size = 256
        self.patch_size = 256
        self.patch_overlap = 32
        self.unpatch_mode = 'discard'

    # @override
    def load_model(self):
        self.model = MulticlassYOLO()
        self.model.load(self.checkpoint)
        self.model.eval()

        label_names = json.load(open(self.label_name_path))
        self.name_to_class = {name: int(label_names[i]["id"]) for i in range(len(label_names)) for name in label_names[i]["names"]}

    # @override
    def inference_with_refactor(self, image, legend_names=None):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""

        # Get the size of the map
        map_width, map_height, map_channels = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!
            image = np.concatenate([image,image,image], axis=2)        

        # Generate patches
        # Pad image so we get a size that can be evenly divided into patches.
        right_pad = self.patch_size - (map_width % self.patch_size)
        bottom_pad = self.patch_size - (map_height % self.patch_size)
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)
        map_patches = patchify(padded_image, (3, self.patch_size, self.patch_size), step=self.patch_size-self.patch_overlap)

        cols = map_patches.shape[1]
        rows = map_patches.shape[2]

        # Flatten row col dims
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        
        log.debug(f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")

        # Perform Inference in batches
        prediction_patches = []
        with torch.no_grad():
            for i in range(0, len(map_patches), self.batch_size):
                prediction = self.model(map_patches[i:i+self.batch_size])
                prediction_patches += prediction
        
        # Merge patches back into single image and remove padding
        prediction_patches = np.stack(prediction_patches, axis=0)
        # prediction_patches = np.array(prediction_patches) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
        prediction_patches = prediction_patches.reshape([1, cols, rows, 1, self.patch_size, self.patch_size])
        unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=self.patch_overlap, mode=self.unpatch_mode)
        #prediction_image = unpatchify(prediction_patches, [image.shape[0], image.shape[1], 1])
        prediction_image = unpatch_image[:,:map_width,:map_height]
        #saveGeoTiff("testdata/debug/sample.tif", prediction_image.astype(np.uint8) * 255, None, None)
        
        if legend_names is not None:
            log.debug(f"Refactoring prediction to known legend names!")
            idx = 1
            final_prediction = np.zeros_like(prediction_image)
            for name in legend_names:
                if name not in self.name_to_class:
                    log.warning(f"Unknown legend name: {name}, return the unrefactored prediction")
                    return prediction_image
                final_prediction[prediction_image == self.name_to_class[name]] = idx
                idx += 1
        else:
            final_prediction = prediction_image
            
        gc.collect() # This is needed otherwise gpu memory is not freed up on each loop
        
        return prediction_image

    # @override
    def inference(self, image, legend_images=None):
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
        # print(map_patches.shape)
        cols = map_patches.shape[1]
        rows = map_patches.shape[2]

        # Flatten row col dims
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        
        # transpose BCHW to BHWC
        map_patches = np.transpose(map_patches, (0, 2, 3, 1))

        log.debug(f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")

        # Perform Inference in batches
        prediction_patches = []
        with torch.no_grad():
            for i in range(0, len(map_patches), self.batch_size):
                prediction,_ = self.model(map_patches[i:i+self.batch_size])
                prediction_patches += prediction
        
        # Merge patches back into single image and remove padding
        prediction_patches = np.stack(prediction_patches, axis=0)
        # print(prediction_patches.shape)
        # prediction_patches = np.array(prediction_patches) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
        prediction_patches = prediction_patches.reshape([1, cols, rows, 1, self.patch_size, self.patch_size])
        unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=self.patch_overlap, mode=self.unpatch_mode)
        #prediction_image = unpatchify(prediction_patches, [image.shape[0], image.shape[1], 1])
        prediction_image = unpatch_image[:,:map_height,:map_width]
        #saveGeoTiff("testdata/debug/sample.tif", prediction_image.astype(np.uint8) * 255, None, None)
        
        final_prediction = prediction_image
            
        gc.collect() # This is needed otherwise gpu memory is not freed up on each loop
        
        return prediction_image