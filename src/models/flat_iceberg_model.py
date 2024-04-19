import gc
import cv2
#import nvtx
import logging
import numpy as np

import torch
from torchvision import transforms

from time import time
from patchify import patchify, unpatchify
from submodules.models.flat_iceberg.inference import OneshotYOLO

from src.pipeline_manager import pipeline_manager
from src.patching import unpatch_img
from .pipeline_pytorch_model import pipeline_pytorch_model
from cmaas_utils.types import MapUnitType

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

    
class flat_iceberg_model(pipeline_pytorch_model):
    def __init__(self):
        super().__init__()
        self.name = 'flat iceberg'
        self.version = '0.1'
        self.feature_type = MapUnitType.POINT
        self.checkpoint = '/projects/bbym/shared/models/flat-iceberg/best.pt'
    
        # Modifiable parameters
        self.device = torch.device("cuda")
        self.batch_size = 256
        self.patch_size = 256
        self.patch_overlap = 32
        self.unpatch_mode = 'discard'

    # @override
    def load_model(self):
        self.model = OneshotYOLO()
        self.model.load(self.checkpoint)
        self.model.eval()

    # @override
    def inference(self, image, legend_images, data_id=-1):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""

        # Get the size of the map
        map_channels, map_height, map_width  = image.shape

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

        # Flatten row col dims
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)

        # transpose BCHW to BHWC
        map_patches = np.transpose(map_patches, (0, 2, 3, 1))
        
        # pipeline_manager.log(logging.DEBUG, f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
        map_prediction = np.zeros((1, map_height, map_width), dtype=np.float32)
        map_confidence = np.zeros((1, map_height, map_width), dtype=np.float32)
        legend_index = 1
        for label, legend_img in legend_images.items():
            # pipeline_manager.log(logging.DEBUG, f'\t\tInferencing legend: {label}')
            lgd_stime = time()
            
            # transpose BCHW to BHWC
            legend_img = np.transpose(legend_img, (1, 2, 0))

            # Resize the legend patch
            legend_img = cv2.resize(legend_img, (self.patch_size, self.patch_size))

            # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
            if legend_img.shape[0] == 1:
                legend_img = np.concatenate([legend_img,legend_img,legend_img], axis=0)

            # Create legend array to merge with patches
            legend_patches = np.array([legend_img for i in range(rows*cols)])

            # Perform Inference in batches
            prediction_patches = []
            with torch.no_grad():
                for i in range(0, len(map_patches), self.batch_size):
                    prediction = self.model(map_patches[i:i+self.batch_size], legend_patches[:len(map_patches[i:i+self.batch_size])])
                    prediction_patches += prediction
            
            # Merge patches back into single image and remove padding
            prediction_patches = np.stack(prediction_patches, axis=0)
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
        
        return map_prediction
