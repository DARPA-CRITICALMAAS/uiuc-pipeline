import gc
import logging
import numpy as np
from time import time
from types import SimpleNamespace
from patchify import patchify, unpatchify

import torch
from torchvision import transforms

from src.pipeline_manager import pipeline_manager
from src.patching import unpatch_img
from .pipeline_pytorch_model import pipeline_pytorch_model
from submodules.models.rigid_wasabi.models import SegmentationModel
from cmaas_utils.types import MapUnitType

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class rigid_wasabi_model(pipeline_pytorch_model):
    def __init__(self):
        self.name = 'rigid wasabi'
        self.version = '0.1'
        self.feature_type = MapUnitType.POLYGON
        self.checkpoint = '/projects/bbym/shared/models/rigid_wasabi/SWIN_jaccard.ckpt'
        self.args = SimpleNamespace(model='swin', superpixel='', edge=False)
        
        # Modifiable parameters
        self.device = torch.device("cuda")
        self.batch_size = 256
        self.patch_size = 256
        self.patch_overlap = 64
        self.unpatch_mode = 'discard'
        
    #@override
    def load_model(self):
        self.model = SegmentationModel.load_from_checkpoint(checkpoint_path=self.checkpoint, args=self.args)
        self.model.eval()
        self.model.to(self.device)

        return self.model
    
    def my_norm(self, data):
        data = data / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)[None, :, None, None].expand(*data.shape)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)[None, :, None, None].expand(*data.shape)
        data = (data - mean)/std
        return data
    
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
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)
        map_patches = patchify(padded_image, (3, self.patch_size, self.patch_size), step=self.patch_size-self.patch_overlap)

        cols = map_patches.shape[1]
        rows = map_patches.shape[2]

        # Flatten row col dims and normalize map patches to [0,1]
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        map_patches = torch.Tensor(map_patches).to(self.device)
        map_patches = self.my_norm(map_patches)

        # pipeline_manager.log(logging.DEBUG, f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
        map_prediction = np.zeros((1, map_height, map_width), dtype=np.float32)
        map_confidence = np.zeros((1, map_height, map_width), dtype=np.float32)
        legend_index = 1
        for label, legend_img in legend_images.items():
            # pipeline_manager.log(logging.DEBUG, f'\t\tInferencing legend: {label}')
            lgd_stime = time()

            # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
            if map_channels == 1: # This is tmp fix!
                legend_img = np.concatenate([legend_img,legend_img,legend_img], axis=0)

            # Resize the legend patch
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
                    prediction = self.model.model(map_patches[i:i+self.batch_size], legend_patches[:len(map_patches[i:i+self.batch_size])])
                    prediction = torch.argmax(prediction, dim=1).cpu().numpy().astype(np.float32)
                    prediction_patches.append(prediction)
                    
            # Merge patches back into single image and remove padding
            prediction_patches = np.concatenate(prediction_patches, axis=0)
            prediction_patches = prediction_patches.reshape([1, cols, rows, 1, self.patch_size, self.patch_size])
            unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=self.patch_overlap, mode=self.unpatch_mode)
            prediction_image = unpatch_image[:, :map_height, :map_width]

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