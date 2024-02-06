import gc
import logging
import numpy as np
from time import time
from types import SimpleNamespace
from patchify import patchify, unpatchify

import torch
from torchvision import transforms

from src.patching import unpatch_img
from .pipeline_pytorch_model import pipeline_pytorch_model
from submodules.models.golden_muscat.models import SegmentationModel

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class golden_muscat_model(pipeline_pytorch_model):
    def __init__(self):
        self.name = 'golden muscat'
        self.checkpoint = '/projects/bbym/shared/models/golden_muscat/jaccard.ckpt'

        self.args = SimpleNamespace(model='Unet', edge=False,superpixel = '')
        self.device = torch.device("cuda")
        self.patch_overlap = 128
        self.unpatch_mode = 'discard'

    #@override
    def load_model(self):
        self.model = SegmentationModel.load_from_checkpoint(checkpoint_path=self.checkpoint, args=self.args)
        self.model.eval()

        return self.model
    
    def my_norm(self, data):
        norm_data = data / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)[None, :, None, None].expand(*norm_data.shape)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)[None, :, None, None].expand(*norm_data.shape)
        norm_data = (norm_data - mean)/std
        return norm_data

    # @override
    def inference(self, image, legend_images, batch_size=16, patch_size=256, patch_overlap=0):
        patch_overlap = self.patch_overlap
        # Pytorch expects image in CHW format
        image = image.transpose(2,0,1)

        # Get the size of the map
        map_channels, map_height, map_width = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!
            image = np.concatenate([image,image,image], axis=2)        

        # Generate patches
        # Pad image so we get a size that can be evenly divided into patches.
        right_pad = patch_size - (map_width % patch_size)
        bottom_pad = patch_size - (map_height % patch_size)
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)
        patches = patchify(padded_image, (3, patch_size, patch_size), step=patch_size-patch_overlap)

        cols = patches.shape[1]
        rows = patches.shape[2]

        # Flatten row col dims and normalize map patches to [0,1]
        norm_patches = patches.reshape(-1, 3, patch_size, patch_size)
        norm_patches = torch.Tensor(norm_patches).to(self.device)
        norm_patches = self.my_norm(norm_patches)

        log.debug(f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
        predictions = {}
        for label, legend_img in legend_images.items():
            log.debug(f'\t\tInferencing legend: {label}')
            lgd_stime = time()

            # Pytorch expects image in CHW format
            legend_img = legend_img.transpose(2,0,1)

            # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
            if map_channels == 1: # This is tmp fix!
                legend_img = np.stack([legend_img,legend_img,legend_img], axis=0)

            # Resize the legend patch
            legend_tensor = torch.Tensor(legend_img).to(self.device)
            resize_legend = transforms.Resize((patch_size, patch_size), antialias=None)
            legend_tensor = resize_legend(legend_tensor)

            # Create legend array to merge with patches
            legend_patches = torch.stack([legend_tensor for i in range(rows*cols)], dim=0)
            legend_patches = self.my_norm(legend_patches)

            # Concatenate the map and legend patches along the third axis (channels) and normalize to [-1,1]
            norm_data = torch.cat([norm_patches, legend_patches], dim=3)

            # Perform Inference in batches
            prediction_patches = []
            with torch.no_grad():
                for i in range(0, len(norm_data), batch_size):
                    prediction = self.model.model(norm_patches[i:i+batch_size], legend_patches[i:i+batch_size])
                    prediction = torch.argmax(prediction, dim=1).cpu().numpy().astype(np.uint8)
                    
                    prediction_patches.append(prediction)
                    
            # Merge patches back into single image and remove padding
            prediction_patches = np.concatenate(prediction_patches, axis=0)
            prediction_patches = prediction_patches.reshape([1, cols, rows, 1, patch_size, patch_size])
            unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=patch_overlap, mode=self.unpatch_mode)
            prediction_image = unpatch_image[:,:map_height,:map_width]

            # Transpose image back to HWC
            prediction_image = prediction_image.transpose(1,2,0)

            # Convert prediction result to a binary format using a threshold
            # prediction_mask = (prediction_image > 0.5)
            predictions[label] = prediction_image
            
            gc.collect() # This is needed otherwise gpu memory is not freed up on each loop

            lgd_time = time() - lgd_stime
            log.debug("\t\tExecution time for {} legend: {:.2f} seconds. {:.2f} patches per second".format(label, lgd_time, (rows*cols)/lgd_time))
        
        
        # prediction_array = np.stack([predictions[k].astype(np.float32) for k in predictions], axis=2)
        # pre_keys = predictions.keys()
        # # del predictions
        # preds_max = np.max(prediction_array, axis=2) - 0.00001
        # # preds_max = preds_max.transpose(1,2,0,3)
        # for i in range(prediction_array.shape[2]):
        #     prediction_array[:,:,i] = (prediction_array[:,:,i] > preds_max) & (preds_max > 0.5)
        # prediction_array = (prediction_array > preds_max) & (preds_max > 0.5)
        # predictions = {}
        # for i, k in enumerate(pre_keys):
        #     predictions[k] = prediction_array[:,:,i]
        # del prediction_array
        # for k in predictions:
        #     predictions[k]
        # predictions[label]
        if len(predictions.keys()) == 0:
            return predictions
        
        cur_max = np.zeros(predictions[list(predictions.keys())[0]].shape)
        for k in predictions:
            cur_max = np.maximum(cur_max, predictions[k])
        cur_max = cur_max

        for k in predictions:
            predictions[k] = (predictions[k] >= cur_max) & (cur_max > 0.5)

        return predictions
    