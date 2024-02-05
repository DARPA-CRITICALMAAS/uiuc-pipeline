import gc
import logging
import numpy as np
import tensorflow as tf
from patchify import patchify
from keras.models import load_model
from src.patching import unpatch_img
from time import time


from .pipeline_tensorflow_model import pipeline_tensorflow_model
from submodules.models.primordial_positron.unet_util import multiplication, multiplication2, dice_coef, dice_coef_loss

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class primordial_positron_model(pipeline_tensorflow_model):
    def __init__(self):
        super().__init__()
        self.unpatch_mode = 'or'
        self.patch_overlap = 30
        self.name = 'primordial positron'
        self.checkpoint = 'submodules/models/primordial_positron/inference_model/Unet-attentionUnet.h5'

    #@override
    def load_model(self):
        # Load the attention Unet model with custom objects for attention mechanisms
        self.model = load_model(self.checkpoint, custom_objects={'multiplication': multiplication,
                                                            'multiplication2': multiplication2,
                                                            'dice_coef_loss':dice_coef_loss,
                                                            'dice_coef':dice_coef})
        return self.model

    #@override 
    def inference(self, image, legend_images, batch_size=16, patch_size=256, patch_overlap=0):
    
        map_width, map_height, map_channels = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!
            image = np.concatenate([image,image,image], axis=2)        

        step_size = patch_size - patch_overlap

        pad_x = (step_size - (image.shape[1] % step_size)) % step_size
        pad_y = (step_size - (image.shape[0] % step_size)) % step_size
        image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant') 

        norm_patches = patchify(image, patch_size, step=step_size) / 255.0
        
        # keep row and columns for unpatchifying
        rows = norm_patches.shape[0]
        cols = norm_patches.shape[1]

        log.debug(f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")

        predictions = {}

        for label, legend_img in legend_images.items():
            log.debug(f'\t\tInferencing legend: {label}')
            lgd_stime = time()

            # Resize the legend patch and normalize to [0,1]
            norm_legend_img = tf.image.resize(legend_img, (patch_size, patch_size)) / 255.0

            # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
            if map_channels == 1: # This is tmp fix!
                norm_legend_img = np.stack([norm_legend_img,norm_legend_img,norm_legend_img], axis=2)
            
            # Create legend array to merge with patches
            norm_legend_patches = np.array([norm_legend_img for i in range(rows*cols)])
            norm_legend_patches = tf.cast(norm_legend_patches, dtype=tf.float32)

            # Concatenate the map and legend patches along the third axis (channels)
            norm_data = tf.concat(axis=3, values=[norm_patches, norm_legend_patches])

            # Perform Inference in batches
            prediction_patches = None
            for i in range(0, len(norm_data), batch_size):
                prediction = self.model.predict(norm_data[i:i+batch_size], verbose=0)
                if prediction_patches is None:
                    prediction_patches = prediction
                else:
                    prediction_patches = tf.concat(axis=0, values=[prediction_patches, prediction])

        # Merge patches back into single image and remove padding
        prediction_patches = np.array(prediction_patches) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
        prediction_patches = prediction_patches.reshape([rows, cols, 1, patch_size, patch_size, 1])
        unpatch_image = unpatch_img(prediction_patches.transpose(2,0,1,5,3,4), [1, image.shape[0], image.shape[1]], overlap=patch_overlap, mode=self.unpatch_mode).transpose(1,2,0)
        prediction_image = unpatch_image[:map_width,:map_height,:]

        # Convert prediction result to a binary format using a threshold
        prediction_mask = (prediction_image > 0.5).astype(np.uint8)
        predictions[label] = prediction_mask
        gc.collect() # This is needed otherwise gpu memory is not freed up on each loop

        lgd_time = time() - lgd_stime
        log.debug("\t\tExecution time for {} legend: {:.2f} seconds. {:.2f} patches per second".format(label, lgd_time, (rows*cols)/lgd_time))
        
        return predictions
