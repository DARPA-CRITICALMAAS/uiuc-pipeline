import gc
import cv2
import nvtx
import logging
import numpy as np
import tensorflow as tf
from time import time
from patchify import patchify, unpatchify

from keras.models import load_model
from .unet_util import dice_coef, dice_coef_loss, multiplication, multiplication2

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

@nvtx.annotate(color="green", domain='DARPA_CMAAS_PIPELINE')
def inference(model, image, legend_images, batch_size=16, patch_size=256, patch_overlap=0):
    map_stime = time()
    # Get the size of the map
    map_width, map_height, map_channels = image.shape

    # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
    if map_channels == 1: # This is tmp fix!
        image = np.concatenate([image,image,image], axis=2)

    # Generate patches
    # Pad image so we get a size that can be evenly divided into patches.
    right_pad = patch_size - (map_width % patch_size)
    bottom_pad = patch_size - (map_height % patch_size)
    image = np.pad(image, ((0, right_pad), (0, bottom_pad), (0,0)), mode='constant', constant_values=0)
    patches = patchify(image, (patch_size, patch_size, 3), step=patch_size-patch_overlap)

    rows = patches.shape[0]
    cols = patches.shape[1]

    # Flatten row col dims and normalize map patches to [0,1]
    norm_patches = patches.reshape(-1, patch_size, patch_size, 3) / 255.0
    norm_patches = tf.cast(norm_patches, dtype=tf.float32) 
    
    log.info(f"Map size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
    predictions = {}
    for label, legend_img in legend_images.items():
        log.info(f'Inferencing legend: {label}')
        lgd_stime = time()

        # Resize the legend patch and normalize to [0,1]
        norm_legend_img = cv2.resize(legend_img, (patch_size, patch_size)) / 255.0

        # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!
            norm_legend_img = np.stack([norm_legend_img,norm_legend_img,norm_legend_img], axis=2)

        # Create legend array to merge with patches
        norm_legend_patches = np.array([norm_legend_img for i in range(rows*cols)])
        norm_legend_patches = tf.cast(norm_legend_patches, dtype=tf.float32)

        # Concatenate the map and legend patches along the third axis (channels) and normalize to [-1,1]
        norm_data = tf.concat(axis=3, values=[norm_patches, norm_legend_patches])
        norm_data = norm_data * 2.0 - 1.0

        prediction_patches = None
        for i in range(0, len(norm_data), batch_size):
            prediction = model.predict(norm_data[i:i+batch_size], verbose=0)
            if prediction_patches is None:
                prediction_patches = prediction
            else:
                prediction_patches = tf.concat(axis=0, values=[prediction_patches, prediction])

        # Merge patches back into single image and remove padding
        prediction_patches = np.array(prediction_patches) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
        prediction_patches = prediction_patches.reshape([rows, cols, 1, patch_size, patch_size, 1])
        prediction_image = unpatchify(prediction_patches, [image.shape[0], image.shape[1], 1])
        prediction_image = prediction_image[:map_width,:map_height,:]

        # Convert prediction result to a binary format using a threshold
        prediction_mask = (prediction_image > 0.5).astype(np.uint8)
        predictions[label] = prediction_mask
        gc.collect() # This is needed otherwise gpu memory is not freed up on each loop

        lgd_time = time() - lgd_stime
        log.info("Execution time for {} legend: {:.2f} seconds. {:.2f} patches per second".format(label, lgd_time, (rows*cols)/lgd_time))
        
    map_time = time() - map_stime
    log.info('Execution time for map: {:.2f} seconds'.format(map_time))
    return predictions
    

def load_primordial_positron_model(checkpoint):
    stime = time()
    if "attention" in checkpoint:
        # Load the attention Unet model with custom objects for attention mechanisms
        log.info(f"Loading model with attention from {checkpoint}")
        model = load_model(checkpoint, custom_objects={'multiplication': multiplication,
                                                            'multiplication2': multiplication2,
                                                            'dice_coef_loss':dice_coef_loss,
                                                            'dice_coef':dice_coef})
    else:
        log.info(f"Loading standard model from {checkpoint}")
        # Load the standard Unet model with custom objects for dice coefficient loss
        model = load_model(checkpoint, custom_objects={'dice_coef_loss':dice_coef_loss, 
                                                            'dice_coef':dice_coef})
    log.info('Model loaded in {:.2f} seconds'.format(time()-stime))
    return model