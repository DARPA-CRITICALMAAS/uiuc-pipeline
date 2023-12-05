import argparse
import math
import cv2
import os
import time
import numpy as np
from PIL import Image
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
import matplotlib.gridspec as gridspec
from data_util import DataLoader
from h5Image import H5Image
from unet_util import (UNET_224, Residual_CNN_block,
                        attention_up_and_concatenate,
                        attention_up_and_concatenate2, dice_coef,
                        dice_coef_loss, evaluate_prediction_result, jacard_coef,
                        multiplication, multiplication2)

import logging


logger = logging.getLogger('primordial-positron')

def prediction_mask(prediction_result, map_array):
    """
    Apply a mask to the prediction image to isolate the area of interest.

    Parameters:
    - prediction_result: numpy array, The output of the model after prediction.
    - map_name: str, The name of the map used for prediction.

    Returns:
    - masked_img: numpy array, The masked prediction image.
    """

    # Convert the RGB map array to grayscale for further processing
    gray = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)

    # Identify the most frequent pixel value, which will be used as the background pixel value
    pix_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    background_pix_value = np.argmax(pix_hist, axis=None)

    # Flood fill from the corners to identify and modify the background regions
    height, width = gray.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(gray, None, (c[0],c[1]), 255)

    # Adaptive thresholding to remove small noise and artifacts
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Detect edges using the Canny edge detection method
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Detect contours in the edge-detected image
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Retain only the largest contour
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    # Create an empty mask of the same size as the prediction_result
    wid, hight = prediction_result.shape[0], prediction_result.shape[1]
    mask = np.zeros([wid, hight])
    mask = cv2.fillPoly(mask, pts=[contour], color=(1)).astype(np.uint8)

    # Convert prediction result to a binary format using a threshold
    prediction_result_int = (prediction_result > 0.5).astype(np.uint8)

    # Apply the mask to the thresholded prediction result
    masked_img = cv2.bitwise_and(prediction_result_int, mask)

    return masked_img


def inference_image(image, legends, model, featureType, patch_size=256):
    predictions = {}

    # Filter the legends based on the feature type
    # if featureType == "Polygon":
    #     map_legends = [legend for legend in all_map_legends if "_poly" in legend]
    # elif featureType == "Point":
    #     map_legends = [legend for legend in all_map_legends if "_pt" in legend]
    # elif featureType == "Line":
    #     map_legends = [legend for legend in all_map_legends if "_line" in legend]
    # elif featureType == "All":
    #     map_legends = all_map_legends
    map_legends = legends.keys()

    # Get the size of the map
    map_width, map_height, _ = image.shape
    logger.debug(f"Map size: {map_width}, {map_height}")

    # Calculate the number of patches based on the patch size and border
    num_rows = math.ceil(map_width / patch_size)
    num_cols = math.ceil(map_height / patch_size)

    # Loop through the patches and perform inference
    for legend in (map_legends):
        logger.debug(f"Processing legend: {legend}")
        start_time = time.time()
        
        # Create an empty array to store the full prediction
        full_prediction = np.zeros((map_width, map_height))

        # Get the legend patch
        legend_patch = legends[legend]

        # Iterate through rows and columns to get map patches
        for row in range(num_rows):
            for col in range(num_cols):

                # Calculate starting indices for rows and columns
                x_start = row * patch_size
                y_start = col * patch_size

                # Calculate ending indices for rows and columns
                x_end = x_start + patch_size
                y_end = y_start + patch_size

                # Adjust the ending indices if they go beyond the image size
                x_end = min(x_end, map_width)
                y_end = min(y_end, map_height)

                map_patch = image[:x_end-x_start, :y_end-y_start]

                # Get the prediction for the current patch
                prediction = np.zeros(map_patch.shape[:2])
                #prediction = perform_inference(legend_patch, map_patch, model)
                # logger.debug(f"Prediction for patch ({row}, {col}) completed.")

                # Adjust the shape of the prediction if necessary
                prediction_shape_adjusted = prediction[:x_end-x_start, :y_end-y_start]

                # Assign the prediction to the correct location in the full_prediction array
                full_prediction[x_start:x_end, y_start:y_end] = prediction_shape_adjusted
            
        # Mask out the map background pixels from the prediction
        logger.debug("Applying mask to the full prediction.")
        masked_prediction = prediction_mask(full_prediction, image)
        predictions[legend] = masked_prediction
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.debug(f"Execution time for 1 legend: {total_time} seconds")


def inference(images, legends, checkpoint, **kwargs):

    featureType = kwargs.get('featureType', 'Polygon') # Polygon or Point

    if "attention" in checkpoint:
        # Load the attention Unet model with custom objects for attention mechanisms
        logger.info(f"Loading model with attention from {checkpoint}")
        model = load_model(checkpoint, custom_objects={'multiplication': multiplication,
                                                            'multiplication2': multiplication2,
                                                            'dice_coef_loss':dice_coef_loss,
                                                            'dice_coef':dice_coef})
    else:
        logger.info(f"Loading standard model from {checkpoint}")
        # Load the standard Unet model with custom objects for dice coefficient loss
        model = load_model(checkpoint, custom_objects={'dice_coef_loss':dice_coef_loss, 
                                                            'dice_coef':dice_coef})

    # loop over all images
    outputs = []
    for idx, image in enumerate(images):
        # Loop through the patches and perform inference
        outputs.append(inference_image(image, legends[idx], model, featureType))

    return outputs
