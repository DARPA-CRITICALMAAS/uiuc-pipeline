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

from sklearn.cluster import KMeans
import cv2

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
    def __init__(self, patch_size=1024):
        self.name = 'seasoned_lynx'
        self.feature_type = MapUnitType.POLYGON

        self.recoloring = True

        self._args = SimpleNamespace(arch_testing=False,
                                     dino_lora=True,
                                     clip_applied=False, # True for 1024, False for 256
                                     pattern_text=False,
                                     lora_r=128,
                                     prompt_backbone_trainable=False,
                                     num_query=50, # 100 for 1024, 50 for 256
                                     lora=True,
                                     if_encoder_lora_layer=True,
                                     encoder_lora_layer=[],
                                     if_decoder_lora_layer=True,
                                     mask_decoder_trainable=False,
                                     backbone='dino_v2')
        
        self.configure_for_patch_size(patch_size)

        self.est_patches_per_sec = 280 # Only used for estimating inference time -- need to modify this

        self.device = torch.device("cuda")

        self.model = VRP_encoder(self._args, self._args.backbone, use_original_imgsize=False, pretrained=False)
        self.sam_model = SAM_pred(self._args.mask_decoder_trainable, self._args.lora, self._args)

        self.model.to(self.device)
        self.sam_model.to(self.device)
        
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

    def configure_for_patch_size(self, patch_size):
        if patch_size == 1024:
            self._args.num_query = 100
            self._checkpoint = 'seasoned-lynx-0.1.pt'
            self.version = '0.1'
            self._args.clip_applied = True
            self.patch_size = patch_size
            self.batch_size = 12
        elif patch_size == 512:
            self._args.num_query = 50
            self._args.lora_r = 64
            self._checkpoint = 'seasoned-lynx-0.3.pt'
            self.version = '0.3'
            self._args.clip_applied = False
            self.patch_size = patch_size
            self.batch_size = 12
        elif patch_size == 256:
            self._args.num_query = 50
            self._checkpoint = 'seasoned-lynx-0.2.pt'
            self.version = '0.2'
            self._args.clip_applied = False
            self.patch_size = patch_size
            self.batch_size = 12
        else:
            raise ValueError(f"Unsupported patch size: {patch_size}")
    
    def my_norm(self, data):
        data = data / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)[None, :, None, None].expand(*data.shape)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)[None, :, None, None].expand(*data.shape)
        data = (data - mean)/std
        return data
    
    def mask_margins(self, image, margin=0.1):
        """Masks the margins of an image (grayscale or color) by setting margin pixels to black."""
        h, w = image.shape[1], image.shape[2]  
        # Calculate 10% margin size
        margin_h = int(h * margin)
        margin_w = int(w * margin)
        
        mask = np.ones((h, w), dtype=bool)
        mask[:margin_h, :] = False  # Top margin
        mask[-margin_h:, :] = False  # Bottom margin
        mask[:, :margin_w] = False  # Left margin
        mask[:, -margin_w:] = False  # Right margin
        
        # Apply the mask: set masked pixels (margins) to black
        masked_image = image.copy()
        masked_image[:, ~mask] = 0
        
        return masked_image
    

    def extract_dominant_colors(self, image, legend=False, num=0, n_colors=1):
        pipeline_manager.log(logging.DEBUG, f"\t\t\t\tExtracting dominant colors from image: {image.shape}")
        
        # Convert image to Lab color space before applying the mask
        lab_image = cv2.cvtColor(image.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2Lab)
        
        # Save the image if required
        if legend:
            cv2.imwrite(f'tmp_legend_{num}.png', cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR))
        else:
            cv2.imwrite(f'tmp_image_{num}.png', cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR))
        
        # Define a threshold for near-black (based on the L* channel in Lab space)
        # L* represents lightness, where 0 is black, so we can threshold it
        lightness_threshold = 5  # You can adjust this value based on the desired precision
        mask = lab_image[:, :, 0] <= lightness_threshold
        
        # Log mask information
        pipeline_manager.log(logging.DEBUG, f"\t\t\t\tMask shape: {mask.shape}, Number of masked pixels: {np.sum(mask)}")
        
        # Reshape image to a 2D array of Lab pixels
        pixels = lab_image.reshape(-1, 3)
        
        # If we are not dealing with a legend, remove near-black pixels using the mask
        if not legend:
            pixels = pixels[~mask.flatten()]
        
        # Check if there are any remaining pixels after masking
        if pixels.shape[0] == 0:
            return np.array([[0, 0, 0]])  # Return black as the default color if no valid pixels are found
        
        pipeline_manager.log(logging.DEBUG, f"\t\t\t\tNon-black pixels: {pixels.shape}")
        pipeline_manager.log(logging.DEBUG, f"\t\t\t\tNum of non-black pixels: {pixels.shape[0]}")
        
        # Use KMeans to extract the dominant colors
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
        
        # Return the dominant color(s) in Lab color space
        return kmeans.cluster_centers_


    # def extract_dominant_colors(self, image, legend=False, num=0, n_colors=1):
    #     pipeline_manager.log(logging.DEBUG, f"\t\t\t\tExtracting dominant colors from image: {image.shape}")
    #     lab_image = image.transpose(1, 2, 0)

    #     # save into the current directory
    #     if legend:
    #         cv2.imwrite(f'tmp_legend_{num}.png', cv2.cvtColor(lab_image, cv2.COLOR_RGB2BGR))
    #     # lab_image = cv2.cvtColor(lab_image, cv2.COLOR_RGB2Lab)

    #     else:
    #         cv2.imwrite(f'tmp_image_{num}.png', cv2.cvtColor(lab_image, cv2.COLOR_RGB2BGR))

    #     threshold = 1e-5
    #     mask = np.all(lab_image <= threshold, axis=-1)  # Pixels that are very close to black
    #     pipeline_manager.log(logging.DEBUG, f"\t\t\t\tMask shape: {mask.shape}, Number of masked pixels: {np.sum(mask)}")

    #     pixels = lab_image.reshape(-1, 3)

    #     if not legend:
    #         pixels = pixels[~mask.flatten()]

    #     if pixels.shape[0] == 0:
    #         return np.array([[0, 0, 0]])  # Return black as the default color if no valid pixels are found
        
    #     non_black_pixels_lab = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2Lab).reshape(-1, 3)

    #     pipeline_manager.log(logging.DEBUG, f"\t\t\t\tNon-black pixels: {pixels.shape}")
    #     pipeline_manager.log(logging.DEBUG, f"\t\t\t\tNum of non-black pixels: {pixels.shape[0]}")
        
    #     # Use KMeans to extract the dominant colors
    #     kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(non_black_pixels_lab)
    #     return kmeans.cluster_centers_
    
    

        # if not legend:
        #     # Remove black pixels (if necessary)
        #     if pixels.dtype == np.uint8:
        #     # For integer values, remove exact black pixels [0, 0, 0]
        #         pixels = pixels[np.any(pixels != [0, 0, 0], axis=-1)]
        #     else:
        #         # For float values, remove near-black pixels (allowing for small precision errors)
        #         pixels = pixels[np.any(pixels > 1e-5, axis=-1)]  # Adjust this threshold as needed

        # non_black_pixels_lab = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2Lab).reshape(-1, 3)

        # pipeline_manager.log(logging.DEBUG, f"\t\t\t\tNon-black pixels: {pixels.shape}")
        # pipeline_manager.log(logging.DEBUG, f"\t\t\t\tNum of non-black pixels: {pixels.shape[0]}")

        # # If no non-black pixels are found, return a default color (e.g., black)
        # if pixels.shape[0] == 0:
        #     return np.array([[0, 0, 0]])  # Returning black as default
        
        # kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(non_black_pixels_lab)
        # return kmeans.cluster_centers_
    


    # @override
    def inference(self, image, legend_images, data_id=-1):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""         
        # For profiling memory usage 
        #torch.cuda.memory._record_memory_history()

        # make sure the device is correctly updated
        self.model.to(self.device)
        self.sam_model.to(self.device)


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
            map_patches_transpose = map_patches.transpose(0, 2, 3, 1)
            map_img_DINO_patches = self.dino_processor(map_patches_transpose, return_tensors="pt")['pixel_values'].to(self.device)       

        map_patches = torch.Tensor(map_patches).to(self.device)
        map_patches_original = map_patches.clone()
        map_patches = self.my_norm(map_patches)

        # pipeline_manager.log(logging.DEBUG, f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
        map_prediction = np.zeros((1, map_height, map_width), dtype=np.float32)
        map_confidence = np.zeros((1, map_height, map_width), dtype=np.float32)

        # domain color extraction from legend images
        map_legend_extraction = {}
        map_legend_extraction_name = {}

        # make sure to check the model and args
        # pipeline_manager.log(logging.DEBUG, f"\t args: {self._args}")
        # pipeline_manager.log(logging.DEBUG, f"\t model: {self.patch_size} {self._checkpoint}")

        # Loop through each legend image to extract the dominant color
        for legend_idx, (label, legend_img) in enumerate(legend_images.items()):
            if legend_img.shape[0] == 1:
                legend_img = np.concatenate([legend_img, legend_img, legend_img], axis=0)

            curr_lgd_img = legend_img.copy()
            # curr_lgd_img = self.mask_margins(curr_lgd_img, margin=0.0)
            dominant_color = self.extract_dominant_colors(curr_lgd_img, legend=True, num=legend_idx+1, n_colors=1)
            map_legend_extraction[str(legend_idx+1)] = dominant_color

            map_legend_extraction_name[str(legend_idx+1)] = label

            pipeline_manager.log(logging.DEBUG, f"\tLegend: {label} - Dominant Color: {dominant_color}")

        
        # loop through each legend image per patch and perform inference
        with torch.no_grad():
            for legend_idx, (label, legend_img) in enumerate(legend_images.items()):
                lgd_stime = time()

                # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
                if legend_img.shape[0] == 1:
                    legend_img = np.concatenate([legend_img, legend_img, legend_img], axis=0)

                with torch.no_grad():
                    legend_img_transpose = legend_img.transpose(1, 2, 0)
                    legend_img_DINO = self.dino_processor(legend_img_transpose, return_tensors="pt")['pixel_values'][0].to(self.device)
                    legend_img_CLIP = self.clip_processor(text=None, images=legend_img_transpose, return_tensors="pt")['pixel_values'][0].to(self.device)
                    legend_mask = torch.ones_like(legend_img_DINO)[0].to(self.device)
                    legend_mask[0:14, :] = 0
                    legend_mask[-14:, :] = 0
                    legend_mask[:, 0:14] = 0
                    legend_mask[:, -14:] = 0

                legend_img_DINO_patches = torch.stack([legend_img_DINO for i in range(self.batch_size)], dim=0)
                legend_img_CLIP_patches = torch.stack([legend_img_CLIP for i in range(self.batch_size)], dim=0)
                legend_mask_patches = torch.stack([legend_mask for i in range(self.batch_size)], dim=0)

                perdiction_mapches = []
                with torch.no_grad():
                    for i in range(0, len(map_patches), self.batch_size):
                        protos, _ = self.model('mask', 
                                        map_img_DINO_patches[i:i+self.batch_size], 
                                        legend_img_DINO_patches[:len(map_patches[i:i+self.batch_size])],
                                        legend_mask_patches[:len(map_patches[i:i+self.batch_size])],
                                        False,
                                        legend_img_CLIP_patches[:len(map_patches[i:i+self.batch_size])])
            
                        _, prediction = self.sam_model(map_patches[i:i+self.batch_size], None, protos)



                        # pipeline_manager.log(logging.DEBUG, f"\t\tprediction values before sigmoid: {prediction}")
                        prediction = torch.sigmoid(prediction)

                        # pipeline_manager.log(logging.DEBUG, f"\t\tprediction values after sigmoid: {prediction}")

                        # do the tresholding here
                        prediction = (prediction > 0.5).float()

                        # pipeline_manager.log(logging.DEBUG, f"\t\tprediction values after tresholding: {prediction}")

                        # expand the prediction to 3 channels
                        prediction = prediction.unsqueeze(1)
                        # prediction_bool = prediction.bool()

                        # pipeline_manager.log(logging.DEBUG, f"\t\tLegend: {label} - Patch: {i} - Prediction: {prediction}")

                        # number of zero values from the prediction
                        # pipeline_manager.log(logging.DEBUG, f"\t\tLegend: {label} - Patch: {i} - Prediction: {torch.count_nonzero(prediction_bool)}")
                        
                        # max and min value of the mask
                        # pipeline_manager.log(logging.DEBUG, f"\t\t\tMax and Min value of the mask: {prediction.max()} - {prediction.min()}")

                        # pipeline_manager.log(logging.DEBUG, f"\t\tLegend: {label} - Patch: {i} - Prediction: {prediction.shape}")
                        # pipeline_manager.log(logging.DEBUG, f"\t\t\tPrediction: {map_patches[i:i+self.batch_size].shape}")
                        
                        # print out preidction and map_patches for debugging, the first one
                        # pipeline_manager.log(logging.DEBUG, f"\t\t\tPrediction: {prediction[0]}")
                        # masks onto the map
                        curr_patch_with_mask = map_patches_original[i:i+self.batch_size] * prediction.expand(-1, 3, -1, -1)

                        # pipeline_manager.log(logging.DEBUG, f"\t\t\t\tCurrent max and min value of the patch: {curr_patch_with_mask.max()} - {curr_patch_with_mask.min()}")

                        # pipeline_manager.log(logging.DEBUG, f"\t\t\t\tCurrent Patch with Mask: {curr_patch_with_mask[0]}")

                        # save the curr patch with mask into the current directory
                        # curr_patch_with_mask_np = curr_patch_with_mask.cpu().numpy()
                        # curr_patch_with_mask_np = curr_patch_with_mask_np.transpose(0, 2, 3, 1)
                        # cv2.imwrite(f'tmp_patch_{i}.png', cv2.cvtColor(curr_patch_with_mask_np[0], cv2.COLOR_RGB2BGR))

                        # # save the masks 
                        # curr_patch_mask_np = prediction_bool.expand(-1, 3, -1, -1).cpu().numpy() * 255
                        # curr_patch_mask_np = curr_patch_mask_np.transpose(0, 2, 3, 1)
                        # cv2.imwrite(f'tmp_mask_{i}.png', curr_patch_mask_np[0])

                        # from ipdb import set_trace; set_trace()




                        # dominant color extraction
                        curr_str = str(self.batch_size) + str(i)
                        dominant_colors = [self.extract_dominant_colors(curr_patch_with_mask[j].cpu().numpy(), 
                                                n_colors=1, num=str(str(curr_str) + str(j))) for j in range(len(curr_patch_with_mask))]

                        #curr_pred_zeros = torch.zeros_like(prediction.shape[0], 1, prediction.shape[2], prediction.shape[3]).to(self.device)
                        curr_pred_zeros = torch.zeros((prediction.shape[0], 1, prediction.shape[2], prediction.shape[3])).to(self.device)
                        # numpy
                        # find the closest legend color to the map color
                        for j in range(len(curr_patch_with_mask)):
                            closest_legend_color = 1
                            closest_legend_color_distance = float('inf')
                            for matching_lgd_idx, legend_color in map_legend_extraction.items():
                                distance = np.linalg.norm(legend_color - dominant_colors[j])
                                if distance < closest_legend_color_distance:
                                    closest_legend_color = matching_lgd_idx
                                    closest_legend_color_distance = distance

                            
                            # update the map_prediction with the closest legend color
                            pipeline_manager.log(logging.DEBUG, f"\t\t\t\tCurr image name : {str(curr_str) + str(j)}")
                            pipeline_manager.log(logging.DEBUG, f"\t\t\t\tCurent Legend Color: {label} - Dominant Color: {dominant_colors[j]}")
                            pipeline_manager.log(logging.DEBUG, f"\t\t\t\tClosest Legend Color: {closest_legend_color} - Distance: {closest_legend_color_distance}")
                            pipeline_manager.log(logging.DEBUG, f"\t\t\t\tClosest Legend Color name: {map_legend_extraction_name[closest_legend_color]}")
                            curr_pred_zeros[j] = prediction[j] * int(closest_legend_color)
                        
                        perdiction_mapches.append(curr_pred_zeros)

                # unpatch
                prediction_patches = torch.cat(perdiction_mapches, dim=0).cpu().numpy()
                prediction_patches = prediction_patches.reshape([1, cols, rows, 1, self.patch_size, self.patch_size])
                unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=self.patch_overlap, mode=self.unpatch_mode)
                prediction_image = unpatch_image[:,:map_height,:map_width]

                # transfer non zero values from prediction_image to map_prediction
                map_prediction[prediction_image > 0] = prediction_image[prediction_image > 0]

                gc.collect()

                lgd_time = time() - lgd_stime

            return map_prediction





                

            