import os
import logging
import traceback
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from math import ceil, floor
import re
import json

import src.cmass_io as io
import src.utils as utils
from src.interprocess_communication import ipq_message_type, ipq_log_message, ipq_work_message

def data_saving_worker(input_queue, log_queue, output_dir, feedback_dir):
    pid = multiprocessing.current_process().pid
    log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_SAVING, logging.DEBUG, None, f'Data saving worker starting'))
    legend_feedback_mode = 'single_image'
    while True:
        try:
            # Wait for work
            if input_queue.empty():
                sleep(1)
                continue

            # Retrive work from queue
            work_message = input_queue.get()
            
            # Check for stop message
            if work_message == 'STOP':
                log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_SAVING, logging.DEBUG, None, f'Data saving worker stopping'))
                break
            map_data = work_message.content

            log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_SAVING, logging.DEBUG, map_data.name, f'Started saving {map_data.name}'))
            # Save Legend preview
            if feedback_dir:
                os.makedirs(os.path.join(feedback_dir, map_data.name), exist_ok=True)
                # Cutout map unit labels
                legend_images = {}
                for label, feature in map_data.legend.features.items():
                    min_pt, max_pt = utils.boundingBox(feature.contour) # Need this as points order can be reverse or could have quad
                    legend_images[feature.name] = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]

                # Save preview of legend labels
                if len(legend_images) > 0:
                    if legend_feedback_mode == 'individual_images':
                        legend_save_path = os.path.join(feedback_dir, map_data.name, 'lgd_' + map_data.name + '_' + feature.name + '.tif')
                        io.saveGeoTiff(legend_save_path, legend_images[feature.name], None, None)
                    if legend_feedback_mode == 'single_image':
                        cols = 4
                        rows = ceil(len(legend_images)/cols)
                        fig, ax = plt.subplots(rows, cols, figsize=(16,16))
                        ax = ax.reshape(rows, cols) # Force 2d shape if less the 4 items
                        for r,c in np.ndindex(ax.shape):
                            ax[r][c].axis('off')
                        for i, label in enumerate(legend_images):
                            row, col  = floor(i/cols), i%cols
                            ax[row][col].set_title(label)
                            ax[row][col].imshow(legend_images[label].transpose(1,2,0))
                        legend_save_path = os.path.join(feedback_dir, map_data.name, map_data.name + '_labels'  + '.png')
                        fig.savefig(legend_save_path)
                        plt.close(fig)

            # Save inference results
            # Save CDR schema
            map_data = generate_geometry(map_data)
            cdr_schema = export_CMAAS_Map_to_cdr_schema(map_data)
            with open(os.path.join(output_dir, f'{map_data.name}_cdr.json'), 'w') as fh:
                fh.write(cdr_schema.model_dump_json())
            # Save raster masks
            legend_index = 1
            for label, feature in map_data.legend.features.items():
                feature_mask = np.zeros_like(map_data.mask, dtype=np.uint8)
                feature_mask[map_data.mask == legend_index] = 1
                filename = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", f'{map_data.name}_{feature.name}.tif')
                filepath = os.path.join(output_dir, filename)
                io.saveGeoTiff(filepath, feature_mask, map_data.georef.crs, map_data.georef.transform)
                legend_index += 1
            log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_SAVING, logging.DEBUG, map_data.name, f'Completed saving {map_data.name}'))
        except Exception as e:
            # Note failure and retry up to 3 times
            log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_SAVING, logging.ERROR, map_data.name, f'Error occured on saving worker'))
            log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_SAVING, logging.ERROR, map_data.name, f'Inference worker failed on {map_data.name} on try {work_message.retries} with exception {e}\n{traceback.format_exc()}'))
            work_message.retries += 1
            if work_message.retries < 3:
                input_queue.put(work_message)
            else:
                log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_SAVING, logging.ERROR, map_data.name, f'MAP {map_data.name} WAS NOT PROCESSED! Could not save data after 3 tries skipping map'))
    return True



from cdr_schemas.common import ModelProvenance
from cdr_schemas.feature_results import FeatureResults
from cdr_schemas.features.point_features import PointLegendAndFeaturesResult, PointFeatureCollection, PointFeature,  Point, PointProperties
from cdr_schemas.features.polygon_features import PolygonLegendAndFeaturesResult, PolygonFeatureCollection, PolygonFeature, Polygon, PolygonProperty

from src.cmass_types import CMASS_Map, CMASS_Legend, CMASS_Feature
from rasterio.features import shapes, sieve 
from shapely.geometry import shape
from typing import List

# region Tmp things to get this to work
def generate_geometry(map_data, noise_threshold=10):
    for _, feature in map_data.legend.features.items():
        # Get mask of feature
        feature_mask = np.zeros_like(map_data.mask, dtype=np.uint8)
        feature_mask[map_data.mask == id] = 1
        # Remove "noise" from mask by removing pixel groups smaller then the threshold
        #sieve_img = sieve(feature_mask, noise_threshold, connectivity=4)
        # Convert mask to vector shapes
        shape_gen = shapes(feature_mask, connectivity=4)
        # Only use Filled pixels (1s) for shapes 
        geometries = [shape(geometry) for geometry, value in shape_gen if value == 1]
        # Change Shapely geometryies to List(List(List(float)))
        feature.geometry = [[[*point] for point in geometry.exterior.coords] for geometry in geometries]
    return map_data
# endregion Tmp things to get this to work

# region CDR Common
def export_CMAAS_Map_to_cdr_schema(map_data, cog_id:str='', system:str='UIUC', system_version:str='0.1'):
    polygon_segmentations = []
    for _, feature in map_data.legend.features.items():
        polygon_segmentations.append(_build_CDR_poly_feature(feature))
    return FeatureResults(cog_id=cog_id, system=system, system_version=system_version, polygon_feature_results=polygon_segmentations)

def saveCDRFeatureResults(filepath, feature_result: FeatureResults):
    # Save CDR schema
    with open(filepath, 'w') as fh:
        fh.write(feature_result.model_dump_json())
# endregion CDR Common

# region CDR Polygon
def _build_CDR_poly_feature(feature : CMASS_Feature) -> PolygonLegendAndFeaturesResult:
    if feature.geometry is not None:
        poly_collection = _build_CDR_poly_feature_collection(feature.geometry)
    else:
        poly_collection = None
    poly_feature = PolygonLegendAndFeaturesResult(
        id="None",
        legend_provenance=ModelProvenance(model='UIUC', model_version='0.1', confidence=None),
        label=feature.name,
        abbreviation=feature.abbreviation,
        description=feature.description,
        legend_contour=feature.contour,
        color=feature.color,
        pattern=feature.pattern,
        polygon_features=poly_collection)
    return poly_feature

def _build_CDR_poly_feature_collection(geometry) -> PolygonFeatureCollection:
    poly_features = []
    for poly in geometry:
        poly_features.append(PolygonFeature(
            id='None', 
            geometry=Polygon(coordinates=[poly]),
            properties=PolygonProperty(model='golden_muscat', model_version='0.0.3')
        ))
    
    return PolygonFeatureCollection(features=poly_features)

# endregion CDR Polygon