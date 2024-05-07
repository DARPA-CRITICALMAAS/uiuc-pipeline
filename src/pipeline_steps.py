import logging
import numpy as np
import cmaas_utils.io as io
import cmaas_utils.cdr as cdr
from cmaas_utils.types import CMAAS_Map, GeoReference, Layout, MapUnitType, Provenance
from src.utils import boundingBox, sanitize_filename
from src.pipeline_manager import pipeline_manager
from time import time
import multiprocessing as mp

def load_data(data_id, image_path:str, legend_dir:str=None, layout_dir:str=None):
    """Wrapper with a custom display for the monitor"""
    map_name = os.path.splitext(os.path.basename(image_path))[0]
    pipeline_manager.log(logging.DEBUG, f'{map_name} - Started processing', pid=mp.current_process().pid)
    if len(map_name) > 50:
        pipeline_manager.log_to_monitor(data_id, {'Name' : map_name[:24] + '...' + map_name[-24:]})
    else:
        pipeline_manager.log_to_monitor(data_id, {'Name': map_name})
    legend_path = None
    layout_path = None
    if legend_dir:
        legend_path = os.path.join(legend_dir, map_name + '.json')
        if not os.path.exists(legend_path):
            legend_path = None
    if layout_dir:
        layout_path = os.path.join(layout_dir, map_name + '.json')
        if not os.path.exists(layout_path):
            layout_path = None
    map_data = io.loadCMAASMapFromFiles(image_path, legend_path, layout_path)
    pipeline_manager.log_to_monitor(data_id, {'Shape': map_data.image.shape})
    return map_data

def amqp_load_data(data_id, cog_tuple):
    """Wrapper with a custom display for the monitor"""
    cog_id, image_path, cdr_json_path = cog_tuple
    map_name = os.path.splitext(os.path.basename(image_path))[0]
    pipeline_manager.log(logging.DEBUG, f'{map_name} - Started processing', pid=mp.current_process().pid)
    if len(map_name) > 50:
        pipeline_manager.log_to_monitor(data_id, {'Name' : map_name[:24] + '...' + map_name[-24:]})
    else:
        pipeline_manager.log_to_monitor(data_id, {'Name': map_name})
    # Load CDR data
    fr = io.loadCDRFeatureResults(cdr_json_path)
    map_data = cdr.importCDRFeatureResults(fr)
    map_data.name = map_name
    map_data.cog_id = cog_id
    map_data.layout = tmp_fix_layout(map_data.layout)
    image, crs, transform = io.loadGeoTiff(image_path)
    map_data.image = image
    map_data.georef = GeoReference(provenance=Provenance(name='GeoTIFF'), crs=crs, transform=transform)
    pipeline_manager.log_to_monitor(data_id, {'Shape': map_data.image.shape})
    return map_data

def gen_layout(data_id, map_data:CMAAS_Map):
    # Generate layout if not precomputed
    if map_data.layout is None:
        pipeline_manager.log(logging.WARNING, f'{map_data.name} - No layout found, generating layout in pipeline not implemented yet')
        # TODO Implement layout generation
        pass
    return map_data

def gen_legend(data_id, map_data:CMAAS_Map, max_legends=300, drab_volcano_legend:bool=False):
    from submodules.legend_extraction.src.extraction import extractLegends
    def convertLegendtoCMASS(legend):
        from cmaas_utils.types import Legend, MapUnit
        features = []
        for feature in legend:
            features.append(MapUnit(type=MapUnitType.POLYGON, label=feature['label'], bounding_box=feature['points']))
        return Legend(provenance=Provenance(name='UIUC Heuristic Model', version='0.1'), features=features)

    if drab_volcano_legend:
        map_data.legend = io.loadLegendJson('src/models/drab_volcano_legend.json')
    else:
        # Generate legend if not precomputed
        if map_data.legend is None:
            pipeline_manager.log(logging.DEBUG, f'{map_data.name} - No legend data found, generating legend', pid=mp.current_process().pid)
            lgd = extractLegends(map_data.image.transpose(1,2,0))
            map_data.legend = convertLegendtoCMASS(lgd)

    # Count distribution of map units for log.
    pt, ln, py, un = 0,0,0,0
    for feature in map_data.legend.features:
        if feature.type == MapUnitType.POINT:
            pt += 1
        if feature.type == MapUnitType.LINE:
            ln += 1
        if feature.type == MapUnitType.POLYGON:
            py += 1
        if feature.type == MapUnitType.UNKNOWN:
            un += 1

    # TMP solution for maps with too many features (most likely from bad legend extraction)
    if len(map_data.legend.features) > max_legends:
        raise Exception(f'{map_data.name} - Too many features found in legend. Found {len(map_data.legend.features)} features. Max is {max_legends}')

    pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Found {len(map_data.legend.features)} Total map units. ({pt} pt, {ln} ln, {py} poly, {un} unknown)', pid=mp.current_process().pid)
    pipeline_manager.log_to_monitor(data_id, {'Map Units': len(map_data.legend.features)})
    
    return map_data

def save_legend(data_id, map_data:CMAAS_Map, feedback_dir:str, legend_feedback_mode:str = 'single_image'):
    # Create directory for that map
    os.makedirs(os.path.join(feedback_dir, map_data.name), exist_ok=True)

    # Cutout map unit labels from image
    legend_images = {}
    for feature in map_data.legend.features:
        min_pt, max_pt = boundingBox(feature.bounding_box) # Need this as points order can be reverse or could have quad
        legend_images[feature.label] = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]

    # Save preview of legend labels
    if len(legend_images) > 0:
        if legend_feedback_mode == 'individual_images':
            legend_save_path = os.path.join(feedback_dir, map_data.name, sanitize_filename('lgd_' + map_data.name + '_' + feature.label + '.tif'))
            io.saveGeoTiff(legend_save_path, legend_images[feature.label], None, None)
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
            legend_save_path = os.path.join(feedback_dir, map_data.name, sanitize_filename(map_data.name + '_labels'  + '.png'))
            fig.savefig(legend_save_path)
            plt.close(fig)
        # pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Saved legend preview to "{legend_save_path}"', pid=mp.current_process().pid)

def segmentation_inference(data_id, map_data:CMAAS_Map, model, devices=None):
    # Device is set on a per process basis
    from torch import device
    previous_device = model.device
    target_device = device(devices[mp.current_process().pid % len(devices)])
    if model.device != target_device: # Move model to device if needed
        model.device = target_device
        model.model.to(target_device)

    # Cutout Legends
    legend_images = {}
    for feature in map_data.legend.features:
        if feature.type == model.feature_type:
            min_pt, max_pt = boundingBox(feature.bounding_box) # Need this as points order can be reverse or could have quad
            legend_images[feature.label] = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]
        # else:
        #     pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Skipping inference for {feature.label} as it is not a {model.feature_type.name} feature', pid=mp.current_process().pid)

    # Cutout map portion of image
    if map_data.layout is not None and map_data.layout.map is not None:
        min_pt, max_pt = boundingBox(map_data.layout.map)
        image = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]].copy()
    else:
        image = map_data.image

    # Log how many map units are being processed and the estimated time to perform inference
    est_patches = ceil(image.shape[1]/model.patch_size)*ceil(image.shape[2]/model.patch_size)
    est_time = (est_patches*len(legend_images))/model.est_patches_per_sec
    pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Preforming inference on {len(legend_images)} {model.feature_type.to_str().capitalize()} features, Estimated time: {est_time:.2f} secs', pid=mp.current_process().pid)
    pipeline_manager.log_to_monitor(data_id, {'Map Units': f'{len(map_data.legend.features)} ({len(legend_images)} {model.feature_type.to_str().capitalize()}s)'})
    pipeline_manager.log_to_monitor(data_id, {'Est Infer Time': f'{est_time:.2f} secs'})

    # Perform Inference
    s_time = time()
    result_mask = model.inference(image, legend_images, data_id=data_id)
    real_time = time()-s_time
    pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Real inference time: {real_time:.2f} secs, {(est_patches*len(legend_images))/real_time:.2f} patches/sec', pid=mp.current_process().pid)

    # Resize cutout to full map
    if map_data.layout is not None and map_data.layout.map is not None:
        result_image = np.zeros((1, *map_data.image.shape[1:]), dtype=np.float32)
        result_image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]] = result_mask
        result_mask = result_image
    
    # Save mask to appropriate feature type
    if model.feature_type == MapUnitType.POINT:
        map_data.point_segmentation_mask = result_mask
    if model.feature_type == MapUnitType.POLYGON:
        map_data.poly_segmentation_mask = result_mask

    return map_data

def generate_geometry(data_id, map_data:CMAAS_Map, model_name, model_version):
    model_provenance = Provenance(name=model_name, version=model_version)
    if map_data.point_segmentation_mask is not None:
        map_data.generate_point_geometry(model_provenance)
    if map_data.poly_segmentation_mask is not None:
        map_data.generate_poly_geometry(model_provenance)
    total_occurances = 0 
    for feature in map_data.legend.features:
        if feature.segmentation is not None and feature.segmentation.geometry is not None:
            total_occurances += len(feature.segmentation.geometry)

    pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Prediction contained {total_occurances} total segmentations', pid=mp.current_process().pid)
    pipeline_manager.log_to_monitor(data_id, {'Segments': total_occurances})
    
    return map_data

import os
from math import ceil, floor
import matplotlib.pyplot as plt
def save_output(data_id, map_data: CMAAS_Map, output_dir, feedback_dir, output_types, system, system_version):
    # Save CDR schema
    if 'cdr_json' in output_types:
        cog_id = None
        if map_data.cog_id is not None:
            cog_id = map_data.cog_id
        cdr_schema = cdr.exportMapToCDR(map_data, cog_id=cog_id, system=system, system_version=system_version)
        cdr_filename = os.path.join(output_dir, sanitize_filename(f'{map_data.name}_cdr.json'))
        io.saveCDRFeatureResults(cdr_filename, cdr_schema)
        # pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Saved CDR schema to "{cdr_filename}"', pid=mp.current_process().pid)

    # Save GeoPackage
    if 'geopackage' in output_types:
        gpkg_filename = os.path.join(output_dir, sanitize_filename(f'{map_data.name}.gpkg'))
        coord_type = 'pixel'
        if map_data.georef is not None:
            if map_data.georef.crs is not None and map_data.georef.transform is not None:
                coord_type = 'georeferenced'
        for feature in map_data.legend.features:
            feature.label = sanitize_filename(feature.label) # Need to sanitize feature names before saving geopackage
        io.saveGeoPackage(gpkg_filename, map_data, coord_type)
        # pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Saved GeoPackage to "{gpkg_filename}"', pid=mp.current_process().pid)

    # Save Raster masks
    if 'raster_masks' in output_types:
        legend_index = 1
        for feature in map_data.legend.features:
            pipeline_manager.log(logging.DEBUG, f'Saving raster_mask for {map_data.name}, {feature.label} {feature.type}')
            if feature.type == MapUnitType.LINE:
                continue
            if feature.type == MapUnitType.POINT:
                if map_data.point_segmentation_mask is None:
                    pipeline_manager.log(logging.WARNING, f"Can\'t save feature {feature.label}. No predicted point_segmentation mask present.")
                    continue
                feature_mask = np.zeros_like(map_data.point_segmentation_mask, dtype=np.uint8)
                feature_mask[map_data.poly_segmentation_mask == legend_index] = 1
                #filepath = os.path.join(output_dir, sanitize_filename(f'{map_data.name}_{feature.label}.tif'))
                #io.saveGeoTiff(filepath, feature_mask, map_data.georef.crs, map_data.georef.transform)
            if feature.type == MapUnitType.POLYGON:
                if map_data.poly_segmentation_mask is None:
                    pipeline_manager.log(logging.WARNING, f"Can\'t save feature {feature.label}. No predicted poly_segmentation mask present.")
                    continue
                feature_mask = np.zeros_like(map_data.poly_segmentation_mask, dtype=np.uint8)
                feature_mask[map_data.poly_segmentation_mask == legend_index] = 1
            filepath = os.path.join(output_dir, sanitize_filename(f'{map_data.name}_{feature.label}.tif'))
            io.saveGeoTiff(filepath, feature_mask, map_data.georef.crs, map_data.georef.transform)
            legend_index += 1
    return map_data.name

import pandas as pd
from submodules.validation.src.grading import grade_poly_raster, usgs_grade_poly_raster, usgs_grade_pt_raster   
def validation(data_id, map_data: CMAAS_Map, true_mask_dir, feedback_dir, use_usgs_scores=False):
    # Build results dataframe
    results_df = pd.DataFrame(columns = ['Map', 'Feature', 'F1 Score', 'Precision', 'Recall', 'IoU Score (polys)',
                                         'USGS F1 Score (polys)', 'USGS Precision (polys)', 'USGS Recall (polys)', 
                                         'Mean matched distance (pts)', 'Matched (pts)', 'Missing (pts)', 
                                         'Unmatched (pts)'])

    legend_index = 1
    for feature in map_data.legend.features:
        if feature.type in [MapUnitType.LINE, MapUnitType.UNKNOWN]:
            continue
        # Get predicted mask
        if feature.segmentation is not None and feature.segmentation.mask is not None:
            feature_mask = feature.segmentation.mask
        else:
            if feature.type == MapUnitType.POINT:
                if map_data.point_segmentation_mask is None:
                    continue
                feature_mask = np.zeros_like(map_data.point_segmentation_mask, dtype=np.uint8)
                feature_mask[map_data.point_segmentation_mask == legend_index] = 1

            if feature.type == MapUnitType.POLYGON:
                if map_data.poly_segmentation_mask is None:
                    #pipeline_manager.log(logging.WARNING, f'Can\'t validate feature {feature.label}. No predicted segmentation mask present.')
                    continue
                feature_mask = np.zeros_like(map_data.poly_segmentation_mask, dtype=np.uint8)
                feature_mask[map_data.poly_segmentation_mask == legend_index] = 1

        # Get true mask
        true_mask_path = os.path.join(true_mask_dir, f'{map_data.name}_{feature.label.replace(" ","_")}_{feature.type}.tif')
        # Skip features that don't have a true mask available
        if not os.path.exists(true_mask_path):
            pipeline_manager.log(logging.WARNING, f'{map_data.name} - Can\'t validate feature {feature.label}. No true segmentation mask found at {true_mask_path}.', pid=mp.current_process().pid)
            results_df[len(results_df)] = {'Map' : map_data.name, 'Feature' : feature.label, 'F1 Score' : np.nan,
                                           'Precision' : np.nan, 'Recall' : np.nan}
            continue
        true_mask, _, _ = io.loadGeoTiff(true_mask_path)

        # Create feedback image if needed
        feedback_image = None
        if feedback_dir:
            feedback_image = np.zeros((3, *feature_mask.shape[1:]), dtype=np.uint8)

        # Grade image
        if feature.type == MapUnitType.POINT:
            feature_score = usgs_grade_pt_raster(feature_mask, true_mask, feedback_image=feedback_image)
            results_df.loc[len(results_df)] = {'Map' : map_data.name,
                                               'Feature' : feature.label, 
                                               'F1 Score' : feature_score[0],
                                               'Precision' : feature_score[1],
                                               'Recall' : feature_score[2],
                                               'Mean matched distance (pts)' : feature_score[3],
                                               'Matched (pts)' : feature_score[4],
                                               'Missing (pts)' : feature_score[5],
                                               'Unmatched (pts)' : feature_score[6],
                                               'USGS F1 Score (polys)' : np.nan,
                                               'USGS Precision (polys)' : np.nan,
                                               'USGS Recall (polys)' : np.nan,
                                               'IoU Score (polys)' : np.nan
                                               }

        if feature.type == MapUnitType.POLYGON:
            feature_score = grade_poly_raster(feature_mask, true_mask, feedback_image=feedback_image)
            usgs_score = (np.nan, np.nan, np.nan, np.nan, None)
            if use_usgs_scores:
                usgs_score = usgs_grade_poly_raster(feature_mask, true_mask, map_data.image, map_data.legend, difficult_weight=0.7)
                feature_score = {**feature_score, **usgs_score}
            results_df.loc[len(results_df)] = {'Map' : map_data.name,
                                            'Feature' : feature.label, 
                                            'F1 Score' : feature_score[0],
                                            'Precision' : feature_score[1],
                                            'Recall' : feature_score[2],
                                            'IoU Score (polys)' : feature_score[3],
                                            'USGS F1 Score (polys)' : usgs_score[0],
                                            'USGS Precision (polys)' : usgs_score[1],
                                            'USGS Recall (polys)' : usgs_score[2],
                                            'Mean matched distance (pts)' : np.nan,
                                            'Matched (pts)' : np.nan,
                                            'Missing (pts)' : np.nan,
                                            'Unmatched (pts)' : np.nan 
                                            }

        # Save feature feedback image
        if feedback_dir:
            feedback_path = os.path.join(feedback_dir, sanitize_filename(f'val_{map_data.name}_{feature.label}.tif'))
            io.saveGeoTiff(feedback_path, feedback_image, map_data.georef.crs, map_data.georef.transform)
        legend_index += 1

    # Save map scores
    if feedback_dir:
        os.makedirs(os.path.join(feedback_dir, map_data.name), exist_ok=True)
        csv_path = os.path.join(feedback_dir, map_data.name, f'#{map_data.name}_scores.csv')
        results_df.to_csv(csv_path, index=False)

    # Average validation results for map
    results_df = results_df[results_df['F1 Score'].notna()]
    f1s, pre, rec, iou = results_df["F1 Score"].mean(), results_df["Precision"].mean(), results_df["Recall"].mean(), results_df["IoU Score (polys)"].mean()
    uf1, upr, urc = results_df["USGS F1 Score (polys)"].mean(), results_df["USGS Precision (polys)"].mean(), results_df["USGS Recall (polys)"].mean()
    mpt, fpt, upt, dpt = sum(results_df["Matched (pts)"]), sum(results_df["Missing (pts)"]), sum(results_df["Unmatched (pts)"]), results_df["Mean matched distance (pts)"].mean()
    pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Average validation scores | F1 : {f1s:.2f}, Precision : {pre:.2f}, Recall : {rec:.2f}, IoU : {iou:.2f}', pid=mp.current_process().pid)
    pipeline_manager.log_to_monitor(data_id, {'F1 Score': f'{f1s:.2f}'})

    return 

from time import sleep
def test_step(data_id, filename):
    raise Exception('Test Error')
    pipeline_manager.log_to_monitor(data_id, {'filename' :os.basename(filename)})
    sleep(1)
    return filename

def tmp_fix_layout(layout : Layout):
    if layout.map is not None:
        layout.map = layout.map[0]
    if layout.polygon_legend is not None:
        layout.polygon_legend = layout.polygon_legend[0]
    if layout.point_legend is not None:
        layout.point_legend = layout.point_legend[0]
    if layout.line_legend is not None:
        layout.line_legend = layout.line_legend[0]
    if layout.cross_section is not None:
        layout.cross_section = layout.cross_section[0]
    if layout.correlation_diagram is not None:
        layout.correlation_diagram = layout.correlation_diagram[0]
    return layout
