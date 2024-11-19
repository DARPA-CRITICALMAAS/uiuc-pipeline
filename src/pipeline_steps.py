import os
import logging
import numpy as np
import cmaas_utils.io as io
import cmaas_utils.cdr as cdr
from cmaas_utils.types import CMAAS_Map, GeoReference, MapSegmentation, MapUnitType, Provenance
from cmaas_utils.utilities import generate_point_geometry, generate_poly_geometry, mask_and_crop
from src.utils import boundingBox, sanitize_filename
from src.pipeline_manager import pipeline_manager
from time import time
from math import ceil, floor
import multiprocessing as mp
import matplotlib.pyplot as plt

# region Load Data
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
    pipeline_manager.log(logging.WARNING, f'Map loaded with {len(map_data.legend.features)} Map units')
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
    with open(cdr_json_path, 'r') as fh:
        map_data = CMAAS_Map.parse_raw(fh.read())
    map_data.name = map_name
    map_data.cog_id = cog_id
    image, crs, transform = io.loadGeoTiff(image_path)
    map_data.image = image
    map_data.georef = GeoReference(provenance=Provenance(name='GeoTIFF'), crs=crs, transform=transform)
    pipeline_manager.log_to_monitor(data_id, {'Shape': map_data.image.shape})
    return map_data
# endregion Load Data


def gen_layout(data_id, map_data:CMAAS_Map):
    # Generate layout if not precomputed
    if map_data.layout is None:
        pipeline_manager.log(logging.WARNING, f'{map_data.name} - No layout found, generating layout in pipeline not implemented yet')
        # TODO Implement layout generation
        pass
    return map_data

# region Legend Generation
def gen_legend(data_id, map_data:CMAAS_Map, model, max_legends=300, drab_volcano_legend:bool=False):
    # Generate legend if not precomputed
    if map_data.legend is None:
        if drab_volcano_legend:
            map_data.legend = io.loadLegendJson('src/models/drab_volcano_legend.json')
        else:
            pipeline_manager.log(logging.DEBUG, f'{map_data.name} - No legend data found, generating legend', pid=mp.current_process().pid)
            
            # Generate legend
            map_data.legend = model.inference(map_data.image, map_data.layout, data_id=data_id)

    # Reduce duplicates
    if map_data.legend.provenance.name != 'polymer': # Skip de-duplication for polymer legends
        legend_features = {}
        for feature in map_data.legend.features:
            legend_features[feature.label] = feature
    if isinstance(map_data.legend.features, dict):
        map_data.legend.features = list(legend_features.values())
    pipeline_manager.log(logging.WARNING, f'Map features after de duplication : {len(map_data.legend.features)} Map units')

    # TMP solution for maps with too many features (most likely from bad legend extraction)
    if len(map_data.legend.features) > max_legends:
        raise Exception(f'{map_data.name} - Too many features found in legend. Found {len(map_data.legend.features)} features. Max is {max_legends}')

    # Remove features with no label swatch
    map_data.legend.features = [f for f in map_data.legend.features if feature.label_bbox] # Remove empty labels

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
    
    pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Found {len(map_data.legend.features)} Total map units. ({pt} pt, {ln} ln, {py} poly, {un} unknown)', pid=mp.current_process().pid)
    pipeline_manager.log_to_monitor(data_id, {'Map Units': len(map_data.legend.features)})
    
    return map_data

def save_legend(data_id, map_data:CMAAS_Map, feedback_dir:str, legend_feedback_mode:str = 'single_image'):
    # Create directory for that map
    os.makedirs(os.path.join(feedback_dir, map_data.name), exist_ok=True)

    # Cutout map unit labels from image
    legend_images = []
    for feature in map_data.legend.features:
        min_pt, max_pt = boundingBox(feature.label_bbox) # Need this as points order can be reverse or could have quad
        legend_images.append((feature.label, map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]))

    # Save preview of legend labels
    if len(legend_images) > 0:
        if feedback_dir:
            os.makedirs(os.path.join(feedback_dir, map_data.name), exist_ok=True)
            with open(os.path.join(feedback_dir, map_data.name, sanitize_filename(map_data.name + '_legend.json')), 'w') as fh:
                fh.write(map_data.legend.model_dump_json())
        if legend_feedback_mode == 'individual_images':
            for label, image in legend_images:
                legend_save_path = os.path.join(feedback_dir, map_data.name, sanitize_filename('lgd_' + map_data.name + '_' + label + '.tif'))
                io.saveGeoTiff(legend_save_path, image, None, None)
        if legend_feedback_mode == 'single_image':
            cols = 4
            rows = ceil(len(legend_images)/cols)
            fig, ax = plt.subplots(rows, cols, figsize=(16,16))
            ax = ax.reshape(rows, cols) # Force 2d shape if less the 4 items
            for r,c in np.ndindex(ax.shape):
                ax[r][c].axis('off')
            for i, f_tuple in enumerate(legend_images):
                label, image = f_tuple
                row, col  = floor(i/cols), i%cols
                ax[row][col].set_title(label)
                ax[row][col].imshow(image.transpose(1,2,0))
            legend_save_path = os.path.join(feedback_dir, map_data.name, sanitize_filename(map_data.name + '_labels'  + '.png'))
            fig.savefig(legend_save_path)
            plt.close(fig)
        # pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Saved legend preview to "{legend_save_path}"', pid=mp.current_process().pid)
# endregion Legend Generation

# region Segmentation
def segmentation_inference(data_id, map_data:CMAAS_Map, model, devices=None):
    # Device is set on a per process basis
    from torch import device
    previous_device = model.device
    target_device = device(devices[mp.current_process().pid % len(devices)])
    if model.device != target_device: # Move model to device if needed
        model.device = target_device
        model.model.to(target_device)

    # Cutout Legends
    legend_images = []
    for feature in map_data.legend.features:
        if feature.type == model.feature_type:
            min_pt, max_pt = boundingBox(feature.label_bbox) # Need this as points order can be reverse or could have quad
            legend_images.append(map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]])
        # else:
        #     pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Skipping inference for {feature.label} as it is not a {model.feature_type.name} feature', pid=mp.current_process().pid)

    # Cutout map portion of image
    if len(map_data.layout.map) > 0:
        image, offset = mask_and_crop(map_data.image, map_data.layout.map)
    else:
        image = map_data.image
        offset = (0,0)

    # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
    map_channels, map_height, map_width = image.shape
    if map_channels == 1: 
        image = np.concatenate([image,image,image], axis=0)

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
    if len(map_data.layout.map) > 0:
        result_image = np.zeros((1, *map_data.image.shape[1:]), dtype=np.float32)
        result_image[:,offset[1]:offset[1]+result_mask.shape[1], offset[0]:offset[0]+result_mask.shape[2]] = result_mask
        result_mask = result_image
    
    # Save mask to appropriate feature type
    map_data.segmentations.append(MapSegmentation(provenance=Provenance(name=model.name, version=model.version), type=model.feature_type, image=result_mask)) # confidence=?

    # Drab volcano only : Filter out features that were not present
    if model.name == 'drab volcano':
        legend_index = 1
        features_present = []
        for feature in map_data.legend.features:
            if np.any(result_mask == legend_index):
                features_present.append(feature)
            legend_index += 1
        # Set legend to only features that had predictions
        map_data.legend.features = features_present
        pipeline_manager.log_to_monitor(data_id, {'Map Units': f'{len(features_present)} ({len(features_present)} {model.feature_type.to_str().capitalize()}s)'})
        pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Drab Volcano predicted {len(features_present)}/48 features', pid=mp.current_process().pid)

    return map_data
# endregion Segmentation

# region Vectorization
def generate_geometry(data_id, map_data:CMAAS_Map):
    for mask in map_data.segmentations:
        if mask.type == MapUnitType.POINT:
            generate_point_geometry(mask, map_data.legend)
        if mask.type == MapUnitType.POLYGON:
            generate_poly_geometry(mask, map_data.legend)

    total_occurances = 0 
    for feature in map_data.legend.features:
        if feature.segmentation is not None and feature.segmentation.geometry is not None:
            total_occurances += len(feature.segmentation.geometry)

    pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Prediction contained {total_occurances} total segmentations', pid=mp.current_process().pid)
    pipeline_manager.log_to_monitor(data_id, {'Segments': total_occurances})
    
    return map_data
# endregion Vectorization

# region Output
def save_output(data_id, map_data: CMAAS_Map, output_dir, feedback_dir, output_types, system, system_version):
    # pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Started save_output', pid=mp.current_process().pid)
    # Save CDR schema
    if 'cdr_json' in output_types:
        cog_id = None
        if map_data.cog_id is not None:
            cog_id = map_data.cog_id
        else:
            # setting cog_id to string for "local" processing
            # TODO: Set this default in cmaas_utils, or perhaps add and overrige --cog_id to pipeline.py
            cog_id = "-1"
        cdr_schema = cdr.exportMapToCDR(map_data, cog_id=cog_id, system=system, system_version=system_version)
        cdr_filename = os.path.join(output_dir, sanitize_filename(f'{map_data.name}_cdr.json'))
        io.saveCDRFeatureResults(cdr_filename, cdr_schema)
        pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Saved CDR schema to "{cdr_filename}"', pid=mp.current_process().pid)

    # Save GeoPackage
    if 'geopackage' in output_types:
        gpkg_filename = os.path.join(output_dir, sanitize_filename(f'{map_data.name}.gpkg'))
        coord_type = 'pixel'
        if map_data.georef is not None:
            if map_data.georef.crs is not None and map_data.georef.transform is not None:
                coord_type = 'georeferenced'
        for feature in map_data.legend.features:
            # pipeline_manager.log(logging.WARNING, f'{map_data.name} - Feature label before sanitization: {feature.label}')
            feature.label = sanitize_filename(feature.label).replace(' ', '_') # Need to sanitize feature names before saving geopackage
            # pipeline_manager.log(logging.WARNING, f'{map_data.name} - Feature label after sanitization: {feature.label}')
        # io.saveGeoPackage(gpkg_filename, map_data)
        test_saveGeoPackage(gpkg_filename, map_data)
        pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Saved GeoPackage to "{gpkg_filename}"', pid=mp.current_process().pid)

    # Save Raster masks
    if 'raster_masks' in output_types:
        legend_index = 1
        for feature in map_data.legend.features:
            if feature.type in [MapUnitType.LINE, MapUnitType.UNKNOWN]:
                continue
            if feature.type == MapUnitType.POINT:
                point_mask = None
                for mask in map_data.segmentations:
                    if mask.type == MapUnitType.POINT:
                        point_mask = mask.image
                        break
                if point_mask is None:
                    # pipeline_manager.log(logging.WARNING, f"{map_data.name} - Can\'t save feature {feature.label}. No predicted point_segmentation mask present.")
                    continue
                feature_mask = np.zeros_like(point_mask, dtype=np.uint8)
                feature_mask[point_mask == legend_index] = 1
            if feature.type == MapUnitType.POLYGON:
                poly_mask = None
                for mask in map_data.segmentations:
                    if mask.type == MapUnitType.POLYGON:
                        poly_mask = mask.image
                        break
                if poly_mask is None:
                    # pipeline_manager.log(logging.WARNING, f"{map_data.name} - Can\'t save feature {feature.label}. No predicted poly_segmentation mask present.")
                    continue
                feature_mask = np.zeros_like(poly_mask, dtype=np.uint8)
                feature_mask[poly_mask == legend_index] = 1
            filepath = os.path.join(output_dir, sanitize_filename(f'{map_data.name}_{feature.label}_{feature.type}.tif'))
            io.saveGeoTiff(filepath, feature_mask, map_data.georef.crs, map_data.georef.transform)
            # pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Saved raster_mask to "{filepath}", pid=mp.current_process().pid')
            legend_index += 1
    return map_data.name
# endregion Output

# region Validation
import pandas as pd
from submodules.validation.src.grading import grade_point_raster, grade_poly_raster, usgs_grade_poly_raster    
def validation(data_id, map_data: CMAAS_Map, true_mask_dir, output_dir, feedback_dir, use_usgs_scores=False):
    # Build results dataframe
    results_df = pd.DataFrame(columns = [
        'Map', 'Feature', 'F1 Score', 'Precision', 'Recall', 'IoU Score',
        'USGS F1 Score', 'USGS Precision', 'USGS Recall', 
        'Mean matched distance Points', 'Matched Points', 'Unmatched Points', 'Missing Points'
    ])

    legend_index = 1
    for feature in map_data.legend.features:
        # Skip features that we don't make predictions for
        if feature.type in [MapUnitType.LINE, MapUnitType.UNKNOWN]:
            continue

        # Get predicted mask
        if feature.segmentation is not None and feature.segmentation.mask is not None:
            feature_mask = feature.segmentation.mask
        else:
            if feature.type == MapUnitType.POINT:
                # Skip features there isn't a predicted mask for
                point_mask = None
                for mask in map_data.segmentations:
                    if mask.type == MapUnitType.POINT:
                        point_mask = mask.image
                        break
                if point_mask is None:
                    continue
                feature_mask = np.zeros_like(point_mask, dtype=np.uint8)
                feature_mask[point_mask == legend_index] = 1

            if feature.type == MapUnitType.POLYGON:
                # Skip features there isn't a predicted mask for
                poly_mask = None
                for mask in map_data.segmentations:
                    if mask.type == MapUnitType.POLYGON:
                        poly_mask = mask.image
                        break
                if poly_mask is None:
                    continue
                feature_mask = np.zeros_like(poly_mask, dtype=np.uint8)
                feature_mask[poly_mask == legend_index] = 1

        # Get true mask
        true_mask_path = os.path.join(true_mask_dir, f'{map_data.name}_{feature.label.replace(" ","_")}_{feature.type}.tif')
        
        # Skip features that don't have a true mask available
        if not os.path.exists(true_mask_path):
            alias_found = False
            if feature.aliases is not None:
                for alias in feature.aliases:
                    true_mask_path = os.path.join(true_mask_dir, f'{map_data.name}_{alias.replace(" ","_")}_{feature.type}.tif')
                    if os.path.exists(true_mask_path):
                        pipeline_manager.log(logging.WARNING, f'{map_data.name} - Using alias {alias} for feature {feature.label}', pid=mp.current_process().pid)
                        alias_found = True
                        break
            if not alias_found:
                pipeline_manager.log(logging.WARNING, f'{map_data.name} - Can\'t validate feature {feature.label}. No true segmentation mask found at {true_mask_path}.', pid=mp.current_process().pid)
                results_df.loc[len(results_df)] = {'Map' : map_data.name, 'Feature' : feature.label}
                legend_index += 1
                continue
        true_mask, _, _ = io.loadGeoTiff(true_mask_path)

        # Create feedback image if needed
        feedback_image = None
        if feedback_dir:
            feedback_image = np.zeros((3, *feature_mask.shape[1:]), dtype=np.uint8)

        # Grade image
        if feature.type == MapUnitType.POINT:
            results, feedback_image = grade_point_raster(feature_mask, true_mask, feedback_image=feedback_image)
            results['Map'] = map_data.name
            results['Feature'] = feature.label
            results['USGS F1 Score'] = results['F1 Score']
            results['USGS Precision'] = results['Precision']
            results['USGS Recall'] = results['Recall']

            results_df.loc[len(results_df)] = results

        if feature.type == MapUnitType.POLYGON:
            results, feedback_image = grade_poly_raster(feature_mask, true_mask, feedback_image=feedback_image)
            results['Map'] = map_data.name
            results['Feature'] = feature.label
            if use_usgs_scores:
                usgs_results, _ = usgs_grade_poly_raster(feature_mask, true_mask, map_data.image, map_data.legend, difficult_weight=0.7)
                results['USGS F1 Score'] = usgs_results['F1 Score']
                results['USGS Precision'] = usgs_results['Precision']
                results['USGS Recall'] = usgs_results['Recall']

            results_df.loc[len(results_df)] = results

        # Save feature feedback image
        if feedback_dir:
            os.makedirs(os.path.join(feedback_dir, map_data.name), exist_ok=True)
            feedback_path = os.path.join(feedback_dir, map_data.name, sanitize_filename(f'val_{map_data.name}_{feature.label}_{feature.type}.tif'))
            io.saveGeoTiff(feedback_path, feedback_image, map_data.georef.crs, map_data.georef.transform)
        legend_index += 1

    # Save map scores
    full_csv_path = os.path.join(output_dir, f'#validation_scores.csv')
    results_df.to_csv(full_csv_path, index=False, mode='a', header=not os.path.exists(full_csv_path))
    if feedback_dir:
        os.makedirs(os.path.join(feedback_dir, map_data.name), exist_ok=True)
        csv_path = os.path.join(feedback_dir, map_data.name, f'#{map_data.name}_scores.csv')
        results_df.to_csv(csv_path, index=False)

    # Average validation results for map
    results_df = results_df[results_df['F1 Score'].notna()]
    f1s, pre, rec, iou = results_df["F1 Score"].mean(), results_df["Precision"].mean(), results_df["Recall"].mean(), results_df["IoU Score"].mean()
    uf1, upr, urc = results_df["USGS F1 Score"].mean(), results_df["USGS Precision"].mean(), results_df["USGS Recall"].mean()
    mpt, fpt, upt, dpt = sum(results_df["Matched Points"]), sum(results_df["Missing Points"]), sum(results_df["Unmatched Points"]), results_df["Mean matched distance Points"].mean()
    pipeline_manager.log(logging.DEBUG, f'{map_data.name} - Average validation scores | F1 : {f1s:.2f}, Precision : {pre:.2f}, Recall : {rec:.2f}, IoU : {iou:.2f}', pid=mp.current_process().pid)
    pipeline_manager.log_to_monitor(data_id, {'F1 Score': f'{f1s:.2f}'})

    return 
# endregion Validation

# region Testing
def test_saveGeoPackage(filepath, map_data: CMAAS_Map, coord_type='pixel'):
    import geopandas as gpd
    from shapely.geometry import Polygon, Point, LineString
    from shapely.affinity import affine_transform
    from rasterio.crs import CRS
    
    # Create a GeoDataFrame to store all features
    gdf = gpd.GeoDataFrame()
    
    # Get the crs
    if map_data.georef and map_data.georef.crs:
        crs = map_data.georef.crs
    else:
        crs = CRS.from_epsg(4326)

    # Process each feature in the legend
    for feature in map_data.legend.features:
        if feature.segmentation and feature.segmentation.geometry:
            geometries = feature.segmentation.geometry
            
            # Apply transform
            if map_data.georef and map_data.georef.transform:
                transform = map_data.georef.transform
                affine_params = [transform.a, transform.b, transform.d, transform.e, transform.xoff, transform.yoff]
                geometries = [affine_transform(geom, affine_params) for geom in geometries]
            
            # Create a GeoDataFrame for this feature    
            gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)

            # Save to GeoPackage
            gdf.to_file(filepath, layer=feature.label, driver="GPKG")

from time import sleep
def test_step(data_id, filename):
    raise Exception('Test Error')
    pipeline_manager.log_to_monitor(data_id, {'filename' :os.basename(filename)})
    sleep(1)
    return filename
# endregion Testing
