import logging
import numpy as np
import cmaas_utils.io as io
import cmaas_utils.cdr as cdr
from cmaas_utils.types import CMAAS_Map, MapUnitType, Provenance
from src.utils import boundingBox
from src.pipeline_manager import pipeline_manager

def load_data(data_id, image_path, legend_dir=None, layout_dir=None):
    """Wrapper with a custom display for the monitor"""
    map_name = os.path.splitext(os.path.basename(image_path))[0]
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

def gen_layout(data_id, map_data):
    # Generate layout if not precomputed
    if map_data.layout is None:
        pipeline_manager.log(logging.WARNING, f'No layout found for {map_data.name}, skipping as generating layout in pipeline not implemented yet')
        # TODO Implement layout generation
        pass
    return map_data

def gen_legend(data_id, map_data, drab_volcano_legend=False):
    from submodules.legend_extraction.src.extraction import extractLegends
    def convertLegendtoCMASS(legend):
        from cmaas_utils.types import Legend, MapUnit
        features = {}
        for feature in legend:
            features[feature['label']] = MapUnit(label=feature['label'], contour=feature['points'], contour_type='rectangle')
        return Legend(features=features, origin='UIUC Heuristic Model')

    if drab_volcano_legend:
        map_data.legend = io.loadLegendJson('src/models/drab_volcano_legend.json')
    else:
        # Generate legend if not precomputed
        if map_data.legend is None:
            pipeline_manager.log(logging.DEBUG, f'No legend found for {map_data.name}, generating legend')
            lgd = extractLegends(map_data.image.transpose(1,2,0))
            map_data.legend = convertLegendtoCMASS(lgd)

    pipeline_manager.log_to_monitor(data_id, {'Map Units': len(map_data.legend.features)})
    pipeline_manager.log(logging.DEBUG, f'{len(map_data.legend.features)} map units for {map_data.name}')
    return map_data

def save_legend(data_id, map_data: CMAAS_Map, feedback_dir: str, legend_feedback_mode: str = 'single_image'):
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
            legend_save_path = os.path.join(feedback_dir, map_data.name, 'lgd_' + map_data.name + '_' + feature.label + '.tif')
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
            legend_save_path = os.path.join(feedback_dir, map_data.name, map_data.name + '_labels'  + '.png')
            fig.savefig(legend_save_path)
            plt.close(fig)

def segmentation_inference(data_id, map_data:CMAAS_Map, model):
    # Cutout Legends
    legend_images = {}
    for feature in map_data.legend.features:
        if feature.type == model.feature_type:
            min_pt, max_pt = boundingBox(feature.bounding_box) # Need this as points order can be reverse or could have quad
            legend_images[feature.label] = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]
        else:
            pipeline_manager.log(logging.DEBUG, f'Skipping inference of {feature.label} as it is not a {model.feature_type.name} feature')
    
    # Update Map Units with how many are actually being processed
    pipeline_manager.log_to_monitor(data_id, {'Map Units': f'{len(map_data.legend.features)} ({len(legend_images)} {model.feature_type.to_str().capitalize()}s)'})

    # Cutout map portion of image
    if map_data.layout is not None and map_data.layout.map is not None:
        min_pt, max_pt = boundingBox(map_data.layout.map)
        image = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]].copy()
    else:
        image = map_data.image

    result_mask = model.inference(image, legend_images, data_id=data_id)

    # Resize cutout to full map
    if map_data.layout is not None and map_data.layout.map is not None:
        result_image = np.zeros((1, *map_data.image.shape[1:]), dtype=np.float32)
        result_image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]] = result_mask
        result_mask = result_image
    
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
    pipeline_manager.log_to_monitor(data_id, {'Segments': total_occurances})
    pipeline_manager.log(logging.DEBUG, f'Generated {total_occurances} polygons for {map_data.name}')
    return map_data

import os
from math import ceil, floor
import matplotlib.pyplot as plt
def save_output(data_id, map_data: CMAAS_Map, output_dir, feedback_dir):
    # Save CDR schema
    cdr_schema = cdr.exportMapToCDR(map_data)
    cdr_filename = os.path.join(output_dir, f'{map_data.name}_cdr.json')
    io.saveCDRFeatureResults(cdr_filename, cdr_schema)

    # Save GeoPackage
    gpkg_filename = os.path.join(output_dir, f'{map_data.name}.gpkg')
    coord_type = 'pixel'
    if map_data.georef is not None:
        if map_data.georef.crs is not None and map_data.georef.transform is not None:
            coord_type = 'georeferenced'
    
    #saveGeoPackage(gpkg_filename, map_data, coord_type)
    io.saveGeoPackage(gpkg_filename, map_data, coord_type)

    # Save Raster masks
    # legend_index = 1
    # for feature in map_data.legend.features:
    #     feature_mask = np.zeros_like(map_data.poly_segmentation_mask, dtype=np.uint8)
    #     feature_mask[map_data.poly_segmentation_mask == legend_index] = 1
    #     filepath = os.path.join(output_dir, f'{map_data.name}_{feature.label}.tif')
    #     io.saveGeoTiff(filepath, feature_mask, map_data.georef.crs, map_data.georef.transform)
    #     legend_index += 1
    return

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
            pipeline_manager.log(logging.WARNING, f'Can\'t validate feature {feature.label}. No true segmentation mask found at {true_mask_path}.')
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
            feedback_path = os.path.join(feedback_dir, f'val_{map_data.name}_{feature.label}.tif')
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
    pipeline_manager.log(logging.DEBUG, f"""
        Average scores for {map_data.name} | F1 : {f1s:.2f}, Precision : {pre:.2f}, Recall : {rec:.2f}, IoU : {iou:.2f}
        USGS F1 Score : {uf1:.2f}, USGS Precision : {upr:.2f}, USGS Recall : {urc:.2f}, Matched pts : {mpt},
        Missing pts : {fpt}, Unmatched pts : {upt}, Mean matched distance : {dpt:.2f}""")
    pipeline_manager.log_to_monitor(data_id, {'F1 Score': f'{f1s:.2f}'})

    return 