import os
import logging
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time, sleep
from math import ceil, floor

import src.cmass_io as io
import src.utils as utils

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

from submodules.validation.src.grading import grade_poly_raster, usgs_grade_poly_raster, usgs_grade_pt_raster

def single_gpu_inference(args, map_files, legends, layouts, model):
    # Perform Inference
    log.info(f'Starting Inference run of {len(map_files)} maps')
    # Create shared queues
    m = multiprocessing.Manager()
    file_queue = m.Queue() # map file paths
    map_queue = m.Queue(maxsize=4) # Return map sturcts
    save_queue = m.Queue()  # map structs with masks 
    validate_queue = m.Queue() # map structs with masks
    save_output_queue = m.Queue() # Just names of completed maps
    validation_output_queue = m.Queue() # Csvs so we can cat at end
    
    # Start workers
    log.info(f'Starting worker processes')
    for f in map_files:
        file_queue.put(f)
    map_worker_count = 2
    map_workers = [multiprocessing.Process(target=data_loader_worker, args=(file_queue, map_queue, legends, layouts))for i in range(map_worker_count)]
    [w.start() for w in map_workers]
    save_worker_count = 4
    save_workers = [multiprocessing.Process(target=saving_worker, args=(save_queue, save_output_queue, args.output, args.feedback)) for i in range(save_worker_count)]
    [w.start() for w in save_workers]
    valid_worker_count = 4
    valid_workers = [multiprocessing.Process(target=validation_worker, args=(validate_queue, validation_output_queue, args.validation, args.feedback)) for i in range(valid_worker_count)]
    [w.start() for w in valid_workers]

    pbar = tqdm(total=len(map_files))
    while True:
        if map_queue.empty():
            sleep(1)

        map_data = map_queue.get()
        if map_data == 'STOP':
            pbar.close()
            break

        log.info(f'Performing inference on {map_data.name}')
        pbar.update(1)
        pbar.set_description(f'Performing inference on {map_data.name}')
        pbar.refresh()

        # Perform inference
        map_stime = time()
        map_data = process_map(map_data, model)
        map_time = time() - map_stime
        log.info(f'Map processing time for {map_data.name} : {map_time:.2f} seconds')

        save_queue.put(map_data)
        validate_queue.put(map_data)
        
    # When all queues are empty can end
    results_df = pd.DataFrame()
    log.info('Waiting for workers to finish')
    while not save_queue.empty() or not validate_queue.empty() or not save_output_queue.empty() or not validation_output_queue.empty():
        if not save_output_queue.empty():
            save_output_queue.get()
        if not validation_output_queue.empty():
            results_df = pd.concat([results_df, validation_output_queue.get()])

    [save_queue.put('STOP') for i in range(save_worker_count)]
    [validate_queue.put('STOP') for i in range(valid_worker_count)]

    for w in map_workers:
        w.join()
    for w in save_workers:
        w.join()
    for w in valid_workers:
        w.join()
    log.info('Finishing')

    # TODO
    # Track remaining and completed maps        

def process_map(map_data, model):
    # Cutout Legends
    legend_images = {}
    for label, feature in map_data.legend.features.items():
        min_pt, max_pt = utils.boundingBox(feature.contour) # Need this as points order can be reverse or could have quad
        legend_images[feature.name] = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]
    
    # Cutout map portion of image
    if map_data.map_contour is not None:
        min_pt, max_pt = utils.boundingBox(map_data.map_contour)
        image = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]].copy()
    else:
        image = map_data.image

    # Run Model
    infer_start_time = time()
    #results = {list(map_data.legend.features.keys())[0] : np.ones_like(image, dtype=np.uint8)} # Mock prediction
    results = model.inference(image, legend_images, batch_size=256)
    log.info("Execution time for {}: {:.2f} seconds".format(model.name, time() - infer_start_time))

    # Resize cutout to full map
    if map_data.map_contour is not None:
        for feature, feature_mask in results.items():
            feature_image = np.zeros((1, *map_data.image.shape[1:]), dtype=np.uint8)
            feature_image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]] = feature_mask
            results[feature] = feature_image
    
    # Save mask to map_data
    for label, feature_mask in results.items():
        map_data.legend.features[label].mask = feature_mask

    return map_data

def multi_process_inference():
    #TODO: Implement multi-process inference
    raise NotImplementedError        

def data_loader_worker(file_queue, output_queue, legends, layouts, max_queued_maps=4):
    log.debug(f'{multiprocessing.current_process().name} Data loader starting')
    try:
        while True:
            if output_queue.qsize() < max_queued_maps:
                map_path = file_queue.get(0)
                map_name = os.path.basename(os.path.splitext(map_path)[0])
                map_data = io.loadCMASSMap(map_path)
                map_data.legend = legends[map_name]
                if map_name in layouts and 'map' in layouts[map_name]:
                    map_data.map_contour = layouts[map_name]['map']['bounds']
                if map_name in layouts and 'legend_polygons' in layouts[map_name]:
                    map_data.legend_contour = layouts[map_name]['legend_polygons']['bounds']
                output_queue.put(map_data)
                log.debug(f'{multiprocessing.current_process().name} Loaded {map_name}')
            else:
                sleep(1)
            if file_queue.empty():
                sleep(10)
                output_queue.put('STOP')
                log.debug(f'{multiprocessing.current_process().name} Data loader stopping')
                break
        return True
    except Exception as e:
        log.error(f'{multiprocessing.current_process().name} Data loader failed on {map_name}')
        log.exception(e)
        return False

def saving_worker(work_queue, out_queue, output_dir, feedback_dir):
    log.debug(f'{multiprocessing.current_process().name} Save worker starting')
    try:
    
        legend_feedback_mode = 'single_image'
        while True:
            if work_queue.empty():
                sleep(1)

            map_data = work_queue.get()
            if map_data == 'STOP':
                log.debug(f'{multiprocessing.current_process().name} Save worker stopping')
                break
            # Save Legend preview
            legend_images = {}
            for label, feature in map_data.legend.features.items():
                min_pt, max_pt = utils.boundingBox(feature.contour) # Need this as points order can be reverse or could have quad
                legend_images[feature.name] = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]
                if feedback_dir:
                    os.makedirs(os.path.join(feedback_dir, map_data.name), exist_ok=True)
                    if legend_feedback_mode == 'individual_images':
                        legend_save_path = os.path.join(feedback_dir, map_data.name, 'lgd_' + map_data.name + '_' + feature.name + '.tif')
                        io.saveGeoTiff(legend_save_path, legend_images[feature.name], None, None)
            if feedback_dir and len(legend_images) > 0:
                os.makedirs(os.path.join(feedback_dir, map_data.name), exist_ok=True)
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
            for label, feature in map_data.legend.features.items():
                if feature.mask is not None:
                    filepath = os.path.join(output_dir, f'{map_data.name}_{feature.name}.tif')
                    io.saveGeoTiff(filepath, feature.mask, map_data.crs, map_data.transform)
            log.debug(f'{multiprocessing.current_process().name} Saved {map_data.name}')
            out_queue.put(map_data.name)

        return True
    except Exception as e:
        log.error(f'{multiprocessing.current_process().name} Save worker failed on {map_data.name}')
        log.exception(e)
        return False

def validation_worker(work_queue, out_queue, true_dir, feedback_dir):
    try:
        log.debug(f'{multiprocessing.current_process().name} Validation worker starting')
        while True:
            if work_queue.empty():
                sleep(1)

            map_data = work_queue.get()
            if map_data == 'STOP':
                log.debug(f'{multiprocessing.current_process().name} Validation worker stopping')
                break

            results = validate_map(map_data, true_dir, feedback=feedback_dir)
            log.debug(f'{multiprocessing.current_process().name} Validated {map_data.name}')
            out_queue.put(results)
        return True
    except Exception as e:
        log.error(f'{multiprocessing.current_process().name} Validation worker failed on {map_data.name}')
        log.exception(e)
        return False

def validate_map(map_data, true_path, usgs_scores=False, feedback=None):
    val_stime = time()

    results_df = pd.DataFrame(columns = ['Map', 'Feature', 'F1 Score', 'Precision', 'Recall', 'IoU Score (polys)',
                                         'USGS F1 Score (polys)', 'USGS Precision (polys)', 'USGS Recall (polys)', 
                                         'Mean matched distance (pts)', 'Matched (pts)', 'Missing (pts)', 
                                         'Unmatched (pts)'])
    for label, feature in map_data.legend.features.items():
        if feature.mask is None:
            log.warning(f'No mask found for {feature.name}. Skipping validation of feature')
            results_df.loc[len(results_df)] = {'Map' : map_data.name, 'Feature' : feature.name, 'F1 Score' : np.nan,
                                                  'Precision' : np.nan, 'Recall' : np.nan}
            continue
        true_mask_path = os.path.join(true_path, f'{map_data.name}_{feature.name}.tif')
        # Skip features that don't have a true mask available
        if not os.path.exists(true_mask_path):
            log.warning(f'No true segementation map found for {feature.name}. Skipping validation of feature')
            results_df[len(results_df)] = {'Map' : map_data.name, 'Feature' : feature.name, 'F1 Score' : np.nan,
                                           'Precision' : np.nan, 'Recall' : np.nan}
            continue

        true_mask, _, _ = io.loadGeoTiff(true_mask_path)

        # Create feedback image if needed
        feedback_image = None
        if feedback:
            feedback_image = np.zeros((3, *feature.mask.shape[1:]), dtype=np.uint8)

        # Grade image
        if feature.type == 'pt':
            feature_score = usgs_grade_pt_raster(feature.mask, true_mask, feedback_image=feedback_image)
            results_df.loc[len(results_df)] = {'Map' : map_data.name,
                                               'Feature' : feature.name, 
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

        if feature.type == 'poly':
            feature_score = grade_poly_raster(feature.mask, true_mask, feedback_image=feedback_image)
            usgs_score = (np.nan, np.nan, np.nan, np.nan, None)
            if usgs_scores:
                usgs_score = usgs_grade_poly_raster(feature.mask, true_mask, map_data.image, map_data.legend, difficult_weight=0.7)
                feature_score = {**feature_score, **usgs_score}
            results_df.loc[len(results_df)] = {'Map' : map_data.name,
                                            'Feature' : feature.name, 
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
        if feedback:
            feedback_path = os.path.join(feedback, f'val_{map_data.name}_{feature.name}.tif')
            io.saveGeoTiff(feedback_path, feedback_image, map_data.crs, map_data.transform)

     # Average validation results for map
    f1s, pre, rec, iou = results_df["F1 Score"].mean(), results_df["Precision"].mean(), results_df["Recall"].mean(), results_df["IoU Score (polys)"].mean()
    uf1, upr, urc = results_df["USGS F1 Score (polys)"].mean(), results_df["USGS Precision (polys)"].mean(), results_df["USGS Recall (polys)"].mean()
    mpt, fpt, upt, dpt = sum(results_df["Matched (pts)"]), sum(results_df["Missing (pts)"]), sum(results_df["Unmatched (pts)"]), results_df["Mean matched distance (pts)"].mean()
    log.info(f'{map_data.name} average scores |\n' +
             f'F1 : {f1s:.2f}, Precision : {pre:.2f}, Recall : {rec:.2f} IoU : {iou:.2f}\n' +
             f'USGS F1 Score : {uf1:.2f}, USGS Precision : {upr:.2f}, USGS Recall : {urc:.2f}\n' +
             f'Matched pts : {mpt}, Missing pts : {fpt}, Unmatched pts : {upt}, Mean matched distance : {dpt:.2f}')

    # Save map scores
    if feedback:
        os.makedirs(os.path.join(feedback, map_data.name), exist_ok=True)
        csv_path = os.path.join(feedback, map_data.name, f'#{map_data.name}_scores.csv')
        results_df.to_csv(csv_path, index=False)

    val_time = time() - val_stime
    log.debug(f'Time to validate {map_data.name} : {val_time:.2f} seconds')
    return results_df
        
    
