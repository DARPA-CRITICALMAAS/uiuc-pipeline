import os
import logging
import traceback
import multiprocessing
import numpy as np
import pandas as pd
from time import time, sleep

import src.cmass_io as io
from src.interprocess_communication import ipq_message_type, ipq_log_message, ipq_work_message
from submodules.validation.src.grading import grade_poly_raster, usgs_grade_poly_raster, usgs_grade_pt_raster   

def validation_worker(input_queue, log_queue, true_dir, feedback_dir):
    pid = multiprocessing.current_process().pid
    log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.DEBUG, None, f'Validation worker starting'))
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
                log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.DEBUG, None, f'Validation worker stopping'))
                break
            map_data = work_message.content

            # Perform validation
            log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.DEBUG, map_data.name, f'Started validation on {map_data.name}'))
            validate_map(map_data, true_dir, log_queue, feedback=feedback_dir)
            log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.DEBUG, map_data.name, f'Completed validation on {map_data.name}'))
        
        except Exception as e:
            # Note failure and retry up to 3 times
            log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.ERROR, map_data.name, f'Error occured on validation worker'))
            log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.ERROR, map_data.name, f'Validation worker failed on {map_data.name} on try {work_message.retries} with exception {e}\n{traceback.format_exc()}'))
            work_message.retries += 1
            if work_message.retries < 3:
                input_queue.put(work_message)
            else:
                log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.ERROR, map_data.name, f'MAP {map_data.name} WAS NOT PROCESSED! Could not perform validation after 3 tries skipping map'))

    return True

def validate_map(map_data, true_path, log_queue, usgs_scores=False, feedback=None):
    pid = multiprocessing.current_process().pid
    val_stime = time()

    results_df = pd.DataFrame(columns = ['Map', 'Feature', 'F1 Score', 'Precision', 'Recall', 'IoU Score (polys)',
                                         'USGS F1 Score (polys)', 'USGS Precision (polys)', 'USGS Recall (polys)', 
                                         'Mean matched distance (pts)', 'Matched (pts)', 'Missing (pts)', 
                                         'Unmatched (pts)'])
    
    legend_index = 1
    for label, feature in map_data.legend.features.items():
        #if feature.mask is None:
        #    log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.WARNING, map_data.name, f'No mask found for {feature.name}. Skipping validation of feature'))
        #    results_df.loc[len(results_df)] = {'Map' : map_data.name, 'Feature' : feature.name, 'F1 Score' : np.nan,
        #                                          'Precision' : np.nan, 'Recall' : np.nan}
        #    continue
        true_mask_path = os.path.join(true_path, f'{map_data.name}_{feature.name}.tif')
        # Skip features that don't have a true mask available
        if not os.path.exists(true_mask_path):
            log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.WARNING, map_data.name, f'No true segementation map found for {feature.name}. Skipping validation of feature'))
            results_df[len(results_df)] = {'Map' : map_data.name, 'Feature' : feature.name, 'F1 Score' : np.nan,
                                           'Precision' : np.nan, 'Recall' : np.nan}
            continue


        feature_mask = np.zeros_like(map_data.mask, dtype=np.uint8)
        feature_mask[map_data.mask == legend_index] = 1
        true_mask, _, _ = io.loadGeoTiff(true_mask_path)

        # Create feedback image if needed
        feedback_image = None
        if feedback:
            feedback_image = np.zeros((3, *feature_mask.shape[1:]), dtype=np.uint8)

        # Grade image
        if feature.type == 'pt':
            feature_score = usgs_grade_pt_raster(feature_mask, true_mask, feedback_image=feedback_image)
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
            feature_score = grade_poly_raster(feature_mask, true_mask, feedback_image=feedback_image)
            usgs_score = (np.nan, np.nan, np.nan, np.nan, None)
            if usgs_scores:
                usgs_score = usgs_grade_poly_raster(feature_mask, true_mask, map_data.image, map_data.legend, difficult_weight=0.7)
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
            io.saveGeoTiff(feedback_path, feedback_image, map_data.georef.crs, map_data.georef.transform)
        legend_index += 1

     # Average validation results for map
    f1s, pre, rec, iou = results_df["F1 Score"].mean(), results_df["Precision"].mean(), results_df["Recall"].mean(), results_df["IoU Score (polys)"].mean()
    uf1, upr, urc = results_df["USGS F1 Score (polys)"].mean(), results_df["USGS Precision (polys)"].mean(), results_df["USGS Recall (polys)"].mean()
    mpt, fpt, upt, dpt = sum(results_df["Matched (pts)"]), sum(results_df["Missing (pts)"]), sum(results_df["Unmatched (pts)"]), results_df["Mean matched distance (pts)"].mean()
    log_queue.put(ipq_log_message(pid, ipq_message_type.VALIDATION, logging.WARNING, map_data.name, f'{map_data.name} average scores | ' +
             f'F1 : {f1s:.2f}, Precision : {pre:.2f}, Recall : {rec:.2f} IoU : {iou:.2f} ' +
             f'USGS F1 Score : {uf1:.2f}, USGS Precision : {upr:.2f}, USGS Recall : {urc:.2f} ' +
             f'Matched pts : {mpt}, Missing pts : {fpt}, Unmatched pts : {upt}, Mean matched distance : {dpt:.2f}'))

    # Save map scores
    if feedback:
        os.makedirs(os.path.join(feedback, map_data.name), exist_ok=True)
        csv_path = os.path.join(feedback, map_data.name, f'#{map_data.name}_scores.csv')
        results_df.to_csv(csv_path, index=False)

    val_time = time() - val_stime
    #log.debug(f'Time to validate {map_data.name} : {val_time:.2f} seconds')
    return results_df