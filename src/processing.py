import os

import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

import src.cmass_io as io
import multiprocessing
from time import time

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

from submodules.validation.src.grading import grade_poly_raster, usgs_grade_poly_raster, usgs_grade_pt_raster

def single_process_inference(map_files, legends, layouts, model):
    # Perform Inference
    log.info(f'Starting Inference run of {len(map_files)} maps')
    pbar = tqdm(total=len(map_files))
    for map_path in pbar:
        map_name = os.path.basename(os.path.splitext(map_path)[0])
        log.info(f'Performing inference on {map_name}')
        pbar.set_description(f'Performing inference on {map_name}')
        pbar.refresh()

        map_request_queue = [] # map paths to be loaded
        map_queue = [] # Return map sturcts
        save_queue = [] # map structs with masks 
        save_output_queue = [] # Just names of completed maps
        validate_queue = [] # map structs with masks 
        validation_output_queue = [] # Csvs so we can cat at end
        #with ThreadPoolExecutor() as executor:
            # TODO
            # start a map_loader worker

            # TODO
            # start a single inference process

            # TODO
            # start a save worker

            # TODO
            # start some validation workers

            # TODO
            # When all queues are empty can end

        # TODO
        # Track remaining and completed maps        

def multi_process_inference():
    #TODO: Implement multi-process inference
    raise NotImplementedError        

class map_load_worker():
    def __init__(self, work_queue, output_queue, legends, layouts):
        self.work_queue = work_queue
        self.output_queue = output_queue
        self.legends = legends
        self.layouts = layouts
        self.running = False
    def run(self):
        self.running = True
        while self.running:
            if not self.work_queue.empty():
                map_path = self.work_queue.pop(0)
                map_name = os.path.basename(os.path.splitext(map_path)[0])
                log.debug(f'{multiprocessing.current_process().name} Loading {map_name}')
                map_data = io.loadCMASSMap(map_path)
                map_data.legend = self.legends[map_name]
                if map_name in self.layouts and 'map' in self.layouts[map_name]:
                    map_data.map_contour = self.layouts[map_name]['map']['bounds']
                if map_name in self.legends and 'legend_polygons' in self.legends[map_name]:
                    map_data.legend_contour = self.legends[map_name]['legend_polygons']['bounds']
                self.output_queue.put(map_data)
            else:
                time.sleep(1)
    def stop(self):
        self.running = False

class save_worker():
    def __init__(self, work_queue, output_dir):
        self.work_queue = work_queue
        self.outputDir = output_dir
        self.running = False
    
    def run(self):
        self.running = True
        while self.running:
            if not self.work_queue.empty():
                map_data = self.work_queue.pop(0)
                log.debug(f'{multiprocessing.current_process().name} Saving {map_data.name}')
                for feature in map_data.legend.features:
                    filepath = os.path.join(self.outputDir, f'{map_data.name}_{feature.name}.tif')
                    io.saveGeoTiff(filepath, feature.mask, map_data.crs, map_data.transform)
                log.debug(f'{multiprocessing.current_process().name} finished saving {map_data.name}')
            else:
                time.sleep(1)
    def stop(self):
        self.running = False

class validate_worker():
    def __init__(self, work_queue, true_dir, feedback_dir):
        self.work_queue = work_queue
        self.true_dir = true_dir
        self.feedback_dir = feedback_dir
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            if not self.work_queue.empty():
                map_data = self.work_queue.pop(0)
                log.debug(f'{multiprocessing.current_process().name} Validating {map_data.name}')
                validate_map(map_data, self.true_dir, feedback=self.feedback_dir)
            else:
                time.sleep(1)

def validate_map(map_data, true_path, usgs_scores=False, feedback=None):
    val_stime = time()

    results_df = pd.DataFrame(columns = ['Map', 'Feature', 'F1 Score', 'Precision', 'Recall', 'IoU Score (polys)',
                                         'USGS F1 Score (polys)', 'USGS Precision (polys)', 'USGS Recall (polys)', 
                                         'Mean matched distance (pts)', 'Matched (pts)', 'Missing (pts)', 
                                         'Unmatched (pts)'])
    for feature in map_data.legend.features:
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
            feedback_image = np.zeros((*feature.mask.shape[:2], 3), dtype=np.uint8)

        # Grade image
        if feature.type == 'pt':
            feature_score = usgs_grade_pt_raster(feature.mask, true_mask, feedback=feedback_image)
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
            feature_score = grade_poly_raster(feature.mask, true_mask, feedback=feedback_image)
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
    log.debug(f'Time to validate {map.name} : {val_time:.2f} seconds')
        
    
