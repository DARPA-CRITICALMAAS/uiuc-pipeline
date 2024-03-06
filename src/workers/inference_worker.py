import logging
import traceback
import multiprocessing
import numpy as np
from time import sleep, time

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

import src.utils as utils
from src.interprocess_communication import ipq_message_type, ipq_log_message, ipq_work_message

def inference_worker(input_queue, save_queue, validation_queue, log_queue, model):
    pid = multiprocessing.current_process().pid
    log_queue.put(ipq_log_message(pid, ipq_message_type.INFERENCE, logging.DEBUG, None, f'Inference worker starting'))
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
                log_queue.put(ipq_log_message(pid, ipq_message_type.INFERENCE, logging.DEBUG, None, f'Inference worker stopping'))
                break
            map_data = work_message.content
            
            # Perform inference
            log_queue.put(ipq_log_message(pid, ipq_message_type.INFERENCE, logging.DEBUG, map_data.name, f'Started inference on {map_data.name}'))
            map_stime = time()
            map_data = process_map(map_data, model)
            map_time = time() - map_stime
            log_queue.put(ipq_log_message(pid, ipq_message_type.INFERENCE, logging.DEBUG, map_data.name, f'Map processing time for {map_data.name} : {map_time:.2f} seconds'))

            # Put map data on output queue
            save_queue.put(ipq_work_message(map_data))
            validation_queue.put(ipq_work_message(map_data))
            log_queue.put(ipq_log_message(pid, ipq_message_type.INFERENCE, logging.DEBUG, map_data.name, f'Completed inference on {map_data.name}'))
        
        except Exception as e:
            # Note failure and retry up to 3 times
            log_queue.put(ipq_log_message(pid, ipq_message_type.INFERENCE, logging.ERROR, map_data.name, f'Inference worker failed on {map_data.name} on try {work_message.retries} with exception {e}\n{traceback.format_exc()}'))
            work_message.retries += 1
            if work_message.retries < 3:
                input_queue.put(work_message)
            else:
                log_queue.put(ipq_log_message(pid, ipq_message_type.INFERENCE, logging.ERROR, map_data.name, f'MAP {map_data.name} WAS NOT PROCESSED! Could not perform inference after 3 tries skipping map'))
    return True

def process_map(map_data, model):
    # Cutout Legends
    legend_images = {}
    for label, feature in map_data.legend.features.items():
        min_pt, max_pt = utils.boundingBox(feature.contour) # Need this as points order can be reverse or could have quad
        legend_images[feature.name] = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]
    
    # Cutout map portion of image
    if map_data.layout is not None and map_data.layout.map is not None:
        min_pt, max_pt = utils.boundingBox(map_data.layout.map)
        image = map_data.image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]].copy()
    else:
        image = map_data.image

    # Run Model
    infer_start_time = time()
    #results = {list(map_data.legend.features.keys())[0] : np.ones_like(image, dtype=np.uint8)} # Mock prediction
    result_mask = model.inference(image, legend_images)
    #log.info("Execution time for {}: {:.2f} seconds".format(model.name, time() - infer_start_time))

    # Resize cutout to full map
    if map_data.layout is not None and map_data.layout.map is not None:
        result_image = np.zeros((1, *map_data.image.shape[1:]), dtype=np.float32)
        result_image[:,min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]] = result_mask
        map_data.mask = result_image

    return map_data