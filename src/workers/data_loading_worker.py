import os
import logging
import multiprocessing
from time import sleep

import src.cmass_io as io
import src.utils as utils
from src.interprocess_communication import ipq_message_type, ipq_log_message, ipq_work_message, ipq_data_load_message

def get_map_region_size(map_data):
    _, height, width = map_data.image.shape
    if map_data.layout is not None:
        if map_data.layout.map is not None:
            min_xy, max_xy = utils.boundingBox(map_data.layout.map)
            height = max_xy[1] - min_xy[1]
            width = max_xy[0] - min_xy[0]
    return height, width

def data_loading_worker(input_queue, output_queue, log_queue):
    pid = multiprocessing.current_process().pid
    log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_LOADING, logging.DEBUG, None, f'Data loading worker starting'))
    while True:
        try:
            # Wait for work
            if input_queue.empty() or output_queue.full():
                sleep(1)
                continue

            # Retrive work from queue
            work_message = input_queue.get()

            # Check for stop message
            if work_message == 'STOP':
                log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_LOADING, logging.DEBUG, None, f'Data loading worker stopping'))
                break
            
            # Load map data
            map_name = os.path.basename(os.path.splitext(work_message.map_path)[0])
            log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_LOADING, logging.DEBUG, map_name, f'Started Loading {map_name}'))
            map_data = io.loadCMASSMap(work_message.map_path)
            map_data.legend = work_message.legend
            map_data.layout = work_message.layout

            # Put data on output queue
            output_queue.put(ipq_work_message(map_data))
            height, width = get_map_region_size(map_data)
            log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_LOADING, logging.DEBUG, map_name, f'Completed Loading {map_name} with shape {map_data.image.shape} and {len(map_data.legend.features)} map units. Map region = ({height}, {width})'))
        
        except Exception as e:
            # Note failure and retry up to 3 times
            log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_LOADING, logging.ERROR, map_name, f'Data loader failed loading {map_name} on try {work_message.retries} with exception {e}'))
            work_message.retries += 1
            if work_message.retries < 3:
                input_queue.put(work_message)
            else:
                log_queue.put(ipq_log_message(pid, ipq_message_type.DATA_LOADING, logging.ERROR, map_name, f'MAP {map_name} WAS NOT PROCESSED! Could not load data after 3 tries skipping map'))
    return True

    # if map_name in layouts and 'map' in layouts[map_name]:
    #     map_data.map_contour = layouts[map_name]['map']['bounds']
    # if map_name in layouts and 'legend_polygons' in layouts[map_name]:
    #     map_data.legend_contour = layouts[map_name]['legend_polygons']['bounds']