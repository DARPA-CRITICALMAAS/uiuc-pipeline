import os
import logging
import multiprocessing
import threading
import pandas as pd
from rich import box
from rich.live import Live
from rich.table import Table
from time import time, sleep

from src.utils import RichHandler
from src.interprocess_communication import ipq_data_load_message, ipq_message_type
from src.workers import data_loading_worker, inference_worker, data_saving_worker, validation_worker

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

A100_patches_per_sec = 225

class pipeline_manager():
    def __init__(self, args, model, legends, layouts):
        self._running = False
        self._model = model
        self.legends = legends
        self.layouts = layouts
        self.output_dir = args['output']
        self.feedback_dir = args['feedback']
        self.true_segmentation_dir = args['validation']

        # Workers
        self.data_loading_worker_count = 1
        self.inference_worker_count = 1 # Can't run more then one currently.
        self.data_saving_worker_count = 2
        self.validation_worker_count = 2

        # Shared queues
        self._mpm = multiprocessing.Manager()
        self._logging_queue = self._mpm.Queue() # Log messages
        self._data_loading_queue = self._mpm.Queue() # File paths of data to load
        self._inference_queue = self._mpm.Queue(maxsize=4) # Loaded maps ready for inference
        self._data_saving_queue = self._mpm.Queue() # Maps with masks ready to save
        self._validation_queue = self._mpm.Queue() # Maps with masks ready to validate

        # Worker pools
        self._data_loading_workers = []
        self._inference_workers = []
        self._data_saving_workers = []
        self._validation_workers = []

        # Initalize Data queue
        for map_path in args['data']:
            map_name = os.path.basename(os.path.splitext(map_path)[0])
            lay = None
            if map_name in self.layouts:
                lay = self.layouts[map_name]
            self._data_loading_queue.put(ipq_data_load_message(map_path, self.legends[map_name], lay, None))

    def start(self):
        if self._running:
            log.warning('Start was called when Inference pipeline already running. Ignoring call to start')
            return False
        self._running = True

        # Start workers
        log.info(f'Starting worker processes')
        self._data_loading_workers = [multiprocessing.Process(target=data_loading_worker, args=(self._data_loading_queue, self._inference_queue, self._logging_queue))for i in range(self.data_loading_worker_count)]
        self._inference_workers = [threading.Thread(target=inference_worker, args=(self._inference_queue, self._data_saving_queue, self._validation_queue, self._logging_queue, self._model))]
        #self._inference_workers = [multiprocessing.Process(target=ipw_inference, args=(self._inference_queue, self._data_saving_queue, self._validation_queue, self._logging_queue, self._model))for i in range(self.inference_worker_count)]
        self._data_saving_workers = [multiprocessing.Process(target=data_saving_worker, args=(self._data_saving_queue, self._logging_queue, self.output_dir, self.feedback_dir)) for i in range(self.data_saving_worker_count)]
        self._validation_workers = [multiprocessing.Process(target=validation_worker, args=(self._validation_queue, self._logging_queue, self.true_segmentation_dir, self.feedback_dir)) for i in range(self.validation_worker_count)]
        
        [w.start() for w in self._data_loading_workers]
        [w.start() for w in self._inference_workers]
        [w.start() for w in self._data_saving_workers]
        [w.start() for w in self._validation_workers]

        return True
    
    def running(self):
        """Returns True if the Inference pipeline is running. False otherwise."""
        return self._running 

    def file_monitor(self, timeout=1):
        global A100_patches_per_sec
        # Internal progress bar update function
        def update_sub_task(message, active_workers):
            if message[:7] == 'Started':
                active_workers += 1
            if message[:9] == 'Completed':
                active_workers -= 1
            return active_workers
        
        active_workers = 0 
        last_update = time()
        while self._running:
            if active_workers != 0:
                last_update = time()
            if time() - last_update > timeout:
                log.info(f'Inference pipeline has stalled. No updates in the last {timeout} seconds')
                self.stop()
                break

            # Sleep while queue is empty
            if self._logging_queue.empty():
                sleep(0.1)
                continue
            # Retieve other processes messages and pass to file log
            record = self._logging_queue.get()
            log.log(record.level, record.message)

            # Handle Error Messages
            if record.level >= logging.ERROR:
                # TODO
                continue
        
            # Update progress bar with status message
            if record.type == ipq_message_type.DATA_LOADING:
                if record.message.startswith("Completed"):
                    map_units = int(record.message.split(' and ')[1].split(' map units')[0])
                    map_region = record.message.split('Map region = ')[1]
                    shape = map_region.replace('(','').replace(')','').replace(',','').split(' ')
                    shape = [int(i) for i in shape]
                    log.info(f"Finished loading {record.map_name} : map_region={map_region} units={map_units}")
                active_workers = update_sub_task(record.message, active_workers)
            if record.type == ipq_message_type.INFERENCE:
                if record.message.startswith("Started") and map_units and shape:
                    patch_size = 256
                    patch_overlap = 32
                    cols = shape[0] / (patch_size-patch_overlap)
                    rows = shape[1] / (patch_size-patch_overlap)
                    patches = (cols * rows * int(map_units))
                    est_time = patches / A100_patches_per_sec
                    log.info(f"Started processing {record.map_name}, estimated time is {est_time:.2f} seconds")
                    start_time = time()
                if record.message.startswith("Completed"):
                    total_time = time() - start_time
                    A100_patches_per_sec = patches / total_time
                    log.info(f"Finished processing {record.map_name}, time was {total_time:.2f} seconds = {A100_patches_per_sec:.2f} patches/sec")
                active_workers = update_sub_task(record.message, active_workers)
            if record.type == ipq_message_type.DATA_SAVING:
                active_workers = update_sub_task(record.message, active_workers)
            if record.type == ipq_message_type.VALIDATION:
                active_workers = update_sub_task(record.message, active_workers)

    def console_monitor(self, timeout=1):
        def estimate_inference_time(map_region_str, map_units, patch_size=256, patch_overlap=32):
            A100_patches_per_sec = 225
            shape = map_region_str.replace('(','').replace(')','').replace(',','').split(' ')
            shape = [int(i) for i in shape]
            cols = shape[0] / (patch_size-patch_overlap)
            rows = shape[1] / (patch_size-patch_overlap)
            est_time = (cols * rows * int(map_units)) / A100_patches_per_sec
            return est_time

        def generate_monitor_table(df:pd.DataFrame) -> Table:
            _typeToStatus = {
                None : 'Waiting in queue',
                'FINISHED' : 'Done processing',
                'ERROR': 'ERROR',
                ipq_message_type.DATA_LOADING: 'Loading Data',
                ipq_message_type.INFERENCE: 'Performing Inference',
                ipq_message_type.DATA_SAVING: 'Saving Data',
                ipq_message_type.VALIDATION: 'Validating Data',
                'DATA_SAVING AND VALIDATION': 'Saving and Validating Data'
            }

            # Build table structure
            table = Table(title="Inference Pipeline", expand=True)
            table.box = box.MINIMAL
            table.add_column("Map")
            table.add_column("Shape")
            table.add_column("Map Units")
            table.add_column("Status")
            table.add_column("Inference Time | Est Time")

            # Calculate inference time
            for index, row in df.iterrows():
                now = time()
                if row['status'] == ipq_message_type.INFERENCE:
                    df.loc[df['map'] == row['map'], 'inference_time'] += now - row['last_update']
                df.loc[df['map'] == row['map'], 'last_update'] = now

            # Add active maps to table
            for index, row in df.iterrows():
                if row['status'] == 'FINISHED' or row['status'] == 'ERROR':
                    continue
                color = ''
                if row['status'] is not None:
                    color = '[green]'
                if row['estimated_time'] == 'N/A':
                    table.add_row(f'{color}{row["map"]}', f'{color}{row["shape"]}', f'{color}{row["map_units"]}', f'{color}{_typeToStatus[row["status"]]}', f'{row["estimated_time"]}')
                else:
                    table.add_row(f'{color}{row["map"]}', f'{color}{row["shape"]}', f'{color}{row["map_units"]}', f'{color}{_typeToStatus[row["status"]]}', f'{color}{row["inference_time"]:.2f} | {row["estimated_time"]:.2f} secs')
            # Add completed maps at bottom
            for index, row in df.iterrows():
                if row['status'] != 'FINISHED' and row['status'] != 'ERROR':
                    continue
                color = ''
                if row['status'] == 'FINISHED':
                    color = '[bright_black]'
                if row['status'] == 'ERROR':
                    color = '[red]'
                table.add_row(f'{color}{row["map"]}', f'{color}{row["shape"]}', f'{color}{row["map_units"]}', f'{color}{_typeToStatus[row["status"]]}', f'{color}{row["inference_time"]:.2f} secs')
            return table
        
        def parse_map_load_message(message):
            # This is how the message is built
            #f'Completed Loading {map_name} with shape {map_data.image.shape} and {len(map_data.legend.features)} map units. Map region = ({height}, {width})'
            shape = message.split('with shape ')[1].split(' and ')[0]
            map_units = int(message.split(' and ')[1].split(' map units')[0])
            map_region = message.split('Map region = ')[1]
            return shape, map_units, map_region

        # Internal progress bar update function
        def update_df(df, record):
            # Step Started
            if record.message[:7] == 'Started':
                # If an item is not in the table, add it.
                if record.map_name not in df['map'].values:
                    df.loc[len(df)] = {'map': record.map_name, 'shape': '', 'map_units': '', 'status': record.type, 'start_time': time(), 'end_time': None, 'inference_time':0.0, 'last_update': time(), 'estimated_time' : 'N/A'}
                # If an item is already in the table update its status
                else:
                    # # Handle special case when saving and validation are done at the same time
                    # if df.loc[df['map'] == record.map_name, 'status'] == ipq_message_type.DATA_SAVING or df.loc[df['map'] == record.map_name, 'status'] == ipq_message_type.VALIDATION:
                    #     if record.type == ipq_message_type.DATA_SAVING or record.type == ipq_message_type.VALIDATION:
                    #         df.loc[df['map'] == record.map_name, 'status'] = 'DATA_SAVING AND VALIDATION'
                    # else:
                        df.loc[df['map'] == record.map_name, 'status'] = record.type

            # Step Completed
            elif record.message[:9] == 'Completed':
                # Update an item's status in the table
                # Handle special case when saving and validation are done at the same time
                # if df.loc[df['map'] == record.map_name, 'status'] == 'DATA_SAVING AND VALIDATION':
                #     if record.type == ipq_message_type.DATA_SAVING:
                #         df.loc[df['map'] == record.map_name, 'status'] = ipq_message_type.VALIDATION
                #     if record.type == ipq_message_type.VALIDATION:
                #         df.loc[df['map'] == record.map_name, 'status'] = ipq_message_type.DATA_SAVING
                # else:
                df.loc[df['map'] == record.map_name, 'status'] = None
                # Special case for noting when the map is fully processed.
                if record.type == ipq_message_type.DATA_SAVING or record.type == ipq_message_type.VALIDATION:
                    df.loc[df['map'] == record.map_name, 'end_time'] = time()
                    df.loc[df['map'] == record.map_name, 'status'] = 'FINISHED'
                # Special case to update the map data once its loaded
                if record.type == ipq_message_type.DATA_LOADING:
                    shape, map_units, map_region = parse_map_load_message(record.message)
                    df.loc[df['map'] == record.map_name, 'shape'] = shape
                    df.loc[df['map'] == record.map_name, 'map_units'] = map_units
                    df.loc[df['map'] == record.map_name, 'estimated_time'] = estimate_inference_time(map_region, map_units)

            # Error Encountered in Step
            if record.level >= logging.ERROR:
                df.loc[df['map'] == record.map_name, 'status'] = 'ERROR'
                if record.message[:3] == 'MAP': # MAP {map_name} WAS NOT PROCESSED
                    df.loc[df['map'] == record.map_name, 'end_time'] = time()
            return df

        df = pd.DataFrame(columns=['map', 'shape', 'map_units', 'status', 'start_time', 'end_time', 'inference_time', 'last_update', 'estimated_time'])
        with Live(generate_monitor_table(df), refresh_per_second=4) as live:
            table_handler = RichHandler(live)
            console_handler = log.handlers[1] # Change console handler to progress bar
            console_formatter = log.handlers[1].formatter
            console_level = log.handlers[1].level
            log.handlers[1] = table_handler
            log.handlers[1].setFormatter(console_formatter)
            log.handlers[1].setLevel(console_level)
            last_update=time()
            while self._running:
                # Check if there are any statuses that are not finished or errors
                for value in df['status'].unique():
                    if value != 'FINISHED' and value != 'ERROR':
                        last_update= time()
                if time() - last_update > timeout:
                     break
                # Sleep while queue is empty
                if self._logging_queue.empty():
                    live.update(generate_monitor_table(df))
                    sleep(0.25)
                    continue

                # Retieve other processes messages and pass to file log
                record = self._logging_queue.get()
                log.log(record.level, record.message)
                #log.debug(f'{df}')
                
                # Update table with log message information
                df = update_df(df, record)
                live.update(generate_monitor_table(df))

        log.handlers[1] = console_handler # Change back to regular console handler
        log.info(f'Inference pipeline has detected that there are no more maps to process. No updates in the last {timeout} seconds')
        self.stop()

    def stop(self):
        if not self._running:
            log.warning('Stop was called when Inference pipeline already stopped. Ignoring call to stop')
            return False
        log.debug('Stopping Inference pipeline')
        [self._data_loading_queue.put('STOP') for w in self._data_loading_workers]
        [self._inference_queue.put('STOP') for w in self._inference_workers]
        [self._data_saving_queue.put('STOP') for w in self._data_saving_workers]
        [self._validation_queue.put('STOP') for w in self._validation_workers]

        # Clear queues
        # self._data_loading_queue.join()
        # self._inference_queue.join()
        # self._data_saving_queue.join()
        # self._validation_queue.join()

        # Stop workers
        log.debug('Waiting for data loading workers to finish')
        [w.join() for w in self._data_loading_workers]
        log.debug('Waiting for inference workers to finish')
        [w.join() for w in self._inference_workers]
        log.debug('Waiting for data saving workers to finish')
        [w.join() for w in self._data_saving_workers]
        log.debug('Waiting for validation workers to finish')
        [w.join() for w in self._validation_workers]
        log.debug('Finished stopping pipeline')

        self._running = False
        return True