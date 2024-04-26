import logging
import traceback
import numpy as np
import pandas as pd
import multiprocessing as mp

from rich import box
from rich.live import Live
from rich.table import Table
from time import time, sleep

from src.utils import RichHandler, swap_console_handler
from src.pipeline_communication import data_message, worker_status_message, worker_status, parameter_data_stream

log = logging.getLogger('DARPA_CMAAS_PIPELINE')


class pipeline_step():
    def __init__(self, id, func, args, display='', workers=1, max_output_size=4):
        self.id = id
        self.func = func
        self.args = args
        self.name = display # Displayed as status in monitor
        self.workers = workers
        self.max_output_size = max_output_size
        self._output_subscribers = []
        
    def output(self):
        """Returns a stream to the output of this step."""
        output_stream = parameter_data_stream(max_size=self.max_output_size)
        self._output_subscribers.append(output_stream)
        return output_stream

class file_monitor():
    def __init__(self, title='Pipeline Monitor', refesh=0.25, timeout=1):
        self.title = title
        self.timeout = timeout
        self.refesh = refesh
        self.completed_items = []
        
        self._user_col = 2
        self._final_step = None
        self._data_df = pd.DataFrame(columns=['id', 'step', 'processing_time', 'last_update'])
        self._step_name_lookup = {None : 'Waiting in queue', 'FINISHED' : 'Done processing', 'ERROR': 'ERROR'}

    def active(self):
        # Stop pipeline being marked as inactive at startup
        if len(self._data_df) == 0:
            return True
        for entry in self._data_df['step']:
            for status in entry:
                if status not in ['FINISHED', 'ERROR']:
                    return True
        return False

    def add_step(self, id, name):
        self._step_name_lookup[str(id)] = name
        self._final_step = id

    def update_data(self, record:worker_status_message) -> pd.DataFrame:
        df = self._data_df
        # Get entry for existing data  
        if record.data_id in df['id'].values:
            irow = df[df['id'] == record.data_id].index[0]
        else:
            irow = None

        # Don't updated finished items
        # if irow is not None and df.at[irow, 'step'] == ['FINISHED']:
        #     return

        if record.status == worker_status.STARTED_PROCESSING:
            if irow is not None:
                df.at[irow, 'step'].append(str(record.step_id))
                df.at[irow, 'last_update'] = time()
            if record.data_id not in df['id'].values:
                df.loc[len(df)] = {'id': record.data_id, 'step': [str(record.step_id)], 'processing_time': 0.0, 'last_update': time()}
        
        elif record.status == worker_status.COMPLETED_PROCESSING:
            df.at[irow, 'step'].remove(str(record.step_id))
            if record.step_id == self._final_step: # Last step
                df.at[irow, 'step'].append('FINISHED')
                self.completed_items.append(record.data_id)

        elif record.status == worker_status.ERROR:
            df.at[irow, 'step'] = ['ERROR']
        
        elif record.status == worker_status.USER_MESSAGE:
            if type(record.message) == dict:
                for key, value in record.message.items():
                    if key not in df.columns: # If column does not exist, add it
                        df.insert(self._user_col, key, [None for _ in range(len(df))])
                        self._user_col += 1
                    irow = df[df['id'] == record.data_id].index[0]
                    df.at[irow, key] = value

        df.replace(np.nan, '', inplace=True)
        self._data_df = df
    

class console_monitor():
    def __init__(self, title='Pipeline Monitor', timeout=1, refesh=0.25, max_lines=20):
        self.title = title
        self.timeout = timeout
        self.refesh = refesh
        self.max_lines = max_lines
        self.completed_items = []
        
        self._user_col = 2
        self._final_step = None
        self._data_df = pd.DataFrame(columns=['id', 'step', 'processing_time', 'last_update'])
        self._step_name_lookup = {None : 'Waiting in queue', 'FINISHED' : 'Done processing', 'ERROR': 'ERROR'}

    def active(self):
        # Stop pipeline being marked as inactive at startup
        if len(self._data_df) == 0:
            return True
        for entry in self._data_df['step']:
            for status in entry:
                if status not in ['FINISHED', 'ERROR']:
                    return True
        return False

    def add_step(self, id, name):
        self._step_name_lookup[str(id)] = name
        self._final_step = id

    def update_data(self, record:worker_status_message) -> pd.DataFrame:
        df = self._data_df
        # Get entry for existing data  
        if record.data_id in df['id'].values:
            irow = df[df['id'] == record.data_id].index[0]
        else:
            irow = None

        # Don't updated finished items
        # if irow is not None and df.at[irow, 'step'] == ['FINISHED']:
        #     return

        if record.status == worker_status.STARTED_PROCESSING:
            if irow is not None:
                df.at[irow, 'step'].append(str(record.step_id))
                df.at[irow, 'last_update'] = time()
            if record.data_id not in df['id'].values:
                df.loc[len(df)] = {'id': record.data_id, 'step': [str(record.step_id)], 'processing_time': 0.0, 'last_update': time()}
        
        elif record.status == worker_status.COMPLETED_PROCESSING:
            df.at[irow, 'step'].remove(str(record.step_id))
            if record.step_id == self._final_step: # Last step
                df.at[irow, 'step'].append('FINISHED')
                self.completed_items.append(record.data_id)

        elif record.status == worker_status.ERROR:
            df.at[irow, 'step'] = ['ERROR']
        
        elif record.status == worker_status.USER_MESSAGE:
            if type(record.message) == dict:
                for key, value in record.message.items():
                    if key not in df.columns: # If column does not exist, add it
                        df.insert(self._user_col, key, [None for _ in range(len(df))])
                        self._user_col += 1
                    irow = df[df['id'] == record.data_id].index[0]
                    df.at[irow, key] = value

        df.replace(np.nan, '', inplace=True)
        self._data_df = df

    def generate_table(self) -> Table:
        # Bulid table structure
        # Count in progress items
        inprogress_items = 0
        for index, row in self._data_df.iterrows():
            if row['step'] != ['FINISHED'] and 'ERROR' not in row['step'] and len(row['step']) != 0:
                inprogress_items += 1
        title = f'{self.title}\n( {inprogress_items} Processing - {len(self.completed_items)} Completed )'
        table = Table(title=title, expand=True, box=box.MINIMAL)
        table.add_column('ID') # Name first.
        for col in self._data_df.columns: # User supplied columns next
            if col not in ['id', 'step', 'processing_time', 'last_update']:
                table.add_column(col)
        table.add_column('Status')
        table.add_column('Processing Time')

        # Update active item's processing time
        for index, row in self._data_df.iterrows():
            if row['step'] != ['FINISHED'] and 'ERROR' not in row['step'] and len(row['step']) != 0:
                now = time()
                self._data_df.at[index, 'processing_time'] += now - row['last_update']
                self._data_df.at[index, 'last_update'] = now

        # Populate table data
        total_lines = 0
        for index, row in self._data_df.iterrows():
            if total_lines >= self.max_lines:
                break
            if row['step'] == ['FINISHED'] or 'ERROR' in row['step']: # Skip completed items
                continue
            color = ''
            if len(row['step']) != 0:
                color = '[green]'
        
            item_cols = []
            item_cols.append(f'{color}{row["id"]}')
            for col in self._data_df.columns:
                if col not in ['id', 'step', 'processing_time', 'last_update']:
                    item_cols.append(f'{color}{row[col]}')
            item_cols.append(f'{color}{self._step_name_lookup[row["step"][0] if len(row["step"]) > 0 else None]}')
            item_cols.append(f'{color}{row["processing_time"]:.2f} seconds')
            table.add_row(*item_cols)
            total_lines += 1

        # Add completed items at the bottom
        for index, row in self._data_df[::-1].iterrows():
            if total_lines >= self.max_lines:
                break
            if row['step'] != ['FINISHED'] and 'ERROR' not in row['step']: # Skip in progress items
                continue
            color = '[bright_black]'
            if 'ERROR' in row['step']:
                color = '[red]'

            item_cols = []
            item_cols.append(f'{color}{row["id"]}')
            for col in self._data_df.columns:
                if col not in ['id', 'step', 'processing_time', 'last_update']:
                    item_cols.append(f'{color}{row[col]}')
            item_cols.append(f'{color}{self._step_name_lookup[row["step"][0] if len(row["step"]) > 0 else None]}')
            item_cols.append(f'{color}{row["processing_time"]:.2f} seconds')
            table.add_row(*item_cols)
            total_lines += 1

        return table
    
    
def _start_worker(step:pipeline_step, log_stream:mp.Queue, management_stream:mp.Queue):
    def work_ready(args):
        for arg in args:
            if isinstance(arg, parameter_data_stream):
                if arg.empty():
                    return False
        return True
    
    def output_ready(subscribers):
        for subscriber in subscribers:
            if subscriber.full():
                return False
        return True

    # Expose the pipeline log stream to the new process
    global pipeline_log_stream
    pipeline_log_stream = log_stream

    pid = mp.current_process().pid
    msg = worker_status_message(pid, None, None, worker_status.WORKER_STARTING, log_level=logging.DEBUG, message=f'Worker Process {pid} - Starting')
    log_stream.put(msg)
    while True:
        try:
            # Check for stop message
            if not management_stream.empty():
                message = management_stream.get()
                if message == 'STOP':
                    msg = worker_status_message(pid, None, None, worker_status.WORKER_STOPPING, log_level=logging.DEBUG, message=f'Worker Process {pid} - Stopping')
                    log_stream.put(msg)
                    break

            # Wait for work
            if not work_ready(step.args):
                sleep(0.1)
                continue
            if not output_ready(step._output_subscribers):          
                sleep(0.1)
                continue

            # Retrive work from queue
            func_args = []
            arg_data = None
            for arg in step.args:
                if isinstance(arg, parameter_data_stream):
                    arg_data = arg.get(block=False)
                    if arg_data is None:
                        break
                    func_args.append(arg_data.data)
                else:
                    func_args.append(arg)
            if arg_data is None:
                continue

            # Run function
            msg = worker_status_message(pid, step.id, arg_data.id, worker_status.STARTED_PROCESSING, log_level=None, message=f'Process {pid} - Started {step.name} : {arg_data.id}')
            log_stream.put(msg)

            result = step.func(arg_data.id, *func_args)

            # Send data to subscribers
            for subscriber in step._output_subscribers:
                subscriber.append(result, id=arg_data.id)

            msg = worker_status_message(pid, step.id, arg_data.id, worker_status.COMPLETED_PROCESSING, log_level=None, message=f'Process {pid} - Completed {step.name} : {arg_data.id}')
            log_stream.put(msg) 

        except Exception as e:
            # Just Log errors
            msg = worker_status_message(pid, step.id, arg_data.id, worker_status.ERROR, log_level=logging.ERROR, message=f'Process {pid} - Error in step {step.name} on {arg_data.id} : {e}\n{traceback.format_exc()}')
            log_stream.put(msg)

class pipeline_manager():
    def __new__(cls, **kwargs): # Singleton Pattern
        if not hasattr(cls, 'instance'):
            cls.instance = super(pipeline_manager, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, max_processes=mp.cpu_count(), monitor=console_monitor()):
        self.steps = []
        self.step_dict = {}
        self.max_processes = max_processes
        self._workers = []
        self._running = False
        self._monitor = monitor

        mpm = mp.Manager()
        self._log_stream = mpm.Queue()
        self._management_stream = mpm.Queue()
        self.error_stream = mpm.Queue() # This is purely for provided a way for users to get error 

    def next_step_id(self):
        return len(self.steps)
    
    def add_step(self, func, args, display='', workers=1, max_output_size=4):
        if self._running:
            log.error('Cannot add step while running pipeline. Please stop pipeline before adding steps')
            return
        sid = self.next_step_id()
        step = pipeline_step(sid, func, args, display, workers, max_output_size)
        self.steps.append(step)
        self.step_dict[display] = sid
        self._monitor.add_step(sid, display)
        return step

    def __getitem__(self, key):
        return self.steps[self.step_dict[key]]
    
    def running(self):
        """Returns True if the pipeline is running."""
        return self._running 

    def _create_worker(self, step):
        w = mp.Process(target=_start_worker, args=(step, self._log_stream, self._management_stream))
        w.start()
        self._workers.append(w)
        log.debug(f'Created worker for step {step.name} with pid {w.pid}')
    
    def start(self):
        if self._running:
            log.warning('Start was called when Inference pipeline already running. Ignoring call to start')
            return False
        
        for step in self.steps:
            for i in range(step.workers):
                self._create_worker(step)
        self._running = True
        log.info(f'Starting pipeline manager with {len(self.steps)} steps and {len(self._workers)} worker processes')
        return True
    
    def stop(self):
        if not self._running:
            log.warning('Stop was called when Inference pipeline already stopped. Ignoring call to stop')
            return False

        log.info('Stopping pipeline')
        for w in range(len(self._workers)*2):
            self._management_stream.put('STOP')

        # Wait for workers to stop
        log.debug('Waiting for pipeline workers to finish')
        [w.join() for w in self._workers]
        self._running = False
        return True

    def set_inactivity_timeout(self, timeout):
        self._monitor.timeout = timeout

    def monitor(self):
        _monitor = self._monitor

        # Console monitor
        if isinstance(_monitor, console_monitor):
            with Live(_monitor.generate_table(), refresh_per_second=(1/_monitor.refesh)) as live:
                logging_handler = swap_console_handler(log, RichHandler(live))
                last_activity = time()
                while self._running:
                    # Check if there is any in progress maps
                    if _monitor.active():
                        last_activity = time()
                    if time() - last_activity > _monitor.timeout:
                        break

                    # Sleep while no new messages are available
                    if self._log_stream.empty():
                        live.update(_monitor.generate_table())
                        # log.warning(_monitor._data_df)
                        sleep(_monitor.refesh)
                        continue

                    # Retieve worker messages
                    record = self._log_stream.get()
                    if record.log_level is not None and record.message is not None:
                        log.log(record.log_level, record.message)

                    # Pass onto user error stream
                    if record.log_level == worker_status.ERROR and record.message is not None:
                        if not self._error_stream.full(): # if a user is not pulling them out, just ignore them so that it doesn't block. 
                            self._error_stream.put(record)
                    
                    # Update monitor table
                    _monitor.update_data(record)
                    live.update(_monitor.generate_table())
                swap_console_handler(log, logging_handler)
        # File monitor
        elif isinstance(_monitor, file_monitor):
            last_activity = time()
            while self._running:
                # Check if there is any in progress maps
                if _monitor.active():
                    last_activity = time()
                if time() - last_activity > _monitor.timeout:
                    break

                # Sleep while no new messages are available
                if self._log_stream.empty():
                    sleep(_monitor.refesh)
                    continue

                # Retieve worker messages
                record = self._log_stream.get()
                if record.log_level is not None and record.message is not None:
                    log.log(record.log_level, record.message)

                # Pass onto user error stream
                if record.log_level == worker_status.ERROR and record.message is not None:
                    if not self._error_stream.full(): # if a user is not pulling them out, just ignore them so that it doesn't block. 
                        self._error_stream.put(record)
                
                # Update monitor table
                _monitor.update_data(record)
        # Shouldn't ever happen
        else:
            msg = 'No monitor attached to pipeline manager. Cannot start monitor'
            raise Exception(msg)

        # Stop pipeline when done
        log.info(f'Pipeline Manager has detected that there are no more maps to process. No updates in the last {_monitor.timeout} seconds')
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Send stop message to workers
            executor.submit(self.stop)
            # Keep flushing process logging statements till pipeline fully stops
            while self._running:
                if self._log_stream.empty():
                    sleep(_monitor.refesh)
                    continue
                # Retieve worker messages
                record = self._log_stream.get()
                if record.log_level is not None and record.message is not None:
                    log.log(record.log_level, record.message)

        # Final flush
        while not self._log_stream.empty():
            record = self._log_stream.get()
            if record.log_level is not None and record.message is not None:
                log.log(record.log_level, record.message)


    def log_to_monitor(data_id, dict):
        # Put message into pipeline log stream
        # This global variable gets set when the worker is started.
        # If you find a better way to do this please fix But this is the only way i could figure out to make the api look nice.
        # so that a user just has to call pipeline_manager.log_to_monitor in their own functions
        pipeline_log_stream.put(worker_status_message(None, None, data_id, worker_status.USER_MESSAGE, log_level=None, message=dict))

    def log(level, message, pid=None, step_id=None, item_id=None):
        # Put a message into the pipeline log stream
        if pid is not None:
            message = f'P-{pid} - ' + message
        pipeline_log_stream.put(worker_status_message(pid, step_id, item_id, worker_status.USER_MESSAGE, log_level=level, message=message))
                    