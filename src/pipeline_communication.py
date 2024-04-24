import logging
import multiprocessing
from enum import Enum
from typing import List

class worker_status(Enum):
    WORKER_STARTING = 1
    WORKER_STOPPING = 2
    STARTED_PROCESSING = 3
    COMPLETED_PROCESSING = 4
    ERROR = 5
    LOG_MESSAGE = 6
    USER_MESSAGE = 7

class worker_status_message():
    def __init__(self, process_id:int, step_id:int, data_id:int, status:worker_status, log_level:int=logging.DEBUG, message:str=None):
        self.pid = process_id
        self.step_id = step_id
        self.data_id = data_id
        self.status = status
        self.log_level = log_level
        self.message = message

    def __str__(self):
        outstr = 'worker_status_message{'
        outstr += f'pid : {self.pid}, sid : {self.step_id}, did : {self.data_id}, status : {self.status}, log_level : {self.log_level}, message : {self.message}'
        outstr += '}'    
        return outstr

class data_message():
    def __init__(self, id:int, data):
        self.id = id
        self.data = data

class parameter_data_stream():
    def __init__(self, data:List[any]=[], names:List[str]=None, max_size:int=None):
        mpm = multiprocessing.Manager()
        if max_size is None:
            self._queue = mpm.Queue()
        else:
            self._queue = mpm.Queue(maxsize=max_size)
        self._id_counter = 0

        for item in data:
            dm = data_message(self._id_counter, item)
            self._queue.put(dm)
            self._id_counter += 1

    def append(self, data:any, id:int=None):
        if id is None:
            id = self._id_counter
            self._id_counter += 1
        dm = data_message(id, data)
        self._queue.put(dm)

    def put(self, data):
        self._queue.put(data)

    def get(self, block=True):
        try:
            return self._queue.get(block=block)
        except:
            return None

    def __len__(self):
        return self._queue.qsize()
    
    def empty(self):
        return self._queue.empty()

    def full(self):
        return self._queue.full()