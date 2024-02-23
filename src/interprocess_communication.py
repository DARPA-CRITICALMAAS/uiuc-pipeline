from enum import Enum
from pathlib import Path
from src.cmass_types import CMASS_Legend, CMASS_Layout

class ipq_message_type(Enum):
    DATA_LOADING = 1
    INFERENCE = 2
    DATA_SAVING = 3
    VALIDATION = 4

class ipq_log_message():
    def __init__(self, process_id:int, worker_type:ipq_message_type, log_level, map_name:str, message:str):
        self.pid = process_id
        self.type = worker_type
        self.level = log_level
        self.map_name = map_name
        self.message = message

class ipq_work_message():
    def __init__(self, content:any):
        self.content = content
        self.retries = 0

class ipq_data_load_message():
    def __init__(self, map_path:Path, legend:CMASS_Legend, layout:CMASS_Layout, geo_ref:Path):
        self.map_path = map_path
        self.legend = legend
        self.layout = layout
        self.geo_ref = geo_ref
        self.retries = 0