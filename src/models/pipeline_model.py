import logging

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class pipeline_model(object):
    def __init__(self):
        self.name = 'base pipeline model'
        self.version = '0.0'
        self.feature_type = None
        self.model = None
        self.estimated_time_per_patch = 0.005 # seconds

    def load_model(self):
        raise NotImplementedError
    
    def inference(self, image, legend_images, data_id=-1):
        raise NotImplementedError
