import logging

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class pipeline_model(object):
    def __init__(self):
        self.name = 'base pipeline model'
        self.model = None

    def load_model(self):
        raise NotImplementedError
    
    def inference(self, image, legend_images, batch_size=16, patch_size=256, patch_overlap=0):
        raise NotImplementedError
