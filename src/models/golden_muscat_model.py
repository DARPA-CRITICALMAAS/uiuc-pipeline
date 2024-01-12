import logging


from .pipeline_model import pipeline_tensorflow_model

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class customer_backpack_model(pipeline_tensorflow_model):
    def __init__(self):
        self.name = 'customer backpack'
        self.checkpoint = 'submodules/models/customer_backpack/inference_model/UNET_seresnet50.h5'

    #@override
    def load_model(self):
        self.model = SegmentationModel.load_from_checkpoint(self.checkpoint, args = args)
        self.model.eval()

        return self.model

