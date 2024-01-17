import logging
from types import SimpleNamespace

from .pipeline_model import pipeline_pytorch_model
from submodules.models.golden_muscat.models import SegmentationModel

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class golden_muscat_model(pipeline_pytorch_model):
    def __init__(self):
        self.name = 'golden muscat'
        self.checkpoint = 'submodules/models/golden_muscat/inference_model/UNET_seresnet50.h5'

        self.args = SimpleNamespace(model='Unet')

    #@override
    def load_model(self):
        self.model = SegmentationModel.load_from_checkpoint(self.checkpoint, self.args)
        self.model.eval()

        return self.model
    