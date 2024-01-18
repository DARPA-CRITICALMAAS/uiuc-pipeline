import logging
from types import SimpleNamespace

from .pipeline_pytorch_model import pipeline_pytorch_model
from submodules.models.rigid_wasabi.models import SegmentationModel

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class rigid_wasabi_model(pipeline_pytorch_model):
    def __init__(self):
        self.name = 'rigid wasabi'
        self.checkpoint = 'src/models/checkpoints/SWIN_jaccard.ckpt'

        self.args = SimpleNamespace(model='Unet')

    #@override
    def load_model(self):
        self.model = SegmentationModel.load_from_checkpoint(self.checkpoint, self.args)
        self.model.eval()

        return self.model