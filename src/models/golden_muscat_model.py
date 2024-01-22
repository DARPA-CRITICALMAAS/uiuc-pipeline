import logging
import torch
from types import SimpleNamespace

from .pipeline_pytorch_model import pipeline_pytorch_model
from submodules.models.golden_muscat.models import SegmentationModel

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class golden_muscat_model(pipeline_pytorch_model):
    def __init__(self):
        self.name = 'golden muscat'
        self.checkpoint = 'src/models/checkpoints/jaccard.ckpt'

        #self.args = self.parse_args()
        self.args = SimpleNamespace(model='Unet')

    #@override
    def load_model(self):
        self.model = SegmentationModel.load_from_checkpoint(self.checkpoint, self.args)
        self.model.eval()

        return self.model
    