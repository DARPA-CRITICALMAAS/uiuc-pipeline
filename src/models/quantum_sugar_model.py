import logging

from submodules.models.quantum_sugar.inference import OneshotYOLO

from .pipeline_pytorch_model import pipeline_pytorch_model
from cmaas_utils.types import MapUnitType

log = logging.getLogger('DARPA_CMAAS_PIPELINE')
    
class quantum_sugar_model(pipeline_pytorch_model):
    def __init__(self):
        super().__init__()
        self.name = 'quantum sugar'
        self.version = '0.1'
        self.feature_type = MapUnitType.POINT
        self.checkpoint = '/projects/bbym/shared/models/quantum_sugar/best.pt'
    
    # @override
    def load_model(self):
        self.model = OneshotYOLO()
        self.model.load(self.checkpoint)
        self.model.eval()

    # @override
    def inference(self, image, legend_images, data_id=-1):
        #TODO implement point dectection inference
        raise NotImplementedError