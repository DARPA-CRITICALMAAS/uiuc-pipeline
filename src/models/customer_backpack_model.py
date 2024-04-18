import logging
from keras.models import load_model

from .pipeline_tensorflow_model import pipeline_tensorflow_model
from submodules.models.customer_backpack.unet_util import dice_coef_loss, dice_coef
from cmaas_utils.types import MapUnitType

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class customer_backpack_model(pipeline_tensorflow_model):
    def __init__(self):
        super().__init__()
        self.name = 'customer backpack'
        self.version = '0.1'
        self.feature_type = MapUnitType.POLYGON
        self.checkpoint = 'submodules/models/customer_backpack/inference_model/UNET_seresnet50.h5'

    #@override
    def load_model(self):
        self.model = load_model(self.checkpoint, custom_objects={'dice_coef_loss':dice_coef_loss, 
                                                            'dice_coef':dice_coef})

        return self.model
    