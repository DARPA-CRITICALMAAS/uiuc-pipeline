import logging
from keras.models import load_model

from .pipeline_tensorflow_model import pipeline_tensorflow_model
from submodules.models.primordial_positron.unet_util import multiplication, multiplication2, dice_coef, dice_coef_loss

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class primordial_positron_model(pipeline_tensorflow_model):
    def __init__(self):
        self.name = 'primordial positron'
        self.checkpoint = 'submodules/models/primordial_positron/inference_model/Unet-attentionUnet.h5'

    #@override
    def load_model(self):
        # Load the attention Unet model with custom objects for attention mechanisms
        self.model = load_model(self.checkpoint, custom_objects={'multiplication': multiplication,
                                                            'multiplication2': multiplication2,
                                                            'dice_coef_loss':dice_coef_loss,
                                                            'dice_coef':dice_coef})
        return self.model
