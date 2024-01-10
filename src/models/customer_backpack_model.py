import logging
from keras.models import load_model

from .pipeline_model import pipeline_model

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class customer_backpack_model(pipeline_model):
    def __init__(self):
        self.name = 'customer backpack'
        self.checkpoint = 'submodules/models/customer_backpack/inference_model/UNET_seresnet50.h5'

    #@override
    def load_model(self):
        self.model = load_model(self.checkpoint, custom_objects={'dice_coef_loss':dice_coef_loss, 
                                                            'dice_coef':dice_coef})

        return self.model
    

### customer backpack utils ###
# Load all the dependencies
from keras import backend as K
# Use dice coefficient function as the loss function 
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

# calculate loss value
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)