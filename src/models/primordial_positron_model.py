import logging
from keras.models import load_model

from .pipeline_model import pipeline_model

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

class primordial_positron_model(pipeline_model):
    def __init__(self):
        self.name = 'primordial positron'
        self.checkpoint = 'submodules/models/primordial-positron/inference_model/Unet-attentionUnet.h5'

    #@override
    def load_model(self):
        # Load the attention Unet model with custom objects for attention mechanisms
        self.model = load_model(self.checkpoint, custom_objects={'multiplication': multiplication,
                                                            'multiplication2': multiplication2,
                                                            'dice_coef_loss':dice_coef_loss,
                                                            'dice_coef':dice_coef})
        return self.model

### primordial positron utils ###
# Load all the dependencies
import copy
import numpy as np
from PIL import Image as im
from keras import backend as K
from tensorflow.keras.layers import Layer, multiply, dot, add, Input
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import f1_score, precision_score,recall_score, cohen_kappa_score
from datetime import datetime, timezone, timedelta

# Use dice coefficient function as the loss function 
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

# Jacard coefficient
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# calculate loss value
def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

# calculate loss value
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def Residual_CNN_block(x, size, dropout=0.0, batch_norm=True):
    if K.image_data_format() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    return conv

class multiplication(Layer):
    def __init__(self,inter_channel = None,**kwargs):
        super(multiplication, self).__init__(**kwargs)
        self.inter_channel = inter_channel
    def build(self,input_shape=None):
        self.k = self.add_weight(name='k',shape=(1,),initializer='zeros',dtype='float32',trainable=True)
    def get_config(self):
        base_config = super(multiplication, self).get_config()
        config = {'inter_channel':self.inter_channel}
        return dict(list(base_config.items()) + list(config.items()))  
    def call(self,inputs):
        g,x,x_query,phi_g,x_value = inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]
        h,w,c = int(x.shape[1]),int(x.shape[2]),int(x.shape[3])
        x_query = K.reshape(x_query, shape=(-1,h*w, self.inter_channel//4))
        phi_g = K.reshape(phi_g,shape=(-1,h*w,self.inter_channel//4))
        x_value = K.reshape(x_value,shape=(-1,h*w,c))
        scale = dot([K.permute_dimensions(phi_g,(0,2,1)), x_query], axes=(1, 2))
        soft_scale = Activation('softmax')(scale)
        scaled_value = dot([K.permute_dimensions(soft_scale,(0,2,1)),K.permute_dimensions(x_value,(0,2,1))],axes=(1, 2))
        scaled_value = K.reshape(scaled_value, shape=(-1,h,w,c))        
        customize_multi = self.k * scaled_value
        layero = add([customize_multi,x])
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
        concate = my_concat([layero,g])
        return concate 
    def compute_output_shape(self,input_shape):
        ll = list(input_shape)[1]
        return (None,ll[1],ll[1],ll[3]*3)
    def get_custom_objects():
        return {'multiplication': multiplication}

def attention_up_and_concatenate(inputs):
    g,x = inputs[0],inputs[1]
    inter_channel = g.get_shape().as_list()[3]
    g = Conv2DTranspose(inter_channel, (2,2), strides=[2, 2],padding='same')(g)
    x_query = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(x)
    phi_g = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(g)
    x_value = Conv2D(inter_channel//2, [1, 1], strides=[1, 1], data_format='channels_last')(x)
    inputs = [g,x,x_query,phi_g,x_value]
    concate = multiplication(inter_channel)(inputs)
    return concate

class multiplication2(Layer):
    def __init__(self,inter_channel = None,**kwargs):
        super(multiplication2, self).__init__(**kwargs)
        self.inter_channel = inter_channel
    def build(self,input_shape=None):
        self.k = self.add_weight(name='k',shape=(1,),initializer='zeros',dtype='float32',trainable=True)
    def get_config(self):
        base_config = super(multiplication2, self).get_config()
        config = {'inter_channel':self.inter_channel}
        return dict(list(base_config.items()) + list(config.items()))  
    def call(self,inputs):
        g,x,rate = inputs[0],inputs[1],inputs[2]
        scaled_value = multiply([x, rate])
        att_x =  self.k * scaled_value
        att_x = add([att_x,x])
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
        concate = my_concat([att_x, g])
        return concate 
    def compute_output_shape(self,input_shape):
        ll = list(input_shape)[1]
        return (None,ll[1],ll[1],ll[3]*2)
    def get_custom_objects():
        return {'multiplication2': multiplication2}

def attention_up_and_concatenate2(inputs):
    g, x = inputs[0],inputs[1]
    inter_channel = g.get_shape().as_list()[3]
    g = Conv2DTranspose(inter_channel//2, (3,3), strides=[2, 2],padding='same')(g)
    g = Conv2D(inter_channel//2, [1, 1], strides=[1, 1], data_format='channels_last')(g)
    theta_x = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(x)
    phi_g = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format='channels_last')(f)
    rate = Activation('sigmoid')(psi_f)
    concate =  multiplication2()([g,x,rate])
    return concate

def UNET_224(IMG_WIDTH=224, INPUT_CHANNELS=8, OUTPUT_MASK_CHANNELS=1, weights=None):
    inputs = Input((IMG_WIDTH, IMG_WIDTH, INPUT_CHANNELS))
    filters = 32
    last_dropout = 0.2
# convolutiona and pooling level 1
    conv_224 = Residual_CNN_block(inputs,filters)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)
# convolutiona and pooling level 2
    conv_112 = Residual_CNN_block(pool_112,2*filters)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)
# convolutiona and pooling level 3
    conv_56 = Residual_CNN_block(pool_56,4*filters)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)
# convolutiona and pooling level 4
    conv_28 = Residual_CNN_block(pool_28,8*filters)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)
# convolutiona and pooling level 5
    conv_14 = Residual_CNN_block(pool_14,16*filters)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)
# Conlovlution and feature concatenation
    conv_7 = Residual_CNN_block(pool_7,32*filters)
# Upsampling with convolution 
    up_14 = attention_up_and_concatenate([conv_7, conv_14]) 
    up_conv_14 = Residual_CNN_block(up_14,16*filters)
# Upsampling with convolution 2
    up_28 = attention_up_and_concatenate([up_conv_14, conv_28])
    up_conv_28 = Residual_CNN_block(up_28,8*filters)
# Upsampling with convolution 3
    up_56 = attention_up_and_concatenate2([up_conv_28, conv_56])
    up_conv_56 = Residual_CNN_block(up_56,4*filters)
# Upsampling with convolution 4
    up_112 = attention_up_and_concatenate2([up_conv_56, conv_112])
    up_conv_112 = Residual_CNN_block(up_112,2*filters)
# Upsampling with convolution 5
    up_224 = attention_up_and_concatenate2([up_conv_112, conv_224])
    #up_224 = attention_up_and_concatenate2(up_conv_112, conv_224)
    up_conv_224 = Residual_CNN_block(up_224,filters,dropout = last_dropout)
# 1 dimensional convolution and generate probabilities from Sigmoid function
    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)  
    conv_final = Activation('sigmoid')(conv_final)
# Generate model
    model = Model(inputs, conv_final, name="UNET_224")
    return model

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def evaluate_prediction_result(location, pred_npy, mask_npy, label_npy, model_path, txt_path):
    prediction_npy = np.load(pred_npy)
    prediction_mask_npy = np.load(mask_npy)
    predition_label_npy = np.load(label_npy)

    #print("All data are loaded")
    
    dim = predition_label_npy.shape
    buf = 30
    numr = 41
    numc = (dim[0]//41)-1
    count = -1
    for i in range(numr):
        
        # if(location == 'covington' and i == 20):
        #     break 

        # Concate each column to create row ith
        # numc_con = int(numc/2)-1 if(location == 'covington') else numc
        numc_con = int(numc)-1
        for j in range(numc_con):
            count += 1    
            temp = prediction_npy[count][buf:-buf,buf:-buf]
            if j == 0:
                rows = temp
            else:
                rows = np.concatenate((rows,temp),axis = 1)

        # Concate the row ith to the total preditcion
        if i == 0:
            prediction_map = copy.copy(rows)
        else:
            prediction_map = np.concatenate((prediction_map,rows),axis = 0)
    
    #print("prediction_map",prediction_map.shape)
    prediction_map = prediction_map[:,:,0]

    # mask
    mask = prediction_mask_npy[:prediction_map.shape[0],:prediction_map.shape[1]]
    [lr,lc] = np.where(mask == 1)
    # print("mask",mask.shape)

    # Read reference data
    groundtruth = predition_label_npy[:prediction_map.shape[0],:prediction_map.shape[1]]
    groundtruthlist = predition_label_npy[:prediction_map.shape[0],:prediction_map.shape[1]][lr,lc]
    prediction = np.logical_and(prediction_map,mask)
    predictionlist = np.logical_and(prediction_map,mask)[lr,lc]

    f1_nonstream = f1_score(groundtruthlist, predictionlist,labels=[0], average = 'micro')
    f1_stream = f1_score(groundtruthlist, predictionlist,labels=[1], average = 'micro')
    precision_nonstream = precision_score(groundtruthlist, predictionlist,labels=[0], average = 'micro')
    precision_stream = precision_score(groundtruthlist, predictionlist,labels=[1], average = 'micro')
    recall_nonstream = recall_score(groundtruthlist, predictionlist,labels=[0], average = 'micro')
    recall_stream = recall_score(groundtruthlist, predictionlist,labels=[1], average = 'micro')
    cohen_kappa = cohen_kappa_score(groundtruthlist, predictionlist)

    append_new_line(txt_path, 'Model path:' + model_path + '  Run at: ' + str(datetime.now()))
    append_new_line(txt_path, 'F1 score of Nonstream: '+str(f1_nonstream))
    append_new_line(txt_path, 'F1 score of Stream: '+str(f1_stream))
    append_new_line(txt_path, 'Precision of Nonstream: '+str(precision_nonstream))
    append_new_line(txt_path, 'Precision of Stream: '+str(precision_stream))
    append_new_line(txt_path, 'Recall of Nonstream: '+str(recall_nonstream))
    append_new_line(txt_path, 'Recall of Stream: '+str(recall_stream))
    append_new_line(txt_path, 'Cohen Kappa: '+str(cohen_kappa))


    print('Model path:' + model_path + '  Run at: ' + str(datetime.now(timezone(timedelta(hours=-6), 'utc'))))
    print('F1 score of Nonstream: '+str(f1_nonstream))
    print('F1 score of Stream: '+str(f1_stream))
    print('Precision of Nonstream: '+str(precision_nonstream))
    print('Precision of Stream: '+str(precision_stream))
    print('Recall of Nonstream: '+str(recall_nonstream))
    print('Recall of Stream: '+str(recall_stream))
    print('Cohen Kappa: '+str(cohen_kappa))

    data = im.fromarray(prediction.astype('float'))
    data.save(pred_npy[:-4]+".tif")