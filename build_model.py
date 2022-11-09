## Date:2022-11-09
## author: Junmei

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import os

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

# from tensorflow import random
##os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def build_deconv(input_shape=(60,30,1), filters = [32,64,64,32], out_classes = 4, drop_rate=0.4, is_training=True):
 
    bulk_layer = Input(input_shape)
    ksize_first_layer = (5,5)
    ksize_other = (3,3)
    c1 = Conv2D(filters[0],kernel_size=ksize_first_layer, padding="same",    activation="tanh", name="c1_1")(bulk_layer)
    c1 = MaxPooling2D((2, 2), name="p1_1")(c1)
    c1 = Conv2D(filters[1], kernel_size=ksize_other, padding="same",  activation="tanh",  name="c1_2")(c1)
    c1 = MaxPooling2D((2, 2), name="p1_2")(c1)
    c1 = Conv2D(filters[2],kernel_size=ksize_other, padding="same",  activation="tanh",  name="c1_3")(c1)
    c1 = MaxPooling2D((2, 2), name="p1_3")(c1)
    c1 = Conv2D(filters[3],kernel_size=ksize_other, padding="same",  activation="tanh",  name="c1_4")(c1)
    
    x = Flatten()(c1)
    ## if is_training then fro training mode; else inference mode
    x = Dropout(drop_rate, name="drop_af_flat")(x, training=is_training)
    x = Dense(filters[-1], activation="tanh", name="hidden1")(x)
    x = Dense(out_classes, activation="softmax", name="output_prob")(x)
    dec_md = Model(inputs=bulk_layer, outputs= x) 
    return dec_md

es_patience_val = 60
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',patience=patience_val)
rlp_factor = 0.98
rlp_patience = 4
callback=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_root_mean_squared_error', patience=rlp_patience,  
                                               mode='min', factor=rlp_factor)

kcnn = build_deconv() #[32,32,32,16]
kcnn.compile(loss=keras.losses.mse,optimizer = keras.optimizers.Adam(0.001, beta_1=0.9),\
         metrics=tf.keras.metrics.RootMeanSquaredError()) 
### for training 
scra_hist = kcnn.fit(train_sets, train_y,
                    shuffle=True, epochs=600,
                    use_multiprocessing=True,
                        verbose=0,
                    workers=12,validation_data=(val_set, val_y), 
                    callbacks=[callback, early_stop],)


### traing in simulated data; then fine-tune using real mixed samples

callback_ft=tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3,  
                                               mode='min', factor=0.98)
finetune_models = {}
for i in range(5):
    tmp_md = scrath_models[i]
    tmp_finet = cnn_scratch(Input((60,30,1)))
    tmp_finet.set_weights(tmp_md.get_weights())
    for l in tmp_finet._layers[:10]:
        l.trainable=False
    for l in tmp_finet._layers[10:]:
        l.trainable=True

    tmp_finet.compile(optimizer=Adam(lr=0.0009), ## better than beta_1=0.9
                     loss=[tf.keras.losses.MeanSquaredError()],#, tf.keras.losses.MeanSquaredError()
                      # tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM)
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])

    hist2 = tmp_finet.fit(real_tr_sets_gp1,real_tr_y_gp1, 
                        shuffle=True, epochs=216, #296,
#                         validation_data=(real_val_set_gp1,real_val_y_gp1),
                        use_multiprocessing=True,
                        workers=22, verbose=0,
                        callbacks=[callback_ft],
                       )
    finetune_models[i] = tmp_finet



