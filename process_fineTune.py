## Date: 2022-11-09
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
import tensorflow.keras as K
from tensorflow.keras.optimizers import Adam
from build_model import *
#  use CUDA or not
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




### traing in simulated data; then fine-tune using real mixed samples
fine_tune_rlp_factor = 0.98
fine_tune_rlp_patience = 3
callback_ft=tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=fine_tune_rlp_patience,  
                                               mode='min', factor=fine_tune_rlp_factor)

trained_md = K.models.load_model("./model/trained_model.h5")
## when in  inference mode use the is_training as None
fine_tune_md =  build_deconv(is_training=None)


fine_tune_lr_first = 0.0005
fine_tune_epoch = 180
fine_tune_md.set_weights(trained_md.get_weights())
## fine-tune: fix the previous layers (extracting featires) so that the params  will not be updated;
## only update params of last two layers (one hidden dense and output layer)
for l in fine_tune_md._layers[:10]:
    l.trainable=False
for l in fine_tune_md._layers[10:]:
    l.trainable=True

fine_tune_md.compile(optimizer=Adam(lr=fine_tune_lr_first), 
                     loss=[K.losses.MeanSquaredError()],
                      metrics=[K.metrics.RootMeanSquaredError()])

hist = fine_tune_md.fit(real_tr_sets,real_tr_y, 
                    shuffle=True, epochs=fine_tune_epoch, 
                    use_multiprocessing=True,
                    workers=12, verbose=0,
                    callbacks=[callback_ft],
                   )
loss = hist.history["root_mean_squared_error"]
plt.figure(figsize=(3,3))

plt.plot(loss, label="Train loss")
plt.legend(loc="upper right")
plt.ylabel("loss")
##plt.ylim([min(plt.ylim()), 0.05])
plt.title("Train loss")
plt.xlabel("epoch")
plt.show()

# fine_tune_md.save("./model/fine_tune_model.h5",save_format='h5')
