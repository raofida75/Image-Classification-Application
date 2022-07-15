import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation,Dense, Dropout


def convolution_layer(x, filter, dropout_):
    weight_decay = 1e-4
    L2 = tf.keras.regularizers.l2(weight_decay)
    for i in range(2):  
        x = Conv2D(filter, (3,3), padding='same', kernel_regularizer=L2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Dropout(dropout_)(x)
    return x

def fully_dense_layer(x_in, units, dropout_):
    x = Dense(units)(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_)(x)
    return x