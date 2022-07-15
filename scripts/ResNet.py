import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Input, Flatten, Dense, ZeroPadding2D, Add



def identity_block(x, filters, f):
    F1, F2, F3 = filters
    x_in = x
    
    # layer 1
    x = Conv2D(F1, (1,1),strides=(1,1),padding='valid', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # layer 2
    x = Conv2D(F2, (f,f),strides=(1,1), padding='same', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # layer 3
    x = Conv2D(F3, (1,1),strides=(1,1),padding='valid', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    
    # add
    x = Add()([x_in, x])
    x = Activation('relu')(x)
    return x


def convulation_block(x, filters, f, s):

    F1, F2, F3 = filters
    x_shortcut = x

    # layer 1
    x = Conv2D(F1, (1,1), (s,s), kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # layer 2
    x = Conv2D(F2, (f,f),strides=(1,1), padding='same', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # layer 3
    x = Conv2D(F3, (1,1),strides=(1,1), kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)

    # reshape input so it's dimension matches up to the dimensions of final x value 
    x_shortcut = Conv2D(F3, (1,1), (s,s), kernel_initializer=glorot_uniform(seed=0) )(x_shortcut)
    x_shortcut = BatchNormalization(axis=3)(x_shortcut)
    
    # add
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x