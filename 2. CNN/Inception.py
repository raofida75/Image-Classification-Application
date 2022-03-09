import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Concatenate


def inception_layer(x, num_filter, add=False):
    x1 = Conv2D(num_filter, kernel_size=1)(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
      
    x2 = Conv2D(num_filter, kernel_size=1)(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(num_filter, kernel_size=3, padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
      
    x3 = Conv2D(num_filter, kernel_size=1)(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(num_filter, kernel_size=5, padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
      
    x4 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    x4 = Conv2D(num_filter, kernel_size=1)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
      
    # concatenate
    x_output = Concatenate()([x1,x2,x3,x4])
      
    if add == True:
      x_output = Add()([x,x_output])
    return x_output