import tensorflow as tf
import numpy as np
import os
from functools import reduce
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,LeakyReLU,Concatenate,Add,ZeroPadding2D,BatchNormalization,Input
from keras.models import Model
from tensorflow.keras.regularizers import L2



def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def conv2d(no_filter,filter_size, **kwargs):
    """ Convolutional layer of the darknet"""
    if kwargs.get("stride") == (2,2):
        y = "valid"
        x = (2,2)
    else:
        y = "same"
        x = (1,1)
    conv = Conv2D(no_filter, filter_size, strides = x, padding= y ,use_bias=False , kernel_regularizer=L2(0.0005 ))
    conv = BatchNormalization() (conv)
    conv = LeakyReLU(alpha = 0.1)(conv)
    return conv


def residual_block(x,no_filter,no_times):
    """residual block which is used in the darknet"""
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = conv2d(no_filter,(3 , 3),stride=(2,2))(x)
    for a in range(no_times):
        y = compose(conv2d(no_filter//2 , (1,1)) , conv2d(no_filter , (3 ,3))) (x)

        x = Add()([x , y])
    return x


def darknet53(x):
    """Darknet Body having 52 Conv2d layers"""
    x = conv2d(32,(3,3))(x)
    x = residual_block(x , 64 , 1)
    x = residual_block(x , 128 , 2)
    x = residual_block(x , 256 , 8)
    x = residual_block(x , 512 , 8)
    x = residual_block(x , 1024 , 4)
    return x


x = Input(shape=(416,416,3))

darknet53(x)


print(x)

