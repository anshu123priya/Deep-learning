import random
from keras.layers.core import *
from keras.layers import *
import sys
import cv2
import matplotlib
matplotlib.use('Agg')
import _pickle as cPickle
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Add, GlobalAveragePooling2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D,Convolution2D
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import model_from_json,model_from_config,load_model
from keras.optimizers import SGD,RMSprop,adam,Adam
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.preprocessing import image
from keras import backend as K
from keras.initializers import random_uniform, RandomNormal
from sklearn.metrics import mean_squared_error
from collections import OrderedDict as od
import tensorflow as tf
import json
import math
from skimage.measure import compare_ssim as ssim


def identity(e):
  skip_conn =tf.identity(e, name='identity')
  return skip_conn

def adaptive_global_average_pool_2d(x):
    c = x.get_shape()[-1]
    return tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))

	
def channel_attention(input_layer_2):
	skip_conn = Lambda(identity)(input_layer_2)
	input_layer_2 = Lambda(adaptive_global_average_pool_2d)(input_layer_2)
	#x = GlobalAveragePooling2D()(input_layer_2)
	input_layer_2 = Conv2D(64 // 16, (1, 1), activation='relu', padding='same',use_bias=True)(input_layer_2)
	input_layer_2 = Conv2D(64, (1, 1), activation='sigmoid', padding='same',use_bias=True)(input_layer_2)
	#y =  Multiply()([input_layer_2, skip_conn])
	return multiply([skip_conn,input_layer_2])
	
def residual_channel_attention_block(input_layer_1):
  skip_conn = Lambda(identity)(input_layer_1)
  input_layer_1 = Conv2D(64, (3, 3), activation='relu', padding='same',use_bias=True)(input_layer_1)
  input_layer_1 = Conv2D(64, (3, 3), padding='same',use_bias=True)(input_layer_1)
  input_layer_1 = channel_attention(input_layer_1)
  #y = Add()([x, skip_conn])
  return Add()([input_layer_1, skip_conn])

def residual_group(input_layer):
  skip_conn = Lambda(identity)(input_layer)
  count = 1
  for i in range(7):
    input_layer = residual_channel_attention_block(input_layer)
    #print('loop_doe')
  input_layer = Conv2D(64, (3, 3), padding='same',use_bias=True)(input_layer)
  print(count)
  y = Add()([input_layer, skip_conn])
  return y,input_layer


def residual_channel_attention_network():
	inputs = Input(shape=image_shape)
	head = Conv2D(64,(3,3),padding='same',use_bias=True)(inputs)
	#x = Lambda(identity)(head)
	x = head
	count = 0
	for i in range(3):
		x = residual_group(x)[0]
		loss = residual_group(x)[1]
		count = count+1
	print(count)
	body = Conv2D(64,(3,3),padding='same',use_bias=True)(x)
	body = Add()([body, head])
	tail = Conv2D(1,(3,3),padding='same',activation='softsign',use_bias=True)(body)
	tail = Lambda(lambda z: z)(tail)
	#tail = Conv2D(1,(3,3),padding='same',use_bias=True)(body)
	output_model = Model(inputs=inputs, outputs= tail,name='Generator')
	#print(len(output_model))
	return output_model
	
	
	
ndf = 64
output_nc = 3


def discriminator_model():
    """Build discriminator architecture."""
    n_layers, use_sigmoid = 3, False
    #inputs = Input(shape=input_dim)
    inputs = Input(shape=image_shape )

    x = Conv2D(filters=ndf, kernel_size=(4,4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4,4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model

  
def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_images = generator(inputs)
    outputs = discriminator(generated_images)
    model = Model(inputs=inputs, outputs=[generated_images, outputs])
    return model


  
image_shape = (145,145, 1)
##input_shape = (145,145, 1)
g = residual_channel_attention_network()
#g.summary()
d = discriminator_model()
#d.summary()
d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
d_on_g.summary()