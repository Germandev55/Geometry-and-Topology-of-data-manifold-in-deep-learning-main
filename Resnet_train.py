import os
import numpy as np
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D,Flatten, ReLU, Concatenate
from tensorflow.keras.layers import Flatten, add
from utils import load_Cifar10
from Topological_descriptors_ID import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--net',  help='resnet32, resnet56, resnet110')
parser.add_argument('--epochs', default=100, type=str, )
parser.add_argument('--path', default='resnet_.h5', type=str,  help='path to save model')
args = parser.parse_args()

def regularized_padded_conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, padding='same', kernel_regularizer=_regularizer,
                                  kernel_initializer='he_normal', use_bias=False)
def bn_relu(x):
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)
def shortcut(x, filters, stride, mode):
    if x.shape[-1] == filters:
        return x
    elif mode == 'B':
        return regularized_padded_conv(filters, 1, strides=stride)(x)
    elif mode == 'B_original':
        x = regularized_padded_conv(filters, 1, strides=stride)(x)
        return tf.keras.layers.BatchNormalization()(x)
    elif mode == 'A':
        return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride>1 else x,
                      paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])
    else:
        raise KeyError("Parameter shortcut_type not recognized!")
    

def original_block(x, filters, stride=1, **kwargs):
    c1 = regularized_padded_conv(filters, 3, strides=stride)(x)
    c2 = regularized_padded_conv(filters, 3)(bn_relu(c1))
    c2 = tf.keras.layers.BatchNormalization()(c2)
    
    mode = 'B_original' if _shortcut_type == 'B' else _shortcut_type
    x = shortcut(x, filters, stride, mode=mode)
    return tf.keras.layers.ReLU()(x + c2)
    
    
def preactivation_block(x, filters, stride=1, preact_block=False):
    flow = bn_relu(x)
    if preact_block:
        x = flow
        
    c1 = regularized_padded_conv(filters, 3, strides=stride)(flow)
    if _dropout:
        c1 = tf.keras.layers.Dropout(_dropout)(c1)
        
    c2 = regularized_padded_conv(filters, 3)(bn_relu(c1))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c2

def bootleneck_block(x, filters, stride=1, preact_block=False):
    flow = bn_relu(x)
    if preact_block:
        x = flow
         
    c1 = regularized_padded_conv(filters//_bootleneck_width, 1)(flow)
    c2 = regularized_padded_conv(filters//_bootleneck_width, 3, strides=stride)(bn_relu(c1))
    c3 = regularized_padded_conv(filters, 1)(bn_relu(c2))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c3


def group_of_blocks(x, block_type, num_blocks, filters, stride, block_idx=0):
    global _preact_shortcuts
    preact_block = True if _preact_shortcuts or block_idx == 0 else False
    
    x = block_type(x, filters, stride, preact_block=preact_block)
    for i in range(num_blocks-1):
        x = block_type(x, filters)
    return x


def Resnet(input_shape, n_classes, l2_reg=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64), strides=(1, 2, 2),
           shortcut_type='B', block_type='preactivated', first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
           dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True):
    
    global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, _bootleneck_width, _preact_shortcuts
    _bootleneck_width = bootleneck_width # used in ResNeXts and bootleneck blocks
    _regularizer = tf.keras.regularizers.l2(l2_reg)
    _shortcut_type = shortcut_type # used in blocks
    _cardinality = cardinality # used in ResNeXts
    _dropout = dropout # used in Wide ResNets
    _preact_shortcuts = preact_shortcuts
    
    block_types = {'preactivated': preactivation_block,
                   'bootleneck': bootleneck_block,
                   'original': original_block}
    
    selected_block = block_types[block_type]
    inputs = tf.keras.layers.Input(shape=input_shape)
    flow = regularized_padded_conv(**first_conv)(inputs)
    
    if block_type == 'original':
        flow = bn_relu(flow)
    
    for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
        flow = group_of_blocks(flow,
                               block_type=selected_block,
                               num_blocks=group_size,
                               block_idx=block_idx,
                               filters=feature,
                               stride=stride)
    
    if block_type != 'original':
        flow = bn_relu(flow)
    
    flow = tf.keras.layers.GlobalAveragePooling2D()(flow)
    flow = tf.keras.layers.BatchNormalization()(flow)
    flow = tf.keras.layers.Dense(128)(flow)
    flow = Activation('relu')(flow)
    flow = tf.keras.layers.BatchNormalization()(flow)
    outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(flow)
    outputs = Activation('softmax')(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def cifar_resnet32(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False):
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(5, 5, 5), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type=block_type, preact_shortcuts=False)
    return model

def cifar_resnet56(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False):
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(9, 9, 9), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type=block_type, preact_shortcuts=False)
    return model

def cifar_resnet110(block_type='preactivated', shortcut_type='B', l2_reg=1e-4, load_weights=False):
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(18, 18, 18), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type=block_type, preact_shortcuts=False)
    return model

def train_resnet(model_name:"resnet32, resnet56, resnet110", epochs, path='resnet_.h5'):
  '''
  Training Resnet model
  '''
  if model_name == 'resnet32': model = cifar_resnet32()
  if model_name == 'resnet56': model = cifar_resnet56()
  if model_name == 'resnet110': model = cifar_resnet110()
  BATCH_SIZE = 128
  x_train, y_train, x_test, y_test = load_Cifar10()
  OUTPUT_CLASSES = y_train.max() + 1
  y_train = tf.keras.utils.to_categorical(y_train, OUTPUT_CLASSES)
  y_test = tf.keras.utils.to_categorical(y_test, OUTPUT_CLASSES)
  imagegen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, 
                                                            width_shift_range=0.25,
                                                            height_shift_range=0.25,
                                                            horizontal_flip=True,
                                                            zoom_range=0.25,
                                                            shear_range=0.15)
  imagegen.fit(x_train)
  dataflow = imagegen.flow(x_train, y_train, batch_size=BATCH_SIZE)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0,  patience=3, min_lr=0.5e-6)
  earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)
  history = model.fit(dataflow, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), shuffle=True, callbacks=[earlystopping, lr_reducer])
  model.save(path)


train_resnet(args.net, args.epochs, args.path)