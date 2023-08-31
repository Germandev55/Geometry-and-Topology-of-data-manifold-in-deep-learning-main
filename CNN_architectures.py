import numpy as np
import tensorflow as tf
import argparse
from utils import load_Cifar10
from models.VGG import *
from models.ResNet import *
from models.SEResnet import *
from models.MobileNetV2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--net',  help='CNN arhitecture: Resnet, VGG, MobileNetV2, SEResnet')
parser.add_argument('--epochs', default=100, type=int,  help='epochs')
parser.add_argument('--path', default='model.h5', type=str, help='Path to save tf.model')
args = parser.parse_args()

def train_model(model, epochs, path):
  print('loading data...')
  x_train, y_train, x_test, y_test = load_Cifar10()
  OUTPUT_CLASSES = y_train.max() + 1
  y_train = tf.keras.utils.to_categorical(y_train, OUTPUT_CLASSES)
  y_test = tf.keras.utils.to_categorical(y_test, OUTPUT_CLASSES)
  BATCH_SIZE = 128
  print('augmentation...')
  imagegen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,   width_shift_range=0.25, height_shift_range=0.25,
                                                            horizontal_flip=True, zoom_range=0.25, shear_range=0.15)
  imagegen.fit(x_train)
  dataflow = imagegen.flow(x_train, y_train, batch_size=BATCH_SIZE)

  lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0,  patience=3, min_lr=0.5e-6)
  earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)
  print('model training...')
  history = model.fit(dataflow, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), shuffle=True, callbacks=[earlystopping, lr_reducer])
  print('saving the model...')
  model.save(path)

if args.net == 'Resnet': 
  model = Resnet()
  train_model(model, args.epochs, args.path)

elif args.net == 'VGG': 
  model = VGG()
  train_model(model, args.epochs, args.path)

elif args.net == 'MobileNetV2': 
  model = MobileNetV2()
  train_model(model, args.epochs, args.path)

elif args.net == 'SEResnet': 
  model = SEResnet()
  train_model(model, args.epochs, args.path)

else:
  print('Error! Choose one of: Resnet, VGG, MobileNetV2, SEResnet')






