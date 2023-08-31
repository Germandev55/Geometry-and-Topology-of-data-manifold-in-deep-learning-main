import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir

def Manifolds_embeddings(data, model, index_layer):

  '''
  data: Data lie on n-dim manifold (for example: x_train)
  model: Functional tensorflow model
  index_layer: layer's number; output = -2, softmax activation = -1
  return: data manifold embeddings from layer internel representation
  '''
  layer = model.get_layer(index=index_layer)
  features_layer = tf.keras.models.Model(inputs=model.inputs, outputs = layer.output)
  output = features_layer(data)
  try:
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    X = global_average_layer(output)
  except:
    X = output
  return X, layer

def load_model(path_):
  '''
  load tf.keras model
  '''
  return tf.keras.models.load_model(path_)

def models_layers(model, layer_name_):
  '''
  model: Functional tensorflow model
  layer_name_: layer's name
  layers list with specific name
  '''
  elist = []
  for e, layer in enumerate(model.layers):
    layer_name = str(layer).split('.')[3].split(' ')[0]
    if (str(layer_name_) in layer_name):
      elist.append(e)
  print(elist+[-2, -1])

def unistscount(layer_list: list) -> list:
  return [round((e+1)/len(layer_list),2) for e,i in enumerate(layer_list)]

def plot_utils(data, x_axis_name, y_axis_name, title):
  '''
  plot changing geometry or topology properties of data manifold
  '''
  fig, ax = plt.subplots(figsize=(5, 5))
  layers = np.array(list(range(len(data))))
  axisx = unistscount(layers)
  ax.grid(which='major', color='#CCCCCC', linestyle='--', alpha=0.35)
  plt.plot(axisx[:], data[:], '|-', linewidth=5.5, markersize=10.0)
  plt.ylabel(y_axis_name)
  plt.xlabel(x_axis_name)
  plt.title(title)
  plt.show()

def load_Cifar10():
  '''
  load and prepare cifar-10 dataset
  '''
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255
  mean = np.mean(x_train, axis=(0, 1, 2, 3))
  std = np.std(x_train, axis=  (0, 1, 2, 3))
  x_train = (x_train-mean)/(std + 1e-7)
  x_test = (x_test-mean)/(std + 1e-7)

  return x_train, y_train, x_test, y_test

def models_dir(path_of_dir):
  '''
  load models list from dir
  '''
  basepath = Path(path_of_dir)
  models_dir_ = []
  epochs_list = []
  acc_list = []
  for filename in listdir(basepath):
    epochs_number = filename.split('_')[2]
    acc = filename.split('_')[-1].split('.')[1]
    acc = round(int(acc)/1000, 3)
    print(acc, epochs_number)
    epochs_list.append(epochs_number)
    acc_list.append(acc)
    model_ = load_model(path_of_dir + filename)
    models_dir_.append(model_)
  return models_dir_, epochs_list, acc_list

