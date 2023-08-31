from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix
import tensorflow as tf
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import argparse

from utils import Manifolds_embeddings
from Topological_descriptors_ID import *
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from ripser import ripser

min_max_scaler = preprocessing.MinMaxScaler()
parser = argparse.ArgumentParser()
parser.add_argument('--data',  help='path/to/data')
parser.add_argument('--model', type=str,  help='path/to/model.h5')
parser.add_argument('--homdim', default=0, type=int,  help='dimension of homology group')
args = parser.parse_args()

def open_dataset(path):
  f =  open(path, 'rb')
  X = pickle.load(f)
  Builds = X.reshape(-1, *X.shape[-3:])
  y_train = [[i]*4500 for i in range(10)]
  y_train = sum(y_train, [])
  x_train, x_test, y_train, y_test = train_test_split(Builds, y_train, train_size=0.8, random_state=42)
  x_train = np.array(x_train, dtype=np.float32) / 255
  mean = np.mean(x_train,axis=(0, 1, 2, 3))
  std = np.std(x_train,axis=(0, 1, 2, 3)) 
  x_train = (x_train-mean)/(std + 1e-7)
  y_train = np.array(y_train)
  return x_train, y_train

def Topo_layers_adv(model, data, layers, hom_dim) -> list:
  MGST_layers_list = []
  for lr in layers:
    X_emb, layer_name = Manifolds_embeddings(data, model, index_layer=lr) 
    X_emb = min_max_scaler.fit_transform(X_emb)
    MGST_layers_list.append(compute_topo_summary(np.array(X_emb), hom_dim))
  return MGST_layers_list

def unistscount(layer_list: list) -> list:
  return [round((e+1)/len(layer_list),2) for e,i in enumerate(layer_list)]

def plot_adversarial(data, f1list, homdim):
  fig, ax = plt.subplots(figsize=(5, 5))
  colors = plt.cm.magma_r(np.linspace(0,1,len(data[:])))
  e = 0
  layers = np.array(list(range(len(data[0]))))
  axisx = unistscount(layers)
  for MGST_l in data[:]:
    plt.plot(axisx, MGST_l[:] , '-',  color=colors[e], linewidth=8.0, markersize=15, markeredgewidth = 3)
    e = e + 1  
  ax.grid(which='major', color='#CCCCCC', linestyle='--', alpha=0.35)
  plt.title('Adversarial manifold')
  ylabel = "Lifespans sum " + str(homdim)
  plt.ylabel(ylabel)
  plt.xlabel("Depth")
  cmap = plt.get_cmap('magma_r', 200)
  norm = mpl.colors.Normalize(vmin=min(f1list), vmax=max(f1list))
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  cb = plt.colorbar(sm, label="f1 score", ticks=np.linspace(min(f1list), max(f1list), len(f1list)))
  cb.set_label(r'f1 score', labelpad=-38, y=-0.015, rotation=0)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  plt.show()

def adversarial_manifold(path_dataset, path_model, homdim):
  print('load model...')
  MobileNet_B = tf.keras.models.load_model(path_model)
  logits_model = tf.keras.Model(MobileNet_B.input, MobileNet_B.layers[-1].output)
  layers_check_ = [36, 43, 56, 63, 76, 83, 96, -2]
  epsilons_list = [0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095]
  test_examples = 800 #number of images in batch
  f1scores_list = []
  MGST_advr_eps = []
  target = 8 #class 0..9
  target_label = np.reshape(target, (1,)).astype('int64')
  print('unpacking dataset...')
  x_train, y_train = open_dataset(path_dataset)
  print('generating attacks ...')
  for epsilon in epsilons_list:
    advr_img_list = []
    input_images = x_train[:test_examples]
    for original_image in input_images: 
      original_image = tf.convert_to_tensor(original_image.reshape((-1,64,64,3)))
      advr_img_list.append(fast_gradient_method(logits_model, original_image, epsilon, np.inf, y=target_label, targeted=True))
    adv_imgs = np.array(advr_img_list)
    adv_imgs_ = np.squeeze(adv_imgs, axis=1)
    adv_example_targeted_label_pred = MobileNet_B.predict(adv_imgs_)
    prediction = tf.math.argmax(adv_example_targeted_label_pred, axis=1)
    f1 = f1_score(y_train[:test_examples], prediction, average='macro')
    f1scores_list.append(f1) 
    MGST_layers_adv_result = Topo_layers_adv(MobileNet_B, adv_imgs_, layers = layers_check_, hom_dim=homdim)
    MGST_advr_eps.append(MGST_layers_adv_result)
    print('eps = ' , epsilon, 'f1 score = ', f1)

  plot_adversarial(MGST_advr_eps, f1scores_list, homdim)

adversarial_manifold(args.data, args.model, args.homdim)
