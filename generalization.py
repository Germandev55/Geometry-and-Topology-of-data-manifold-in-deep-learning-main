from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from os import listdir
from utils import load_Cifar10, load_model, Manifolds_embeddings
from Topological_descriptors_ID import *
import random
import argparse
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--path',  help='path/to/models dir')
parser.add_argument('--homdim', default=0, type=int,  help='dimension of homology group')
args = parser.parse_args()

def generalization_experiments(path_of_dir:'path/to/models dir', homdim):
  print('loading data...')
  x_data, y_data, x_test, y_test = load_Cifar10()
  basepath = Path(path_of_dir)
  Resnet_models = []
  acc_list = []
  model_name = []
  print('unpacking dataset...')
  for filename in listdir(basepath):
    acc = filename.split('_')[-1]
    acc_list.append(float(acc[:-3]))
    model_ = load_model(path_of_dir + filename)
    Resnet_models.append(model_)
    model_name.append(filename)
   
  print('computing topological descriptors...')
  Resnet_gen = []
  x_train, y_train = zip(*random.sample(list(zip(x_data, y_data)), 2500))
  for model in Resnet_models: 
    cl = generaization_layers(model, np.array(x_train),  y_train, -2, homdim, 1) 
    Resnet_gen.append(cl)

  colors = []
  for n in model_name:
    n = n.split('_')[0]
    if n == 'Resnet32': colors.append(0)
    if n == 'Resnet56': colors.append(1)
    if n == 'Resnet110': colors.append(2)
  colours = ListedColormap(['r','g','b'])

  Y  =  Resnet_gen
  X  =  acc_list
  fig, ax = plt.subplots(figsize=(6, 6))
  corr, p_value = pearsonr(X, Y)
  print(corr)

  scatter = ax.scatter(X, Y, alpha=0.5, s = 700 , c=colors, cmap=colours )
  ax.grid(which='major', color='#CCCCCC', linestyle='--', alpha=0.35) 
  models_blocks = ['0.2','0.18','0.16','0.14','0.12','0.1','0.08','0.06']
  ax.set_xticks([0.78,0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94])
  ax.plot([min(X),max(X)], [max(Y),min(Y)], linestyle='--', color='black', linewidth=0.75, markersize=40)
  ax.set_xticklabels(models_blocks)
  ax.set_xlim(xmin=0.77,xmax=0.94)
  y_label = 'Lifespans sum '+str(homdim)
  ax.set_ylabel(y_label)
  ax.set_xlabel("Test accuracy error")
  red_patch1 = mpatches.Patch(color='red',alpha=0.5, label='Resnet-32')
  red_patch2 = mpatches.Patch(color='blue', alpha=0.5, label='Resnet-56')
  red_patch3 = mpatches.Patch(color='green', alpha=0.5, label='Resnet-110')
  ax.legend(handles=[red_patch1,red_patch2,red_patch3])
  plt.show()

def generaization_measure(model_path, x_data, y_data):
  '''
  model_path: tf.model path
  x_data, y_data: train dataset and labels
  '''
  tf_model = load_model(model_path)
  hom_dim = 0
  class_list = []
  for c in range(10):
    Lifespans_class = Lifespans_layers(tf_model, x_data, y_data, -2, hom_dim, n_class=c, mode='local') 
    class_list.append(Lifespans_class[0])
  gen_measure_topo = sum(class_list)/10
  return gen_measure_topo

def generaization_layers(model, x_data, y_data, layers, hom_dim, n_class=1) -> float:
    min_max_scaler = preprocessing.MinMaxScaler()
    X_emb, layer_name = Manifolds_embeddings(x_data, model, index_layer=layers) 
    X_emb = min_max_scaler.fit_transform(X_emb)   
    X_emb_class = []
    for e, i in enumerate(y_data[:len(x_data)]):
        if i == n_class: X_emb_class.append(X_emb[e]) 
    return compute_topo_summary(np.array(X_emb_class), hom_dim)

generalization_experiments(args.path, args.homdim)