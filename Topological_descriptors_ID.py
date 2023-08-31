import numpy as np
import tensorflow as tf
from scipy.spatial import distance_matrix
from sklearn import preprocessing
from utils import load_Cifar10, plot_utils, load_model, Manifolds_embeddings
from sklearn.linear_model import LinearRegression
import random
from ripser import ripser #pip install ripser


def get_Lifespans_sum(data, eps, alpha, m, maxdim=2):
  
  '''
    eps: fixed epsilon in filtration or np.inf
    alpha: power of lifespan
    maxdim: dimension of homology group
    return: topology descriptor
    '''
  dgms = ripser(data, maxdim=maxdim, distance_matrix=m, thresh=eps)['dgms']
  list_Hom = []
  mean_Hom = []
  for e, Hom in enumerate(dgms):
    B_Hom = 0
    for i in Hom:
      if (np.isinf(i).any() == False):
        B_Hom += (((i[1]-i[0]))**alpha)*0.5
      else:
        continue
    list_Hom.append(B_Hom)
    mean_Hom.append(B_Hom/len(Hom))
  return list_Hom, mean_Hom

def Lifespans_layers(model, X_data, y_data, layers, hom_dim,  mode='local') -> list:

    '''
    X_data, y_data: train dataset manifold and labels
    eps: fixed epsilon in filtration or np.inf
    alpha: power of lifespan
    hom_dim: dimension of homology group
    layers: list of layers
    return: topology descriptor of each layers
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    Topo_invariant_layers_list = []
    for lr in layers:
        X_emb, layer_name = Manifolds_embeddings(X_data, model, index_layer=lr)
        X_emb = min_max_scaler.fit_transform(X_emb)
        class_allclass = []
        if mode == 'local': 
            for n_class in range(0,9):
              X_emb_class = []
              for e, i in enumerate(y_data[:len(X_emb)]):
                  if (i == n_class) and (len(X_emb_class) < 250):
                      X_emb_class.append(X_emb[e])    
              class_allclass.append(compute_topo_summary(np.array(X_emb_class), hom_dim))
            topo_emb_data = sum(class_allclass)/len(class_allclass)
        else:
            topo_emb_data  = compute_topo_summary(np.array(X_emb), hom_dim) 
        Topo_invariant_layers_list.append(topo_emb_data)

    return Topo_invariant_layers_list

def compute_topo_summary(X, hom_dim) -> float:
    distances_X = distance_matrix(X, X)
    MGST, MGST_mean = get_Lifespans_sum(distances_X  , eps=np.inf, alpha=1, m=True, maxdim=hom_dim)
    lfts = round(MGST[hom_dim], 3)
    return lfts

def PHdim(X, hom_dim=0, n_start = 500, alpha=1):
  '''
  X: data manifold
  eps: fixed epsilon in filtration or np.inf
  alpha: power of lifespan
  hom_dim: dimension of homology group, our experiments work with 0-PHdim
  n_start: length of first sample 
  return: Persistent Homological fractal dimension(PHdim) of data X
  '''
  n = n_start
  n_array = []
  E_array = []
  n_step = 350
  while X.shape[0] > n:
    x_data = X[np.random.choice(X.shape[0], n, replace=False), :]
    #x_data = distance_matrix(x_data, x_data)
    E, _ = get_Lifespans_sum(x_data, np.inf, alpha, 0, hom_dim)
    E_array.append(E[hom_dim])
    n_array.append(n)
    n += n_step

  model = LinearRegression().fit(np.log(np.array(n_array).reshape((-1, 1))), np.log(np.array(E_array)))
  d = alpha/(alpha-model.coef_[0])
  return round(d, 2)

def PHDim_layers(model, data, layers, n_start, alpha):

  '''
  data: 
  eps: fixed epsilon in filtration or np.inf
  alpha: power of lifespan
  '''
  PHDim_layers_list = []
  for lr in layers:
    X_emb, layer_name = Manifolds_embeddings(data, model, index_layer=lr) 
    X_emb_class = np.array(X_emb)
    PHdim_ = PHdim(X_emb_class, hom_dim = 0, n_start = n_start, alpha=alpha)
    PHDim_layers_list.append(PHdim_)
  return PHDim_layers_list


def topo_manifold_layers(tf_model, layers_):
  x_train, y_train, x_test, y_test = load_Cifar10()
  examples = 3000
  mode = 'local'
  hom_dim = 0
  x_data, y_data = zip(*random.sample(list(zip(x_train, y_train)), examples))
  Topo_layers = Lifespans_layers(tf_model, np.array(x_data), y_data, layers_, hom_dim, mode=mode)
  return Topo_layers

def PHdim_manifold_layers(tf_model, layers_):
  x_train, y_train, x_test, y_test = load_Cifar10()
  examples = 3000
  x_data, y_data = zip(*random.sample(list(zip(x_train, y_train)), examples))
  return PHDim_layers(tf_model, np.array(x_data), layers_, n_start = 500, alpha=1)

def Experiment_VGG(model_path:"path to tf.model file", mode:'Topology or PHdim'):
    tf_model = load_model(model_path)
    layers_ = [2, 5, 7, 10, 12, 15, 17, 20, 22, -2] # VGG: models_layers(tf_model, 'Conv')
    #plot_graph(topo_manifold_layers(tf_model, layers_), PHdim_manifold_layers(tf_model, layers_), 'VGG')
    if mode == 'Topology': 
      print('computing topological descriptors')
      plot_utils(topo_manifold_layers(tf_model, layers_),  'Depth', 'Lifespans sum 0','VGG_'+mode)
    elif mode == 'PHdim': 
      print('computing Persistent Homological fractal dimension')
      plot_utils(PHdim_manifold_layers(tf_model, layers_), 'Depth', mode,  'VGG_'+mode)
    else: print('Error! mode must be "Topology" or "PHdim"')

def Experiment_Resnet(model_path:"path to tf.model file", mode:'Topology or PHdim'):
    tf_model = load_model(model_path)
    layers_ = [16, 26, 38, 48, 60, 70, 82, 92, 104, -1] # Resnet: models_layers(tf_model, 'Add')
    #plot_graph(topo_manifold_layers(tf_model, layers_), PHdim_manifold_layers(tf_model, layers_), 'Resnet')
    if mode == 'Topology': 
      print('computing topological descriptors')
      plot_utils(topo_manifold_layers(tf_model, layers_), 'Depth', 'Lifespans sum 0', 'Resnet_'+mode)
    elif mode == 'PHdim': 
      print('computing Persistent Homological fractal dimension')
      plot_utils(PHdim_manifold_layers(tf_model, layers_), 'Depth', mode, 'Resnet_'+mode)
    else: print('Error! mode must be "Topology" or "PHdim"')

def Topo_models(layers_, model_list, X_data, y_data, hom_dim = 0, n_class = 1):
  Topo_model_list = []
  for model in model_list:
    Topo_results = Lifespans_layers(model, X_data, y_data, layers_, hom_dim=hom_dim, n_class=1, mode='local')
    Topo_model_list.append(Topo_results)
  return Topo_model_list

def PHDim_models(layers, model_list, data):
  Phdim_model_list = []
  for model in model_list:
    Phdim_results = PHDim_layers(model, data, layers, n=500, alpha=1)
    Phdim_model_list.append(Phdim_results)
  return Phdim_model_list
