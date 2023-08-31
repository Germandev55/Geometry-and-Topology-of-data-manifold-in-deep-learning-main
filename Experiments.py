import argparse
from Topological_descriptors_ID import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--net',  help='CNN arhitecture')
parser.add_argument('--mode', default='Topology', type=str,  help='Topology or PHdim')
parser.add_argument('--path', default=1, type=str, help='Path to tf.model')
args = parser.parse_args()

if args.net == 'Resnet': 
    Experiment_Resnet(args.path, args.mode)
    
if args.net == 'VGG':
    Experiment_VGG(args.path, args.mode)
