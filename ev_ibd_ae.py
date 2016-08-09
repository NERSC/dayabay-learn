
import matplotlib; matplotlib.use("agg")


__author__ = 'racah'
import numpy as np
import os
import pickle
import sys
import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from util.data_loaders import load_ibd_pairs, get_ibd_data
from networks.evdcae_net import DenoisingConvAe
from util.helper_fxns import create_run_dir, make_accidentals
from networks.print_n_plot import  calc_plot_n_save_tsne
import logging
import argparse

'''1) Primary AD           10000 or 1
 2) Delayed AD response  01000 or 2
 3) Muon decay           00100 or 3
 4) Flasher              00010 or 4
 5) Other (background noise) 00001 or 5'''






# if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=200,
    help='number of epochs for training')

parser.add_argument('-l', '--learn_rate', default=0.01, type=float,
    help='the learning rate for the network')

parser.add_argument('-n', '--num_ims', default=200, type=int,
    help='number of total images')

parser.add_argument('-f', '--num_filters', default=128, type=int,
    help='number of filters in each conv layer')

parser.add_argument( '--fc', default=1024, type=int,
    help='number of fully connected units')

parser.add_argument('-c','--num_extra_conv', default=0, type=int,
    help='conv layers to add on to each conv layer before max pooling')

parser.add_argument('-b','--batch_size', default=128, type=int,
help='batch size')

parser.add_argument('--momentum', default=0.9, type=float,
    help='momentum')


args = parser.parse_args()



kwargs = dict(args._get_kwargs())



run_dir = create_run_dir()



x_train, x_val, x_test = get_ibd_data(tot_num_pairs=kwargs['num_ims'], preprocess=True, just_charges=True)


dca = DenoisingConvAe(save_dir=run_dir, network_kwargs=kwargs)

dca.fit(x_train, x_train, x_val, x_val)















