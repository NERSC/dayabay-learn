
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





if __name__ == "__main__":
    epochs =1
    numpairs = 200
    learn_rate = 0.01
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=1,
        help='number of epochs for training')
    parser.add_argument('-w', '--bottleneck-width', type=int, default=10,
        help='number of features in the bottleneck layer')
    parser.add_argument('-n', '--numpairs', type=int, default=200,
        help='number of IBD pairs to use')
    parser.add_argument('-l', '--learn_rate', default=0.01, type=float,
        help='the learning rate for the network')

    parser.add_argument('--accidental-fraction', type=float, default=0,
        help='fraction of train, test, and val sets that are' +
        ' intentionally accidentals')
    args = parser.parse_args()
    epochs = args.epochs
    numpairs = args.numpairs
    learn_rate = args.learn_rate
    
    run_dir = create_run_dir()
    
    
    
    x_train, x_val, x_test = get_ibd_data(tot_num_pairs=numpairs, preprocess=True, just_charges=True)
    
    make_accidentals(x_train)
    
    dca = DenoisingConvAe(network_kwargs={'learning_rate':learn_rate}, 
                          train_kwargs={'num_epochs': epochs, 'save_path': run_dir})

    dca.fit(x_train,x_train,x_val,x_val)

    rec= dca.predict(x_train)

    hlayer = dca.extract_hidden_layer(x_train)
    
    calc_plot_n_save_tsne(x_train, hlayer, run_dir)
    
    





