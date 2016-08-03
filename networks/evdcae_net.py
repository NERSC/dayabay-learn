


import sys
import theano
import os
import time
import numpy as np
import theano.tensor as T
import lasagne
import matplotlib
import matplotlib.pyplot as plt
import h5py

import time
import preprocessing
from sklearn.manifold import TSNE

#temporary!
if __name__ == "__main__":
    #just a little trick for testing in order to get the data loader
    sys.path.insert(0,os.path.dirname(os.getcwd()))
    

from util.data_loaders import get_ibd_data
from networks.Network import AbstractNetwork
#enable importing of notebooks
from print_n_plot import plot_reconstruction
from build_evdcae import build_network
from train_val import train
from networks.preprocessing import scale, scale_min_max



class DenoisingConvAe(AbstractNetwork):
    def __init__(self, *args, **kwargs):
        network_kwargs = kwargs['network_kwargs'] if 'network_kwargs' in kwargs else {}
        self.train_kwargs = kwargs['train_kwargs'] if 'train_kwargs' in kwargs else {}
        
        self.train_fn,         self.val_fn,         self.pred_fn,         self.hlayer_fn,        self.network = build_network(**network_kwargs)
        
        

    def fit(self, x_train, y_train, x_val,y_val):
        x_train, x_val = self.preprocess_data(x_train, x_val)
        train((x_train, x_train, x_val,x_val),
                             self.network, 
                             self.train_fn, 
                             self.val_fn,
                             pred_fn = self.pred_fn,
                             **self.train_kwargs)
        

    def predict(self, x):
        return self.pred_fn(x)
        

    def extract_hidden_layer(self, data):
        return self.hlayer_fn(data)
        

    def minibatch_iterator(self, x, y):
        '''iterate over minibatches'''
        raise NotImplemented()

    def preprocess_data(self, train,val ):
        mins, maxes = scale_min_max(train)
        scale_min_max(val,mins=mins,maxes=maxes)
        return train, val



if __name__ == "__main__":
    x_train, x_val, x_test = get_ibd_data(tot_num_pairs=200, preprocess=True, just_charges=True)

    dca = DenoisingConvAe(network_kwargs={'learning_rate':0.01}, 
                          train_kwargs={'num_epochs': 11, 'save_path': os.path.dirname(os.getcwd()) + '/results'})

    dca.fit(x_train,x_train,x_val,x_val)

    rec= dca.predict(x_train)

    hlayer = dca.extract_hidden_layer(x_train)



    ts = TSNE(perplexity=50).fit_transform(hlayer)

    plt.scatter(ts[:,0], ts[:,1])

