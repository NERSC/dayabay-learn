
# coding: utf-8

# In[1]:

import sys


# In[2]:

import inspect


# In[3]:

sys.path.insert(0,'/global/common/cori/software/theano/0.8.2/lib/python2.7/site-packages/')


# In[4]:

import theano


# In[5]:

inspect.getfile(theano)


# In[6]:

sys.path.insert(0,'/global/common/cori/software/lasagne/0.1/lib/python2.7/site-packages/')


# In[7]:

import lasagne


# In[8]:

import numpy as np


# In[33]:

__author__ = 'racah'
import os
import pickle
import sys
#from neon.util.argparser import NeonArgparser

import h5py
import matplotlib
from sklearn.manifold import TSNE
import numpy as np
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from networks.LasagneConv import IBDPairConvAe
from vis.viz import Viz
from util.helper_fxns import adjust_train_val_test_sizes

from util.data_loaders import load_dayabay_conv, load_ibd_pairs


# In[15]:

# 1) Primary AD           10000 or 1
# 2) Delayed AD response  01000 or 2
# 3) Muon decay           00100 or 3
# 4) Flasher              00010 or 4
# 5) Other (background noise) 00001 or 5


# In[24]:

args = sys.argv[3:] if 'ipykernel' in sys.argv[0] else sys.argv


# In[26]:

h5file = args[1] if len(args) > 1 else '/project/projectdirs/dasrepo/single_20000.h5'
batch_size = args[2] if len(args) > 2 else 128


# In[27]:

#load data from hdf5, preprocess and split into train and test
x_train, x_test, x_val  = load_ibd_pairs(h5file, train_frac=0.5, valid_frac=0.25, preprocess=True)


# In[30]:

#class for networks architecture
cae = IBDPairConvAe()

#uses scikit-learn interface (so this trains on X_train)
cae.fit(X_train)


# In[105]:

#extract the hidden layer outputs when running x_val thru autoencoder
feat = cae.extract(X_val)


# In[31]:

gr_truth = y_val #convert from one-hot to normal


v = Viz(gr_truth)

# take first two principal components of features, so we can plot easily
#normally we would do t-SNE (but takes too long for quick demo)
x_pc = v.get_pca(feat)

#plot the 2D-projection of the features
v.plot_features(x_pc,save=False)


# In[32]:

#get reconstruction of X_val from autoencoder
x_rec = cae.predict(X_val)

#fromat X_val
x_orig = X_val.reshape(X_val.shape[0], 192)

#plot reconstruction
v.plot_reconstruction(x_orig[2], x_rec[2], indx=10, save=False)

