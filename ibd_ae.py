__author__ = 'racah'
import numpy as np
import os
import pickle
import sys
import h5py
import matplotlib
from sklearn.manifold import TSNE
import numpy as np
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from vis.viz import Viz
from util.helper_fxns import adjust_train_val_test_sizes
from util.helper_fxns import center, scale
from util.data_loaders import load_ibd_pairs
from util.data_loaders import load_dayabay_conv
from LasagneConv import IBDPairConvAe
import logging



# 1) Primary AD           10000 or 1
# 2) Delayed AD response  01000 or 2
# 3) Muon decay           00100 or 3
# 4) Flasher              00010 or 4
# 5) Other (background noise) 00001 or 5


# In[118]:

if __name__ == "__main__":

    #class for networks architecture
    cae = IBDPairConvAe(epochs=10)
    
    #load data from hdf5, preprocess and split into train and test
    train = np.zeros((20000, 4, 8, 24))
    val = np.zeros((10000, 4, 8, 24))
    test = np.zeros((10000, 4, 8, 24))
    h5files = []
    for i in range(4):
        name = "../dayabay-data-conversion/extract_ibd/ibd_yasu_%d_%d.h5"
        h5file = name % (i*10000, (i+1)*10000-1)
        (train[i*5000:(i+1)*5000], val[i*2500:(i+1)*2500],
            test[i*2500:(i+1)*2500]) = load_ibd_pairs(h5file)

    center(train)
    center(val)
    center(test)
    scale(train, 1)
    scale(val, 1)
    scale(test, 1)

    train, _, val, _, test, _  = adjust_train_val_test_sizes(cae.minibatch_size,
        train, train, val, val, test, test)

    #uses scikit-learn interface (so this trains on X_train)
    cae.fit(train)

    #extract the hidden layer outputs when running x_val thru autoencoder
    feat = cae.extract_layer(val, 'bottleneck')[:, :, 0, 0]
    logging.debug('feat.shape = %s', str(feat.shape))
    gr_truth = np.ones(val.shape[0])

    v = Viz(gr_truth)

    # take first two principal components of features, so we can plot easily
    #normally we would do t-SNE (but takes too long for quick demo)
    x_pc = v.get_pca(feat)

    #plot the 2D-projection of the features
    v.plot_features(x_pc,save=True)

