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
from util.data_loaders import load_ibd_pairs, get_ibd_data
from networks.LasagneConv import IBDPairConvAe
import argparse
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s')



# 1) Primary AD           10000 or 1
# 2) Delayed AD response  01000 or 2
# 3) Muon decay           00100 or 3
# 4) Flasher              00010 or 4
# 5) Other (background noise) 00001 or 5


# In[118]:

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10,
        help='number of epochs for training')
    parser.add_argument('-w', '--bottleneck-width', type=int, default=10,
        help='number of features in the bottleneck layer')
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    #class for networks architecture
    logging.info('Constructing untrained ConvNet')
    cae = IBDPairConvAe(bottleneck_width=args.bottleneck_width,
        epochs=args.epochs)
    logging.info('Preprocessing data files')
    train, val, test = get_ibd_data(tot_num_pairs=10000)

    #uses scikit-learn interface (so this trains on X_train)
    logging.info('Training network')
    cae.fit(train)

    #extract the hidden layer outputs when running x_val thru autoencoder
    logging.info('Extracting bottleneck layer')
    feat = cae.extract_layer(val, 'bottleneck')[:, :, 0, 0]
    logging.debug('feat.shape = %s', str(feat.shape))
    gr_truth = np.ones(val.shape[0])

    logging.info('Constructing visualization')
    v = Viz(gr_truth)

    # take first two principal components of features, so we can plot easily
    #normally we would do t-SNE (but takes too long for quick demo)
    x_pc = v.get_pca(feat)

    #plot the 2D-projection of the features
    v.plot_features(x_pc,save=True)
