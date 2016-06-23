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
from networks.LasagneConv import IBDPairConvAe, IBDPairConvAe2
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
    parser.add_argument('-n', '--numpairs', type=int, default=-1,
        help='number of IBD pairs to use')
    parser.add_argument('-o', '--output', default=None,
        help='optionally save AE prediction to specified h5 file')
    parser.add_argument('-l', '--learn_rate', default=0.001, type=float,
        help='the learning rate for the network')
    parser.add_argument('--network', default='IBDPairConvAe',
        choices=[
            'IBDPairConvAe',
            'IBDPairConvAe2',
        ],
        help='network to use')
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    #class for networks architecture
    logging.info('Constructing untrained ConvNet of class %s', args.network)
    convnet_class = eval(args.network)
    cae = convnet_class(bottleneck_width=args.bottleneck_width,
        epochs=args.epochs, learn_rate=args.learn_rate)
    logging.info('Preprocessing data files')
    train, val, test = get_ibd_data(tot_num_pairs=args.numpairs)
    preprocess = cae.preprocess_data(train)
    preprocess(val)
    preprocess(test)

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

    if args.output is not None:
        logging.info('Saving autoencoder output')
        outdata = np.vstack((cae.predict(train)[1], cae.predict(val)[1],
            cae.predict(test)[1]))
        filename = args.output
        outfile = h5py.File(filename, 'w')
        outdset = outfile.create_dataset("ibd_pair_predictions", data=outdata,
            compression="gzip", chunks=True)

