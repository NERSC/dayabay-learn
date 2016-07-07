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
from networks.LasagneConv import IBDChargeDenoisingConvAe
import argparse
import logging
logging.basicConfig(format='%(levelname)s:\t%(message)s')
logging.getLogger().setLevel(logging.DEBUG)



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
    parser.add_argument('-p', '--save-prediction', default=None,
        help='optionally save AE prediction to specified h5 file')
    parser.add_argument('-s', '--save-model', default=None,
        help='optionally save the trained model parameters')
    parser.add_argument('-m', '--load-model', default=None,
        help='optionally load a previously saved set of model parameters')
    parser.add_argument('-l', '--learn_rate', default=0.001, type=float,
        help='the learning rate for the network')
    parser.add_argument('--tsne', action='store_true',
        help='do t-SNE visualization')
    parser.add_argument('--network', default='IBDPairConvAe',
        choices=[
            'IBDPairConvAe',
            'IBDPairConvAe2',
            'IBDChargeDenoisingConvAe',
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
    if args.load_model:
        logging.info('Loading model parameters from file %s', args.load_model)
        cae.load(args.load_model)
    logging.info('Preprocessing data files')
    only_charge = getattr(cae, 'only_charge', False)
    train, val, test = get_ibd_data(tot_num_pairs=args.numpairs,
        just_charges=only_charge)
    preprocess = cae.preprocess_data(train)
    preprocess(val)
    preprocess(test)

    #uses scikit-learn interface (so this trains on X_train)
    epochs = []
    costs = []
    def saveprogress(**kwargs):
        epochs.append(kwargs['epoch'])
        costs.append(kwargs['cost'])
    def plotcomparisons(**kwargs):
        if kwargs['epoch'] % 10 != 0:
            return
        numevents = 4
        plotargs = {
            'interpolation': 'nearest',
            'aspect': 'auto',
        }
        for i in range(numevents):
            plt.subplot(2, numevents, i + 1)
            plt.imshow(kwargs['input'][i, 0], **plotargs)
            plt.title('input %d' % i)
            plt.subplot(2, numevents, i + numevents + 1)
            plt.imshow(kwargs['output'][i, 0], **plotargs)
            plt.title('output %d' % i)
        plt.savefig('results/progress/reco%d.pdf' % kwargs['epoch'])
        plt.clf()
    def log_message_cost(**kwargs):
        logging.debug('Loss after epoch %d is %f', kwargs['epoch'],
            kwargs['cost'])
    cae.epoch_loop_hooks.append(log_message_cost)
    cae.epoch_loop_hooks.append(saveprogress)
    cae.epoch_loop_hooks.append(plotcomparisons)
    logging.info('Training network with %d samples', train.shape[0])
    cae.fit(train)

    if args.save_model:
        logging.info('Saving model parameters to %s', args.save_model)
        cae.save(args.save_model)

    plt.plot(epochs, costs)
    plt.savefig('test.pdf')
    plt.clf()

    if args.tsne:
        logging.info('Constructing visualization')
        v = Viz(gr_truth,nclass=1)

        # take first two principal components of features, so we can plot easily
        #normally we would do t-SNE (but takes too long for quick demo)
        #x_pc = v.get_pca(feat)

        num_feats = 500 if feat.shape[0] > 500 else feat.shape[0]
        x_ts = v.get_tsne(feat[:num_feats])

        #plot the 2D-projection of the features
        v.plot_features(x_ts,save=True)

    if args.save_prediction is not None:
        logging.info('Saving autoencoder output')
        outdata = np.vstack((cae.predict(train)[1], cae.predict(val)[1],
            cae.predict(test)[1]))
        filename = args.output
        outfile = h5py.File(filename, 'w')
        outdset = outfile.create_dataset("ibd_pair_predictions", data=outdata,
            compression="gzip", chunks=True)

