'''Construct a t-SNE visualization of the bottleneck layer of an
autoencoder.'''

from networks.LasagneConv import *
from sklearn.manifold import TSNE
import numpy as np
from util.data_loaders import get_ibd_data
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cae = IBDChargeDenoisingConvAe(bottleneck_width=256, minibatch_size=2000)
    cae.load('batch/tmp_output/e100_w256_n20000/model_e100_w256_n20000.npz')
    # load the data
    train, val, test = get_ibd_data(tot_num_pairs=4000, just_charges=True)

    # Get bottleneck layer output for each set
    preprocess = cae.preprocess_data(train)
    train_features = cae.extract_layer(train, 'bottleneck')
    print train_features.shape
    train_features = train_features[:, :, 0, 0]
    #val_features = cae.extract_layer(val, 'bottleneck')
    #test_features = cae.extract_layer(test, 'bottleneck')

    train_tsne = TSNE(random_state=0)
    result = train_tsne.fit_transform(train_features)

    plt.plot(result[:, 0], result[:, 1], 'ro')
    plt.savefig('tsne.pdf')
