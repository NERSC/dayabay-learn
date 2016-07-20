'''Construct a t-SNE visualization of the bottleneck layer of an
autoencoder.'''

from networks.LasagneConv import *
from sklearn.manifold import TSNE
import numpy as np
from util.data_loaders import get_ibd_data
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='IBDPairConvAe', choices=[
            'IBDPairConvAe',
            'IBDPairConvAe2',
            'IBDChargeDenoisingConvAe',
        ], help='network architecture to use')
    parser.add_argument('-w', '--bottleneck-width', type=int,
        help='width of bottleneck layer')
    parser.add_argument('--minibatch-size', type=int, required=True,
        help='minibatch size (must match original architecture)')
    parser.add_argument('-n', '--num-batches', type=int, default=1,
        help='number of minibatches to process')
    parser.add_argument('-m', '--model', required=True,
        help='location of model parameters from Network.save()')
    parser.add_argument('-o', '--output', required=True,
        help='location to save the output plot')
    parser.add_argument('--save-data', type=str, default=None,
        help='if specified, save the TSNE coordinates to the given location')
    parser.add_argument('-l', '--layer-name', default='bottleneck',
        help='the name given to the layer to extract')
    parser.add_argument('-c', '--condition', nargs='*',
        help='the name(s) of the condition(s) to use to split/color the' +
        ' data points')
    args = parser.parse_args()


    cae = IBDChargeDenoisingConvAe(bottleneck_width=args.bottleneck_width,
        minibatch_size=args.minibatch_size)
    cae.load(args.model)
    tot_num_pairs = args.minibatch_size * args.num_batches
    # load the data
    data, _, _ = get_ibd_data(tot_num_pairs=tot_num_pairs, just_charges=True,
        train_frac=1, valid_frac=0)

    # Get bottleneck layer output for each set
    preprocess = cae.preprocess_data(data)
    for i in range(args.num_batches):
        range_to_feed = slice(i*args.minibatch_size, (i+1)*args.minibatch_size)
        features_tmp = cae.extract_layer(data[range_to_feed], args.layer_name)
        if i == 0:
            features = features_tmp[:, :, 0, 0]
        else:
            features = np.vstack((features, features_tmp[:, :, 0, 0]))

    # set up different colors
    conditions = {}
    def isaccidental(batch):
        '''Approximate isaccidental based on the distance between max pixels.'''
        flat_ish = batch.reshape(batch.shape[0], batch.shape[1], -1)
        maxes = flat_ish.argmax(axis=2)
        flat_indexes = np.unravel_index(maxes.flat, (batch.shape[2],
            batch.shape[3]))
        # The goal is to have an array with contents
        # [
        #     [   [prompt_x, prompt_y],
        #         [delayed_x, delayed_y],
        #         (etc. over all channels)   ],
        # ...
        # ]
        # (or perhaps more 
        flat_index_pairs = np.vstack(flat_indexes).T.reshape(
            batch.shape[0], batch.shape[1], 2)
        distances = np.hypot(
            flat_index_pairs[:, :, 0],
            flat_index_pairs[:, :, 1])
        return np.fabs(distances[:, 0] - distances[:, 1]) > 5


    conditions['accidental'] = isaccidental

    myTSNE = TSNE(random_state=0)
    result = myTSNE.fit_transform(features)

    if args.condition is None:
        plt.plot(result[:, 0], result[:, 1], 'ro')
        plt.savefig(args.output)
    else:
        for condition in args.condition:
            f = conditions[condition]
            mask = f(data)
            plt.plot(result[mask, 0], result[mask, 1], 'ro')
            plt.plot(result[~mask, 0], result[~mask, 1], 'bo')
            plt.legend([condition, 'not %s' % condition])
            plt.savefig(condition + args.output)

    if args.save_data is not None:
        np.save(args.save_data, result)
