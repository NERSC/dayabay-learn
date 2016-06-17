# Evan Racah
import h5py
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import glob
import os
from util.helper_fxns import center, scale
from operator import mul



# def do_geom_preproc(X, filter_size):
#     """preprocess data to take into account cylinder in info
#     Parameters
#     ----------
#     X : array-like shape = [n_samples,n_channels,y_dim, x_dim]
#     Returns
#     filter_size: scalar (number of filters in the first conv layer
#     -------
#     X_t : array_like, shape = [n_samples,n_channels * 2, y_dim, x_dim]
#         For each datapoint x in X and for each tree in the forest,
#         return the index of the leaf x ends up in.
#     """

    
#     #pad the right w ith the first filter_size-1 columns from the left
#     X_p = np.lib.pad(X, ((0, 0), (0, 0), (0, filter_size - 1)), 'wrap')

#     #add another 8,24 array that is shifted by 12
#     X_s_p = np.lib.pad(X_p, ((0,0), (0,0), (0, 11)), 'wrap')[:,:,11:]
#     X_t = np.hstack((X_p, X_s_p))

#     return X_t

def load_ibd_pairs(path, train_frac=0.5, valid_frac=0.25, tot_num_pairs=-1):
    '''Load up the hdf5 file given into a set of numpy arrays suitable for
    convnets.

    The output is a tuple of (train, valid, test). Each set has shape
    (n_pairs, nchannels, xsize, ysize) where
        (nchannels, xsize, ysize) = (4, 8, 24).

    The relative size of each set can be specified in the arguments.'''
    h5file = h5py.File(path, 'r')
    h5set = h5file['ibd_pair_data']
    
    if tot_num_pairs == -1:
        npairs = h5set.shape[0]
    else:
        npairs = tot_num_pairs
    ntrain = int(train_frac * npairs)
    nvalid = int(valid_frac * npairs)
    ntest = npairs - ntrain - nvalid

    train = np.asarray(h5set[:ntrain])
    valid = np.asarray(h5set[ntrain:(ntrain + nvalid)])
    test = np.asarray(h5set[(ntrain + nvalid):])

    imageshape = (4, 8, 24)
    nfeatures = reduce(mul, imageshape)
    # Don't use all of the array since it contains the metadata as well as the
    # pixels
    train = train[:, :nfeatures].reshape(ntrain, *imageshape)
    valid = valid[:, :nfeatures].reshape(nvalid, *imageshape)
    test = test[:, :nfeatures].reshape(ntest, *imageshape)

    return (train, valid, test)


def get_ibd_data(path_prefix="/global/homes/s/skohn/ml/dayabay-data-conversion/extract_ibd", mode='standardize',
                tot_num_pairs=-1):
    
    
    #load data from hdf5, preprocess and split into train and test
    train = np.zeros((20000, 4, 8, 24))
    val = np.zeros((10000, 4, 8, 24))
    test = np.zeros((10000, 4, 8, 24))
    #do we need this preallocation?
    h5files = []
    for i in range(4):
        name = os.path.join(path_prefix,"ibd_yasu_%d_%d.h5")
        h5file = name % (i*10000, (i+1)*10000-1)
        (train[i*5000:(i+1)*5000], val[i*2500:(i+1)*2500],
            test[i*2500:(i+1)*2500]) = load_ibd_pairs(h5file, tot_num_pairs=tot_num_pairs)

    center(train)
    center(val)
    center(test)
    scale(train, 1, mode=mode)
    scale(val, 1, mode=mode)
    scale(test, 1, mode=mode)
    
    return train, val, test


