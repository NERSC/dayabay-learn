# Evan Racah
import h5py
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import glob
import os
from helper_fxns import center, scale, fix_time_zeros
from operator import mul

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
    test = np.asarray(h5set[(ntrain + nvalid):(ntrain + nvalid + ntest)])

    imageshape = (4, 8, 24)
    nfeatures = reduce(mul, imageshape)
    # Don't use all of the array since it contains the metadata as well as the
    # pixels
    train = train[:, :nfeatures].reshape(ntrain, *imageshape)
    valid = valid[:, :nfeatures].reshape(nvalid, *imageshape)
    test = test[:, :nfeatures].reshape(ntest, *imageshape)

    return (train, valid, test)


def get_ibd_data(path_prefix="/project/projectdirs/dasrepo/ibd_pairs", mode='standardize',
                tot_num_pairs=-1):
    
    h5filename = "all_pairs.h5"
    train, val, test = load_ibd_pairs(path=os.path.join(path_prefix, h5filename), tot_num_pairs=tot_num_pairs)
    fix_time_zeros(train)
    fix_time_zeros(val)
    fix_time_zeros(test)
    center(train)
    center(val)
    center(test)
    scale(train, 1, mode=mode)
    scale(val, 1, mode=mode)
    scale(test, 1, mode=mode)
    
    return train, val, test
