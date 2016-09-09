# Evan Racah
import h5py
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import glob
import os
from operator import mul
def load_ibd_pairs(path, train_frac=0.5, valid_frac=0.25, tot_num_pairs=-1,
        h5dataset='ibd_pair_data'):
    '''Load up the hdf5 file given into a set of numpy arrays suitable for
    convnets.

    The output is a tuple of (train, valid, test). Each set has shape
    (n_pairs, nchannels, xsize, ysize) where
        (nchannels, xsize, ysize) = (4, 8, 24).

    The relative size of each set can be specified in the arguments.'''
    h5file = h5py.File(path, 'r')
    h5set = h5file[h5dataset]
    
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


def get_ibd_data(path="/project/projectdirs/dasrepo/ibd_pairs/all_pairs.h5", preprocess=False, mode='normalize',
                tot_num_pairs=-1, just_charges=False, train_frac=0.5,
                valid_frac=0.25, h5dataset='ibd_pair_data'):
    
    train, val, test = load_ibd_pairs(path=path,
        tot_num_pairs=tot_num_pairs, train_frac=train_frac,
        valid_frac=valid_frac, h5dataset=h5dataset)
    #would be nice to optionally preprocess upon loading (mostly for unit testing)
    if just_charges:
        train = train[:,[0,2]]
        val = val[:,[0,2]]
        test = test[:,[0,2]]
    if preprocess:
        from networks.preprocessing import scale, scale_min_max
        if mode == 'normalize':
            mins, maxes = scale_min_max(train)
            scale_min_max(test,mins=mins,maxes=maxes)
            scale_min_max(val,mins=mins,maxes=maxes)
        
            
        
    return train, val, test

def load_predictions(filepath=None, tot_num_pairs=-1):
    if filepath is None:
        filepath = os.path.join(os.environ['PWD'], 'prediction.h5')
    infile = h5py.File(filepath, 'r')
    h5set = infile['ibd_pair_predictions']
    if tot_num_pairs==-1:
        tot_num_pairs = h5set.shape[0]
    return np.asarray(h5set[:tot_num_pairs])
