# Evan Racah
import h5py
#from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import glob
import os
from os.path import join

from operator import mul

def filter_out_zeros(X,y):
    
    #get indices of all rows that don't contain all zeroes
    nonzero_rows = ~np.all(X[:, : ]==0., axis=1)
    
    #filter for these nonzero rows
    X = X[nonzero_rows]
    y = y[nonzero_rows]
    return X,y

def get_equal_per_class(X,y, nclass):
#     #get the number of rows in X that correspond to a y for each given class, then take the class that 
#     #corresponds to the fewest rows in X
#     min_classes = [X[y == float(cl)].shape[0] for cl in range(nclass)]
#     print min_classes
#     min_class_count = min(min_classes)
#     #get min_class_count rows of X for each corresponding class
#     print y.shape
#     X = np.vstack(tuple([X[y == float(cl)][:min_class_count, :] for cl in range(nclass)]))
#     y = np.vstack(tuple([y[y == float(cl)][:min_class_count] for cl in range(nclass)]))
#     print y.shape
    
    return X,y


def make_divisible_by_batch_size(X,y,batch_size):
    if batch_size:
        num = batch_size * (X.shape[0] / batch_size)
        X = X[:num]
        y = y[:num]
    return X, y

def split_dataset_in_2(X, y, proportion, seed=3, batch_size=None):
    
    #make indices for every row
    num_ex = X.shape[0]
    i = np.arange(X.shape[0])
    
    #shuffle the indices
    #no shuffle. file is already shuffled 
    #np.random.RandomState(seed).shuffle(i)
    
    #allocate proportion*100 percent of the indices for first dataset
    #and the rest for second
    first_inds = i[:proportion * num_ex]
    second_inds = i[proportion * num_ex:]
    
    #indexes the correct rows from X and y and make sure the number of examples is divisible batch size so no weird erros
    #with X and y
    X_first, y_first = make_divisible_by_batch_size(X[first_inds], y[first_inds].reshape(first_inds.shape[0]), batch_size)
    X_second, y_second = make_divisible_by_batch_size(X[second_inds], y[second_inds].reshape(second_inds.shape[0]), batch_size)
    
    return X_first, y_first, X_second, y_second


def do_geom_preproc(X):
    X = X.reshape(X.shape[0],8,24)
    #pad the right w ith the first filter_size-1 columns from the left
    X_p = np.lib.pad(X, ((0, 0), (0, 0), (0, filter_size - 1)), 'wrap')

    #add another 8,24 array that is shifted by 12
    X_s_p = np.lib.pad(X_p, ((0,0), (0,0), (0, 11)), 'wrap')[:,:,11:]
    X_t = np.hstack((X_p, X_s_p))

    #flatten
    X_t = X_t.reshape(X_t.shape[0], reduce(mul, X_t.shape[1:]))
    return X_t

def mean_subtract_unit_var(X):
    X -= np.mean(X)
    X /= np.std(X)
    return X
    

def load_dayabay_conv(path,geom_preproc=False,filter_size=3, just_test=False, test_prop=0.2, validation=True, val_prop=0.2, seed=3, get_y=True, eq_class=True, tr_size=None, batch_size=None):
    '''Expects hdf5 with
            dataset inputs: (n,192+), where first 192 columns are the charge deposits
            dataset label_one_hot: (n,5), where each row is a one hot encoding of the class (1-5)'''
    nclass=5
    
    h5_dataset = h5py.File(path)
    
    if not tr_size:
        tr_size = h5_dataset['inputs'].shape[0]
        
    X = np.asarray(h5_dataset['inputs'][:tr_size,:192]).astype('float64')
    if get_y:
        y = np.asarray(h5_dataset['targets'][:tr_size]).astype('float64')

    y = np.argmax(y,axis=1).reshape(y.shape[0],)
    

    #just to make sure there are no events with all zeroes
    X, y = filter_out_zeros(X,y)
    #X = mean_subtract_unit_var(X)
    
    #do geometry preprocessing
    if geom_preproc:
        X = do_geom_preproc(X)
    

    #make an equal number of examples for each class
    if eq_class:
        X, y = get_equal_per_class(X, y, nclass)



    #reshape into 8,24
    
    X = X.reshape(X.shape[0], 1, 8, 24)
    
    #split into train and test
    X_test, y_test, X_train, y_train = split_dataset_in_2(X, y, test_prop, seed, batch_size=batch_size)


    #split train up again into train and val
    if validation:
         X_val, y_val, X_train, y_train = \
                split_dataset_in_2(X_train, y_train, val_prop, seed, batch_size=batch_size)


    return (X_train, y_train), (X_val,y_val), (X_test, y_test), nclass