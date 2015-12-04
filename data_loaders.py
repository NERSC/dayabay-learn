# Evan Racah
import h5py
#from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import glob
import os
from os.path import join
from peter_old_code.dayabay_dataset_code import dayabay_rotate as dbr
from operator import mul

def filter_out_zeros(X,y):
    nonzero_rows = ~np.all(X[:, :-1]==0, axis=1) #cuz label will not be zero
    X = X[nonzero_rows]
    y = y[nonzero_rows]
    return X,y

def get_equal_per_class(X,y, nclass):
    #get equal number of examples
    #we assume that y is in one hot encoding form, so the if class is 3 the third element of a row in y is 1.
    min_class_count = min([X[y[:, cl] == 1.].shape[0] for cl in range(nclass)])

    #a tuple of nlcass arrays where each array is examples from one of the classes and each array is same size
    X_eq_cl = tuple([X[y[:, cl] == 1.][:min_class_count, :] for cl in range(nclass)])
    y_eq_cl = tuple([y[y[:, cl] == 1.][:min_class_count, :] for cl in range(nclass)])
    return X_eq_cl, y_eq_cl, min_class_count

def split_train_test(X,y,nclass, test_prop, seed):

    X_eq_cl, y_eq_cl, min_class_count = get_equal_per_class(X,y,nclass)
    num_ex_per_class = min_class_count
    num_ex = num_ex_per_class * nclass
    num_tr_per_class = int((1-test_prop) * num_ex_per_class)
    num_tr = int( (1-test_prop) * num_ex)
    i_cl = np.arange(min_class_count)
    np.random.RandomState(seed).shuffle(i_cl)
    X_train = np.vstack(tuple([x_cl[i_cl[:num_tr_per_class]] for x_cl in X_eq_cl]))
    y_train = np.vstack(tuple([y_cl[i_cl[:num_tr_per_class]] for y_cl in y_eq_cl]))
    X_test = np.vstack(tuple([x_cl[i_cl[num_tr_per_class:]] for x_cl in X_eq_cl]))
    y_test = np.vstack(tuple([y_cl[i_cl[num_tr_per_class:]] for y_cl in y_eq_cl]))
    return X_train, y_train, X_test, y_test, num_tr

def split_train_val(X_train, y_train, seed, val_prop, num_tr):
    ix = np.arange(X_train.shape[0])
    np.random.RandomState(seed).shuffle(ix)
    X_train = X_train[ix]
    y_train = y_train[ix]
    num_val = val_prop * num_tr
    X_val = X_train[:num_val]
    y_val = y_train[:num_val]
    X_train = X_train[num_val:]
    y_train = y_train[num_val:]
    return X_train, y_train, X_val, y_val




#todo rotate/center data
def load_dayabaysingle(path,
                       validation=True
                       ,
                       cross_validate=False
                       ,
                       normalize=True
                       ,
                       just_test=False
                       ,
                       test_prop=0.2
                       ,
                       val_prop=0.2
                       ,
                       seed=3):

    '''val_prop is proprotion of train that is val
    test_prop is proportion of total that is test'''
    pkl_dir = './pkled_standard_scaler'
    h5_dataset = h5py.File(path)
    X = np.asarray(h5_dataset['inputs']).astype('float64')
    y = np.asarray(h5_dataset['targets']).astype('float64')
    nclass = 5

    #filter out zero rows
    X,y = filter_out_zeros(X,y)


    #cetner the highest sensor value X
    X = dbr.rotate_tr(X, dbr.get_rot_amount_maxelem_tr(X))







    if not just_test:
        X_train, y_train, X_test, y_test, num_tr = \
            split_train_test(X,y,nclass, test_prop, seed)




        if validation:
            X_train, y_train, X_val, y_val = \
                split_train_val(X_train, y_train, seed, val_prop, num_tr)

        if normalize:
            #mean center data
            tr_mean = np.mean(X_train)


            if not os.path.exists(pkl_dir):
                os.mkdir(pkl_dir)
            pickle.dump(tr_mean, open(join(pkl_dir, str(num_tr) + '.pkl'), 'w'))
            X_train -= tr_mean
            X_test -= tr_mean
            if validation:
                 X_val -= tr_mean

    else:
        # X_test = np.vstack(X_eq_cl)
        # y_test = np.vstack(y_eq_cl)
        #
        # if normalize:
        #     # assert os.path.exists(pkl_dir), "No standard scaler pkl files exist. Exiting!"
        #     # #upload standard scaler that was fit on training data and transform test based on that
        #     # newest_file = max(glob.iglob(join(pkl_dir,'*.pkl')), key=os.path.getctime)
        #     # ss = pickle.load(open(join(pkl_dir,newest_file)))
        #     # X_test = ss.transform(X_test)
        #     pass
        pass






    if validation:
        ret = [(X_train, y_train), (X_val, y_val), (X_test, y_test), nclass]

    elif cross_validate:
        pass
    else:
        ret = [(X_train, y_train), (X_test, y_test), nclass]

    return ret

def load_dayabay_conv(path,clev_preproc=False,filter_size=3, just_test=False, test_prop=0.2, validation=True, val_prop=0.2, seed=3):
    nclass=5
    h5_dataset = h5py.File(path)
    X = np.asarray(h5_dataset['inputs']).astype('float64')
    y = np.asarray(h5_dataset['targets']).astype('float64')

    X -= np.mean(X)
    X /= np.std(X)
    X, y = filter_out_zeros(X,y)
    if clev_preproc:
        X = X.reshape(X.shape[0],8,24)
        #pad the right with the first filter_size-1 columns from the left
        X_p = np.lib.pad(X, ((0, 0), (0, 0), (0, filter_size - 1)), 'wrap')

        #add another 8,24 array that is shifted by 12
        X_s_p = np.lib.pad(X_p, ((0,0), (0,0), (0, 11)), 'wrap')[:,:,11:]
        X_t = np.hstack((X_p, X_s_p))

        #flatten
        X_t = X_t.reshape(X_t.shape[0], reduce(mul, X_t.shape[1:]))
    else:
        X_t = X


    if not just_test:
        X_train, y_train, X_test, y_test, num_tr = split_train_test(X_t,y,nclass, test_prop, seed)




    if validation:
        X_train, y_train, X_val, y_val = \
                split_train_val(X_train, y_train, seed, val_prop, num_tr)
    return (X_train, y_train), (X_val,y_val), (X_test, y_test), nclass

















