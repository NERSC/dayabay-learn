# Evan Racah
import h5py
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import glob
import os
from os.path import join


def load_dayabaysingle(path,
                       validation=True,
                       cross_validate=False,
                       normalize=True,
                       just_test=False,
                       test_prop=0.2,
                       val_prop=0.2,
                       seed=3):

    '''val_prop is proprotion of train that is val
    test_prop is proportion of total that is test'''
    pkl_dir = './pkled_standard_scaler'
    h5_dataset = h5py.File(path)
    X = np.asarray(h5_dataset['inputs']).astype('float64')
    y = np.asarray(h5_dataset['targets']).astype('float64')
    nclass = 5

    #filter out zero rows
    nonzero_rows = ~np.all(X[:, :-1]==0, axis=1) #cuz label will not be zero
    X = X[nonzero_rows]
    y = y[nonzero_rows]


    #get equal number of examples
    #we assume that y is in one hot encoding form, so the if class is 3 the third element of a row in y is 1.
    min_class_count = min([X[y[:, cl] == 1.].shape[0] for cl in range(nclass)])

    #a tuple of nlcass arrays where each array is examples from one of the classes and each array is same size
    X_eq_cl = tuple([X[y[:, cl] == 1.][:min_class_count, :] for cl in range(nclass)])
    y_eq_cl = tuple([y[y[:, cl] == 1.][:min_class_count, :] for cl in range(nclass)])

    X = np.vstack(X_eq_cl)
    y = np.vstack(y_eq_cl)


    num_ex = X.shape[0]
    num_ex_per_class = min_class_count
    num_tr_per_class = int((1-test_prop) * num_ex_per_class)
    num_tr = int( (1-test_prop) * num_ex)




    if just_test:
        X_test = X
        y_test = y

        if normalize:
            assert os.path.exists(pkl_dir), "No standard scaler pkl files exist. Exiting!"
            #upload standard scaler that was fit on training data and transform test based on that
            newest_file = max(glob.iglob(join(pkl_dir,'*.pkl')), key=os.path.getctime)
            ss = pickle.load(open(join(pkl_dir,newest_file)))
            X_test = ss.transform(X_test)
    else:

        #shuffle indices
        #ix = np.arange(X.shape[0])
        i_cl = np.arange(min_class_count)
        np.random.RandomState(seed).shuffle(i_cl)
        #np.random.RandomState(seed).shuffle(ix)
        X_train = np.vstack(tuple([x_cl[i_cl[:num_tr_per_class]] for x_cl in X_eq_cl]))
        y_train = np.vstack(tuple([y_cl[i_cl[:num_tr_per_class]] for y_cl in y_eq_cl]))
        X_test = np.vstack(tuple([x_cl[i_cl[num_tr_per_class:]] for x_cl in X_eq_cl]))
        y_test = np.vstack(tuple([y_cl[i_cl[num_tr_per_class:]] for y_cl in y_eq_cl]))
        # tr_i = num_tr
        # X_train = X[i[:tr_i]]
        # y_train = y[i[:tr_i]]
        #
        # X_test = X[i[tr_i:]]
        # y_test = y[i[tr_i:]]

        if validation:
            ix = np.arange(X_train.shape[0])
            np.random.RandomState(seed).shuffle(ix)
            X_train = X_train[ix]
            y_train = y_train[ix]
            num_val = val_prop * num_tr
            X_val = X_train[:num_val]
            y_val = y_train[:num_val]
            X_train = X_train[num_val:]
            y_train = y_train[num_val:]

        if normalize:
            ss = StandardScaler()
            #subtracts out mean and divide by stdv
            X_train = ss.fit_transform(X_train)

            #save standard scaler to file
            if not os.path.exists(pkl_dir):
                os.mkdir(pkl_dir)
            pickle.dump(ss, open(join(pkl_dir,str(num_tr) + '.pkl'), 'w'))

            #shouldnt need the lvalue here, cuz does it in place
            X_test = ss.transform(X_test)
            if validation:
                X_val = ss.transform(X_val)




    if validation:
        ret = [(X_train, y_train), (X_val, y_val), (X_test, y_test), nclass]

    elif cross_validate:
        pass
    else:
        ret = [(X_train, y_train), (X_test, y_test), nclass]

    return ret







