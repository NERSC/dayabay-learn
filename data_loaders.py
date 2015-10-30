# Evan Racah
import h5py
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import glob
import os
import os.path.join as join

def load_dayabaysingle(path, validation=True, cross_validate=False, normalize=True, just_test=False, test_prop=0.2, val_prop=0.2, seed=None):
    '''val_prop is proprotion of train that is val
    test_prop is proportion of total that is test'''
    pkl_dir = './pkled_standard_scaler'
    h5_dataset = h5py.File(path)
    X = np.asarray(h5_dataset['inputs'])
    y = np.asarray(h5_dataset['targets'])
    nclass = 5

    #filter out zero rows
    nonzero_rows = ~np.all(X[:, :-1]==0, axis=1) #cuz label will not be zero
    X = X[nonzero_rows]
    y = y[nonzero_rows]

    #get equal number of examples
    min_class_count = min([X[X[:,-1] == float(cl)].shape[0] for cl in range(1, nclass+1) ])
    X = np.vstack(tuple([X[X[:,-1] == float(cl)][:,:min_class_count] for cl in range(1, nclass+1)]))


    num_ex = X.shape[0]
    num_tr = int( (1-test_prop) * num_ex)



    #data is shuffled before being written to hdf5, but if we continually are using
    #the same dataset we might want to do some shuffling with a seed
    if not seed:
        if just_test:
            X_test = X
            y_test = y

            if normalize:
                #upload standard scaler that was fit on training data and transform test based on that
                newest_file = max(glob.iglob(join(pkl_dir,'*.pkl')), key=os.path.getctime)
                if not os.path.exists(pkl_dir):
                    os.mkdir(pkl_dir)
                ss = pickle.load(open(join(pkl_dir,newest_file)))
                X_test = ss.transform(X_test)
        else:

            tr_i = num_tr
            X_train = X[:tr_i]
            y_train = y[:tr_i]

            X_test = X[tr_i:]
            y_test = y[tr_i:]

            if validation:
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
                pickle.dump(ss, join(pkl_dir, '_' + str(num_tr)))

                #shouldnt need the lvalue here, cuz does it in place
                X_test = ss.transform(X_test)
                if validation:
                    X_val = ss.transform(X_val)



    else:
        pass


    if validation:
        ret = [(X_train, y_train), (X_val, y_val), (X_test, y_test), nclass]

    elif cross_validate:
        pass
    else:
        ret = [(X_train, y_train), (X_test, y_test), nclass]

    return ret







