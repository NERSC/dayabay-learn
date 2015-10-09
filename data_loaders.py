# Evan Racah
import h5py


def load_dayabaysingle(path, validation=True, cross_validate=False, normalize=False, test_prop=0.2, val_prop=0.2, seed=None):
    '''val_prop is proprotion of train that is val
    test_prop is proportion of total that is test'''

    h5_dataset = h5py.File(path)
    X = h5_dataset['inputs']
    y = h5_dataset['targets']
    nclass = 5
    num_ex = X.shape[0]
    num_tr = int( (1-test_prop) * num_ex)


    if normalize:
        pass
    #data is shuffled before being written to hdf5, but if we continually are using
    #the same dataset we might want to do some shuffling with a seed
    if not seed:
        tr_i = num_tr
        X_train = X[:tr_i]
        y_train = y[:tr_i]

        X_test = X[tr_i:]
        y_test = y[tr_i:]

    else:
        pass

    if validation:
        num_val = val_prop * num_tr
        X_val = X_train[:num_val]
        y_val = y_train[:num_val]
        X_train = X_train[num_val:]
        y_train = y_train[num_val:]

        ret = [(X_train, y_train), (X_val, y_val), (X_test, y_test), nclass]

    elif cross_validate:
        pass
    else:
        ret = [(X_train, y_train), (X_test, y_test), nclass]

    return ret







