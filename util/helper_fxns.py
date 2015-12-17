import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np

def save_orig_data(h5fin,X_train, y_train, X_val,y_val, X_test, y_test):
    h5fin.create_dataset('raw/train/x', data=X_train)
    h5fin.create_dataset('raw/train/y', data=y_train)
    h5fin.create_dataset('raw/test/x', data=X_test)
    h5fin.create_dataset('raw/test/y', data=y_test)
    h5fin.create_dataset('raw/val/x', data=X_val)
    h5fin.create_dataset('raw/val/y', data=y_val)


def plot_train_val_learning_curve(h5fin,final_h5_file):
    plt.clf()
    v_loss = np.asarray(h5fin['valid_loss/cost/loss'])
    t_loss = np.asarray(h5fin['train_loss/cost/loss'])
    plt.plot(range(v_loss.shape[0]), v_loss)
    plt.plot(range(t_loss.shape[0]), t_loss)
    plt.xlabel('epochs')
    plt.ylabel('sum squared error')
    plt.legend(['val_loss', 'tr_loss'])
    plt.savefig(final_h5_file + '.pdf')


def adjust_train_val_test_sizes(batch_size, X_train, y_train, X_val, y_val, X_test, y_test ):
    #make sure size of validation data a multiple of batch size (otherwise tough to match labels)

    train_end = batch_size * (X_train.shape[0] / batch_size)
    X_train = X_train[:train_end]
    y_train = y_train[:train_end]

    val_end = batch_size * (X_val.shape[0] / batch_size)
    X_val = X_val[:val_end]
    y_val = y_val[:val_end]

    #make sure size of test data a multiple of batch size
    test_end = batch_size * (X_test.shape[0] / batch_size)
    X_test = X_test[:test_end]
    y_test = y_test[:test_end]

    return X_train, y_train, X_val, y_val, X_test, y_test


def save_middle_layer_output(dataset, dset, model, bneck_width):
    ix = 0
    for (x, t) in dataset:
        for l in model.layers.layers:
                #forward propagate the data through the deepnet
                x = l.fprop(x)
                if l.name == 'middleLayer' or l.name == 'middleActivationLayer': #trying to get middle layer here
                    ae_x = x.asnumpyarray()

                    #ae_x is transposed from what we expect, so its of size (bneck_width, batch_size)
                    layer_width, batch_size = ae_x.shape
                    assert layer_width == bneck_width

                    #data iterator will do wrap around, so in the last batch's last several items
                    # will be the first several from the first batch
                    if ix + batch_size > dataset.ndata:
                        dset[ix:dataset.ndata] = ae_x.T[:dataset.ndata - ix]
                        # val_arr[ix:dataset.ndata, :] = ae_x.T[:dataset.ndata - ix]

                    else:
                        dset[ix:ix+batch_size] = ae_x.T
                        # val_arr[ix:ix+batch_size, :] = ae_x.T
                        ix += batch_size
                    break;


def get_eq_classes_of(y, points_per_class, nclass):
    y_ind = np.arange(y.shape[0])
    indices = np.asarray([y_ind[y[:, cl] == 1.][:points_per_class] for cl in range(nclass)]).reshape(nclass * points_per_class)

    return indices