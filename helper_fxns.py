import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np

def save_orig_data(h5fin,X_train, y_train, X_val,y_val, X_test, y_test):
    h5fin.create_dataset('train_raw_x', data=X_train)
    h5fin.create_dataset('train_raw_y', data=y_train)
    h5fin.create_dataset('test_raw_x', data=X_test)
    h5fin.create_dataset('test_raw_y', data=y_test)
    h5fin.create_dataset('val_raw_x', data=X_val)
    h5fin.create_dataset('val_raw_y', data=y_val)


def plot_train_val_learning_curve(h5fin,final_h5_file):
    plt.clf()
    v_loss = np.asarray(h5fin['valid_loss/cost/loss'])
    t_loss = np.asarray(h5fin['train_loss/cost/loss'])
    plt.plot(range(v_loss.shape[0]), v_loss)
    plt.plot(range(t_loss.shape[0]), t_loss)
    plt.xlabel('epochs')
    plt.ylabel('sum squared error')
    plt.legend('val_loss', 'tr_loss')
    plt.savefig(final_h5_file + '.pdf')