#Evan

import matplotlib
matplotlib.use('agg')
import numpy as np
import logging
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from data_loaders import load_dayabaysingle
from neon.data import DataIterator
from neon.initializers import Uniform
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Affine, Linear, GeneralizedCost
from neon.transforms.activation import Tanh, Identity
from neon.transforms.cost import SumSquared, MeanSquared
from neon.models import Model
import os
from matplotlib import pyplot as plt
from neon.callbacks.callbacks import Callbacks, LossCallback
import glob
from os.path import join
import pickle
import h5py
from sklearn.decomposition import PCA
from tsne_source_code import tsne
import time


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

def main():
    #####setup directories and args and logging
    logger = logging.getLogger()

    parser = NeonArgparser()

    model_files_dir = './model_files'
    final_dir = './results/fc-ae'
    output_dir = './intermediate_metrics'
    dirs = [model_files_dir, final_dir, output_dir]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


    parser.add_argument('--h5file')
    parser.add_argument('--test')
    parser.add_argument('--just_test')

    parser.set_defaults(batch_size=128,
                        test=False,
                        save_path=model_files_dir,
                        h5file='/global/homes/p/pjsadows/data/dayabay/single/single_20000.h5',
                        serialize=2,
                        epochs=100,
                        progress_bar=True,
                        datatype='f64',
                        output_file=output_dir,
                        model_file=False,
                        just_test=False,
                        eval_freq=1)

    args = parser.parse_args()
    if args.just_test:
        assert args.model_file, "If just testing, you must specify a model file to load weights from"
        args.test=True
    ##################


    ######set up

    #load and split all data from file
    (X_train, y_train), (X_val, y_val), (X_test, y_test), nclass = load_dayabaysingle(path=args.h5file)
    nin = X_train.shape[1]



    #make sure size of validation data a multiple of batch size
    val_end = args.batch_size * (X_val.shape[0] / args.batch_size)
    X_val = X_val[:val_end]
    y_val = y_val[:val_end]


    #make sure size of test data a multiple of batch size
    test_end = args.batch_size * (X_test.shape[0] / args.batch_size)
    X_test = X_test[:test_end]
    y_test = y_test[:test_end]

    #DataIterator(X, y=None, nclass=None, lshape=None, make_onehot=True)
    #if y unspecified -> works like autoencoder
    train_set = DataIterator(X_train)

    valid_set = DataIterator(X_val)


    if args.test:
        test_set = DataIterator(X_test)

    ########


    ############ set up network

    #Peter initially used AutoUniformGen(), which calculated weights based on activation fxn and size of layer, is deprecated
    init_uni = Uniform(low=-0.1, high=0.1)

    lr_decay_factor = -0.85 #Peter had this as spearmint determined value in old code
    opt_gdm = GradientDescentMomentum(learning_rate=0.001,
                                      momentum_coef=0.5, #Peter had a more complicated momentum set up which is not implemented in new neon
                                      schedule=Schedule(step_config=1,
                                                        change=1.0 - 10 ** lr_decay_factor))
    activation = Tanh()
    bneck_width = 10
    n_layers = 3 #another variable determiend by spearmint
    layers = []
    layers.append(Affine(nout=284, init=init_uni, batch_norm=True, activation=Tanh()))
    layers.append(Affine(nout=284, init=init_uni, batch_norm=True, activation=Tanh()))
    layers.append(Linear(nout=bneck_width, init=init_uni, name='middleLayer'))
    layers.append(Affine(nout=284, init=init_uni, batch_norm=True, activation=Tanh()))
    layers.append(Affine(nout=284, init=init_uni, batch_norm=True, activation=Tanh()))
    layers.append(Linear(nout=nin, init=init_uni))

    cost = GeneralizedCost(costfunc=SumSquared())


    ae = Model(layers=layers)
    ##################


    ##########
    # set up saving data

    #name this model by its widths for the first half of its layers (b/c second half is the same as first -> symmetrical) and the activation fxn and training set size
    #key of form 192-284-284-10-Tanh-26272 for example

    ae_model_key = '{0}-{1}-{2}-{3}-{4}'.format(str(nin),
                                            '-'.join([str(l[0].nout) if isinstance(l, list) else str(l.nout) for l in layers[:(len(layers) / 2)]]),
                                            str(activation).split('object')[0].split('.')[-1][:-1],
                                            os.path.splitext(os.path.basename(args.h5file))[0],'rot')

    args.save_path += '/' + ae_model_key + '-checkpt.pkl'
    args.output_file += '/' + ae_model_key + '-metrics.h5'
    h5fin = h5py.File(join(final_dir, ae_model_key + '-' + str(args.epochs) + ('-test' if args.test else '') + '-final.h5'), 'w')

    #by not specifying metric and adding in eval_freq to args we should get the val loss saved every <eval_freq> epoch
    callbacks = Callbacks(ae, train_set, args, eval_set=valid_set)


    #add callback for calculating train loss every every <eval_freq> epoch
    callbacks.add_callback(LossCallback(h5fin.create_group('train_loss'), ae, eval_set=train_set, epoch_freq=args.eval_freq))

    callbacks.add_callback(LossCallback(h5fin.create_group('valid_loss'), ae, eval_set=valid_set, epoch_freq=args.eval_freq))

    callbacks.add_save_best_state_callback(join(model_files_dir,  ae_model_key + "-best_state"))

    ###################


    #### fit
    if args.just_test:
        args.epochs = 0
    ae.fit(train_set, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

    ######
    #
    # for i, (x,t) in enumerate(valid_set):
    #     x_i = x.asnumpyarray().T
    #     x_t = X_val[i*args.batch_size:i*args.batch_size+args.batch_size]
    #     if np.array_equal(x_i,x_t):
    #         print i
    #     else:
    #         print 'no!', i
    #     # for k, row in enumerate(x_i):
    #     #     for j,row_t in enumerate(X_val):
    #     #         if np.array_equal(row,row_t):
    #     #             print i +k,j
    #
    # assert False








    h5fin.create_dataset('train_raw_x', data=X_train)
    h5fin.create_dataset('train_raw_y', data=y_train)
    h5fin.create_dataset('test_raw_x', data=X_test)
    h5fin.create_dataset('test_raw_y', data=y_test)
    h5fin.create_dataset('val_raw_x', data=X_val)
    h5fin.create_dataset('val_raw_y', data=y_val)

    #save intermeidate layer values
    # h5ae_tr = h5fin.create_dataset('train_ae_x', (X_train.shape[0], bneck_width))
    # save_middle_layer_output(train_set, h5ae_tr, ae)


    h5ae_val = h5fin.create_dataset('val_ae_x', (X_val.shape[0], bneck_width))
    save_middle_layer_output(valid_set, h5ae_val, ae, bneck_width)

    #val_loss_data = h5py.File(args.output_file)['cost/loss'][:]

    #todo add val intermediate metrics to results file which already contains train loss metrics
    #h5fin.create_dataset('cost/loss_val',(args.epochs / args.eval_freq,), data=val_loss_data)



    h5fin.close()

    #save model file
    pickle.dump(ae.serialize(), open(join(model_files_dir, '%s-%s.pkl'%(ae_model_key, str(args.epochs))), 'w'))

if __name__ == "__main__":
    main()












