#Evan

# import matplotlib
# matplotlib.use('agg')
import numpy as np
import logging
from neon.util.argparser import NeonArgparser
from data_loaders import load_dayabaysingle
from neon.data import DataIterator
from neon.initializers import Uniform
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Affine, Linear, GeneralizedCost
from neon.transforms.activation import Tanh, Identity
from neon.transforms.cost import SumSquared, MeanSquared
from neon.models import Model
import os
from neon.callbacks.callbacks import Callbacks, LossCallback
import glob
from os.path import join
import pickle
import h5py
from tsne_visualize import Vis
from util.helper_fxns import save_orig_data, adjust_train_val_test_sizes, save_middle_layer_output





def main(bneck_width=10, n_layers=3):
    #####setup directories and args and logging
    logger = logging.getLogger()

    parser = NeonArgparser()

    model_files_dir = './model_files/fc-ae'
    final_dir = './results/fc-ae'
    output_dir = './intermediate_metrics'
    dirs = [model_files_dir, final_dir, output_dir]

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    parser.add_argument('--h5file')
    parser.add_argument('--test')
    parser.add_argument('--just_test')

    parser.set_defaults(batch_size=100,
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
    X_train, y_train, X_val, y_val, X_test, y_test = adjust_train_val_test_sizes(args.batch_size, X_train, y_train, X_val, y_val, X_test, y_test)
    nin = X_train.shape[1]

    train_set = DataIterator(X_train)
    valid_set = DataIterator(X_val)
    if args.test:
        train_set = DataIterator(np.vstack((X_train, X_val)))
        test_set = DataIterator(X_test)



    #Peter initially used AutoUniformGen(), which calculated weights based on activation fxn and size of layer, is deprecated
    init_uni = Uniform(low=-0.1, high=0.1)

    lr_decay_factor = -0.85 #Peter had this as spearmint determined value in old code
    opt_gdm = GradientDescentMomentum(learning_rate=0.001,
                                      momentum_coef=0.5, #Peter had a more complicated momentum set up which is not implemented in new neon
                                      schedule=Schedule(step_config=1,
                                                        change=1.0 - 10 ** lr_decay_factor))
    activation = Tanh()
    conv = dict(init=init_uni, batch_norm=True, activation=Tanh())
    layers = \
    [Affine(nout=284, **conv),
    Affine(284, **conv),
    Affine(nout= bneck_width, init=init_uni, act_name='middleLayer',activation=Tanh()),
    Affine(nout=284, **conv),
    Affine(nout=284, **conv),
    Affine(nout=nin, init=init_uni,activation=Tanh())]

    cost = GeneralizedCost(costfunc=SumSquared())
    ae = Model(layers=layers)
    ##################

    ae_model_key = '{0}-{1}-{2}-{3}-{4}'.format(str(nin),
                                            '-'.join([str(l[0].nout) if isinstance(l, list) else str(l.nout) for l in layers[:(len(layers) / 2)]]),
                                            str(activation).split('object')[0].split('.')[-1][:-1],
                                            os.path.splitext(os.path.basename(args.h5file))[0],'rot')

    args.save_path += '/' + ae_model_key + '-checkpt.pkl'
    args.output_file += '/' + ae_model_key + '-metrics.h5'
    final_h5_file = join(final_dir, ae_model_key + '-' + str(args.epochs) + ('-test' if args.test else '') + '-final.h5')
    h5fin = h5py.File(final_h5_file, 'w')

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


    save_orig_data(h5fin,X_train, y_train, X_val,y_val, X_test, y_test)

    #save intermeidate layer values
    h5ae_tr = h5fin.create_dataset('fc-ae/train/x', (X_train.shape[0], bneck_width))
    save_middle_layer_output(train_set, h5ae_tr, ae, bneck_width)


    h5ae_val = h5fin.create_dataset('fc-ae/val/x', (X_val.shape[0], bneck_width))
    save_middle_layer_output(valid_set, h5ae_val, ae, bneck_width)

    #val_loss_data = h5py.File(args.output_file)['cost/loss'][:]

    #todo add val intermediate metrics to results file which already contains train loss metrics
    #h5fin.create_dataset('cost/loss_val',(args.epochs / args.eval_freq,), data=val_loss_data)

    ts = Vis(final_h5_file, reconstruct=False)
    ts.plot_tsne()

    h5fin.close()

    #save model file
    pickle.dump(ae.serialize(), open(join(model_files_dir, '%s-%s.pkl'%(ae_model_key, str(args.epochs))), 'w'))

if __name__ == "__main__":

    main()












