#Evan
import numpy as np
import logging
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from data_loaders import load_dayabaysingle
from neon.data import DataIterator
from neon.initializers import Uniform
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Affine, Linear, GeneralizedCost
from neon.transforms.activation import Tanh
from neon.transforms.cost import SumSquared
from neon.models import Model
import os
from neon.callbacks.callbacks import Callbacks


logger = logging.getLogger()

parser = NeonArgparser()

parser.add_argument('--h5file', '--batch_size', '--test')
parser.set_defaults(batch_size=100, test=False) #h5file=?

args = parser.parse_args()

be = gen_backend(backend=args.backend,
                 batch_size=args.batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype,
                 stochastic_round=False)


#load and split all data from file
(X_train, y_train), (X_val, y_val), (X_test, y_test), nclass = load_dayabaysingle(path=args.h5file)

nin = X_train.shape[1]
#DataIterator(X, y=None, nclass=None, lshape=None, make_onehot=True)
#if y unspecified -> works like autoencoder
train_set = DataIterator(X_train)
valid_set = DataIterator(X_val)

#Bengio recommends
#Peter initially used AutoUniformGen(), which calculated weights based on activation fxn and size of layer, is deprecated
init_uni = Uniform(low=-0.1, high=0.1)

lr_decay_factor = -0.85 #Peter had this as spearmint determined value in old code
opt_gdm = GradientDescentMomentum(learning_rate=0.001,
                                  momentum_coef=0.5, #Peter had a more complicated momentum set up which is not implemented in new neon
                                  schedule=Schedule(step_config=1,
                                                    change=1.0 - 10 ** lr_decay_factor))

n_layers = 3 #another variable determiend by spearmint
layers = []
wide = Affine(nout=284, init=init_uni, batch_norm=True, activation=Tanh())
for i in range(3):
    layers.append(wide)
layers.append(Linear(nout=10, init=init_uni))
for i in range(3):
    layers.append(wide)
layers.append(Linear(nout=nin, init=init_uni))

cost = GeneralizedCost(costfunc=SumSquared())

ae = Model(layers=layers)

if args.model_file:
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    logger.info('loading initial model state from %s' % args.model_file)
    ae.load_weights(args.model_file)

callbacks = Callbacks(ae, train_set, output_file=args.output_file, progress_bar=args.progress_bar)

if args.serialize > 0:
    checkpoint_schedule = args.serialize
    checkpoint_model_path = args.save_path
    callbacks.add_serialize_callback(checkpoint_schedule, checkpoint_model_path)

ae.fit(train_set, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)


#test
#get output from
for (x, t) in X_val:
    for i, l in enumerate(ae.layers):
        x =l.fprop(x)
        if i == (len(layers) - 1) / 2: #trying to get middle layer here
            b_neck = x
            break

print b_neck









