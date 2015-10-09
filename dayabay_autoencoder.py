#Evan
import numpy as np
import logging
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from data_loaders import load_dayabaysingle
from neon.data import DataIterator
from neon.initializers import Uniform
from neon.optimizers import GradientDescentMomentum, Schedule

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

#DataIterator(X, y=None, nclass=None, lshape=None, make_onehot=True)
#if y unspecified -> works like autoencoder
train_set = DataIterator(X_train)

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
