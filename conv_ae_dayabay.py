__author__ = 'racah'
from neon.util.argparser import NeonArgparser
from data_loaders import load_dayabay_conv
import numpy as np

from neon.data import DataIterator, load_mnist
from neon.initializers import Uniform
from neon.layers import Conv, Pooling, GeneralizedCost, Deconv, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, SumSquared, Tanh
from neon.callbacks.callbacks import Callbacks, LossCallback
from neon.util.argparser import NeonArgparser
from he_initializer import HeWeightInit
import h5py
import os
import pickle
from dayabay_autoencoder import save_middle_layer_output
# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

num_epochs = args.epochs
final_dir='./results'
model_files_dir='./model_files'
bneck_width = 10
path='./'
(X_train, y_train), (X_val,y_val), (X_test, y_test), nclass = load_dayabay_conv(path=path)

# Set input and target to X_train
train_set = DataIterator(X_train, lshape=(2, 8, 24))
valid_set = DataIterator(X_test, lshape=(2, 8, 24))

# Initialize the weights and the learning rule
w_init = HeWeightInit() #Uniform(low=-0.1, high=0.1)
opt_gdm = GradientDescentMomentum(learning_rate=0.001, momentum_coef=0.9, wdecay=0.0005)

conv = dict(strides=1, init=w_init, activation=Tanh())#, batch_norm=True)
dconv = dict(init=w_init, strides=2, padding=1)

#TODO: figure out layers
# Define the layers
layers = [Conv((3,3,8), **conv),
          Pooling((2,2), strides=2),
          Conv((4, 4, 32), **conv),
          Pooling((2,2), strides=2),
          Affine(bneck_width,init=w_init,bias=None, batch_norm=False,activation=Tanh()),
          Deconv((3, 3, 8), **dconv),
          Deconv((3, 3, 8), **dconv),
          Deconv((4, 4, 1), **dconv)]

# Define the cost
cost = GeneralizedCost(costfunc=SumSquared())

mlp = Model(layers=layers)
# Fit the model

# configure callbacks
callbacks = Callbacks(mlp, train_set, **args.callback_args)
model_key = '{0}-{1}-{2}-{3}'.format(X_train.shape[1],'-'.join([(l.name[0] if 'Bias' not in l.name and 'Activation' not in l.name else '') +
 ('-' + str(l.fshape) if 'Pooling' in l.name or 'Conv' in l.name else '') for l in mlp.layers.layers]), str(args.epochs), str(X_train.shape[0]))

h5fin = h5py.File(os.path.join(final_dir,'ConvAE' + '-' + model_key + '-' + str(args.epochs) + ('-test' if args.test else '') + '-final.h5'), 'w')
h5fin.create_dataset('train_raw_x', data=X_train)
h5fin.create_dataset('train_raw_y', data=y_train)
h5fin.create_dataset('test_raw_x', data=X_test)
h5fin.create_dataset('test_raw_y', data=y_test)
h5fin.create_dataset('val_raw_x', data=X_val)
h5fin.create_dataset('val_raw_y', data=y_val)



#add callback for calculating train loss every every <eval_freq> epoch
callbacks.add_callback(LossCallback(h5fin.create_group('train_loss'), mlp, eval_set=train_set, epoch_freq=args.eval_freq))

callbacks.add_callback(LossCallback(h5fin.create_group('valid_loss'), mlp, eval_set=valid_set, epoch_freq=args.eval_freq))



mlp.fit(train_set, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)


h5ae_val = h5fin.create_dataset('val_ae_x', (X_val.shape[0], bneck_width))
save_middle_layer_output(valid_set, h5ae_val, mlp)

reconstructed_val = mlp.get_outputs(valid_set)
h5fin.create_dataset('reconstructed_val', data=reconstructed_val)

h5fin.close()
pickle.dump(mlp.serialize(), open(os.path.join(model_files_dir, '%s-%s.pkl'%(model_key, str(args.epochs))), 'w'))

