__author__ = 'racah'
from neon.util.argparser import NeonArgparser
from data_loaders import load_dayabay_conv
import numpy as np
from neon.data import DataIterator, load_mnist
from neon.initializers import Uniform, Gaussian, GlorotUniform
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
from tsne_visualize import TsneVis
from dayabay_autoencoder import save_middle_layer_output
import matplotlib
from helper_fxns import plot_train_val_learning_curve, save_orig_data
matplotlib.use('agg')
from matplotlib import pyplot as plt
# parse the command line arguments
parser = NeonArgparser(__doc__)

parser.add_argument('--h5file')
parser.add_argument('--test')
final_dir='./results/conv-ae'
model_files_dir='./model_files/conv-ae'
dirs = [final_dir, model_files_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)
bneck_width = 10

parser.set_defaults(batch_size=128,h5file='/global/homes/p/pjsadows/data/dayabay/single/single_20000.h5',
                    serialize=2,epochs=100,model_file=False,eval_freq=1, test=False)
args = parser.parse_args()
num_epochs = args.epochs

(X_train, y_train), (X_val,y_val), (X_test, y_test), nclass = load_dayabay_conv(path=args.h5file,clev_preproc=True, seed=5)

train_set = DataIterator(X_train, lshape=(2, 8, 26), make_onehot=False)
valid_set = DataIterator(X_val, lshape=(2, 8, 26), make_onehot=False)


w_init = HeWeightInit()
opt_gdm = GradientDescentMomentum(learning_rate=0.001, momentum_coef=0.9)

conv = dict(strides=1, init=w_init, padding={'pad_w': 0, 'pad_h':1}, activation=Rectlin())#, batch_norm=True)
dconv = dict(init=w_init, strides=2, padding=0)

#change kernal size to 5x5
layers = [Conv((3, 3, 16), **conv), #8,26,2 -> 8,24,
          Pooling((2, 2), strides=2),# -> 4,12,
          Conv((3, 3, 2), **conv), # -> 4,10,
          Pooling((2, 2), strides=2), #-> 2,5
          Conv((2, 5, bneck_width), init=w_init, strides=1, padding=0, activation=Rectlin(), conv_name="middleLayer"),#-> 1,1,10 like an FC layer
          Deconv((2, 4, 16), **dconv), #-> 2,4,
          Deconv((2, 5, 16), init=w_init, strides=2, padding=0), #-> 4,12
          Deconv((2, 6, 2), **dconv)] #->8,26,


# Define the cost
cost = GeneralizedCost(costfunc=SumSquared())
mlp = Model(layers=layers)



model_key = '{0}-{1}-{2}-{3}-{4}'.format(X_train.shape[1],'-'.join([(l.name[0] if 'Bias' not in l.name and 'Activation' not in l.name else '') +
 ('-' + '_'.join(str(l.fshape).split(' ')) if 'Pooling' in l.name or 'Conv' in l.name or 'conv' in l.name else '') for l in mlp.layers.layers]), str(args.epochs), str(X_train.shape[0]),str(X_train.shape[1]))

args.save_path = model_files_dir + '/' + model_key + '.pkl'
final_h5_file = os.path.join(final_dir,'ConvAE' + '-' + model_key + '-' + str(args.epochs) + ('-test' if args.test else '') + '-final.h5')
h5fin = h5py.File(final_h5_file, 'w')

callbacks = Callbacks(mlp, train_set, args) #**args.callback_args)
callbacks.add_callback(LossCallback(h5fin.create_group('train_loss'), mlp, eval_set=train_set, epoch_freq=args.eval_freq))
callbacks.add_callback(LossCallback(h5fin.create_group('valid_loss'), mlp, eval_set=valid_set, epoch_freq=args.eval_freq))

# Fit the model
mlp.fit(train_set, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

h5ae_val = h5fin.create_dataset('val_ae_x', (X_val.shape[0], bneck_width))
save_middle_layer_output(valid_set, h5ae_val, mlp, bneck_width)
reconstructed_val = mlp.get_outputs(valid_set)
h5fin.create_dataset('reconstructed_val', data=reconstructed_val)

save_orig_data(h5fin,X_train, y_train, X_val,y_val, X_test, y_test)
plot_train_val_learning_curve(h5fin,final_h5_file)
h5fin.close()

ts = TsneVis(final_h5_file)
ts.plot_tsne()

pickle.dump(mlp.serialize(), open(os.path.join(model_files_dir, '%s-%s.pkl'%(model_key, str(args.epochs))), 'w'))


