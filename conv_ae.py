__author__ = 'racah'
import os
import pickle
import sys
import datetime
from neon.data import ArrayIterator
from neon.layers import Conv, Pooling, GeneralizedCost, Deconv
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, SumSquared
from neon.callbacks.callbacks import Callbacks, LossCallback
from neon.util.argparser import NeonArgparser
import h5py
import matplotlib
import sklearn
from sklearn.manifold import TSNE
from util.data_loaders import load_dayabay_conv
from util.he_initializer import HeWeightInit
from vis.tsne_visualize import Vis
from util.helper_fxns import plot_train_val_learning_curve,\
    save_orig_data, \
    save_middle_layer_output, \
    adjust_train_val_test_sizes, create_h5_file, stop_func, get_middle_layer_output
import numpy as np
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# 1) Primary AD           10000 or 1
# 2) Delayed AD response  01000 or 2
# 3) Muon decay           00100 or 3
# 4) Flasher              00010 or 4
# 5) Other (background noise) 00001 or 5

class ConvAe(object):
    def __init__(self, args):
        self.final_dir='./results/output'
        self.model_files_dir='./results/model_files/conv-ae'
        self.setup_dirs(final_dir=self.final_dir, model_files_dir=self.model_files_dir)
        self.args = args
        self.eval_data_type = 'test' if self.args.test else 'val'
        #self.h5fin, self.final_h5_filename = create_h5_file(self.final_dir,
                                                            #self.args.epochs, self.args.learn_rate)

        self.train_set, self.eval_set = self._retrieve_data()

    def setup_dirs(self,**kwargs):
        for dir in kwargs.values():
            if not os.path.exists(dir):
                os.makedirs(dir)

                
    def _retrieve_data(self):
        (X_train, y_train), (X_val,y_val), \
        (X_test, y_test), nclass = load_dayabay_conv(path=self.args.h5file,
                                                    clev_preproc=False, seed=6, eq_class=True, get_y=True )

        X_train, y_train, X_val, y_val,\
        X_test, y_test = adjust_train_val_test_sizes(self.args.batch_size, X_train,
                                                     y_train, X_val, y_val, X_test, y_test)


        self.data_dict = {'train':(X_train, y_train), 'val': (X_val, y_val),'test': (X_test, y_test)}
        
        #save_orig_data(self.h5fin, X_train, y_train, X_val, y_val, X_test, y_test)
        self.num_tr_examples = X_train.shape[0]
        self.data_dim = X_train.shape[1]


        train_set = ArrayIterator(X_train, lshape=(1, 8, 24))
        valid_set = ArrayIterator(X_val, lshape=(1, 8, 24))

        if self.args.test:
            train_set = ArrayIterator(np.vstack((X_train, X_val)), lshape=(1, 8, 24), make_onehot=False)
            test_set = ArrayIterator(X_test, lshape=(1, 8, 24), make_onehot=False)
            return train_set, test_set
        else:
            return train_set, valid_set

    def get_data(self,key):
        return self.data_dict[key]
    
    def get_data_iterator(self, key):
        pass
    
    def setup_network(self):

        w_init = HeWeightInit()

        opt_gdm = GradientDescentMomentum(learning_rate=self.args.learn_rate, momentum_coef=0.9)

        conv = dict(strides=1, init=w_init, padding={'pad_w': 0, 'pad_h':1}, activation=Rectlin(),
                    batch_norm=False)#, batch_norm=True)
        dconv = dict(init=w_init, strides=2, padding=0, batch_norm=False)
        layers = [Conv((5, 5, 16), strides=1, init=w_init, padding=2, activation=Rectlin(), batch_norm=False)] #8,24,1-> 8,24,

        layers.extend([
                  Pooling((2, 2), strides=2),# -> 4,12,
                  Conv((3, 3, 16), **conv), # -> 4,10,
                  Pooling((2, 2), strides=2), #-> 2,5
                  Conv((2, 5, self.args.bneck_width), init=w_init, strides=1, padding=0, activation=Rectlin(name="middleLayer"),batch_norm=False),#-> 1,1,10 like an FC layer
                  Deconv((2, 4, 16), **dconv), #-> 2,4,
                  Deconv((2, 5, 16), init=w_init, strides=2, padding=0, batch_norm=False), #-> 4,11
                  Deconv((2, 4, 1), **dconv)] )#->8,24,


        # Define the cost
        cost = GeneralizedCost(costfunc=SumSquared())
        mlp = Model(layers=layers)
        return mlp, cost, opt_gdm

    def get_model_key(self, mlp):
        '''For naming of files. Should be changed'''
        model_key = 'conv' + str(datetime.date.today())
        return model_key

    def add_callbacks(self,mlp, eval_set):
        callbacks = Callbacks(mlp, eval_set=eval_set, **self.args.callback_args) # eval_set=eval_set the two callbacks below do this

        return callbacks
        

    def setup_results(self, mlp, train_set, eval_set):
        model_key = self.get_model_key(mlp)
        self.args.save_path = self.model_files_dir + '/' + model_key + '.pkl'

        return self.add_callbacks(mlp,eval_set)




    def evaluate(self,mlp, eval_set):
        pass

    def get_reconstructed(self):
        reconstructed_eval = self.mlp.get_outputs(self.eval_set)
        return reconstructed_eval
        
    def get_reconsruction_error(self):
        pass
    
    def save_and_plot(self,mlp):
        self.h5fin.close()
        plot_train_val_learning_curve(self.final_h5_filename)



        v = Vis(self.final_h5_filename, old=False, plot_tsne=True, reconstruct=False, pp_types='conv-ae,raw', data_types=self.eval_data_type, max_iter=self.args.max_tsne_iter)
        v.plot()

        pickle.dump(mlp.serialize(), open(os.path.join(self.model_files_dir, '%s-%s-%s.pkl'%(self.get_model_key(mlp), str(self.args.epochs), str(self.args.learn_rate))), 'w'))

    def extract_features(self):
        feat = get_middle_layer_output(self.eval_set, self.mlp, self.args.bneck_width)
        return feat
        
    
    def train(self):
        mlp, cost, opt_gdm = self.setup_network()
        callbacks = self.setup_results(mlp, self.train_set, self.eval_set)
        print "Training with %d training example" % (self.train_set.ndata)
        mlp.fit(self.train_set,  optimizer=opt_gdm, num_epochs=self.args.epochs, cost=cost, callbacks=callbacks)
        self.mlp = mlp