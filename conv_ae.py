from Network import AbstractNetwork
from neon.data import ArrayIterator
from neon.layers import Conv, Pooling, GeneralizedCost, Deconv
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, SumSquared
from neon.callbacks.callbacks import Callbacks, LossCallback
from neon.util.argparser import NeonArgparser
from util.helper_fxns import save_middle_layer_output, get_middle_layer_output
import numpy as np
from operator import mul
from util.he_initializer import HeWeightInit
from neon.backends import gen_backend

# 1) Primary AD           10000 or 1
# 2) Delayed AD response  01000 or 2
# 3) Muon decay           00100 or 3
# 4) Flasher              00010 or 4
# 5) Other (background noise) 00001 or 5

class ConvAe(AbstractNetwork):
    def __init__(self, args, nchannels=1):
        self.args = args
        self.nchannels = nchannels
        self.model,self.cost, self.opt_gdm = self.setup_network()
        self.network = self.model
        self.optimizer = self.opt_gdm

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
                  Conv((2, 5, self.args.bneck_width),name='bottleneck', init=w_init, strides=1, padding=0, activation=Rectlin(),batch_norm=False),#-> 1,1,10 like an FC layer
                  Deconv((2, 4, 16), **dconv), #-> 2,4,
                  Deconv((2, 5, 16), init=w_init, strides=2, padding=0, batch_norm=False), #-> 4,11
                  Deconv((2, 4, self.nchannels), **dconv)] )#->8,24,


        cost = GeneralizedCost(costfunc=SumSquared())
        model = Model(layers=layers)
        return model, cost, opt_gdm
        
    def preprocess_data(self, x,y=None):
        x_flat = x.reshape(x.shape[0], reduce(mul, x.shape[1:]))
        if y:
            y_flat = y.reshape(y.shape[0], reduce(mul, y.shape[1:]))
            dataset = ArrayIterator(x_flat, y_flat, lshape=x.shape[1:])
        else:
            dataset = ArrayIterator(x_flat, lshape=x.shape[1:])
       
        return dataset

    
    def predict(self,dataset):
        return self._reconstruct(dataset)

    def _reconstruct(self, dataset):
        rec_set = self.preprocess_data(dataset)
        return self.model.get_outputs(rec_set)
        
    def evaluate(self, dataset):
        pass
    
    def extract(self, data, layer=None):
        extract_set = self.preprocess_data(data)
        feats = np.zeros((extract_set.ndata, self.args.bneck_width))
        for i, (x,t) in enumerate(extract_set):
            self.model.fprop(x)
            
            for l in self.model.layers.layers:
                if l.name == 'bottleneck_Rectlin':
                    feats[self.args.batch_size * i:self.args.batch_size * (i+1)] = l.outputs.T.asnumpyarray()
                    
        return feats
        
    
    def fit(self, data):
        train_set = self.preprocess_data(data)
        print "Training with %d training example" % (train_set.ndata)
        self.model.fit(train_set,  optimizer=self.opt_gdm,
        num_epochs=self.args.epochs, cost=self.cost,
            callbacks=Callbacks(self.model, eval_set=train_set, **self.args.callback_args))
