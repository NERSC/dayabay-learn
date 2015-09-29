# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""
Example that creates and uses a network without a configuration file.
"""

import neon2 as neon

import numpy as np
from numpy import array
import logging
import neon.backends
from neon.backends import gen_backend
from neon.layers import FCLayer, DataLayer, CostLayer, DropOutLayer
from neon.models import MLP, Autoencoder
from neon.transforms import RectLeaky, RectLin, Linear, Logistic, CrossEntropy, Tanh, Softmax, SumSquaredDiffs
from neon.experiments import FitPredictErrorExperiment
from neon.metrics import LogLossMean, LogLossSum, MisclassRate, AUC, MSE, SSE
import neon.params
#import ipdbf
#from higgs import Higgs
#import dataset_hdf5
#import dataset_root
import os
# from neon2.datasets import dayabay
from dayabay_dataset_code import dayabay
from neon2.experiments.predict import PredictExperiment
import sys
#logging = {'level': 20, 'format': '%(asctime)-15s %(levelname)s:%(module)s - %(message)s'}
logging.basicConfig(level=20)
logger = logging.getLogger()
print sys.argv[1]
def create_model(hp):
    nin = 192
    weight_init = neon.params.AutoUniformValGen()
    lrule = {'type': 'gradient_descent_momentum','lr_params' : {
                'learning_rate':  0.001, #0.0000001, #0.01,
                'schedule': {'type': 'step','ratio': hp['lr_decay'],'step_epochs': 1},
                #'weight_decay': 0.00001,
                'momentum_params': {'type': 'linear_monotone', 
                                    'initial_coef': 0.5,
                                    'saturated_coef': 0.99,
                                    'start_epoch': 0,
                                    'saturate_epoch': 50,
                                    },
                 }
            }
    #lrule = {'type': 'rmsprop','lr_params': {'learning_rate': 0.001,},}
    layers = []
    layers.append(DataLayer(nout=nin, name='input'))
    
    if False:
        for i in range(hp['nlayers']):
            layername = 'h%d' % i
            layers.append(FCLayer(nout=hp['nhid'], weight_init=weight_init, lrule_init=lrule, activation=hp['activation'], name=layername))
            if i >= hp['nlayers'] - hp['nlayers_dropout']:
                layers.append(DropOutLayer(keep=0.5))
        weight_init_top = weight_init #params.UniformValGen(low=-0.001, high=0.001)
        #layers.append(FCLayer(nout=nin, weight_init=weight_init_top, lrule_init=lrule, activation=Linear(), name='y'))
        #layers.append(FCLayer(nout=nin, weight_init=weight_init, lrule_init=lrule, activation=RectLeaky(slope=0.01), name='y'))
        layers.append(FCLayer(nout=5, weight_init=weight_init_top, lrule_init=lrule, activation=Softmax(), name='y'))
        layers.append(CostLayer(cost=CrossEntropy(), name='costlayer')) # CrossEntropy or SumSquaredDiffs
    
    # Autoencode
    for i in range(hp['nlayers'] * 2 - 1):
        layername = 'h%d' % i
        if i == hp['nlayers'] - 1:        
            layers.append(FCLayer(nout=10, weight_init=weight_init, lrule_init=lrule, activation=Linear(), name=layername))
        else:
            layers.append(FCLayer(nout=hp['nhid'], weight_init=weight_init, lrule_init=lrule, activation=hp['activation'], name=layername))
        #if i >= hp['nlayers'] - hp['nlayers_dropout']:
        #    layers.append(DropOutLayer(keep=0.5))
    weight_init_top = neon.params.UniformValGen(low=-0.001, high=0.001)
    layers.append(FCLayer(nout=nin, weight_init=weight_init_top, lrule_init=lrule, activation=Linear(), name='y'))
    layers.append(CostLayer(cost=SumSquaredDiffs(), name='costlayer'))
    model = MLP(num_epochs=hp['max_epochs'],
                batch_size=hp['batch_size'],
                layers=layers, 
                serialized_path='./saved_params/%s.prm' % hp['string'],
                serialize_schedule=10,
                #deserialized_path='./debug.prm', overwrite_list=['num_epochs'],
                )
    return model


def run(spearminthp):
    # Run experiment.
    # Interpret spearmint params.
    hp = {}
    hp['nlayers'] = int(spearminthp['nlayers'])
    hp['nlayers_dropout'] = 0 #int(spearminthp['nlayers'])
    hp['nhid'] = int(spearminthp['nhid'])
    hp['lr_decay'] = 1.0 - 10**float(spearminthp['lr_decay_factor'])
    hp['activation'] = Tanh() #RectLin()
    hp['batch_size'] = 100
    hp['max_epochs'] = 100
    hp['string'] = 'ae_%d_%d_%0.3f' % (hp['nlayers'], hp['nhid'], hp['lr_decay']) 
    
    # Train model.
    model = create_model(hp)
    backend = gen_backend(rng_seed=0)
    h5file = sys.argv[1]
    print h5file
    #h5file = './jialin_data/small_dayabay1.h5'
   # h5file = '/global/homes/p/pjsadows/data/dayabay/single/single_20000.h5' # 5 classes of 20000 examples each
    dataset = dayabay.Imageset(h5file=h5file, mode='c', save_dir='/global/homes/r/racah/projects/dayabay-learn/intel_data/pkls', autoencode_flag=True)
    #metrics = {'train':[LogLossSum(), MisclassRate()], 'validation':[LogLossMean()], 'test':[]}
    #metrics = {'train':[LogLossMean(), MisclassRate()], 'validation':[LogLossMean(), MisclassRate()], 'test':[]}
    metrics = {'train':[MisclassRate(), MSE()], 'validation':[MisclassRate(), MSE()], 'test':[]}
    #experiment = FitPredictErrorExperiment(model=model,
    #                                       backend=backend,
    #                                       dataset=dataset,
    #                                       metrics=metrics,
    #                                       timing=True,
    #                                       #predictions=['validation']
    #                                       )
    experiment = PredictExperiment(model=model,
                                   backend=backend,
                                   dataset=dataset,
                                   metrics=metrics,
                                   timing=True,
                                   predictions=['train'],
                                   layeridx = 3,
                                   ) 
    # Experiment result is dict: result[metric_set][metric_name]
    #import pdb
    #pdb.set_trace()
    experiment.run()
    #for setname in result.keys():
    #    for metricname in result[setname].keys():
    #        print '%s_%s: %f' % (setname, metricname, result[setname][metricname])
    #loss = result['validation']['MSE']
    #return loss
    return 0

def main(job_id, params):
    '''
    This function is called by spearmint to train and test a network.
    Anything printed here will end up in the output directory for job job_id.
    '''
    print params # params are dict with array values.
    loss = run(params)
    return np.float64(loss) # Note: float32 causes InvalidDocument error in mongodb and spearmint compression.

if __name__=='__main__':
    # Train a single network with metaparameters specified here.
    params = {}
    #ae_3_284_0.990.prm
    params = {
              u'nlayers': array([3]),
              u'nhid': array([284]),
              u'lr_decay_factor': array([-2.0]),
              }
    job_id = 0
    loss = main(job_id, params)
    print loss


