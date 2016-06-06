__author__ = 'racah'
import numpy as np
import os
import pickle
import sys
from neon.util.argparser import NeonArgparser

import h5py
import matplotlib
from sklearn.manifold import TSNE
import numpy as np
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from conv_ae import ConvAe
from vis.viz import Viz
from util.helper_fxns import adjust_train_val_test_sizes
from util.data_loaders import load_ibd_pairs


from util.data_loaders import load_dayabay_conv




# 1) Primary AD           10000 or 1
# 2) Delayed AD response  01000 or 2
# 3) Muon decay           00100 or 3
# 4) Flasher              00010 or 4
# 5) Other (background noise) 00001 or 5


# In[118]:

#neon makes you set things up with command line args, so here we set all the defaults
def setup_parser():
        # parse the command line arguments
        parser = NeonArgparser(__doc__)
        #main data file
        parser.add_argument('--h5file')
        
        #whether you are doing test or just train and val
        parser.add_argument('--test')
        
        parser.add_argument('--learn_rate')
        
        #hidden layer dimension for bottle neck of autoencoder
        parser.add_argument('--bneck_width')
        
        #how many iterations of tsne
        parser.add_argument('--max_tsne_iter')
        parser.set_defaults(batch_size=128,
                            h5file='/project/projectdirs/dasrepo/single_20000.h5',
                            
                            #take a snapshot every n epochs where n is value fo serialize
                            serialize=2, 
                            
                            #how many epochs
                            epochs=100, 
                            learn_rate=0.0001, 
                            
                            #where to save serialized weights for a model snapshot 
                            model_file=False,
                            
                            #how frequently to log validation data
                            eval_freq=1, 
                            test=False, 
                            
                            #where to save any results
                            save_path='./results/model_files/conv-ae2',
                            
                            #these flags are for doing the preprocessing geometrical tricks 
                            wrap_pad_trick=False, 
                            cylinder_local_trick=False, 
                            
                            bneck_width=10, 
                            max_tsne_iter=500)

        args = parser.parse_args()
        
        #convert strings to floats and ints
        args.learn_rate = float(args.learn_rate)
        args.max_tsne_iter = int(args.max_tsne_iter)
        
        return args




if __name__ == "__main__":
    
    #sys.argv = sys.argv[5:] # only for iPython to skip all the ipython command line arguments
    args = setup_parser()
    #load data from hdf5, preprocess and split into train and test
    preprocess = True
    train, val, test = load_ibd_pairs(args.h5file, preprocess=preprocess)

    train, _, val, _, test, _  = adjust_train_val_test_sizes(args.batch_size,
        train, train, val, val, test, test)

    #class for networks architecture
    cae = ConvAe(4, args)

    #uses scikit-learn interface (so this trains on X_train)
    cae.fit(train)

    #extract the hidden layer outputs when running x_val thru autoencoder
    feat = cae.extract(val)
    gr_truth = np.ones(val.shape[0])

    v = Viz(gr_truth)

    # take first two principal components of features, so we can plot easily
    #normally we would do t-SNE (but takes too long for quick demo)
    x_pc = v.get_pca(feat)

    #plot the 2D-projection of the features
    v.plot_features(x_pc,save=True)

