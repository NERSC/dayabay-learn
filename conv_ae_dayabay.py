__author__ = 'racah'
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




# 1) Primary AD           10000 or 1
# 2) Delayed AD response  01000 or 2
# 3) Muon decay           00100 or 3
# 4) Flasher              00010 or 4
# 5) Other (background noise) 00001 or 5




def setup_parser():
        # parse the command line arguments
        parser = NeonArgparser(__doc__)

        parser.add_argument('--h5file')
        parser.add_argument('--test')
        parser.add_argument('--learn_rate')
        parser.add_argument('--wrap_pad_trick')
        parser.add_argument('--cylinder_local_trick')
        parser.add_argument('--bneck_width')
        parser.add_argument('--max_tsne_iter')
        parser.set_defaults(batch_size=128,h5file='/global/homes/p/pjsadows/data/dayabay/single/single_20000.h5',
                    serialize=2, epochs=100, learn_rate=0.0001, model_file=False,eval_freq=1, test=False, save_path='./results/model_files/conv-ae',
                    wrap_pad_trick=False, cylinder_local_trick=False, bneck_width=10, max_tsne_iter=500)

        args = parser.parse_args()
        args.learn_rate = float(args.learn_rate)
        args.max_tsne_iter = int(args.max_tsne_iter)
        return args



if __name__ == "__main__":
    #sys.argv = sys.argv[5:] # only for iPython to skip all the ipython command line arguments
    args = setup_parser()
    #args.epochs = 1
    
    cae = ConvAe(args)
    cae.train()
    
    feat = cae.extract_features()
    x_orig, y_val = cae.get_data('val')
    gr_truth = np.argmax(y_val,axis =1) #convert from one-hot to normal
    
    v = Viz(gr_truth)
    
    x_pc = v.get_pca(feat)
    #x_ts = v.get_tsne(feat,n_iter=args.max_tsne_iter)
    v.plot_features(x_pc,save=True)
    x_rec = cae.get_reconstructed()
    #x_orig, _ = cae.get_data('val')
    v.plot_reconstruction(x_orig[2], x_rec[2], indx=10, save=True)

