import os
import pickle
import sys
import h5py
import matplotlib
from sklearn.manifold import TSNE
import numpy as np
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

class Viz(object):
    def __init__(self, gr_truth=None, nclass=5):
        self.nclass = nclass
        self.gr_truth = gr_truth
        self.labels = ['ibd_prompt', 'ibd_delay', 'muon', 'flasher', 'other']
        self.colors = matplotlib.colors.cnames.keys()
        
    def plot_features(self, x_2d, save='False'):
        assert self.gr_truth is not None, "We need ground truth for plots!"
        plt.rcParams['figure.figsize'] = 15, 10
        for i in range(self.nclass):
            x = x_2d[self.gr_truth == i]
            plt.scatter(x[:,0], x[:,1],color=self.colors[i], label=self.labels[i])
        plt.legend(loc='lower left', ncol= 5, fontsize=12)
        #plt.show()
        if save:
            self.save(name='feat_scatter')

        
    def save(self, name='untitled_plot',savedir='./results/plots'):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        save_path = os.path.join(savedir,name + '.jpg')
        print save_path
        plt.savefig(save_path)
    
#     def postprocess(self, key, feat):
#         if key.lower() == 'pca':
#             return self.get_pca(feat)
    def get_pca(self, feat):
        p = PCA(2) #2 principal components
        x = p.fit_transform(feat)
        return x
    
    def get_tsne(self,feat, **kwargs):
        '''very slow'''
        default_args = dict(n_components=2, perplexity=50,n_iter=1000)
        for k,v in default_args.iteritems():
            if k not in kwargs:
                kwargs[k] = v
        ts = TSNE(**kwargs)
        x_ts = ts.fit_transform(feat.astype('float64'))
        return x_ts
    def plot_reconstruction(self,x_inp,x_reconstructed,indx=0, save=False):
        plt.figure(1)
        plt.clf()
        ims = [x_inp, x_reconstructed]
        info = ['original', 'reconstructed']
    
        if self.gr_truth is not None:
            # the ground truth class number indexes the class names or labels
            cls = self.labels[self.gr_truth[indx]]
        else:
            cls= ''
        for i in range(len(ims)):
            sp = plt.subplot(len(ims),1,i + 1)
            im = ims[i].reshape(8,24)
            sp.imshow(im,interpolation='none' )
            sp.set_title(cls + ' ' + info[i])
        if save:
            self.save(name='_'.join([cls, str(indx), 'reconstruction']))