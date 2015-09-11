# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pickle as pkl
import scipy
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# <codecell>

# Autoencoded
fn = '/home/pjsadows/cloud/desktop/dayabay/train-targets.pkl'
X = pkl.load(open(fn,'r'))
fn = '/home/pjsadows/cloud/desktop/dayabay/train-inference.pkl'
Xae = pkl.load(open(fn,'r'))
print X.shape, Xae.shape
# Raw data loaded from dataset_root.
#Xr,Yr = pkl.load(open('/home/pjsadows/cloud/desktop/dayabay/data1000.pkl','r'))
#Xr,Yr = Xr.T, Yr.T 
#print Xr.shape, Yr.shape
import h5py
h5file= h5py.File('/home/pjsadows/cloud/desktop/dayabay/ibd_100.h5', 'r')
Xa = h5file['charges']
Xa_prompt = Xa[:,:192]
Xa_delayed = Xa[:,192:]
X_combined = np.vstack([X, Xa_prompt, Xa_delayed])

# <codecell>

def preprocess(X):
    # Simply take log of data and divide by scale factor.
    prelog = 1.0
    scale = 10.0 # log(500000) ~= 10
    X = np.maximum(X, np.zeros_like(X))
    X = np.log(X + prelog) / scale
    return X
flashers = np.loadtxt('/home/pjsadows/cloud/desktop/dayabay/flashers.csv')
detector = flashers[:,0]
triggernum = flashers[:,1]
flasher = flashers[:,2]
NominalCharge = flashers[:,3]
Xf = flashers[:,9:]
#Xf = np.log(np.maximum(Xf, np.zeros_like(Xf)) + 1.0)
Xf = preprocess(Xf)
print Xf.shape
#X,Y = pkl.load(open('/home/pjsadows/cloud/desktop/dayabay/data1000.pkl','r'))

# <codecell>

X =  X_combined
y = flasher
y[NominalCharge>=3000] = 2
y = np.hstack([y, 3*np.ones(Xa_prompt.shape[0]), 4*np.ones(Xa_delayed.shape[0])])

from pylab import *
import imp
tsne = imp.load_source('tsne', '/home/pjsadows/local/tsne.py')
#Y = tsne.tsne(X, no_dims, perplexity)
Xtsne = tsne.tsne(X.astype('float64'), 2, 50, 20.0)

figure(1)
clf()
labels = ['non-flasher noise', 'flasher', 'nominalCharge > 3000', 'ibd_prompt', 'ibd_delayed']
colors = ['b','r','g', 'k', 'm']
markers = ['o','o','o','s', 's']
sizes = [20,20,20,200,200]
for i in range(len(np.unique(y))):
    scatter(Xtsne[y==i,0], Xtsne[y==i,1], s=sizes[i], marker=markers[i], c=colors[i], alpha=0.5, label=labels[i])
legend(loc='upper right')

