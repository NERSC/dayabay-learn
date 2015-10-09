
import numpy as np
import pickle as pkl
import scipy
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import h5py
import subprocess
import os
from tsne_python import tsne
from bh_tsne import bhtsne
from matplotlib import pyplot as plt

base_path='/scratch3/scratchdirs/jialin/dayabay/'

file_number = 29155
event_types = ['muo', 'fla', 'oth']
post_process_types = ['ae', 'raw']
ae = {'path': base_path + 'autoencoded', 'prefix':'output-', 'suffix': '', 'h5_key': 'autoencoded'}
ae = {'path': './test-h5_files', 'prefix':'', 'suffix': '', 'h5_key': 'autoencoded'}
raw = {'path': '/scratch3/scratchdirs/jialin/dayabay/preprocess/', 'prefix':'output-', 'suffix': '', 'h5_key': 'inputs'}
raw = {'path': './raw-h5files', 'prefix':'', 'suffix': '', 'h5_key': 'inputs'}


for event in event_types:
    ae_file_path = os.path.join(ae['path'], ae['prefix'] + event  + ae['suffix'], str(file_number) + '.h5')
    raw_file_path = os.path.join(raw['path'], raw['prefix'] + event + raw['suffix'], str(file_number) + '.h5')
    ae[event] = h5py.File(ae_file_path,'r')[ae['h5_key']]
    ae[event + '_key'] = ae[event][0, -1]
    raw[event] = h5py.File(raw_file_path,'r')[raw['h5_key']]
    raw[event + '_key'] = raw[event][0, -1]


ae['combined'] = np.vstack(tuple(ae[k] for k in event_types))
raw['combined'] = np.vstack(tuple(raw[k] for k in event_types))

ae['tsne'] = tsne.tsne(ae['combined'][:, :-1]) #, 2, 10, 10.0) #b/c the last column is the label

raw['tsne'] = tsne.tsne(raw['combined'][:, :-1])#, 2, 10, 10.0) #b/c the last column is the label


colors = ['b','r','g', 'k', 'm']
markers = ['o','o','o','s', 's']
plt.figure(1)
for i, event in enumerate(event_types):
    tsne_event = ae['tsne'][ ae['combined'][:, -1] == ae[event + '_key']]
    plt.scatter(tsne_event[:, 0], tsne_event[:, 1], marker=markers[i], c=colors[i], alpha=0.5, label=event)
plt.legend(loc='upper right')
plt.savefig('./tsne-ae.jpg')

plt.figure(2)
for i, event in enumerate(event_types):
    tsne_event = raw['tsne'][ raw['combined'][:, -1] == raw[event + '_key']]
    plt.scatter(tsne_event[:, 0], tsne_event[:, 1], marker=markers[i], c=colors[i], alpha=0.5, label=event)
plt.legend(loc='upper right')
plt.savefig('./tsne-raw.jpg')

