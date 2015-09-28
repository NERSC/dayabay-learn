import sys
import h5py
import numpy as np
import pickle
import os

path_to_pkl = sys.argv[1]
path_to_h5 = sys.argv[2]
a = pickle.load(open(path_to_pkl))
output_filename = path_to_pkl.split('/')[1].split('.')[0] + '.h5'
h5f = h5py.File(output_filename,'w')
h5f.create_dataset('autoencoded', data=a)
h5f.close()

