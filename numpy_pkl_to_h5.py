import sys
import h5py
import numpy as np
import pickle
import os

path_to_pkl = sys.argv[1]
path_to_h5_input = sys.argv[2]
ae = pickle.load(open(path_to_pkl))
orig_data = h5py.File(path_to_h5_input, 'r')
last_col = np.asarray(orig_data['test'][: , -1])
ae_l = np.hstack((ae, last_col.reshape((last_col.size,1))))
output_filename = os.path.basename(path_to_pkl).split('.')[0] + '.h5'
h5f = h5py.File(output_filename,'w')
h5f.create_dataset('autoencoded', data=ae_l)
h5f.close()



