__author__ = 'racah'
import h5py
import sys
import numpy as np


event_types = ['ibd_prompt', 'ibd_delay', 'muon', 'flasher', 'other' ]
event_dict = {i: ev for i, ev in enumerate(event_types)}

h5file_path = sys.argv[1]
reconstr_val = np.asarray(h5py.File(h5file_path)['reconstructed_val'])
init_val_x = np.asarray(h5py.File(h5file_path)['val_raw_x'])
val_y = np.asarray(h5py.File(h5file_path)['val_raw_y'])

d = {}

