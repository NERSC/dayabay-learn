__author__ = 'racah'
import sys
import h5py
import os
h5file = os.path.expandvars(sys.argv[1])

h5_f = h5py.File(h5file)
h5_data = h5_f[h5_f.keys()[0]]
class_label = int(h5_data[0,-1])
print class_label
