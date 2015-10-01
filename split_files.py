__author__ = 'racah'
import sys
import h5py
import os

h5file = sys.argv[1]
basename = os.path.splitext(os.path.basename(h5file))[0]
extension = os.path.splitext(os.path.basename(h5file))[1]
h5_data = h5py.File(h5file)['inputs']

classes = int(h5_data[-1,-1])
examples = h5_data.shape[0]
ind = examples / classes
for i in range(classes):
    x = h5_data[i * ind:(i+1) * ind]
    h5f = h5py.File(basename + '-' + str(i+1) + '.' + extension, 'w')
    h5f.create_dataset('inputs', data=x)

