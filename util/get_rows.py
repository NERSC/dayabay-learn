__author__ = 'racah'
import sys
import numpy as np
import h5py
import os
import pickle
input_paths = sys.argv[1:]

'''get shape without having to fire up ipython'''
for path in input_paths:
    if os.path.splitext(path)[-1] == '.pkl':
        print pickle.load(open(path)).shape[0]

    elif os.path.splitext(path)[-1] == '.h5':
        h5f = h5py.File(path,'r')
        for key in h5f.keys():
            print h5f[key].shape[0]

    else:
        print "we don't support file format %s" % (os.path.splitext(path)[-1])



