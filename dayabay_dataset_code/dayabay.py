# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Generic image-like dataset able to be processed in macro batches.
Adopted from /usr/common/das/neon/CD/virtualenv/lib/python2.7/site-packages/neon/datasets/imageset.py
"""

import logging
import numpy as np
import os
import sys
from threading import Thread
import h5py

from neon.datasets.dataset import Dataset
from neon.util.param import opt_param, req_param
from neon.util.persist import deserialize

from dayabay_rotate import *

logger = logging.getLogger(__name__)


class MacrobatchDecodeThread(Thread):
    """
    Load and decode a macrobatch of images in a separate thread,
    double buffering.

    Hide the time to transpose and convert (astype).
    """

    def __init__(self, ds):
        ''' Takes Dataset object ds as input. '''
        Thread.__init__(self)
        self.ds = ds

    def preprocess(self):
        ''' Perform preprocessing (centering) on current macrobatch.'''
        b_idx = self.ds.macro_decode_buf_idx
        for mini_idx in range(self.ds.minis_per_macro[b_idx]):
            X = self.ds.img_mini_T[b_idx][mini_idx]
            # All the centering methods try to center the image on column 11.
            if not self.ds.mode:
                pass
            else:
                if self.ds.mode=='v':
                    amount = get_rot_amount(X)
                elif self.ds.mode=='e':
                    amount = get_rot_amount_maxelem(X) 
                elif self.ds.mode=='c':
                    amount = get_rot_amount_maxcol(X)
                X = rotate(X,amount)

            self.ds.img_mini_T[b_idx][mini_idx] = X
            
            #self.ds.img_mini_T[b_idx][mini_idx] = np.ones_like(X)


    def run(self):
        # Load dataset macrobatch from disk and maybe decode.
        bsz = self.ds.batch_size
        b_idx = self.ds.macro_decode_buf_idx
        macro = self.ds.get_macro_batch()
        betype = self.ds.backend_type

        # This macrobatch could be smaller than macro_size for last macrobatch
        mac_sz = macro['inputs'].shape[0] # nrec by nfeatures

        # Data minibatches may be processed before being used.
        img_macro = np.zeros((self.ds.macro_size, self.ds.npixels), dtype='float32')
        img_macro[:mac_sz, :] = macro['inputs']
        if mac_sz < self.ds.macro_size:
            img_macro[mac_sz:, :] = 0
        
        # Leave behind the partial minibatch
        self.ds.minis_per_macro[b_idx] = mac_sz / bsz
        self.ds.img_mini_T[b_idx] = \
            [None for mini_idx in range(self.ds.minis_per_macro[b_idx])]

        # Data is stored to self.ds.img_mini_T.
        for mini_idx in range(self.ds.minis_per_macro[b_idx]):
            s_idx = mini_idx * bsz
            e_idx = (mini_idx + 1) * bsz
            self.ds.img_mini_T[b_idx][mini_idx] = \
                img_macro[s_idx:e_idx].T.astype(betype, order='C')

            if self.ds.img_mini_T[b_idx][mini_idx].shape[1] < bsz:
                tmp = self.ds.img_mini_T[b_idx][mini_idx].shape[0]
                mb_residual = self.ds.img_mini_T[b_idx][mini_idx].shape[1]
                filledbatch = np.vstack((img_macro[s_idx:e_idx],
                                         np.zeros((bsz - mb_residual, tmp))))
                self.ds.img_mini_T[b_idx][mini_idx] = \
                    filledbatch.T.astype(betype, order='C')
        
       # Targets are kept as full macrobatch: ntargets x nrec.
        self.ds.tgt_macro[b_idx] = \
            macro['targets'].T if 'targets' in macro else None


        # Preprocessing (rotation)
        self.preprocess()

        return


class Imageset(Dataset):

    """
    Sets up a macro batched imageset dataset.

    Assumes you have the data already partitioned and in macrobatch format

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Keyword Args:
        repo_path (str, optional): where to locally host this dataset on disk
    """
    def __init__(self, **kwargs):
        
        # Peter params
        opt_param(self, ['h5file'], False)
        opt_param(self, ['mode'], None) # 'c' = center images on column of max charge
        opt_param(self, ['repo_path'], './')
        opt_param(self, ['nrec'], None) # Number of records to use.     
        opt_param(self, ['autoencode_flag'], False) # Return X,X pairs instead of X,y 
        opt_param(self, ['intel'], False)
        opt_param(self, ['preprocess_done'], False)
        opt_param(self, ['mean_norm', 'unit_norm'], False)
	opt_param(self,['all_train'], False)
        opt_param(self, ['num_workers'], 6)
        opt_param(self, ['backend_type'], 'np.float32')

        self.__dict__.update(kwargs)

        if self.backend_type in ['float16', 'np.float16', 'numpy.float16']:
            self.backend_type = np.float16
        elif self.backend_type in ['float32', 'np.float32', 'numpy.float32']:
            self.backend_type = np.float32
        else:
            raise ValueError('Datatype not understood')
        logger.warning("Imageset initialized with dtype %s", self.backend_type)
        req_param(self, ['h5file', 'save_dir'])

                

    def __getstate__(self):
        """
        Defines what and how we go about serializing an instance of this class.
        """
        self.macro_decode_thread = None
        return self.__dict__

    def __setstate__(self, state):
        """
        Defines how we go about deserializing into an instance of this class.
        """
        self.__dict__.update(state)

    def load(self, backend=None, experiment=None, batch_size=None):
        '''
        Called at beginning of experiment. Use this to detect some important 
        quantities:
        1) Number of input output dims. 
        2) Macrobatch size.
        3) Train,test split.
        '''
        f = h5py.File(self.h5file, 'r')
        if self.intel:
            self.nfeatures = f['inputs'].shape[1] - 1
            self.ntargets = 3 #f['targets'].shape[1]

        else:
            self.nfeatures = f['inputs'].shape[1]
            self.ntargets = f['targets'].shape[1]


        if self.autoencode_flag:
            self.ntargets = self.nfeatures    

        if self.nrec:
            self.nrec = min(self.nrec, f['inputs'].shape[0])
        else:
            self.nrec = f['inputs'].shape[0]
        # Parameters for determining the train/test split.
        self.macro_size = experiment.model.batch_size  # Seems reasonable.
        #assert np.mod(self.nrec, self.macro_size) == 0, \
        #'Data partition does not fit nicely: ' + \
        #'nexamples=%d, macrobatchsize=%d, minibatchsize=%d' % \
        #(self.nrec, self.macro_size, experiment.model.batch_size)
        nmacros = self.nrec / self.macro_size
	
        if self.all_train:
            partition = {'train': 1, 'valid': 0.2, 'test': 0.2}
        else:
            partition = {'train': 0.6, 'valid': 0.2, 'test': 0.2}
	
        self.ntrain = int(nmacros * partition['train']) # ntrain is in number of macrobatches
        self.train_start = 0
        self.train_nrec = self.ntrain * self.macro_size
        self.maxtrain = self.ntrain + self.train_start - 1
        
        self.nval = int(nmacros * partition['valid'])
        self.val_start = self.ntrain
        self.val_nrec = self.nval * self.macro_size
        self.maxval = self.nval + self.val_start - 1

        self.ntest = int(nmacros * partition['test'])
        self.test_start = self.ntrain + self.nval
        self.test_nrec = self.ntest * self.macro_size
        self.maxtest = self.ntest + self.test_start - 1

        self.macro_idx = 0
        return

    def get_macro_batch(self):
        self.macro_idx = (self.macro_idx + 1 - self.startb) \
            % self.nmacros + self.startb
        
        file_handle = h5py.File(self.h5file, "r")
        s_idx = self.macro_idx * self.macro_size
        e_idx = (self.macro_idx + 1) * self.macro_size
        macro_batch = {}

        if self.intel:
            macro_batch['inputs'] = file_handle['inputs'][s_idx:e_idx, :-1] #b/c Jialin's data has label last column
            macro_batch['targets'] = file_handle['inputs'][s_idx:e_idx, -1] #jialin's label, not one hot encoding, but does not matter here b/c this isnt used for autoencoding
        else:
            macro_batch['inputs'] = file_handle['inputs'][s_idx:e_idx]
            macro_batch['targets'] = file_handle['targets'][s_idx:e_idx]
        file_handle.close()

        if self.autoencode_flag:
            macro_batch['targets'] = macro_batch['inputs']

        return macro_batch
        #self.macro_idx = (self.macro_idx + 1 - self.startb) \
        #    % self.nmacros + self.startb
        #fname = os.path.join(self.save_dir,
        #                     'data_batch_{:d}'.format(self.macro_idx))
        #return deserialize(os.path.expanduser(fname), verbose=False)
        

    def del_mini_batch_producer(self):
        if self.macro_decode_thread is not None:
            self.macro_decode_thread.join()
        del self.inp_be

    def init_mini_batch_producer(self, batch_size, setname, predict=False):
        # local shortcuts
        sbe = self.backend.empty
        betype = self.backend_type
        sn = 'val' if (setname == 'validation') else setname
        self.npixels = self.nfeatures

        self.startb = getattr(self, sn + '_start')
        self.nmacros = getattr(self, 'n' + sn)
        self.maxmacros = getattr(self, 'max' + sn)

        if self.startb + self.nmacros - 1 > self.maxmacros:
            self.nmacros = self.maxmacros - self.startb + 1
            logger.warning("Truncating n%s to %d", sn, self.nmacros)

        self.endb = self.startb + self.nmacros - 1
        if not self.endb == self.maxmacros:
            nrecs = getattr(self, sn + '_nrec') % self.macro_size + \
                (self.nmacros - 1) * self.macro_size
        else:
            nrecs = self.nmacros * self.macro_size
        num_batches = nrecs / batch_size
        assert num_batches > 0

        #self.mean_img = getattr(self, sn + '_mean')
        #self.mean_img.shape = (self.num_channels, osz, osz)
        #pad = (osz - csz) / 2
        #self.mean_crop = self.mean_img[:, pad:(pad + csz), pad:(pad + csz)]
        #self.mean_be = sbe((self.npixels, 1), dtype=betype)
        #self.mean_be.copy_from(self.mean_crop.reshape(
        #    (self.npixels, 1)).astype(np.float32))

        # Control params for macrobatch decoding thread
        self.macro_active_buf_idx = 0
        self.macro_decode_buf_idx = 0
        self.macro_num_decode_buf = 2
        self.macro_decode_thread = None

        self.batch_size = batch_size
        self.predict = predict
        self.minis_per_macro = [self.macro_size / batch_size
                                for i in range(self.macro_num_decode_buf)]

        if self.macro_size % batch_size != 0:
            raise ValueError('self.macro_size not divisible by batch_size')

        self.macro_idx = self.endb
        self.mini_idx = -1

        # Allocate space for host side image, targets and labels
        self.img_mini_T = [None for i in range(self.macro_num_decode_buf)]
        self.tgt_macro = [None for i in range(self.macro_num_decode_buf)]

        # Allocate space for device side buffers
        inp_shape = (self.npixels, self.batch_size)
        self.inp_be = sbe(inp_shape, dtype=betype)
        self.inp_be.name = "minibatch"

        # Allocate space for device side targets if necessary
        tgt_shape = (self.ntargets, self.batch_size)
        self.tgt_be = sbe(tgt_shape, dtype=betype) if self.ntargets != 0 else None

        return num_batches

    def get_mini_batch(self, batch_idx):
        b_idx = self.macro_active_buf_idx
        self.mini_idx = (self.mini_idx + 1) % self.minis_per_macro[b_idx]

        # Decode macrobatches in a background thread,
        # except for the first one which blocks
        if self.mini_idx == 0:
            if self.macro_decode_thread is not None:
                # No-op unless all mini finish faster than one macro
                self.macro_decode_thread.join()
            else:
                # special case for first run through
                self.macro_decode_thread = MacrobatchDecodeThread(self)
                self.macro_decode_thread.start()
                self.macro_decode_thread.join()

            # usual case for kicking off a background macrobatch thread
            self.macro_active_buf_idx = self.macro_decode_buf_idx
            self.macro_decode_buf_idx = \
                (self.macro_decode_buf_idx + 1) % self.macro_num_decode_buf
            self.macro_decode_thread = MacrobatchDecodeThread(self)
            self.macro_decode_thread.start()

        # All minibatches except for the 0th just copy pre-prepared data
        b_idx = self.macro_active_buf_idx
        s_idx = self.mini_idx * self.batch_size
        e_idx = (self.mini_idx + 1) * self.batch_size

        # See if we are a partial minibatch
        self.inp_be.copy_from(self.img_mini_T[b_idx][self.mini_idx])

        # Try to avoid this if possible as it inhibits async stream copy
        #if self.mean_norm:
        #    self.backend.subtract(self.inp_be, self.mean_be, self.inp_be)

        if self.unit_norm:
            self.backend.divide(self.inp_be, self.norm_factor, self.inp_be)

        if self.tgt_be is not None:
            self.tgt_be.copy_from(
                self.tgt_macro[b_idx][:, s_idx:e_idx]
                    .astype(self.backend_type))

        if self.autoencode_flag:
            return self.inp_be, self.inp_be

        return self.inp_be, self.tgt_be

    def has_set(self, setname):
        return True if (setname in ['train', 'validation', 'test']) else False
