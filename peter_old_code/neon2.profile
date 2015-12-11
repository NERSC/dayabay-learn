#!/bin/bash

# These from https://www.nersc.gov/users/software/data-visualization-and-analytics/neon-scalable-deep-learning-library/
module load python_base/2.7.9
#module load python/2.7.9
module load numpy/1.9.1_mkl
module load scipy/0.14.1_mkl
module load mpi4py/1.3.1

#module load python
#module load numpy/1.9.0_mkl
#module load scipy/0.14.0_mkl
#module load mpi4py

module load pyyaml
module load cray-hdf5-parallel
module load h5py
module load matplotlib
module load cython
module load ipython
#module load neon
module load mongodb
module load virtualenv

# These came from email.

# Spearmint 
#module load spearmint # Contains pymongo
#export SPEARMINT_PATH=/global/homes/p/pjsadows/spearmint/spearmint/bin/
#PYTHONPATH=/global/homes/p/pjsadows/spearmint/spearmint/:$PYTHONPATH
#export HYPEROPT_PATH=/global/homes/p/pjsadows/params # temporary
#PYTHONPATH=/global/homes/p/pjsadows/lib/protobuf/python/google/:$PYTHONPATH
#export PYTHONPATH=/global/homes/p/pjsadows/lib/spearmint/spearmint/schedulers/:$PYTHONPATH

#export PYTHONPATH=/global/homes/p/pjsadows/lib/spearmint/:$PYTHONPATH
#export PYTHONPATH=/global/homes/p/pjsadows/lib/pbs_python-4.6.0/build/lib/python2.7/site-packages/pbs/:$PYTHONPATH
#module load spearmint


export NEON_HOME=/global/homes/p/pjsadows/lib/neon2/
export PYTHONPATH=$NEON_HOME:$PYTHONPATH
export NEON_HOME=/global/homes/p/pjsadows/lib/neon/
export PYTHONPATH=$NEON_HOME:$PYTHONPATH
export SKLEARNHOME=/project/projectdirs/mantissa/modules/
export PYTHONPATH=$SKLEARNHOME:$PYTHONPATH
#PATH=$NEON_HOME/bin:$PATH

# Examples:
# ls -l $NEON_EXAMPLES
# neon $NEON_EXAMPLES/mlp/mnist-small.yaml
# aprun -n 2 -N 2 -S 1 neon-mpi --modelpar $NEON_EXAMPLES/mlp/mnist-small.yaml
# pydoc neon.models.MLP

# Physics
module load root
module  swap PrgEnv-intel PrgEnv-gnu
#module load root
#module load python
export PYTHONPATH=/usr/common/usg/root/5.34/gnu/lib/root/:$PYTHONPATH
export PYTHONPATH=/global/homes/p/pjsadows/lib/:$PYTHONPATH # Includes roottools module by peter
#python SimpleRoot.py /project/projectdirs/dayabay/scratch/ynakajim/mywork/data_samples/recon.Neutrino.0021221.Physics.EH1-Merged.P14A-P._0001.root


