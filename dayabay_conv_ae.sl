#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1
#SBATCH -t 00:50:00

cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
module load neon
python conv_ae_dayabay.py --epochs 30

