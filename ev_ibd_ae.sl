#!/bin/bash -l
#SBBATCH -N 1
#SBATCH -p regular
#SBATCH -t 6:00:00
#SBATCH -o batch_outputs/slurm-%A.out
#SBATCH --qos=premium

module load deeplearning
python ev_ibd_ae.py $@

