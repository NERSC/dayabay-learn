#!/bin/bash -l

# This script will submit a NN training job to the "shared" partition where 1
# core can be requested

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH -J train_dca
#SBATCH -A dasrepo
#SBATCH -L project
#SBATCH -e batch/logs/train_dca.e.log
#SBATCH -o batch/logs/train_dca.o.log

srun batch/run_ibd_ae.py.sh
