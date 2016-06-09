#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1
#SBATCH -t 00:50:00

python conv_ae_dayabay.py --epochs 30

