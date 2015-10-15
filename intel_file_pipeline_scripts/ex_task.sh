#!/bin/sh
echo "my taskid is: " $TF_TASKID
h5dir=$1
offset=$2
cd /global/homes/r/racah/projects/dayabay-learn 
#h5dir=/scratch3/scratchdirs/jialin/dayabay/preprocess/output-fla
h5file=`ls $h5dir | sed -n "$(( $TF_TASKID + $offset ))"p`
./extras_intel_autoencode.sh $h5dir/$h5file  $SCRATCH/intel_data --intel /scratch3/scratchdirs/jialin/dayabay/autoencoded

