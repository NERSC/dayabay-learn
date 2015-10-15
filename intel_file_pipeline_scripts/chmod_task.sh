#!/bin/bash -l
h5dir=$1
echo "my taskid is: " $TF_TASKID
file=`ls $h5dir | sed -n $TF_TASKID"p"`
echo "chmod a+r $h5dir/$file"

