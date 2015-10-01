#!/usr/bin/env bash
#usage intel_autoencode.sh h5file-name savedir-name --intel
h5file=$1
savedir=$2
intel=$3
. neon2.profile
echo $h5file
filename=`basename $h5file`
base_name=${filename%.*}
aprun -n 1 python split_files.py $h5file
for i in 1 2 3
do
name=$base_name"-$i.h5"
aprun -n 1 python predict.py --h5file $name --save_dir $savedir --all_train $intel
mv $savedir/train-inference.pkl $savedir/$name.pkl
#dont need to move the train targets so keep saving over them
#mv $savedir/train-targets.pkl $savedir/targets_$name.pkl
done

#add column for each class and combine these files
python ./util/numpy_pkl_to_h5.py $savedir/$base_name"-1".pkl $savedir/$base_name"-2".pkl $savedir/$base_name"-3".pkl

#remove the pkls
rm $savedir/$base_name"-1".pkl $savedir/$base_name"-2".pkl $savedir/$base_name"-3".pkl