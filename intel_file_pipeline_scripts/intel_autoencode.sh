#!/usr/bin/env bash
#usage intel_autoencode.sh h5file-name savedir-name --intel
if [ "$#" -lt 2 ]
then
echo "Too few arguments! Usage: intel_autoencode.sh h5file-name savedir-name --intel"
exit
fi

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
rm $name
mv $savedir/train-inference.pkl $savedir/pkls/${name%.*}.pkl
#dont need to move the train targets so keep saving over them
#mv $savedir/train-targets.pkl $savedir/targets_$name.pkl
done

#add column for each class and combine these files
aprun -n 1 python ./util/numpy_pkl_to_h5.py $savedir/pkls/$base_name"-1".pkl $savedir/pkls/$base_name"-2".pkl $savedir/pkls/$base_name"-3".pkl
mv $base_name.h5 $savedir/h5

#remove the pkls
#rm $savedir/pkls/$base_name"-1".pkl $savedir/pkls/$base_name"-2".pkl $savedir/pkls/$base_name"-3".pkl
