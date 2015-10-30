#!/usr/bin/env bash
#usage intel_autoencode.sh h5file-name savedir-name --intel
if [ "$#" -lt 2 ]
then
echo "Too few arguments! Usage: intel_autoencode.sh h5file-name savedir-name --intel final-save-loc"
exit
fi

h5file=$1
echo h5:$h5file
savedir=$2
echo sd:$savedir
intel=$3
final_save_loc=$4
. neon2.profile
#echo $h5file
new_dirname="$(basename "$(dirname "$h5file")")"
final_dir=$final_save_loc/$new_dirname
filename=`basename $h5file`
if [ -f $final_dir/$filename ]; then
echo "File $final_dir/$filename already exists! Exiting..."
exit
fi

if [ ! -f $h5file ]; then
echo "File $h5file does not exist!. Exiting..."
exit
fi
rows=`python ./util/get_rows.py $h5file`
if [ $rows -gt 1000 ]; then
batch_size=100
else
batch_size=1
fi
base_name=${filename%.*}
class=`python ./util/get_class.py $h5file`
echo $class
name=$base_name"-$class".h5
cp $h5file $savedir/$name
tmpdir=$savedir/${name%.*}
mkdir $tmpdir
python predict.py --h5file $savedir/$name --save_dir $tmpdir --all_train $intel --batch_size $batch_size
#b/c split files writes to ./ for now
rm $savedir/$name

PKL=$savedir/pkls/${name%.*}.pkl
echo $PKL
mv $tmpdir/train-inference.pkl $PKL
rm -rf $tmpdir
if [ ! -d $final_dir ]; then
mkdir $final_dir
fix_perms -g dasrepo $final_dir
fi
#dont need to move the train targets so keep saving over them
#mv $savedir/train-targets.pkl $savedir/targets_$name.pkl
#done
#add column for each class and combine thesei files
python ./util/numpy_pkl_to_h5.py $final_dir  $PKL
chgrp dasrepo $final_dir/$filename
#remove the pkls
rm $PKL 
