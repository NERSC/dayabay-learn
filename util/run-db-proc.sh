#!/usr/bin/env bash
main_path=$HOME/projects/dayabay-learn/
main_pp_path=$main_path/preprocess
main_ae_path=$main_path/autoencoded
mkdir $main_pp_path
mkdir $main_ae_path


for class in muo fla oth adinit addelay
do
mkdir $main_pp_path/output-$class
python get_all_of_class.py $class ./data/peter_data/single_20000.h5  $main_path/output-$class
./intel_file_pipeline_scripts/extras_intel_autoencode.sh $main_pp_path/output-$class/single_20000.h5 $SCRATCH/intel_data --intel $main_ae_path
done


