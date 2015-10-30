#!/usr/bin/env bash
base_final_h5_dir=/scratch3/scratchdirs/jialin/dayabay/autoencoded
if [[ $# -lt 1 ]]
then
base_h5_dir=/scratch3/scratchdirs/jialin/dayabay/preprocess/large2
else
base_h5_dir=$1
fi
base_dir=/global/homes/r/racah/projects/dayabay-learn
for subdir in `ls $base_h5_dir`
do
	total_files="$(ls $base_h5_dir/$subdir | wc -l)"
	files_finished="$(ls $base_final_h5_dir/$subdir | wc -l)"
	earliest_file_finished_idx=60000 #$(( files_finished - (files_finished % 15800) ))
	num_files_per_tf_job="$(( ( 1 + total_files - earliest_file_finished_idx) / 6 ))"
	start_idx=$earliest_file_finished_idx
	finish_idx=$(( start_idx + num_files_per_tf_job ))
	count=0
	while [[ $start_idx -lt $total_files ]]
	do
		
		new_dir=$base_dir/tf_metadata/$subdir/$count
		mkdir -p $new_dir
		cd $new_dir
		qsub  -F "$base_h5_dir/$subdir $start_idx $finish_idx  " $base_dir/extras_intel_ae_tf.pbs
		start_idx=$finish_idx
		finish_idx=$(( finish_idx + num_files_per_tf_job ))
		if [[ $finish_idx -gt $total_files ]]
		then
			finish_idx=$total_files
		fi
		count="$(( count + 1 ))"
	done
done


