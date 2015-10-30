#! /bin/bash -l

for class in fla muo oth
do
	dir=$HOME/projects/dayabay-learn/tf_metadata/output-$class-large
	for subdir in `ls $dir`
	
	do
	if [[ `ls $dir/$subdir | wc -l` -gt 0 ]]
	then	
	$HOME/projects/dayabay-learn/util/get_files_missed.sh $dir/$subdir $JIALIN_PATH/large2/output-$class-large $HOME/projects/dayabay-learn/missing_files &
	fi
	done
done
