#!/bin/bash -l

qsub_dir=$1
file_dir=$2
output_dir=$3
if [[ $# -lt 3 ]]
then
echo "usage ./get_missed_files <tflog_dir> <files_dir> <dir_where_you_want_output_saved>"
exit
fi
if [[ ! -f $output_dir ]]
then
mkdir -p $output_dir
fi
final_files_dir=$JIALIN_PATH/../autoencoded/"$(basename $file_dir)"
save_file=$output_dir/"$(basename $file_dir)"-missing.txt
tmp_file=shape.txt
for name in `cat $qsub_dir/tf.log | grep "100   /global" | awk '{print $6}'`; 
do 
	file=$file_dir/"$(ls $file_dir | sed -n $name'p')"
	if [[ ! -f $final_files_dir/"$( basename $file )" ]]
	then
		echo "hey"
		$HOME/projects/dayabay-learn/util/get_shape.py $file > $tmp_file  2> /dev/null
       		if [[ -s $tmp_file ]] 	
        	then
			shape="$(cat $tmp_file)"
			if [[ $shape == *"inputs :  ("* ]]
			then
				rows="$(echo $shape | awk '{print $3}'|sed s/\(//g | sed s/,//g)"
				if [[ $rows -gt 0 ]]
				then
					if [[ ! -f $final_files_dir/"$( basename $file )" ]]
					then
						echo "$file is good, but was not processed by the autoencoder for some reason" >> $save_file
					fi
				else
					echo "$file has 0 rows!" >> $save_file
				fi
		fi

		else
			echo "$file couldn't be opened!" >> $save_file
		fi
	fi
done
