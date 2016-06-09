#!/bin/bash -l
ipython_file=$1
module load python
ipython nbconvert --to script $1
python_script="${ipython_file%.*}".py

#filter out ipython lines
cat $python_script | grep -v "ipython" > tmp.txt
mv tmp.txt $python_script
