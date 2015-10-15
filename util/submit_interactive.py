#!/usr/bin/python
"""
Submits an interactive job using qsub
"""

import subprocess
import os
import sys

if len(sys.argv[1:]) > 0:
    cores = sys.argv[1]
else:
    cores = '120'

os.environ["ML_CASP_NUM_CORES"] = cores
os.environ["ML_SPARK_CORES"] = cores
all_args = ["qsub", "-I", "-lwalltime=00:30:00", '-qccm_int', '-lmppwidth='+cores, '-V' ]
print all_args
subprocess.call(all_args)

