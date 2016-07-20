#!/bin/bash -l
echo "Sourcing setup file"
source ~skohn/setup_ml_cori.sh
echo "Finished sourceing setup file"
echo "Start seconds"
date +%s
echo "---"
python $SLURM_SUBMIT_DIR/ibd_ae.py -e 10 -w 256 \
-s model_e10_w256_n20000.npz \
-p reco_e10_w256_n20000.h5 \
-l 0.01 -n 20000 -vvv --network IBDChargeDenoisingConvAe \
--out-dir $SLURM_SUBMIT_DIR/batch/tmp_output
echo "End seconds"
date +%s
echo "---"
echo "Done"
