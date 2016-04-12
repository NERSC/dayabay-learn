conv_ae_dayabay.py takes in an nx192 dayabay data and 
uses a convolutional autoencoder to extract 10-dimensional features from each example and then plots each 10-D example in
2-D using t-SNE.

Instructions for running on Cori:

* module load python
* module load scikit-learn
* module load neon/1.1.0
* python conv_ae_dayabay.py --epochs <number of epochs> --max_tsne_iter <max number of tsne iterations>


The t-SNE step is very slow at this point.
