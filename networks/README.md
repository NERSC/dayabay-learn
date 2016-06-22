Networks Listing
============

This directory contains the neural networks developed to process Daya Bay data.

IBDPairConvAe
-----------

This convolutional autoencoder is a basic "out of the box" autoencoder. It is
a similar architecture to the autoencoder used in the original Daya Bay ML
paper. The bottleneck layer is where features are extracted, clustering is
performed, etc. 

 - File:
[LasagneConv.py](https://github.com/NERSC/dayabay-learn/blob/master/networks/LasagneConv.py)

 - User-specifiable parameters:
    - `minibatch_size` (default = 128)
    - `learn_rate` (default = 1e-3)
    - `bottleneck_depth` (default = 10)
 - Input:
    - Shape (N, 4, 8, 24)
    - 4 channels in the order: prompt charge, prompt time, delayed charge,
      delayed time
 - Preprocessing:
    - Log of charge (performed by data extractor)
    - Remove 0s from time layers by averaging neighbors' times
    - Add a constant to each channel (common over events) so that each channel's
      mean (over all events) is 0
    - Divide each channel by a constant (common over events) so that each
      channel's standard deviation (over all events) is 1
 - Layers:
    - Convolutional layer:
        - 16 filters
        - (5, 5) filter size
        - (2, 2) padding
        - ReLU activation
    - Max pool layer: (2, 2) pool size
    - Convolutional layer:
        - 16 filters
        - (3, 3) filter size
        - (1, 0) padding
        - ReLU activation
    - Max pool layer: (2, 2) pool size
    - Convolutional 'bottleneck' layer:
        - `bottleneck_depth` filters
        - (2, 5) filter size
        - (0, 0) padding
        - ReLU activation
    - Deconvolutional layer:
        - 16 filters
        - (2, 4) filter size
        - (2, 2) stride
    - Deconvolutional layer:
        - 16 filters
        - (2, 5) filter size
        - (2, 2) stride
    - Deconvolutional layer:
        - 4 filters
        - (2, 4) filter size
        - (2, 2) stride
