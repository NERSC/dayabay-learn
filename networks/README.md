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

IBDPairConvAe2
-----------

This convolutional autoencoder is based on
[IBDPairConvAe](https://github.com/NERSC/dayabay-learn/tree/master/networks#ibdpairconvae).
It is identical in all respects except for the preprocessing and final
deconvolutional layer.

 - File:
[LasagneConv.py](https://github.com/NERSC/dayabay-learn/blob/master/networks/LasagneConv.py)

 - Preprocessing changes:
    - Divide each channel by a constant (common over events) so that each
      channel's minimum pixel has value -1 and each channel's maximum has value
      +1. This is instead of rescaling so that the standard deviation is 1.
 - Final deconvolutional layer changes:
    - Tanh activation

IBDChargeDenoisingConvAe
-----------

This denoising convolutional autoencoder is based on
[IBDPairConvAe2](https://github.com/NERSC/dayabay-learn/tree/master/networks#ibdpairconvae2).
It is identical in all respects except for the denoising: before the input is
fed to the network, a random selection of its pixels (per IBD pair, fraction
specified by user) is set to 0.

 - File:
[LasagneConv.py](https://github.com/NERSC/dayabay-learn/blob/master/networks/LasagneConv.py)

 - Input layer changes:
    - The input to the network is a partially corrupted version. For each IBD
      pair and a given corruption probability *p*, assign each pixel to be 0
      with probability *p* (or its uncorrupted value with probability 1-*p*).
      *Note*: the training cost still compares the network's output to the
      uncorrupted input.
