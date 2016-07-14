# Daya Bay Machine Learning Software

This repository contains a set of neural networks developed to learn more about
the data collected by the Daya Bay Reactor Neutrino Experiment.

## Instructions:

There are 2 main scripts: a command-line based python script, `ibd_ae.py`, and an IPython
Notebook, `ibd_ae.ipynb`.

To set up the Cori environment, run the following commands

```
module load python
source activate deeplearning
```

On other environments, ensure the following dependencies are available:

 - python
 - theano
 - lasagne
 - scikit-learn
 - numpy/scipy/matplotlib
 - h5py

To run the IPython notebook, open it up (e.g. at
[ipython.nersc.gov](ipython.nersc.gov)) and follow the input prompts. To run
the command-line script, you can find help by executing the following command:

```
python ibd_ae.py --help
```

There are multiple network architectures located in the networks directory.
View the README there for more information.
