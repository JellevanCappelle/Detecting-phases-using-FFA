# About
This repository contains (most of) the code for my bachelor's thesis "Detecting Phase Transitions using Forward-Forward Neural Networks".
As the name implies, the goal of my project has been to apply the [Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345) to physical models with a phase transition in an attempt to find the critical point, preferably without using any prior knowledge.

This is accomplished by training a discriminative Forward-Forward network to learn features that are unique to specific ranges of temperature or other tuning parameter.
These features can then be used to find the critical point by:
- splitting the dataset into two classes with high mutual similarity of learned features (class-based method)
- finding the temperature/tuning parameter whose features are most similar to every other temperature/parameter (similarity-based method)
- extracting an order parameter from the learned features, and taking its derivative.

# Overview
The folder 'Experimental setup' contains everything needed to run experiments.
`experiment_classes.py` contains a class for each physical model to intialize the base `Experiment` class.
These classes can be used from the main file, `experiment.py`, to run the actual experiments and plot/print the results.

The 'Wolff Algorithm' folder contains a C++ implementation of the [Wolff algorithm](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.62.361) for the Ising and q-state Potts models.
A good explanation of the algorithm by Erik Luijten is available [here](https://csml.northwestern.edu/resources/Reprints/lnp_color.pdf).
The implementation has been designed to completely avoid memory allocations and set-lookups during simulation (of course, some memory has to be allocated to intialize the simulation), and to support multithreading.
To build the simulators, [libnpy](https://github.com/llohse/libnpy) is required, which is also linked as a sub-module in the folder.
I have used the version of libnpy from april 2023 in my project.

# Getting started
## Running an experiment
To run an experiment, simply configure `experiment.py` to the desired physical model, network architecture and hyperparameters, and then execute the script.
Before running experiments, you first need to generate a dataset for the selected physical model.

## Generating a dataset
The files ending in `_dataset.py` are responsible for generating and loading datasets for the various physical models.
Evert van Nieuwenburg has provided `mbl_dataset.py` (with some minor edits by myself), as well as the Kitaev chain dataset.

Below are instructions for each model.

### Ising & Potts
Build the Ising or Potts simulator from the 'Wolff Algorithm' folder.
Then, configure `ising_dataset.py` or `potts_dataset.py` and run the script.
Make sure to change the variable `simulator_path` in each script to point to the location of the simulator executable.

### TFIM
To generate a dataset, install [NetKet](https://netket.readthedocs.io), and configure and run `tfim_dataset.py`.
NetKet doesn't support Windows, so Windows users should run it through WSL or some other alternative.
This is only necessary when generating the dataset, but not when running experiments.

### MBL
Configure and run `mbl_dataset.py`.

### Kitaev
The dataset has been provided by Evert van Nieuwenburg, and does not have to be generated.
