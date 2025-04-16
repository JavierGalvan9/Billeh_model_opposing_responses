# Modeling circuit mechanisms of opposing cortical responses to visual flow perturbations with the Billeh model

A computational framework for simulating and analyzing cortical responses in a mouse V1 column, based on the Allen Institute model from Billeh et al.

![Fig1](https://github.com/user-attachments/assets/540e9030-e460-4ff5-a1a9-576363e62416)

## Overview

This repository implements a computational model of the mouse primary visual cortex (V1), based on the work from the Allen Institute, as described in [Billeh et al. (2020)](https://doi.org/10.1016/j.neuron.2020.01.009). The codebase allows researchers to:

1. Construct and configure a model of the V1 column with customizable parameters.
2. Run simulations of the model responding to different visual stimuli, including:
   - Drifting gratings with configurable orientation and frequency
   - Full field flash stimuli
3. Analyze neuronal responses to these stimuli, including:
   - Classification of neuronal populations based on their response dynamics
   - Analysis of synaptic and input currents
   - Spatial distribution analysis of neuronal populations
   - Network connectivity and topology analysis
4. Generate visualizations of neuronal activity, connectivity patterns, and response properties

The model is particularly useful for studying how V1 neurons respond differently to visual stimuli, including the mechanisms behind opposing cortical responses to visual flow perturbations.

## Installation

### Requirements

This project requires Python 3.8+ and several dependencies. A full list of dependencies is available in the `requirements.txt` file. Key dependencies include:

- tensorflow (2.15.0)
- numpy (1.23.5)
- pandas
- matplotlib
- scipy
- h5py
- absl-py
- bmtk (Brain Modeling Toolkit)
- numba
- scikit-learn/scikit-image

To install the required packages, run:

```bash
pip install -r requirements.txt
```

### GPU Support

The model can take advantage of GPU acceleration for faster simulation. The code has been tested with:
- CUDA 11.8
- cuDNN 8.8.0

Make sure your system has compatible CUDA drivers installed for optimal performance.

## Model Architecture

The model is based on the Allen Institute's V1 cortical column model and consists of:

1. A network of Generalized Leaky Integrate-and-Fire (GLIF) neurons
2. LGN input population providing visual inputs to the model
3. Background input providing noise/baseline activity
4. Recurrent connectivity within the V1 population

The full model contains up to 230,924 neurons, but can be configured to use a smaller subset. The "core" configuration typically uses ~51,978 neurons from the central region of the V1 column.

## Usage

### Basic Workflow

The typical workflow for using this model consists of:

1. Setting up and running a simulation with visual stimuli
2. Analyzing network topology
3. Classifying and analyzing neuronal dynamics
4. Analyzing specific response characteristics

### Running Simulations

To run a simulation with drifting gratings stimulus:

```bash
python Predictive_coding_experiments/drifting_gratings.py --gratings_orientation 0 --gratings_frequency 2 --neurons 5000
```

Parameters:
- `--gratings_orientation`: Orientation of the drifting grating (0-359 degrees, in steps of 45)
- `--gratings_frequency`: Temporal frequency of the grating (Hz)
- `--neurons`: Number of neurons to simulate (5000, 51978 for core-only, or 230924 for full model)
- `--reverse`: Flag to reverse the direction of the grating
- `--n_simulations`: Number of simulation trials to run

### Analyzing Network Topology

After running a simulation, analyze the network topology:

```bash
python network_topology.py --gratings_orientation 0 --gratings_frequency 2 --neurons 5000
```

### Classification and Analysis

Several scripts are available for analyzing the simulation results:

```bash
# Classify neurons based on their dynamic responses
python Predictive_coding_experiments/neurons_dynamic_classification.py --gratings_orientation 0 --gratings_frequency 2 --neurons 5000

# Compare different neuronal populations
python Predictive_coding_experiments/dynamic_populations_comparison.py --gratings_orientation 0 --gratings_frequency 2 --neurons 5000

# Analyze synaptic characteristics
python Predictive_coding_experiments/synapses_analysis.py --gratings_orientation 0 --gratings_frequency 2 --neurons 5000
```

### Full Field Flash Simulation

To run a simulation with full field flash stimulus:

```bash
python Predictive_coding_experiments/full_field_flash.py --neurons 5000
```

### Analyzing Responses to Visual Flow Perturbations

```bash
python Predictive_coding_experiments/perturbation_responsive_neurons_analysis.py --neurons 5000
```

## Workflow Order

The recommended order for running the scripts (as noted in `Predictive_coding_experiments/README`):

1. `drifting_gratings.py` - Run the simulation and store data
2. `network_topology.py` - Analyze structural connectivity
3. `neurons_dynamic_classification.py` - Classify neurons based on dynamics
4. Analysis scripts (run any of these after steps 1-3):
   - `dynamic_populations_comparison.py`
   - `neurons_dynamic_analysis.py`
   - `synapses_analysis.py`
   - `stimulus_classification_comparison.py`
   - `feature_selectivity_analysis.py`
   - `classes_spatial_distribution_analysis.py`
5. `full_field_flash.py` - Run flash stimulus simulation
6. `perturbation_responsive_neurons_analysis.py` - Analyze responses to perturbations

## Model Components

### Key Modules

- `billeh_model_utils/`: Core utilities for the model
  - `load_sparse.py`: Functions to load the network structure
  - `models.py`: Implementation of the GLIF neuron model
  - `models_output_currents.py`: Extended model with output current tracking
  - `other_billeh_utils.py`: Helper functions for the model
  - `plotting_utils.py`: Visualization functions
  
- `general_utils/`: General utility functions
  - `file_management.py`: Functions for data I/O
  - `other_utils.py`: Miscellaneous utilities

- `Predictive_coding_experiments/`: Scripts for specific experiments
  - `drifting_gratings.py`: Simulations using drifting gratings
  - `full_field_flash.py`: Simulations using full field flash
  - Various analysis scripts for different aspects of the model

### Data Structure

Data is stored in:
- `GLIF_network/`: Contains network structure and parameters
- `Simulation_results/`: Stores simulation outputs
- `Topological_analysis_*/`: Network topology analysis results

## Contributing

When contributing to this repository, please ensure that:
1. Your code follows the existing style
2. You thoroughly test your changes
3. You document any new functions or modified behavior

## Citation

If you use this code in your research, please cite:

> Galván Fraile, J., Scherr, F., Ramasco, J. J., Arkhipov, A., Maass, W., & Mirasso, C. R. (2022). Modeling circuit mechanisms of opposing cortical responses to visual flow perturbations.

<!-- ## License

[Specify license or add a LICENSE file to the repository] -->

## Acknowledgments

This work is based on the Allen Institute's mouse V1 model. We thank the Allen Institute for providing the foundational model and data.
