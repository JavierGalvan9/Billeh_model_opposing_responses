#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:22:48 2021

@author: jgalvan
"""

import os
import sys
import h5py
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import plotting_figures as myplots
import seaborn as sns
from sklearn.metrics import confusion_matrix
parentDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management
from other_utils import memory_tracer, timer
sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
import other_billeh_utils
import load_sparse


class StimulusClassificationComparison:    
    def __init__(self, flags):
        self.orientation = flags.gratings_orientation
        self.frequency = flags.gratings_frequency
        self.reverse = flags.reverse
        self.n_neurons = flags.neurons
        self.neuron_population = flags.neuron_population
        self.n_simulations_init = flags.n_simulations
        self.skip_first_simulation = flags.skip_first_simulation
        self.variables = ['input_current']
        self.simulation_results = 'Simulation_results'
        self.directory = f'orien_{str(flags.gratings_orientation)}_freq_{str(flags.gratings_frequency)}_reverse_{str(flags.reverse)}_rec_{flags.neurons}'
        self.full_path = os.path.join(self.simulation_results, self.directory)
        self.comparison_path = os.path.join(self.full_path, self.neuron_population, 'Comparison with other stimulus')
        os.makedirs(self.comparison_path, exist_ok=True)
        self.full_data_path = os.path.join(self.full_path, 'Data', 'simulation_data.hdf5')
        # Load the simulation configuration attributes
        self.sim_metadata = {}
        with h5py.File(self.full_data_path, 'r') as f:
            dataset = f['Data']
            self.sim_metadata.update(dataset.attrs)
        self.data_dir = self.sim_metadata['data_dir']
        self.data_dir = '/home/jgalvan/Desktop/Neurocoding/V1_Billeh_model/GLIF_network'
        
    # @profile    
    def __call__(self):
        # Load the baseline configuration and the classification made
        classification_path = os.path.join(self.full_path, self.neuron_population, 'classification_results')
        self.selected_df = file_management.load_lzma(os.path.join(classification_path, f'{self.neuron_population}_selected_df.lzma'))
        load_fn = load_sparse.cached_load_billeh
        _, self.network, _, _ = load_fn(self.sim_metadata['n_input'], self.n_neurons, self.sim_metadata['core_only'], 
                                        self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                        n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        # Load a mask for the selected neurons in the core
        self.core_mask = other_billeh_utils.isolate_core_neurons(self.network, data_dir=self.data_dir)
        self.n_core_neurons = self.core_mask.sum()
        self.selected_core_mask = np.full(self.n_neurons, False)
        self.selected_core_mask[self.selected_df['Tf index']] = True
        self.selected_core_mask = self.selected_core_mask[self.core_mask]
        
        # Load other configurations which the baseline will be compared to
        orientations = [0, 90, 180, 0] 
        frequencies = [8, 2, 2, 2]
        reverses = [False, False, False, True]
        currents = []
        all_confussion_matrices = []
        for orientation, frequency, reverse in zip(orientations, frequencies, reverses):
            new_directory = f'orien_{str(orientation)}_freq_{str(frequency)}_reverse_{reverse}_rec_{self.n_neurons}'
            new_full_path = os.path.join(self.simulation_results, new_directory)
            new_classification_path = os.path.join(new_full_path, self.neuron_population, 'classification_results')
            self.new_selected_df = file_management.load_lzma(os.path.join(new_classification_path, f'{self.neuron_population}_selected_df.lzma'))
            # Compare the neurons classification between the two configurations with a confussion matrix
            matrix = confusion_matrix(self.selected_df['class'], self.new_selected_df['class'], labels=['dVf', 'unclassified', 'hVf'])
            matrix = np.array(matrix).astype(np.float32)
            matrix = matrix/ matrix.sum(axis=1)[:, np.newaxis]
            cm_df = pd.DataFrame(matrix,
                                 index = ['dVf','unc','hVf'], 
                                 columns = ['dVf','unc','hVf'])
            all_confussion_matrices.append(cm_df)
            # Save the confussion matrix
            path = os.path.join(self.comparison_path, f'or_{orientation}_freq_{frequency}_rev_{reverse}')
            os.makedirs(path, exist_ok=True)
            self.comparison_matrix(cm_df, self.orientation, orientation, 
                                   self.frequency, frequency, path=path)
            # Obtain the input_current of the new configuration
            new_full_data_path = os.path.join(new_full_path, 'Data', 'simulation_data.hdf5')
            self.sim_data, _, _ = other_billeh_utils.load_simulation_results_hdf5(new_full_data_path, n_simulations=self.n_simulations_init, 
                                                                                  skip_first_simulation=self.skip_first_simulation, variables=self.variables)
            # Isolate the selected neuron population and normalize respect to baseline
            new_input_current_selected = self.sim_data['input_current'][:,:, self.selected_core_mask]        
            new_input_current_selected -= self.new_selected_df['baseline_input_current'].values
            currents.append(new_input_current_selected)
        # Compare the inputs currents and classifications between the different configurations              
        myplots.currents_comparison_figure(currents, frequencies, reverses, self.selected_df, path=self.comparison_path)
        myplots.matrices_comparison_figure(all_confussion_matrices, self.orientation, orientations, self.frequency, frequencies, 
                                           path=self.comparison_path)

    def comparison_matrix(self, confussion_matrix, orientation, orientation2, frequency, frequency2, path=''):
        # Create the comparison matrix between configurations
        fig = plt.figure(figsize=(5,4))
        sns.heatmap(confussion_matrix, vmin=0, vmax=1, annot=True, annot_kws={"size": 10}, fmt=".2f", yticklabels=True, cmap='binary')
        plt.ylabel(f'Direction {orientation} - frequency {frequency}')
        plt.xlabel(f'Direction {orientation2} - frequency {frequency2}')
        plt.yticks(va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(path, 'confussion_matrix.png'), dpi=300, transparent=True)
        plt.show()
        
def main(flags):
    comparator = StimulusClassificationComparison(flags)
    comparator()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Define key flags')
    parser.add_argument('--gratings_orientation', type=int, choices=range(0, 360, 45), default=0)
    parser.add_argument('--gratings_frequency', type=int, default=2)
    parser.add_argument('--neurons', type=int, default=230924)
    parser.add_argument('--n_simulations', type=int, default=None)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false')
    parser.set_defaults(reverse=False)
    parser.add_argument('--skip_first_simulation', action='store_true')
    parser.add_argument('--no-skip_first_simulation', dest='skip_first_simulation', action='store_false')
    parser.set_defaults(skip_first_simulation=True)
    parser.add_argument('--neuron_population', type=str, default='e23')
    
    flags = parser.parse_args()
    main(flags)
    
    