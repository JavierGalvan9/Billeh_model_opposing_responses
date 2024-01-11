#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:22:48 2021

@author: jgalvan
"""


import os
import sys
import numpy as np
import pandas as pd
import argparse
from scipy import stats
import plotting_figures as myplots
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
import other_billeh_utils
import load_sparse
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management
from other_utils import memory_tracer, timer


class ClassificationComparison:    
    def __init__(self, flags):
        self.orientation = flags.gratings_orientation
        self.frequency = flags.gratings_frequency
        self.reverse = flags.reverse
        self.n_neurons = flags.neurons
        self.neuron_population_1 = flags.neuron_population_1
        self.neuron_population_2 = flags.neuron_population_2
        self.n_simulations_init = flags.n_simulations
        self.skip_first_simulation = flags.skip_first_simulation
        self.simulation_results = 'Simulation_results'
        self.directory = f'orien_{str(self.orientation)}_freq_{str(self.frequency)}_reverse_{self.reverse}_rec_{flags.neurons}'
        self.full_path = os.path.join(self.simulation_results, self.directory)
        self.comparison_path = os.path.join(self.full_path, f'Comparison {self.neuron_population_1} vs {self.neuron_population_2}')
        os.makedirs(self.comparison_path, exist_ok=True)
        # Load the simulation results
        self.full_data_path = os.path.join(self.full_path, 'Data', 'simulation_data.hdf5')
        
        self.variables = ['input_current']
        self.sim_data, self.sim_metadata, self.n_simulations = other_billeh_utils.load_simulation_results_hdf5(self.full_data_path, n_simulations=self.n_simulations_init, 
                                                                                                               skip_first_simulation=self.skip_first_simulation, variables=self.variables)
        self.data_dir = self.sim_metadata['data_dir']
        # self.data_dir = '/home/jgalvan/Desktop/Neurocoding/V1_Billeh_model/GLIF_network'
    
    # @profile    
    def __call__(self):
        # Load the network of the model and the model core network
        load_fn = load_sparse.cached_load_billeh
        _, self.network, _, _ = load_fn(self.sim_metadata['n_input'], self.n_neurons, self.sim_metadata['core_only'], 
                                        self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                        n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        ### Identify core neurons
        self.core_mask = other_billeh_utils.isolate_core_neurons(self.network, data_dir=self.data_dir)
        self.n_core_neurons = self.core_mask.sum()
        if self.n_core_neurons < 51978:
            self.core_network = self.network
        else:
            _, self.core_network, _, _ = load_fn(self.sim_metadata['n_input'], 51978, self.sim_metadata['core_only'], 
                                                self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                                n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        
        ### Identify selected neurons population 1
        if len(self.neuron_population_1.split('_')) > 1:
            self.neuron_multiple_populations_1 = self.neuron_population_1.split('_')
            self.selected_mask_1 = np.zeros(self.n_core_neurons).astype(bool)
            for neuron_pop in self.neuron_multiple_populations_1:
                pop_mask = other_billeh_utils.isolate_neurons(self.core_network, neuron_population=neuron_pop, data_dir=self.data_dir)
                self.selected_mask_1 = np.logical_or(self.selected_mask_1, pop_mask)
        else:
            self.selected_mask_1 = other_billeh_utils.isolate_neurons(self.core_network, neuron_population=self.neuron_population_1, data_dir=self.data_dir)  
        
        ### Identify selected neurons population 2
        if len(self.neuron_population_2.split('_')) > 1:
            self.neuron_multiple_populations_2 = self.neuron_population_2.split('_')
            self.selected_mask_2 = np.zeros(self.n_core_neurons).astype(bool)
            for neuron_pop in self.neuron_multiple_populations_2:
                pop_mask = other_billeh_utils.isolate_neurons(self.core_network, neuron_population=neuron_pop, data_dir=self.data_dir)
                self.selected_mask_2 = np.logical_or(self.selected_mask_2, pop_mask)
        else:
            self.selected_mask_2 = other_billeh_utils.isolate_neurons(self.core_network, neuron_population=self.neuron_population_2, data_dir=self.data_dir)  
        # Select the input current of each poopulation
        stimuli_init_time=500
        self.baseline_input_current = np.mean(self.sim_data['input_current'][:, 0:stimuli_init_time, :], axis=(0, 1))
        self.input_current = self.sim_data['input_current'] - self.baseline_input_current
        self.input_current_1 = self.input_current[:,:, self.selected_mask_1]
        self.input_current_2 = self.input_current[:,:, self.selected_mask_2]
        # Plot each population traces
        myplots.plot_average_population_comparison(self.input_current_1, self.input_current_2, 
                                                   pop1_name='E23', pop2_name='E56',
                                                   path=self.comparison_path)
        
        ### Compare the distributions of the input current responses for both populations
        input_current_response_data = pd.DataFrame()
        neuron_pop_names = ['E 5/6', 'E 2/3']
        for neuron_pop, neuron_pop_name in zip([self.neuron_population_1, self.neuron_population_2], neuron_pop_names):
            full_path_pop = os.path.join(self.full_path, neuron_pop)
            df_neuron_pop = file_management.load_lzma(os.path.join(full_path_pop, 'classification_results', f'{neuron_pop}_selected_df.lzma'))
            input_current_response_data[neuron_pop_name] = df_neuron_pop['response_input_current']
        # Make a significance Welch t-test to compare the population means
        pairs = [('E 5/6', 'E 2/3')]  
        pvalues = [stats.ttest_ind(input_current_response_data['E 5/6'].dropna().values, 
                                   input_current_response_data['E 2/3'].dropna().values, 
                                   equal_var=False).pvalue]
        statistic = [stats.ttest_ind(input_current_response_data['E 5/6'].dropna().values, 
                                     input_current_response_data['E 2/3'].dropna().values, 
                                     equal_var=False).statistic]
        # Create an array with the keller colors
        classes_colors = ['#92876B', '#808BD0']
        myplots.populations_current_response_violin_hist(input_current_response_data, pairs, pvalues, 
                                                         colors=classes_colors, path=self.comparison_path)
        significance_test_path = os.path.join(self.comparison_path, 'Significance_test.txt')
        with open(significance_test_path, 'w') as out_file:
            out_file.write(f'Results of the Welch t-test comparing the input_current_response distributions of {self.neuron_population_1} and {self.neuron_population_2}\n')
            out_file.write(f'- Statistic: {statistic[0]}\n')
            out_file.write(f'- p-value: {pvalues[0]}\n')

        

# @timer
# @memory_tracer  
def main(flags):
    comparator = ClassificationComparison(flags)
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
    parser.add_argument('--neuron_population_1', type=str, default='e23')
    parser.add_argument('--neuron_population_2', type=str, default='e5_e6')
    
    flags = parser.parse_args()
    main(flags)  
    
    