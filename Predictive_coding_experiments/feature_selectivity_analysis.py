#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:22:48 2021

@author: jgalvan
"""

import os
import sys
import h5py
import argparse
import pandas as pd
import numpy as np
import plotting_figures as myplots
from scipy import stats
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management
sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
import other_billeh_utils
import load_sparse


class FeatureSelectivityAnalysis:    
    def __init__(self, flags):
        self.frequency = flags.gratings_frequency
        self.reverse = flags.reverse
        self.n_neurons = flags.neurons
        self.neuron_population = flags.neuron_population
        self.n_simulations_init = flags.n_simulations
        self.skip_first_simulation = flags.skip_first_simulation
        self.simulation_results = 'Simulation_results'
        self.directory = f'orien_0_freq_{self.frequency}_reverse_{self.reverse}_rec_{self.n_neurons}'
        self.full_path = os.path.join(self.simulation_results, self.directory)
        self.direction_path = os.path.join(self.simulation_results, 'Direction analysis', self.neuron_population)
        os.makedirs(self.direction_path, exist_ok=True)
        self.frequency_path = os.path.join(self.simulation_results, 'Frequency analysis', self.neuron_population)
        os.makedirs(self.frequency_path, exist_ok=True)
        # Load the simulation configuration attributes
        self.full_data_path = os.path.join(self.full_path, 'Data', 'simulation_data.hdf5')
        self.sim_metadata = {}
        with h5py.File(self.full_data_path, 'r') as f:
            dataset = f['Data']
            self.sim_metadata.update(dataset.attrs)
        self.data_dir = self.sim_metadata['data_dir']        
    
    # @profile
    def __call__(self):
        # Load the network of the model
        load_fn = load_sparse.cached_load_billeh
        _, self.network, _, _ = load_fn(self.sim_metadata['n_input'], self.n_neurons, self.sim_metadata['core_only'], 
                                        self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                        n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        ### Identify core neurons
        self.core_mask = other_billeh_utils.isolate_core_neurons(self.network, data_dir=self.data_dir)
        ### Direction analysis
        # Select a set of stimulus directions where classification of the neuron_population was made
        self.stimulus_directions = np.arange(0, 360, 45)
        # Create two dataframes to store the input_current responses and classificaitons
        self.input_current_response_direction_df = pd.DataFrame(columns=self.stimulus_directions.astype(str))
        self.direction_classifications_df = pd.DataFrame(columns=self.stimulus_directions.astype(str))        
        for direction in self.stimulus_directions:
            full_path = os.path.join('Simulation_results', f'orien_{direction}_freq_{self.frequency}_reverse_{self.reverse}_rec_{self.n_neurons}')
            full_path_pop = os.path.join(full_path, self.neuron_population)
            classification_file = os.path.join(full_path_pop, 'classification_results', f'{self.neuron_population}_selected_df.lzma')
            if not os.path.exists(classification_file):
                os.system(f"python neurons_dynamic_classification.py --neurons {self.n_neurons} --gratings_orientation {direction} --gratings_frequency {self.frequency} --classification_criteria input_current --neuron_population {self.neuron_population} --no-reverse")
                print('Lack of results from dynamic simulation. Try again when the dynamic simulation finishes.')
                quit()
            else:
                selected_df = file_management.load_lzma(classification_file)
            self.direction_classifications_df[str(direction)] = selected_df['class']
            self.input_current_response_direction_df[str(direction)] = selected_df['response_input_current']  
        # Set the Tf index as dataframe index
        self.direction_classifications_df['Tf index'] = selected_df['Tf index'].values
        self.direction_classifications_df.set_index('Tf index', inplace=True)
        self.input_current_response_direction_df['Tf index'] = selected_df['Tf index'].values
        self.input_current_response_direction_df.set_index('Tf index', inplace=True)
        # Extract the preferred_angle of the selected_neurons
        preferred_angles = selected_df['preferred_angle']
        file_management.save_lzma(self.direction_classifications_df, 'multiple_stimulus_directions_classifications', self.direction_path) 
        # Plot the input current responses of the selected neurons for every direction
        myplots.current_response_boxplots(self.input_current_response_direction_df, self.direction_classifications_df, path=self.direction_path)
        myplots.differential_current_response_boxplots(self.input_current_response_direction_df, self.direction_classifications_df, path=self.direction_path)
        # Identify pure class neurons as the ones that keep their characteristic response to every direction
        pure_dVf_mask = np.all(self.direction_classifications_df=='dVf', axis=1)
        pure_hVf_mask = np.all(self.direction_classifications_df=='hVf', axis=1)
        print(f'Pure dVf neurons: {np.sum(pure_dVf_mask)}/{len(pure_dVf_mask)}')
        print(f'Pure hVf neurons: {np.sum(pure_hVf_mask)}/{len(pure_hVf_mask)}')
        self.perturbation_neurons_mask = np.logical_or(pure_dVf_mask, pure_hVf_mask).values
        # Plot input_current traces of pure class neurons
        input_current_traces_dict = {}
        neuron_classes = ['dVf', 'hVf']
        pure_classes_mask = [pure_dVf_mask, pure_hVf_mask]
        for neu_class, class_mask in zip(neuron_classes, pure_classes_mask):
            # Isolate the class neurons using a core related mask
            input_current_traces_dict[neu_class] = {}
            class_tf_ids = self.direction_classifications_df.loc[class_mask].index
            V1_class_mask = np.full(self.n_neurons, False)
            V1_class_mask[class_tf_ids] = True
            core_class_mask = V1_class_mask[self.core_mask]
            # Iterate over stimulus_directions to obtain the traces for each class
            for direction in self.stimulus_directions:
                input_current_traces_dict[neu_class][str(direction)] = {}
                full_path = os.path.join('Simulation_results', f'orien_{direction}_freq_{self.frequency}_reverse_{self.reverse}_rec_{self.n_neurons}')
                full_data_path = os.path.join(full_path, 'Data', 'simulation_data.hdf5')
                # Load the simulation results
                self.variables = ['input_current']
                self.sim_data, _, _ = other_billeh_utils.load_simulation_results_hdf5(full_data_path, n_simulations=self.n_simulations_init, 
                                                                                      skip_first_simulation=self.skip_first_simulation, variables=self.variables)
                class_direction_input_current = np.mean(self.sim_data['input_current'][:,:, core_class_mask], axis=0) 
                class_direction_input_current_mean = np.mean(class_direction_input_current, axis=1) 
                class_direction_input_current_sem = stats.sem(class_direction_input_current, axis=1) 
                input_current_traces_dict[neu_class][str(direction)]['mean'] = class_direction_input_current_mean
                input_current_traces_dict[neu_class][str(direction)]['sem'] = class_direction_input_current_sem
        # Plot the average input_current traces of each class        
        myplots.mean_perturbation_responses_figure(input_current_traces_dict, path=self.direction_path)
        myplots.mean_perturbation_responses_composite_figure(input_current_traces_dict, path=self.direction_path)
        # Preferred perturbation direction and comparison
        perturbation_neurons_preferred_angles = preferred_angles[self.perturbation_neurons_mask]
        perturbation_neurons_current_response = self.input_current_response_direction_df[self.perturbation_neurons_mask]
        myplots.visual_preferred_direction(perturbation_neurons_preferred_angles, path=self.direction_path)
        myplots.perturbation_preferred_direction(perturbation_neurons_current_response, perturbation_neurons_preferred_angles, path=self.direction_path)

        ### Analysis of preferred temporal frequencies for the 0º direction
        self.temporal_frequencies = np.arange(0, 10, 1)
        zero_direction = 0
        # Save in a dataframe the input_current responses
        self.input_current_response_freq_df = pd.DataFrame(columns=self.temporal_frequencies.astype(str))
        for frequency in self.temporal_frequencies:
            full_path = os.path.join('Simulation_results', f'orien_{zero_direction}_freq_{frequency}_reverse_{self.reverse}_rec_{self.n_neurons}')
            full_path_pop = os.path.join(full_path, self.neuron_population)
            classification_file = os.path.join(full_path_pop, 'classification_results', f'{self.neuron_population}_selected_df.lzma')
            if not os.path.exists(classification_file):
                os.system(f"python neurons_dynamic_classification.py --neurons {self.n_neurons} --gratings_orientation {zero_direction} --gratings_frequency {frequency} --classification_criteria input_current --neuron_population {self.neuron_population} --no-reverse")
                print('Lack of results from dynamic simulation. Try again when the dynamic simulation finishes.')
                quit()
            else:
                selected_df = file_management.load_lzma(classification_file)
            self.input_current_response_freq_df[str(frequency)] = selected_df['response_input_current']  
        # Find the preferred temporal frequency (the one with the largest response) for each selected neuron
        preferred_temporal_frequency = self.input_current_response_freq_df.abs().idxmax(axis = 1).astype(float)
        myplots.preferred_temporal_frequency_histogram(preferred_temporal_frequency, self.direction_classifications_df, 
                                                       self.temporal_frequencies, path=self.frequency_path)
            
                
def main(flags):
     
    feature_selectivity_analysis = FeatureSelectivityAnalysis(flags)
    feature_selectivity_analysis()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Define key flags')
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
    
    