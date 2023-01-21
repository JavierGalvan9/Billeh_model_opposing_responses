#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:22:48 2021

@author: jgalvan
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import plotting_figures as myplots
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management
sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
import other_billeh_utils
import load_sparse


class FlashResponseAnalysis:    
    def __init__(self, flags):
        self.n_neurons = flags.neurons
        self.black_flash = flags.black_flash
        self.neuron_population = flags.neuron_population
        self.n_simulations_init = flags.n_simulations
        self.skip_first_simulation = flags.skip_first_simulation
        self.simulation_results = 'Simulation_results'
        self.directory = f'full_field_flashes_rec_{self.n_neurons}_black_flash_{self.black_flash}'
        self.full_path = os.path.join(self.simulation_results, self.directory)
        self.images_path = os.path.join(self.simulation_results, 'Perturbation analysis', self.neuron_population)
        os.makedirs(self.images_path, exist_ok=True)
        self.full_data_path = os.path.join(self.full_path, 'Data', 'simulation_data.hdf5')
        # Load the simulation results
        self.variables = ['input_current']
        self.sim_data, self.sim_metadata, self.n_simulations = other_billeh_utils.load_simulation_results_hdf5(self.full_data_path, n_simulations=self.n_simulations_init, 
                                                                                                               skip_first_simulation=self.skip_first_simulation, variables=self.variables)
        self.data_dir = self.sim_metadata['data_dir']
        self.stimulus_init_time = 500
        self.stimulus_end_time = 1500
                
    def __call__(self):
        # Load the network of the model
        load_fn = load_sparse.cached_load_billeh
        _, self.network, _, _ = load_fn(self.sim_metadata['n_input'], self.n_neurons, self.sim_metadata['core_only'], 
                                        self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                        n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        # Load the selected neurons classifications to every gratings drifting direction 
        classifications_df = file_management.load_lzma(os.path.join(self.simulation_results, 'Direction analysis', self.neuron_population, 'multiple_stimulus_directions_classifications.lzma'))
        tf_indices = classifications_df.index.values
        # Load the selected_df of the selected population
        self.tf_id_to_selected_id = np.zeros(self.n_neurons, dtype=np.int32) - 1
        for selected_id, tf_id in enumerate(tf_indices):
            self.tf_id_to_selected_id[tf_id] = selected_id 
        # Identify perturbation responsive neurons as the ones that keep their characteristic response to every direction
        pure_dVf_mask = np.all(classifications_df=='dVf', axis=1).values
        pure_hVf_mask = np.all(classifications_df=='hVf', axis=1).values
        self.perturbation_responsive_neurons_mask = np.logical_or(pure_dVf_mask, pure_hVf_mask)
        n_pop_neurons = len(pure_dVf_mask)
        n_pure_dVf = np.sum(pure_dVf_mask)
        n_pure_hVf = np.sum(pure_hVf_mask)
        print(f'Pure dVf neurons: {n_pure_dVf}/{n_pop_neurons}')
        print(f'Pure hVf neurons: {n_pure_hVf}/{n_pop_neurons}')
        # Identify direction_selective neurons as the ones with just a dvf or hvf response 
        # to one direction and remain unclassified for the others
        dummy_df = pd.get_dummies(classifications_df, prefix_sep='', prefix='').sum(axis=1, level=0)
        direction_selective_mask = dummy_df['unclassified'] == 7
        self.direction_selective_neurons_df = classifications_df.loc[direction_selective_mask]
        selective_dVf_mask = np.logical_and(direction_selective_mask, dummy_df['dVf']==1)
        selective_hVf_mask = np.logical_and(direction_selective_mask, dummy_df['hVf']==1)
        n_selective_dVf = np.sum(selective_dVf_mask)
        n_selective_hVf = np.sum(selective_hVf_mask)
        print(f'Direction selective dVf neurons: {n_selective_dVf}/{n_pop_neurons}')
        print(f'Direction selective hVf neurons: {n_selective_hVf}/{n_pop_neurons}')
        # Load the input current for the flash stimulus and normalize it with respect to its baseline value
        self.baseline_flash_input_current = np.mean(self.sim_data['input_current'][:, :self.stimulus_init_time, :], axis=(0, 1))
        self.input_current = self.sim_data['input_current'] - self.baseline_flash_input_current
        self.input_current_per_res_dvf = self.input_current[:, :, self.tf_id_to_selected_id[tf_indices[pure_dVf_mask]]]
        self.input_current_per_res_hvf = self.input_current[:, :, self.tf_id_to_selected_id[tf_indices[pure_hVf_mask]]]
        self.input_current_dir_sel_dvf = self.input_current[:, :, self.tf_id_to_selected_id[tf_indices[selective_dVf_mask]]]
        self.input_current_dir_sel_hvf = self.input_current[:, :, self.tf_id_to_selected_id[tf_indices[selective_hVf_mask]]]
        
        # Compare the average traces for the different populations in response 
        # to the full field white flash stimulus
        myplots.plot_average_population_comparison(self.input_current_per_res_dvf, self.input_current_per_res_hvf, 
                                                   pop1_name='Pert. resp. dVf', pop2_name='Pert. resp. hVf',
                                                   color1='#33ABA2', color2='#F06233',
                                                   path=self.images_path)
        myplots.plot_average_population_comparison(self.input_current_per_res_dvf, self.input_current_dir_sel_dvf, 
                                                   pop1_name='Pert. resp. dVf', pop2_name='Dir. sel. dVf',
                                                   color1='#33ABA2', color2='b',
                                                   path=self.images_path)
        myplots.plot_average_population_comparison(self.input_current_per_res_hvf, self.input_current_dir_sel_hvf, 
                                                   pop1_name='Pert. resp. hVf', pop2_name='Dir. sel. hVf',
                                                   color1='#F06233', color2='g',
                                                   path=self.images_path)
        myplots.plot_average_population_comparison(self.input_current_dir_sel_hvf, self.input_current_dir_sel_dvf, 
                                                   pop1_name='Dir. sel. hVf', pop2_name='Dir. sel. dVf',
                                                   color1='g', color2='b',
                                                   path=self.images_path)
        myplots.plot_flash_responses_comparison(self.input_current_per_res_dvf, self.input_current_dir_sel_dvf, 
                                                self.input_current_per_res_hvf, self.input_current_dir_sel_hvf, 
                                                path=self.images_path)
        
        ### Compare the distributions of the input current responses for all populations
        classes_names = ['Pert. resp. dVf', 'Dir. sel. dVf', 'Pert. resp. hVf','Dir. sel. hVf' ]
        data = pd.DataFrame(index=np.arange(n_pop_neurons), columns=classes_names)
        data.loc[:n_pure_dVf-1, 'Pert. resp. dVf'] = np.mean(self.input_current_per_res_dvf[:, 500:750,:], axis=(0, 1))
        data.loc[:n_selective_dVf-1, 'Dir. sel. dVf'] = np.mean(self.input_current_dir_sel_dvf[:, 500:750,:], axis=(0, 1))
        data.loc[:n_pure_hVf-1, 'Pert. resp. hVf'] = np.mean(self.input_current_per_res_hvf[:, 500:750,:], axis=(0, 1))
        data.loc[:n_selective_hVf-1, 'Dir. sel. hVf'] = np.mean(self.input_current_dir_sel_hvf[:, 500:750,:], axis=(0, 1))
        # Make a significance Welch t-test to compare the population means
        pairs = [('Dir. sel. dVf', 'Pert. resp. dVf'),
                 ('Dir. sel. hVf', 'Pert. resp. hVf')]
        pvalues = [stats.ttest_ind(data['Pert. resp. dVf'].values, data['Dir. sel. dVf'].values, equal_var=False).pvalue,
                  stats.ttest_ind(data['Pert. resp. hVf'].values, data['Dir. sel. hVf'].values, equal_var=False).pvalue]
        statistic = [stats.ttest_ind(data['Pert. resp. dVf'].values, data['Dir. sel. dVf'].values, equal_var=False).statistic,
                  stats.ttest_ind(data['Pert. resp. hVf'].values, data['Dir. sel. hVf'].values, equal_var=False).statistic]
        classes_colors = ['#808BD0', 'gray', '#92876B', 'brown']
        myplots.populations_current_response_violin_hist(data, pairs, pvalues, colors=classes_colors, 
                                                         path=self.images_path)
        significance_test_path = os.path.join(self.images_path, 'Significance_test.txt')
        with open(significance_test_path, 'w') as out_file:
            out_file.write(f'Results of the Welch t-test comparing the input_current_response distributions of perturbation responsive and direction selective {self.neuron_population} neurons\n')
            for idx, pair in enumerate(pairs):
                out_file.write(f'Pair: {pair[0]} vs {pair[1]}\n')
                out_file.write(f'- Statistic: {statistic[idx]}\n')
                out_file.write(f'- p-value: {pvalues[idx]}\n \n')
        
def main(flags):
    flash_response_analysis = FlashResponseAnalysis(flags)
    flash_response_analysis()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Define key flags')
    parser.add_argument('--neurons', type=int, default=230924)
    parser.add_argument('--n_simulations', type=int, default=None)
    parser.add_argument('--skip_first_simulation', action='store_true')
    parser.add_argument('--no-skip_first_simulation', dest='skip_first_simulation', action='store_false')
    parser.set_defaults(skip_first_simulation=True)
    parser.add_argument('--black_flash', action='store_true')
    parser.add_argument('--no-black_flash', dest='black_flash', action='store_false')
    parser.set_defaults(black_flash=True)
    parser.add_argument('--neuron_population', type=str, default='e23')
    
    flags = parser.parse_args()
    main(flags)  
    