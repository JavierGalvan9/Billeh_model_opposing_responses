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
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import plotting_figures as myplots
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management
sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
import other_billeh_utils
import load_sparse

#To use the line_profiler check:
#https://stackoverflow.com/questions/3927628/how-can-i-profile-python-code-line-by-line

mpl.style.use('default')
rd = np.random.RandomState(seed=42)
# Suppress/hide the warning of zero division
np.seterr(invalid='ignore')


class DynamicNeuronClasses: 
    # @profile
    def __init__(self, flags, voltage_threshold=1, firing_ratio_threshold=2, modulation_index_threshold=0.25):        
        self.orientation = flags.gratings_orientation
        self.frequency = flags.gratings_frequency
        self.reverse = flags.reverse
        self.n_neurons = flags.neurons
        self.neuron_population = flags.neuron_population
        self.n_simulations_init = flags.n_simulations
        self.n_simulations_static_gratings = flags.n_simulations_static_gratings
        self.classify_neurons = flags.classify_neurons
        self.correct_membrane_voltage = flags.correct_membrane_voltage
        self.skip_first_simulation = flags.skip_first_simulation
        self.classification_criteria = flags.classification_criteria
        self.simulation_results = 'Simulation_results'
        self.directory = f'orien_{str(flags.gratings_orientation)}_freq_{str(flags.gratings_frequency)}_reverse_{str(flags.reverse)}_rec_{flags.neurons}'
        self.full_path = os.path.join(self.simulation_results, self.directory)
        self.full_data_path = os.path.join(self.full_path, 'Data', 'simulation_data.hdf5')
        self.pop_full_path = os.path.join(self.full_path, self.neuron_population)
        # Load the simulation results
        self.variables = ['z', 'v', 'input_current', 'recurrent_current', 'bottom_up_current']
        self.sim_data, self.sim_metadata, self.n_simulations = other_billeh_utils.load_simulation_results_hdf5(self.full_data_path, n_simulations=self.n_simulations_init, 
                                                                                                               skip_first_simulation=self.skip_first_simulation, variables=self.variables)
        self.sim_data['asc_bkg'] = self.sim_data['input_current'] - self.sim_data['recurrent_current'] - self.sim_data['bottom_up_current']
        # with open(os.path.join(self.full_path,'flags_config.json'), 'r') as fp:
        #     self.sim_metadata = json.load(fp)
        self.data_dir = self.sim_metadata['data_dir']
        # self.simulation_length = self.sim_metadata['seq_len']

        if self.n_simulations_static_gratings is None:
            self.baseline_method = 'pre_perturbation'
        else:
            self.baseline_method = 'static_gratings'
        self.voltage_threshold = voltage_threshold
        self.firing_ratio_threshold = firing_ratio_threshold 
        self.modulation_index_threshold = modulation_index_threshold
    # @profile
    def __call__(self):
        # Load the model network
        load_fn = load_sparse.cached_load_billeh
        _, self.network, _, _ = load_fn(self.sim_metadata['n_input'], self.n_neurons, self.sim_metadata['core_only'], 
                                        self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                        n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        ### Identify core neurons
        self.core_mask = other_billeh_utils.isolate_core_neurons(self.network, data_dir=self.data_dir)
        self.core_id_to_tf_id = np.arange(self.n_neurons)[self.core_mask]
        self.n_core_neurons = self.core_mask.sum()
        
        if self.n_core_neurons < 51978:
            self.core_network = self.network
        else:
            _, self.core_network, _, _ = load_fn(self.sim_metadata['n_input'], 51978, self.sim_metadata['core_only'], 
                                                self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                                n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        # variables = ['v', 'input_current', 'recurrent_current', 'bottom_up_current']
        # self.sim_data, _ = other_billeh_utils.load_simulation_results(self.full_data_path, n_simulations = self.n_simulations_init, skip_first_simulation=self.skip_first_simulation,
        #                                                               variables=variables, simulation_length=self.simulation_length, n_neurons=self.n_core_neurons)
        # self.sim_data['asc_bkg'] = self.sim_data['input_current'] - self.sim_data['recurrent_current'] - self.sim_data['bottom_up_current']
        # Load the spike trains and calculate the firing rates
        # variables = ['z']
        # z, self.n_simulations = other_billeh_utils.load_simulation_results(self.full_data_path, n_simulations = self.n_simulations_init, skip_first_simulation=self.skip_first_simulation,
        #                                                                       variables=variables, simulation_length=self.simulation_length, n_neurons=self.n_neurons)
        self.sim_data['z'] = self.sim_data['z'][:,:, self.core_mask]
        firing_rate, self.sampling_interval = other_billeh_utils.firing_rates_smoothing(self.sim_data['z']) 
        self.sim_data['firing_rate'] = firing_rate 
        
        ### Plot random V1 core neurons response to stimuli
        path = os.path.join(self.full_path, 'V1 neurons samples')
        for neuron_id in rd.choice(range(self.n_core_neurons), size=2):
            tf_id = self.core_id_to_tf_id[neuron_id]
            myplots.plot_neuron_voltage_and_spikes_and_input_current(self.sim_data, neuron_id, tf_id=tf_id,
                                                                      fig_title=f'V1 neuron id:{tf_id}', path=path)
            myplots.plot_neuron_input_currents(self.sim_data, neuron_id, tf_id=tf_id,
                                                fig_title=f'V1 neuron currents id:{tf_id}', path=path)
            # Plot the effect of the voltage correction
            myplots.spike_effect_correction_plot(self.sim_data, neuron_id, tf_id, pre_spike_gap=1, post_spike_gap=5, path=path)  

        #Make a spike correction to the membrane voltage
        if self.correct_membrane_voltage:
            self.sim_data['v'] = other_billeh_utils.voltage_spike_effect_correction(self.sim_data['v'], self.sim_data['z'])

        ### Identify selected neurons
        if len(self.neuron_population.split('_')) > 1:
            self.neuron_multiple_populations = self.neuron_population.split('_')
            self.selected_mask = np.zeros(self.n_core_neurons).astype(np.bool)
            for neuron_pop in self.neuron_multiple_populations:
                subpop_mask = other_billeh_utils.isolate_neurons(self.core_network, neuron_population=neuron_pop, data_dir=self.data_dir)
                self.selected_mask = np.logical_or(self.selected_mask, subpop_mask)
        else:
            self.selected_mask = other_billeh_utils.isolate_neurons(self.core_network, neuron_population=self.neuron_population, data_dir=self.data_dir)  
        self.sim_data = {k:v[:, :, self.selected_mask] for k, v in self.sim_data.items()}
        # Plot the firing rate heatmap of the selected neurons
        myplots.firing_rate_heatmap(self.sim_data, self.sampling_interval, self.frequency, reverse=self.reverse,
                                    normalize_firing_rates=True, exclude_NS_neurons=True,
                                    path=self.pop_full_path)
        # Create arrays to convert between selected ids and tf ids
        self.n_selected_neurons = np.sum(self.selected_mask)
        self.selected_id_to_tf_id = np.arange(self.n_neurons)[self.core_mask][self.selected_mask]
        self.tf_id_to_selected_id = np.zeros(self.n_neurons, dtype=np.int32) - 1
        for selected_id, tf_id in enumerate(self.selected_id_to_tf_id):
            self.tf_id_to_selected_id[tf_id] = selected_id            
        self.selected_df = pd.DataFrame({'Tf index': self.selected_id_to_tf_id})

        # Plot random selected neurons response to movement
        path = os.path.join(self.pop_full_path, f'{self.neuron_population} neurons samples')
        for neuron_id in rd.choice(range(self.n_selected_neurons), size=10):
            tf_id = self.selected_id_to_tf_id[neuron_id]
            myplots.plot_neuron_voltage_and_spikes_and_input_current(self.sim_data, neuron_id, tf_id=tf_id, 
                                                                      fig_title=f'{self.neuron_population} neuron id:{tf_id}', 
                                                                      path=path)
            myplots.plot_neuron_input_currents(self.sim_data, neuron_id, tf_id=tf_id, 
                                                fig_title=f'{self.neuron_population} neuron currents id:{tf_id}', 
                                                path=path)
        ### Mean pre-stimulus analysis 
        if self.baseline_method == 'pre_perturbation':
            self.baseline_analysis(self.sim_data)
        elif self.baseline_method == 'static_gratings':
            self.baseline_analysis_static_gratings()
        # Histogram of the baseline values for neurons in the population              
        myplots.baseline_histogram(self.selected_df, path=self.pop_full_path)
        # Normalize the features with respect to their baseline and calculate firing rates and modulation index
        self.stimuli_variables_response()
        # Classify the neuron according to the selected criteria
        if self.classify_neurons:
            self.classify_neurons_method()
            # Include the preferred angles of every neuron
            self.selected_df['preferred_angle'] = np.zeros(len(self.selected_df)) 
            angle_tunning = other_billeh_utils.angle_tunning(self.network, data_dir=self.data_dir)
            self.selected_df['preferred_angle'] = angle_tunning[self.core_mask][self.selected_mask]
            myplots.preferred_angle_distribution(self.selected_df, path=self.pop_full_path)
            
            ### Compare the distributions of the input current responses for the three classes
            for feature in ['baseline_input_current', 'baseline_input_current_sd']:
                hVf_values = self.selected_df.loc[self.selected_df['class']=='hVf', feature].values
                dVf_values = self.selected_df.loc[self.selected_df['class']=='dVf', feature].values
                unc_values = self.selected_df.loc[self.selected_df['class']=='unclassified', feature].values
                # One-way ANOVA
                statistic, pvalue = stats.f_oneway(dVf_values, unc_values, hVf_values)
                significance_test_path = os.path.join(self.pop_full_path, 'classification_results', 'Significance test', feature)
                os.makedirs(significance_test_path, exist_ok=True)
                with open(os.path.join(significance_test_path, 'ANOVA_test.txt'), 'w') as out_file:
                    out_file.write(f'Results of the Oneway ANOVA test comparing the three classes of {self.neuron_population}:\n')
                    out_file.write(f'- Statistic: {statistic}\n')
                    out_file.write(f'- p-value: {pvalue}\n')
                    if pvalue > 0.05:
                        out_file.write('Since the pvalue is greater than 0.05 we consider that the ANOVA test is not significant and we will not perform pairwise Welch t-test.\n')
                # If ANOVA test result significative perform Welch t-tests on every pair
                if pvalue < 0.05:
                    pairs = [('dVf', 'unclassified'),
                             ('unclassified', 'hVf'),
                             ('dVf', 'hVf')]
                    significance_results = np.array([stats.ttest_ind(dVf_values, unc_values, equal_var=False),
                                                    stats.ttest_ind(unc_values, hVf_values, equal_var=False),
                                                    stats.ttest_ind(dVf_values, hVf_values, equal_var=False)])
                    statistics, pvalues = significance_results[:, 0], significance_results[:, 1]
                    # Plot the distributions of the feature for each class
                    path = os.path.join(self.pop_full_path, 'Neuron classes', 'Response per neuron class', 'Distributions')
                    os.makedirs(path, exist_ok=True)
                    myplots.input_current_classes_distributions(self.selected_df, feature, pairs, pvalues, 
                                                                filename=f'{feature}_classes_distributions', path=path)
                    with open(os.path.join(significance_test_path, 'Welch_t_test.txt'), 'w') as out_file:
                        out_file.write(f'Results of the Welch t-test comparing the pairs of {self.neuron_population} classes: \n')
                        for idx, pair in enumerate(pairs):
                            out_file.write(f'Pair: {pair[0]} vs {pair[1]}\n')
                            out_file.write(f'Statistic: {statistics[idx]}\n')
                            out_file.write(f'p-value: {pvalues[idx]}\n \n')
                        
            ### Make figures analyzing the different variables for each class
            if self.classification_criteria == 'input_current':
                if len(set(self.input_current_threshold)) == 1:
                    input_current_thresh = self.input_current_threshold[0]
                else:
                    input_current_thresh = None
            # Make plots of the different variables comparing the three classification classes
            path = os.path.join(self.pop_full_path, 'Neuron classes', 'Response per neuron class')
            myplots.plot_average_classes(self.sim_data, self.selected_df, 
                                          voltage_threshold=self.voltage_threshold, input_current_threshold=input_current_thresh,
                                          path=path)
            myplots.subplot_currents_average_classes(self.sim_data, self.selected_df, input_current_threshold=input_current_thresh,
                                                      path=path)
            myplots.plot_current_per_class(self.sim_data, self.selected_df, input_current_threshold=input_current_thresh,
                                            path=path)
            myplots.subplot_current_per_class(self.sim_data, self.selected_df, input_current_threshold=input_current_thresh,
                                              path=path)
            myplots.firing_rate_per_class(self.sim_data, self.selected_df, self.sampling_interval, 
                                          normalize_firing_rates=True, exclude_NS_neurons=True, path=path)
            # Select and plot the heatmap neurons responses
            self.heatmap_neurons_selection(random=True, neurons_per_class=10)
            path = os.path.join(self.pop_full_path, 'Neuron classes', 'Response per neuron class', 'Heatmap neurons traces')
            heatmap_df = self.selected_df.loc[np.logical_not(np.isnan(self.selected_df['heatmap_neurons']))]
            for neu_class in self.neuron_classes:
                tf_indices = heatmap_df.loc[(heatmap_df['class'] == neu_class), 'Tf index']   
                heatmap_indices = heatmap_df.loc[(heatmap_df['class'] == neu_class), 'heatmap_neurons']
                for neu_id, tf_id, heatmap_id in zip(tf_indices.index, tf_indices, heatmap_indices):
                    myplots.plot_neuron_voltage_and_spikes_and_input_current(self.sim_data, neu_id, tf_id=tf_id, 
                                                                              fig_title=f'{neu_class} heatmap id:{heatmap_id}', 
                                                                              path=path)
                    myplots.plot_neuron_input_currents(self.sim_data, neu_id, tf_id=tf_id, 
                                                        fig_title=f'{neu_class} neuron currents heatmap id:{heatmap_id}', 
                                                        path=path)
            # Make voltage heatmap    
            path = os.path.join(self.pop_full_path, 'Neuron classes', 'Response per neuron class', 'Heatmaps')
            myplots.voltage_heatmap(self.sim_data, self.selected_df, path=path)
            # Make heatmaps for every current variable
            for variable_label in ['input_current', 'recurrent_current', 'bottom_up_current', 'asc_bkg']:
                myplots.currents_heatmap(self.sim_data, variable_label, self.selected_df, path=path)
            # Composite figures of heatmaps and traces
            myplots.currents_figure(self.sim_data, self.selected_df, 
                                    input_current_threshold=input_current_thresh, path=path)  
            path = os.path.join(self.pop_full_path, 'Neuron classes', 'Smoothed firing rate')
            myplots.firing_rate_figure(self.sim_data, self.selected_df, self.sampling_interval, self.frequency, 
                                       normalize_firing_rates=True, exclude_NS_neurons=True, reverse=self.reverse,
                                       path=path)
            myplots.firing_rate_figure(self.sim_data, self.selected_df, self.sampling_interval, self.frequency, 
                                       normalize_firing_rates=True, exclude_NS_neurons=False, reverse=self.reverse,
                                       path=path)         
            myplots.firing_rate_figure(self.sim_data, self.selected_df, self.sampling_interval, self.frequency, 
                                       normalize_firing_rates=False, exclude_NS_neurons=True, reverse=self.reverse,
                                       path=path)          
            myplots.firing_rate_figure(self.sim_data, self.selected_df, self.sampling_interval, self.frequency, 
                                       normalize_firing_rates=False, exclude_NS_neurons=False, reverse=self.reverse,
                                       path=path)
            for extra_neuron_pop in ['i23Pvalb', 'i23Sst', 'e4', 'i4Pvalb']:
                extra_selected_mask = other_billeh_utils.isolate_neurons(self.core_network, neuron_population=extra_neuron_pop, data_dir=self.data_dir)
                extra_firing_rate = firing_rate[:,:, extra_selected_mask]
                myplots.firing_rate_figure(self.sim_data, self.selected_df, self.sampling_interval, self.frequency, 
                                           normalize_firing_rates=True, exclude_NS_neurons=True, 
                                           extra_neuron_pop=extra_neuron_pop, extra_firing_rates=extra_firing_rate, 
                                           reverse=self.reverse, path=path)
        else:
            path = os.path.join(self.pop_full_path, 'Figures without classification')
            self.heatmap_neurons_selection(random=True)
            heatmap_df = self.selected_df.loc[np.logical_not(np.isnan(self.selected_df['heatmap_neurons']))]
            myplots.voltage_heatmap(self.sim_data, self.selected_df, path=path)
            variables = ['input_current', 'recurrent_current', 'bottom_up_current', 'asc_bkg']
            for variable in variables:
                myplots.currents_figure_with_stimulus(self.sim_data, self.selected_df, variable, self.frequency, reverse=self.reverse, path=path)
            
        self.create_latex_table()
        path = os.path.join(self.pop_full_path, 'classification_results')
        os.makedirs(path, exist_ok=True)
        file_management.save_lzma(self.selected_df, f'{self.neuron_population}_selected_df', path)

    def baseline_analysis(self, data, stimuli_init_time=500):
        # We take the baseline response 500 ms prior to the stimulus
        for key,val in data.items():
            self.selected_df['baseline_'+key] = np.mean(val[:, 0:stimuli_init_time, :], axis=(0, 1))
        self.selected_df['baseline_input_current_sd'] = np.std(data['input_current'][:, 0:stimuli_init_time, :], axis=(0, 1))
        self.selected_df['baseline_raw_firing_rate'] = np.mean(data['z'][:, 0:stimuli_init_time, :], axis=(0, 1))*stimuli_init_time

    def baseline_analysis_static_gratings(self):
        directory = f'orien_{str(self.orientation)}_freq_0_reverse_{self.reverse}_rec_{self.n_neurons}'    
        full_static_data_path = os.path.join(self.simulation_results, directory, 'Data')
        try:
            os.path.exists(full_static_data_path)
        except ValueError:
            print("The static simulation has not been run before.")
        # Load the simulation results
        static_sim_data, _, _ = other_billeh_utils.load_simulation_results_hdf5(full_static_data_path, n_simulations=self.n_simulations_init, 
                                                                                skip_first_simulation=self.skip_first_simulation, variables=self.variables)
        static_sim_data['asc_bkg'] = static_sim_data['input_current'] - static_sim_data['recurrent_current'] - static_sim_data['bottom_up_current']
        
        
        # variables = ['v', 'input_current', 'recurrent_current', 'bottom_up_current']
        # static_sim_data, _ = other_billeh_utils.load_simulation_results(full_static_data_path, n_simulations = self.n_simulations_init, skip_first_simulation=True,
        #                                                                 variables=variables, simulation_length=self.simulation_length, n_neurons=self.n_core_neurons)
        # static_sim_data['asc_bkg'] = static_sim_data['input_current'] - static_sim_data['recurrent_current'] - static_sim_data['bottom_up_current']
        # Load the spike trains and smooth them
        # variables = ['z']
        # z_static_gratings, _ = other_billeh_utils.load_simulation_results(full_static_data_path, n_simulations = self.n_simulations_init, skip_first_simulation=True,
        #                                                                   variables=variables, simulation_length=self.simulation_length, n_neurons=self.n_neurons)
        static_sim_data['z'] = static_sim_data['z'][:,:, self.core_mask]
        static_firing_rate, static_sampling_interval = other_billeh_utils.firing_rates_smoothing(static_sim_data['z']) 
        static_sim_data['firing_rate'] = static_firing_rate
        # Do spike based correction to the voltages
        static_sim_data['v'] = other_billeh_utils.voltage_spike_effect_correction(static_sim_data['v'], static_sim_data['z'])
        # Select the values for the selected population
        static_sim_data = {k:v[:, :, self.selected_mask] for k, v in static_sim_data.items()}
        # Calculate the baseline values
        self.baseline_analysis(static_sim_data, stimuli_init_time=2500)

    def stimuli_variables_response(self, stimuli_init_time=500, stimuli_end_time=1500):
        # Normalize the variables with respect to their baseline values
        stimuli_dur = stimuli_end_time - stimuli_init_time
        for key in self.sim_data.keys():
            if key not in ['z', 'firing_rate']:
                self.sim_data[key] -= np.array(self.selected_df['baseline_'+key])
                self.selected_df['response_'+key] = np.mean(self.sim_data[key][:, stimuli_init_time:stimuli_end_time, :], axis=(0, 1))
                fig = plt.figure()
                plt.hist(self.selected_df['response_'+key], bins=50)
                fig.savefig(os.path.join(self.pop_full_path, f'{key}_response_histogram.png'), dpi=300, transparent=True)
                plt.close()
        self.selected_df['response_raw_firing_rate'] = np.mean(self.sim_data['z'][:, stimuli_init_time:stimuli_end_time, :], axis=(0, 1))*stimuli_dur
        
        # Calculate the firing rate ratio between the stimuli and baseline values
        firing_rate_ratio = np.divide(self.selected_df['response_raw_firing_rate'], self.selected_df['baseline_raw_firing_rate'], 
                                      out=np.full_like(self.selected_df['response_raw_firing_rate'], np.nan), where=self.selected_df['baseline_raw_firing_rate']!=0)
        self.selected_df['firing_rate_ratio'] = firing_rate_ratio
        myplots.firing_rate_histogram(firing_rate_ratio, self.firing_ratio_threshold, path=self.pop_full_path)
        
        # Identify the most responsive neuron
        most_responsive_neuron_id = np.nanargmax(firing_rate_ratio) 
        most_responsive_neuron_tf_id = self.selected_id_to_tf_id[most_responsive_neuron_id]  
        path = os.path.join(self.pop_full_path, f'{self.neuron_population} neurons samples', 'Most excited neuron')
        myplots.plot_neuron_voltage_and_spikes_and_input_current(self.sim_data, most_responsive_neuron_id, tf_id=most_responsive_neuron_tf_id,
                                                                  fig_title=f'Most excited {self.neuron_population} neuron id:{most_responsive_neuron_tf_id}', 
                                                                  path=path)
        myplots.plot_neuron_input_currents(self.sim_data, most_responsive_neuron_id, tf_id=most_responsive_neuron_tf_id,
                                            fig_title=f'Most excited {self.neuron_population} neuron id:{most_responsive_neuron_tf_id}', 
                                            path=path)
        # Calculate the modulation index
        num = self.selected_df['response_raw_firing_rate'] - self.selected_df['baseline_raw_firing_rate']
        den = self.selected_df['response_raw_firing_rate'] + self.selected_df['baseline_raw_firing_rate']
        modulation_index = np.divide(num, den, out=np.full_like(num, np.nan), where=den!=0)
        self.selected_df['modulation_index'] = modulation_index
        myplots.modulation_index_histogram(modulation_index, path=self.pop_full_path)

    def classify_neurons_method(self):
        if self.classification_criteria == 'v':
            self.hVf_neurons_mask = self.selected_df['response_v'] < -self.voltage_threshold
            self.dVf_neurons_mask = self.selected_df['response_v'] > self.voltage_threshold
            self.unclassified_neurons_mask = np.logical_not(np.logical_or(self.hVf_neurons_mask, self.dVf_neurons_mask))
        
        elif self.classification_criteria == 'firing_rate':
            self.hVf_neurons_mask = self.selected_df['firing_rate_ratio'] < 1/self.firing_ratio_threshold
            self.dVf_neurons_mask = self.selected_df['firing_rate_ratio'] > self.firing_ratio_threshold            
            self.unclassified_neurons_mask = np.logical_not(np.logical_or(self.hVf_neurons_mask, self.dVf_neurons_mask))
            
        elif self.classification_criteria == 'modulation_index':
            self.hVf_neurons_mask = self.selected_df['modulation_index'] < -self.modulation_index_threshold
            self.dVf_neurons_mask = self.selected_df['modulation_index'] > self.modulation_index_threshold
            self.unclassified_neurons_mask = np.logical_not(np.logical_or(self.hVf_neurons_mask, self.dVf_neurons_mask)) 
            
        elif self.classification_criteria == 'input_current':
            # Define the input current threshold based on the cells 
            # rheobase (minimum amount of current required for a spike)
            node_type_ids = self.core_network['node_type_ids']
            rheobase = self.core_network['node_params']['rheobase']*10**12 
            factor = 0.05
            self.input_current_threshold = factor*rheobase[node_type_ids][self.selected_mask]
            self.hVf_neurons_mask = self.selected_df['response_input_current'] < -self.input_current_threshold
            self.dVf_neurons_mask = self.selected_df['response_input_current'] > self.input_current_threshold
            self.unclassified_neurons_mask = np.logical_not(np.logical_or(self.hVf_neurons_mask, self.dVf_neurons_mask))
        else:
            print('The classification criteria chosen is not valid. Try one of the following: v, firing_rate_ratio, modulation_index, input_current')
            
        self.neuron_classes = ['hVf', 'dVf', 'unclassified']
        neurons_mask = [self.hVf_neurons_mask, self.dVf_neurons_mask, self.unclassified_neurons_mask]
        neurons_color = ['#F06233', '#33ABA2', '#9CB0AE']
        for label, mask, color in zip(self.neuron_classes, neurons_mask, neurons_color):
            self.selected_df.loc[mask, 'class'] = label
            self.selected_df.loc[mask, 'color'] = color
    
    def heatmap_neurons_selection(self, random=False, neurons_per_class=10):
        if random:
            n_sample_neurons = 30
            self.neurons_selected_indices = rd.choice(self.selected_df.index, size=n_sample_neurons, replace=False)
        else:
            n_classes = len(self.neuron_classes)
            n_sample_neurons = n_classes*neurons_per_class
            self.neurons_selected_indices = np.zeros(n_sample_neurons)
            for class_id, neu_class in enumerate(self.neuron_classes):
                class_stats = self.selected_df.loc[self.selected_df['class']==neu_class]
                class_indices = class_stats.index
                class_sample_indices = rd.choice(class_indices, size=neurons_per_class, replace=False)
                self.neurons_selected_indices[neurons_per_class*class_id:neurons_per_class*(class_id+1)] = class_sample_indices
        if self.classification_criteria in ['v', 'input_current']:
            mean_response = self.selected_df[f'response_{self.classification_criteria}'][self.neurons_selected_indices]
        else:
            mean_response = self.selected_df[self.classification_criteria][self.neurons_selected_indices]
        self.sorted_selected_indices = [x for _, x in sorted(zip(mean_response, self.neurons_selected_indices), key=lambda element: (element[0]))] 
        self.selected_df['heatmap_neurons'] = np.nan
        self.selected_df.loc[self.sorted_selected_indices, 'heatmap_neurons'] = np.arange(n_sample_neurons) 
    
    def create_latex_table(self):
        variables = ['v', 'input_current', 'recurrent_current', 'bottom_up_current', 'asc_bkg']
        new_df = self.selected_df.copy()
        for variable in variables:
            new_df[f'response_{variable}'] += new_df[f'baseline_{variable}']
        path = os.path.join(self.pop_full_path, 'classification_results')
        os.makedirs(path, exist_ok=True)
        if 'class' not in new_df.keys():
            new_df['class'] = np.full(len(new_df), 'all') 
        baseline_mean_df = new_df.groupby(['class'])[['baseline_v', 'baseline_raw_firing_rate', 'baseline_recurrent_current', 
                                                    'baseline_bottom_up_current', 'baseline_input_current', 'baseline_asc_bkg']].mean().reset_index()
        baseline_sem_df = new_df.groupby(['class'])[['baseline_v', 'baseline_raw_firing_rate', 'baseline_recurrent_current', 
                                                    'baseline_bottom_up_current', 'baseline_input_current', 'baseline_asc_bkg']].sem().reset_index()
        drifting_mean_df = new_df.groupby(['class'])[['response_v','response_raw_firing_rate', 'firing_rate_ratio', 'modulation_index',
                                                     'response_recurrent_current', 'response_bottom_up_current', 
                                                     'response_input_current', 'response_asc_bkg']].mean().reset_index()
        drifting_sem_df = new_df.groupby(['class'])[['response_v','response_raw_firing_rate', 'firing_rate_ratio', 'modulation_index',
                                                    'response_recurrent_current', 'response_bottom_up_current', 
                                                    'response_input_current', 'response_asc_bkg']].sem().reset_index()    
        with open(os.path.join(path,"baseline_latex_table.txt"), "w") as f:
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> New commiting attempt
            f.write(" & ".join(baseline_mean_df.columns))
            for (i, row), (j, row2) in zip(baseline_mean_df.iterrows(), baseline_sem_df.iterrows()):
                f.write(" & ".join([x[:3] if type(x) == str else "%.2f"%x + r"$\pm$" + "%.2f"%row2[idx] for idx, x in enumerate(row.values)]) + " \\\\\n")
        with open(os.path.join(path,"drifting_latex_table.txt"), "w") as f:
<<<<<<< HEAD
=======
=======
            f.write(" & ".join(baseline_mean_df.columns) + " \\\\\n")
            for (i, row), (j, row2) in zip(baseline_mean_df.iterrows(), baseline_sem_df.iterrows()):
                f.write(" & ".join([x[:3] if type(x) == str else "%.2f"%x + r"$\pm$" + "%.2f"%row2[idx] for idx, x in enumerate(row.values)]) + " \\\\\n")
        with open(os.path.join(path,"drifting_latex_table.txt"), "w") as f:
            f.write(" & ".join(drifting_mean_df.columns) + " \\\\\n")
>>>>>>> Last changes, mainly based on perturbation analysis
>>>>>>> New commiting attempt
            for (i, row), (j, row2) in zip(drifting_mean_df.iterrows(), drifting_sem_df.iterrows()):
                f.write(" & ".join([x[:3] if type(x) == str else "%.2f"%x + r"$\pm$" + "%.2f"%row2[idx] for idx, x in enumerate(row.values)]) + " \\\\\n")
        with open(os.path.join(path,"neurons_per_class.txt"), "w") as f:
             class_numbers = new_df.groupby(['class'])['class'].count()  
             f.write(class_numbers.to_frame().style.to_latex())
             f.write((class_numbers/self.n_selected_neurons).to_frame().style.to_latex())
       

def main(flags):
    neurons_classifier = DynamicNeuronClasses(flags)
    neurons_classifier()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define key flags')
    parser.add_argument('--gratings_orientation', type=int, choices=range(0, 360, 45), default=0)
    parser.add_argument('--gratings_frequency', type=int, default=2)
    parser.add_argument('--neurons', type=int, default=230924)
    parser.add_argument('--n_simulations', type=int, default=None)
    parser.add_argument('--n_simulations_static_gratings', type=int, default=None)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false')
    parser.set_defaults(reverse=False)
    parser.add_argument('--skip_first_simulation', action='store_true')
    parser.add_argument('--no-skip_first_simulation', dest='skip_first_simulation', action='store_false')
    parser.set_defaults(skip_first_simulation=True)
    parser.add_argument('--classify_neurons', action='store_true')
    parser.add_argument('--no-classify_neurons', dest='classify_neurons', action='store_false')
    parser.set_defaults(classify_neurons=True)    
    parser.add_argument('--correct_membrane_voltage', action='store_true')
    parser.add_argument('--no-correct_membrane_voltage', dest='correct_membrane_voltage', action='store_false')
    parser.set_defaults(correct_membrane_voltage=True)   
    parser.add_argument('--classification_criteria', type=str, default='input_current')
    parser.add_argument('--neuron_population', type=str, default='e23')
    
    flags = parser.parse_args()
    main(flags)
    
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # # cProfile.run('main(flags)')
    # main(flags)  
    # profiler.disable()
    # profiler.dump_stats('example.stats')
    
    # stats = pstats.Stats('example.stats')
    # stats.sort_stats('cumtime').print_stats(20)
    #stats.print_callees(​"cprofile_example.py:3"​)
    
