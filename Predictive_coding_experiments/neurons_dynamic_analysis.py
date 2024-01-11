#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:49:17 2022

@author: jgalvan
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management
sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
import other_billeh_utils
import load_sparse

mpl.style.use('default')
np.random.seed(3000)


# Define a function to rename the neurons populations
def renaming_neuron_populations(pop_names):
    pop_names = pop_names.astype('<U8')
    u2_pop_names = pop_names.astype('<U2')
    u1_pop_names = u2_pop_names.astype('<U1')
    e_mask = u1_pop_names == 'e'
    pop_names[e_mask] = u2_pop_names[e_mask]
    
    return pop_names


class DynamicNeuronAnalysis:    
    def __init__(self, flags):
        self.orientation = flags.gratings_orientation
        self.frequency = flags.gratings_frequency
        self.reverse = flags.reverse
        self.n_neurons = flags.neurons
        self.neuron_population = flags.neuron_population
        self.compare_neurons = flags.compare_neurons
        self.n_simulations_init = flags.n_simulations
        self.skip_first_simulation = flags.skip_first_simulation
        self.stimuli_init_time = flags.stimuli_init_time
        self.stimuli_end_time = flags.stimuli_end_time
        self.simulation_results = 'Simulation_results'
        self.directory = f'orien_{str(flags.gratings_orientation)}_freq_{str(flags.gratings_frequency)}_reverse_{str(flags.reverse)}_rec_{flags.neurons}'
        self.full_path = os.path.join(self.simulation_results, self.directory)
        self.pop_full_path = os.path.join(self.full_path, self.neuron_population)
        os.makedirs(self.pop_full_path , exist_ok=True)
        # Load the spike trains of the LGN and V1
        self.full_data_path = os.path.join(self.full_path, 'Data', 'simulation_data.hdf5')
        self.variables = ['z', 'z_lgn']
        self.sim_data, self.sim_metadata, self.n_simulations = other_billeh_utils.load_simulation_results_hdf5(self.full_data_path, n_simulations=self.n_simulations_init, 
                                                                                                               skip_first_simulation=self.skip_first_simulation, variables=self.variables)
        self.data_dir = self.sim_metadata['data_dir']
        self.data_dir = '/home/jgalvan/Desktop/Neurocoding/V1_Billeh_model/GLIF_network'
        
    def __call__(self):
        # Load the model network
        load_fn = load_sparse.cached_load_billeh
        _, self.network, _, _ = load_fn(self.sim_metadata['n_input'], self.n_neurons, self.sim_metadata['core_only'], 
                                        self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                        n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
           
        # Calculate the mean spiking activity previous to and during the stimulus for z and z_lgn
        mean_z = np.mean(self.sim_data['z'], axis=0)
        self.mean_z_pre = np.sum(mean_z[:self.stimuli_init_time, :], axis=0)
        self.mean_z_stim = np.sum(mean_z[self.stimuli_init_time:self.stimuli_end_time, :], axis=0)
        mean_z_lgn = np.mean(self.sim_data['z_lgn'], axis=0)
        self.mean_z_lgn_pre = np.sum(mean_z_lgn[:self.stimuli_init_time, :], axis=0)
        self.mean_z_lgn_stim = np.sum(mean_z_lgn[self.stimuli_init_time:self.stimuli_end_time, :], axis=0)
        # Load the population stats dataframe
        classification_path = os.path.join(self.full_path, self.neuron_population, 'classification_results')
        self.selected_df = file_management.load_lzma(os.path.join(classification_path, f'{self.neuron_population}_selected_df.lzma'))
        # Load recurrent network and rename the V1neuron populations
        network_path = os.path.join(os.path.dirname(os.getcwd()), f'Topological_analysis_{self.n_neurons}')
        self.recurrent_network = file_management.load_lzma(os.path.join(network_path, 'recurrent_network.lzma'))
        self.recurrent_network['Target type'] = renaming_neuron_populations(self.recurrent_network['Target type'].values)
        self.recurrent_network['Source type'] = renaming_neuron_populations(self.recurrent_network['Source type'].values)
        # Load input network and rename the V1 neuron populations
        self.input_network = file_management.load_lzma(os.path.join(network_path, 'input_network.lzma'))
        self.input_network['Target type'] = renaming_neuron_populations(self.input_network['Target type'].values)

        # If there exists a classification in the neuron population then analyze each class 
        # and analyze the neuron population otherwise
        if 'class' in self.selected_df.columns:
            classes_labels = set(self.selected_df['class'])
            # Rename the neurons if they belong to a particular class (hVf, dVf or unclassified)
            for neu_class in classes_labels:
                class_stats = self.selected_df.loc[self.selected_df['class']==neu_class]
                class_tf_indices = class_stats['Tf index'].values
                self.recurrent_network.loc[self.recurrent_network['Source'].isin(class_tf_indices), 'Source type'] = neu_class
                self.recurrent_network.loc[self.recurrent_network['Target'].isin(class_tf_indices), 'Target type'] = neu_class
                self.input_network.loc[self.input_network['Target'].isin(class_tf_indices), 'Target type'] = neu_class
            # Analyze the static and dynamic weights of each class
            for neu_class in classes_labels:
                pop_names = [neu_class]
                path = os.path.join(self.pop_full_path, neu_class)
                os.makedirs(path, exist_ok=True)
                class_dynamic_df = self.static_dynamic_analysis(pop_names, neu_class, save_results_df=True, path=path)
                self.static_dynamic_analysis_figures(class_dynamic_df, neu_class)
        else:
            all_populations_name = list(set(self.recurrent_network['Target type']))
            pop_names = [s for s in all_populations_name if self.neuron_population in s]
            dynamic_df = self.static_dynamic_analysis(pop_names, self.neuron_population, save_results_df=True, path=self.pop_full_path)
            self.static_dynamic_analysis_figures(dynamic_df, self.neuron_population)
            
        # Compare the static (fig1) and dynamic (fig2) weights between different populations
        if len(self.compare_neurons.split('_')) == 1:
            comparison_pop = [pop_name for pop_name in all_populations_name if self.compare_neurons in pop_name]
        else:
            comparison_pop = self.compare_neurons.split('_')
        ncols = len(comparison_pop)
        fig1, axes1 = plt.subplots(nrows=1, ncols=ncols, figsize=(5*ncols, 4), sharey=True)
        fig2, axes2 = plt.subplots(nrows=1, ncols=ncols, figsize=(5*ncols, 4), sharey=True)
        for idx, neuron_pop in enumerate(comparison_pop):
            pop_names = [neuron_pop]
            dynamic_df = self.static_dynamic_analysis(pop_names, neuron_pop, save_results_df=False)
            self.static_weights_histogram(axes1[idx], dynamic_df)
            self.dynamic_weights_histogram(axes2[idx], dynamic_df, ordering='laminar')  
            if idx!=0:
                axes1[idx].yaxis.label.set_visible(False)
                axes2[idx].yaxis.label.set_visible(False)

        for ax in axes2:
            # Shade positive and negative weights areas
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            # y_min = min(cdf['influence'])
            # y_max = max(cdf['influence'])
            ax.fill_between(x=[x_min, x_max], y1=0, y2=y_max, color='lightcoral', alpha=0.2, zorder=0)
            ax.fill_between(x=[x_min, x_max], y1=0, y2=y_min, color='lightblue', alpha=0.2, zorder=0)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Save both figures        
        path = os.path.join(self.full_path, 'Topological images', 'Comparison dynamic analysis', self.compare_neurons)
        os.makedirs(path, exist_ok=True)
        for figure, fn in zip([fig1, fig2], ['static_histogram.png', 'dynamic.png']):
            figure.tight_layout()
            figure.savefig(os.path.join(path, fn), dpi=300, transparent=True)
            plt.close(figure)
    
    def dynamic_weight_calculation(self, pop_names, connectivity_network,
                                   pre_presynaptic_spikes, stim_presynaptic_spikes):
        for idx, name in enumerate(pop_names):
            if idx == 0:
                connectivity_network_pop = connectivity_network.loc[connectivity_network['Target type']==name].copy()
            else:
                connectivity_network_pop = pd.concat([connectivity_network_pop, connectivity_network.loc[connectivity_network['Target type']==name].copy()])
        # Determine the source_ids, their spikes and the connection weights
        source_ids = connectivity_network_pop['Source']
        weights = connectivity_network_pop['Weight']
        # Calculate the dynamic weight previous to the stimulus
        pre_influence = pre_presynaptic_spikes[source_ids]*weights/self.stimuli_init_time
        connectivity_network_pop['pre_influence'] = None
        connectivity_network_pop.loc[:, 'pre_influence'] = pre_influence
        # Calculate the dynamic weight during the stimulus
        stimuli_dur = self.stimuli_end_time - self.stimuli_init_time
        stim_influence = stim_presynaptic_spikes[source_ids]*weights/stimuli_dur
        connectivity_network_pop['stim_influence'] = None
        connectivity_network_pop.loc[:, 'stim_influence'] = stim_influence
        
        return connectivity_network_pop
    
    def static_weights_histogram(self, axes, whole_network_pop, label=None, ordering='laminar'):
        whole_network_pop = whole_network_pop.reset_index(level=1)
        
        if ordering == 'laminar':
            true_order = ['LGN unit', 'i1Htr3a', 'dVf', 'hVf', 'unclassified', 
                          'i23Htr3a', 'i23Pvalb', 'i23Sst',
                          'e4', 'i4Htr3a', 'i4Pvalb', 'i4Sst', 
                          'e5', 'i5Htr3a', 'i5Pvalb', 'i5Sst',
                          'e6', 'i6Htr3a', 'i6Pvalb', 'i6Sst', 'Total']
            
        # elif ordering == 'ascent':
        #     selection = synapses_weight_per_neuron.loc[significative_pops]
        #     selection.sort_values(inplace=True)
                        
        sns.boxplot(y='Weight', x='Source type',
            data=whole_network_pop, 
            order=true_order,
            ax=axes,
            showfliers = False,#fliersize=2,
            color="g")
        axes.get_xticklabels()[-1].set_weight("bold")
        axes.set_ylabel(r'Total synaptic weight $[pA]$', fontsize=14)
        axes.xaxis.label.set_visible(False)
        axes.legend().set_visible(False)
        if label is not None:
            axes.set_title(label, fontsize=12)
        plt.setp(axes.get_xticklabels(), rotation=90)
        axes.tick_params(axis='both',          
                         which='both',     
                         labelsize=12) 
        
    def dynamic_weights_histogram(self, axes, whole_network_pop, label=None, ordering='laminar'):
        
        whole_network_pop = whole_network_pop.reset_index(level=1)
        df1 = whole_network_pop[['Source type', 'pre_influence']].assign(Trial='Baseline')
        df1.columns = ['Source type', 'influence', 'Period']
        df2 = whole_network_pop[['Source type', 'stim_influence']].assign(Trial='Drifting gratings')
        df2.columns = ['Source type', 'influence', 'Period']
        cdf = pd.concat([df1, df2])
       
        if ordering == 'laminar':
            true_order = ['LGN unit', 'i1Htr3a', 'dVf', 'hVf', 'unclassified', 
                          'i23Htr3a', 'i23Pvalb', 'i23Sst',
                          'e4', 'i4Htr3a', 'i4Pvalb', 'i4Sst', 
                          'e5', 'i5Htr3a', 'i5Pvalb', 'i5Sst',
                          'e6', 'i6Htr3a', 'i6Pvalb', 'i6Sst', 'Total']
            
        elif ordering == 'relevant':
            drifting_cdf = cdf.loc[cdf['Period']=='Drifting gratings']
            drifting_cdf = drifting_cdf.loc[drifting_cdf['Source type']!='Total']
            drifting_cdf['influence'] = drifting_cdf['influence'].abs()
            drifting_cdf = drifting_cdf.groupby('Source type')['influence'].sum()
            drifting_cdf = drifting_cdf.sort_values(ascending=False)
            drifting_cdf = drifting_cdf.cumsum()/drifting_cdf.sum()
            # select the ones which explain 95% of the total influence
            drifting_cdf = drifting_cdf[drifting_cdf<0.95]
            # isolate this source types from cdf dataframe
            drifting_cdf_index = drifting_cdf.index.values.tolist()
            drifting_cdf_index.append('Total')  
            cdf = cdf.loc[cdf['Source type'].isin(drifting_cdf_index)]

            true_order = ['LGN unit', 'i1Htr3a', 'dVf', 'hVf', 'unclassified', 
                'i23Htr3a', 'i23Pvalb', 'i23Sst',
                'e4', 'i4Htr3a', 'i4Pvalb', 'i4Sst', 
                'e5', 'i5Htr3a', 'i5Pvalb', 'i5Sst',
                'e6', 'i6Htr3a', 'i6Pvalb', 'i6Sst', 'Total']

            # from this list sets , select the ones contained in drifting_cdf source types
            true_order = [x for x in true_order if x in drifting_cdf_index]

        # elif ordering == 'max_variation':
        #     sel_rec['influence abs_change'] = np.abs(sel_rec['stim_influence'] - sel_rec['pre_influence'])
        #     selection = sel_rec.loc[significative_pops]
        #     selection.sort_values(by=['influence abs_change'], inplace=True)

        static_color = '#CCCCCC'  # Calm and subdued color
        moving_color = 'mediumseagreen'     # Vibrant and energetic color

        # Create a custom Seaborn palette with these colors
        custom_palette = [static_color, moving_color]
        # original_palette = 'CMRmap_r'
        sns.boxplot(y='influence', x='Source type',
                    hue = 'Period',
                    data=cdf, 
                    order=true_order,
                    ax=axes,
                    showfliers = False,#fliersize=2,
                    palette=custom_palette)
        
        axes.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=0)

        axes.get_xticklabels()[-1].set_weight("bold")
        axes.set_ylabel(r'$W^{eff} ~~ [pA/ms]$', fontsize=14)
        axes.xaxis.label.set_visible(False)
        # Modify the legend labels
        handles, labels = axes.get_legend_handles_labels()
        # Change the label for the static gratings
        labels[0] = 'Static gratings'
        axes.legend(handles, labels, loc='upper left').set_visible(True)
        if label is not None:
            axes.set_title(label, fontsize=14)
        plt.setp(axes.get_xticklabels(), rotation=90)
        axes.tick_params(axis='both',          
                         which='both',     
                         labelsize=12) 
        # axes.set_xlim(x_min, x_max)
        # axes.set_ylim(y_min, y_max)

    def static_dynamic_analysis(self, pop_names, population_label, save_results_df=False, path=''):
        # Determine the dynamic weights for the chosen neuron population
        self.recurrent_network_pop = self.dynamic_weight_calculation(pop_names, self.recurrent_network, self.mean_z_pre, self.mean_z_stim)
        self.input_network_pop = self.dynamic_weight_calculation(pop_names, self.input_network, self.mean_z_lgn_pre, self.mean_z_lgn_stim)
        whole_network_pop = pd.concat([self.input_network_pop, self.recurrent_network_pop])        
        # Count number of neurons in the population (assuming every neuron has at least one synapse)
        # and normalize per neuron the different features
        # self.n_pop = len(set(whole_network_pop['Target']))       
        #whole_network_pop_grouped = whole_network_pop.groupby(['Source type']).agg('sum')/self.n_pop
        whole_network_pop_grouped = whole_network_pop.groupby(['Target', 'Source type']).agg('sum')#/self.n_pop
        whole_network_pop_grouped.drop('Source', axis=1, inplace=True)
        total_contrib = whole_network_pop_grouped.groupby(level=0).sum()
        total_contrib['Source type'] = 'Total'
        total_contrib.reset_index(inplace=True)
        total_contrib.set_index(['Target', 'Source type'], inplace=True)
        whole_network_pop_grouped = pd.concat([whole_network_pop_grouped, total_contrib])  
        whole_network_pop_grouped.sort_index(inplace=True) 
        if save_results_df:
            file_management.save_lzma(whole_network_pop, f'{population_label}_recurrent_dynamic_network', path)
            file_management.save_lzma(whole_network_pop_grouped, f'{population_label}_static_dynamic_analysis', path)
        
        return whole_network_pop_grouped
    
    def static_dynamic_analysis_figures(self, whole_network_pop, population_label):
        # Save the static weights distribution
        images_path = os.path.join(self.full_path, 'Topological images', 'Individual dynamic analysis', f'{population_label}')
        os.makedirs(images_path, exist_ok=True)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
        self.static_weights_histogram(axes, whole_network_pop)
        fig.tight_layout()
        fig.savefig(os.path.join(images_path, 'static_histogram.png'), dpi=300, transparent=True)
        plt.close(fig)
        # Save the dynamic weights distribution
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
        self.dynamic_weights_histogram(axes, whole_network_pop, ordering='relevant')
        fig.tight_layout()
        fig.savefig(os.path.join(images_path, 'dynamic_histogram.png'), dpi=300, transparent=True)
        plt.close(fig)
        
def main(flags):
    neurons_dynamic_analysis = DynamicNeuronAnalysis(flags)
    neurons_dynamic_analysis()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Define key flags')
    parser.add_argument('--gratings_orientation', type=int, choices=range(0, 360, 45), default=0)
    parser.add_argument('--gratings_frequency', type=int, default=2)
    parser.add_argument('--neurons', type=int, default=230924)
    parser.add_argument('--stimuli_init_time', type=int, default=500)
    parser.add_argument('--stimuli_end_time', type=int, default=1500)
    parser.add_argument('--n_simulations', type=int, default=None)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false')
    parser.set_defaults(reverse=False)
    parser.add_argument('--skip_first_simulation', action='store_true')
    parser.add_argument('--no-skip_first_simulation', dest='skip_first_simulation', action='store_false')
    parser.set_defaults(skip_first_simulation=True) 
    parser.add_argument('--classification_criteria', type=str, default='Our')
    parser.add_argument('--neuron_population', type=str, default='e23')
    parser.add_argument('--compare_neurons', type=str, default='e23')
    
    flags = parser.parse_args()
    main(flags)  
    
    