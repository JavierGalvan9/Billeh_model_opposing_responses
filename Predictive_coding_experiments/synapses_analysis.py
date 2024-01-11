#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:19:51 2021

@author: jgalvan
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import plotting_figures as myplots
parentDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management
sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
import other_billeh_utils
import load_sparse


# Define a function to rename the neurons populations
def renaming_neuron_populations(pop_names):
    pop_names = pop_names.astype('<U8')
    u2_pop_names = pop_names.astype('<U2')
    u1_pop_names = u2_pop_names.astype('<U1')
    e2_mask = u2_pop_names =='e2'
    e_mask = u1_pop_names == 'e'
    pop_names[e_mask] = u2_pop_names[e_mask]
    pop_names[e2_mask] = 'e23'
    
    return pop_names


class SynapsesAnalysis:
    def __init__(self, flags):
        self.orientation = flags.gratings_orientation
        self.frequency = flags.gratings_frequency
        self.reverse = flags.reverse
        self.n_neurons = flags.neurons
        self.neuron_population = flags.neuron_population
        self.simulation_results = 'Simulation_results'
        self.network_path = os.path.join(os.path.dirname(os.getcwd()), f'Topological_analysis_{self.n_neurons}')
        self.directory = f'orien_{str(flags.gratings_orientation)}_freq_{str(flags.gratings_frequency)}_reverse_{str(flags.reverse)}_rec_{flags.neurons}'
        self.full_path = os.path.join(self.simulation_results, self.directory)
        self.pop_full_path = os.path.join(self.full_path, self.neuron_population)
        self.full_data_path = os.path.join(self.full_path, 'Data', 'simulation_data.hdf5')
        self.classification_path = os.path.join(self.pop_full_path, 'classification_results')
        self.topology_dir = os.path.join(self.pop_full_path, 'Topological analysis')
        os.makedirs(self.topology_dir, exist_ok=True)
        self.images_path = os.path.join(self.pop_full_path, 'Topological images')
        os.makedirs(self.images_path, exist_ok=True)
        # Load the simulation configuration attributes
        self.sim_metadata = {}
        with h5py.File(self.full_data_path, 'r') as f:
            dataset = f['Data']
            self.sim_metadata.update(dataset.attrs)
        self.data_dir = self.sim_metadata['data_dir']
        self.data_dir = '/home/jgalvan/Desktop/Neurocoding/V1_Billeh_model/GLIF_network'
        
    def __call__(self):
        # Load the model network and connectivity analysis
        load_fn = load_sparse.cached_load_billeh
        _, self.network, _, _ = load_fn(self.sim_metadata['n_input'], self.n_neurons, self.sim_metadata['core_only'], 
                                        self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                        n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        self.network_info = file_management.load_lzma(os.path.join(self.network_path, 'network_info.lzma'))
        self.weighted_network_info = file_management.load_lzma(os.path.join(self.network_path, 'weighted_network_info.lzma'))
        # Load the recurrent_network and rename the V1 neuron populations
        self.recurrent_network = file_management.load_lzma(os.path.join(self.network_path, 'recurrent_network.lzma'))
        self.recurrent_network['Target type'] = renaming_neuron_populations(self.recurrent_network['Target type'].values)
        self.recurrent_network['Source type'] = renaming_neuron_populations(self.recurrent_network['Source type'].values)
        # Identify all the neuron populations that relate to the given neuron_population
        all_populations_name = list(set(self.recurrent_network['Target type']))
        pop_names = [s for s in all_populations_name if self.neuron_population in s]
        # Load the selected_df of the selected population
        self.core_mask = other_billeh_utils.isolate_core_neurons(self.network, data_dir=self.data_dir)
        classification_file = f'{self.neuron_population}_selected_df.lzma'
        path = os.path.join(self.classification_path, classification_file)
        if not os.path.exists(path):
            os.system(f"python neurons_dynamic_classification.py --neurons {self.n_neurons} --gratings_orientation {self.orientation} --gratings_frequency {self.frequency} --classification_criteria input_current --neuron_population {self.neuron_population} --no-reverse")
            print('Lack of results from dynamic simulation. Try again when the dynamic simulation finishes.')
            quit()
        else:
            self.selected_df = file_management.load_lzma(path)
        # Obtain the local recurrent network and information of selected core neurons
        for idx, name in enumerate(pop_names):
            if idx == 0:
                self.recurrent_network_pop = self.recurrent_network.loc[(self.recurrent_network['Target type']==name)]
                pop_info = pd.DataFrame(self.network_info.loc[np.logical_and(self.network_info['Node type']==name, self.core_mask)])
                w_pop_info = pd.DataFrame(self.weighted_network_info.loc[np.logical_and(self.weighted_network_info['Node type']==name, self.core_mask)])
            else:
                self.recurrent_network_pop = pd.concat([self.recurrent_network_pop, self.recurrent_network.loc[(self.recurrent_network['Target type']==name)]])
                pop_info = pd.concat([pop_info, pd.DataFrame(self.network_info.loc[np.logical_and(self.network_info['Node type']==name, self.core_mask)])])
                w_pop_info = pd.concat([w_pop_info, pd.DataFrame(self.weighted_network_info.loc[np.logical_and(self.weighted_network_info['Node type']==name, self.core_mask)])])
        # Save degrees and weighted degrees of the selected population        
        for key, value in pop_info.items():
            if key not in ['Node', 'Node type']:
                self.selected_df[key] = np.fromiter(value, dtype=np.float32)
        for key, value in w_pop_info.items():
            if key not in ['Node', 'Node type']:
                self.selected_df['Weighted '+key] = np.fromiter(value, dtype=np.float32)
                
        # Save the new selected_df which includes connectivity degrees
        file_management.save_lzma(self.selected_df, f'{self.neuron_population}_selected_df', self.classification_path)
        
        # Analyze the characteristics of the different classes if they exist,
        # otherwise analyze the connectivity of the whole neuron_population
        if 'class' in self.selected_df.columns:
            neuron_classes = set(self.selected_df['class'])
            # Compare the classes connectivity features
            self.hVf_vs_dVf_analysis(path=self.topology_dir)
            myplots.degree_distributions_per_class_plot(self.selected_df, path=self.images_path)
            # Rename the neurons belonging to one of the classes
            self.neurons_per_class = {}
            for neu_class in neuron_classes:
                # Save the number of neurons belonging to each class
                n_class = np.sum(self.selected_df['class']==neu_class)
                self.neurons_per_class[neu_class] = n_class
                class_stats = self.selected_df.loc[self.selected_df['class']==neu_class]
                class_tf_indices = class_stats['Tf index'].values
                self.recurrent_network_pop.loc[self.recurrent_network_pop['Source'].isin(class_tf_indices), 'Source type'] = neu_class
                self.recurrent_network_pop.loc[self.recurrent_network_pop['Target'].isin(class_tf_indices), 'Target type'] = neu_class
            # Compare the degrees of different classes    
            for degree_key, weighted_degree_key in zip(['k_in', 'Total k_in', 'Input k_in', 'k_out'], ['Weighted k_in', 'Weighted Total k_in', 'Weighted Input k_in', 'Weighted k_out']):
                myplots.degree_distributions_classes_comparison_plot(self.selected_df, degree_key, weighted_degree_key, path=self.images_path)
                myplots.new_degree_distributions_classes_comparison_plot(self.selected_df, degree_key, weighted_degree_key, path=self.images_path)
               
            # Analyze the connectivity and weights distribution of each class
            for neu_class in neuron_classes:
                # Get the local recurrent network
                class_sample = self.recurrent_network_pop.loc[self.recurrent_network_pop['Target type']==neu_class]
                class_path = os.path.join(self.images_path, self.neuron_population, neu_class)
                os.makedirs(class_path, exist_ok=True)
                # Obtain the number of synapses
                class_counts = class_sample.groupby(['Source type'])['Target'].agg('count').sort_values(ascending=False)
                class_counts /= self.neurons_per_class[neu_class]
                myplots.synapses_histogram(class_counts, y_label='# synapses per neuron', path=class_path)
                # Obtain the weights for each presynaptic class
                class_counts = class_sample.groupby(['Source type'])['Weight'].agg('sum')
                class_counts /= self.neurons_per_class[neu_class]
                myplots.synapses_histogram(class_counts, y_label='Mean input weight', path=class_path)
               
            # Isolate the recurrent network where the targets are the different classes and plot 
            self.recurrent_network_pop = self.recurrent_network_pop.loc[(self.recurrent_network_pop['Target type']=='dVf')|
                                                                        (self.recurrent_network_pop['Target type']=='hVf')|
                                                                        (self.recurrent_network_pop['Target type']=='unclassified')]
            myplots.classes_connectivity_figure(self.recurrent_network_pop, weight=False, path=self.images_path)
            myplots.classes_connectivity_figure(self.recurrent_network_pop, weight=True, path=self.images_path)
            
            # Isolate the local recurrent network
            self.recurrent_network_pop = self.recurrent_network_pop.loc[(self.recurrent_network_pop['Source type']=='dVf')|
                                                                        (self.recurrent_network_pop['Source type']=='hVf')|
                                                                        (self.recurrent_network_pop['Source type']=='unclassified')]
            # Represent different features for the connectivity within the layer
            for feature in ['connection_probability', 'in_degree', 'weighted_in_degree', 'average_synaptic_weight']:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
                myplots.classes_interconnection_matrix(self.selected_df, self.recurrent_network_pop, self.neurons_per_class, ax, feature=feature)
                plt.tight_layout()
                fig.savefig(os.path.join(self.images_path, f'{feature}.png'), dpi=300, transparent=True)
                plt.close(fig)
                 
            # Create compound figure with two connectivity features
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3), sharey=True)
            for idx, feature in enumerate(['connection_probability', 'average_synaptic_weight']):
                myplots.classes_interconnection_matrix(self.selected_df, self.recurrent_network_pop, self.neurons_per_class, axes[idx], feature=feature)
                if idx!=0:
                    axes[idx].yaxis.label.set_visible(False)
            plt.tight_layout()
            fig.savefig(os.path.join(self.images_path, 'connection_prob_and_synaptic_weight_matrices.png'), dpi=300, transparent=True)
            plt.close(fig)
        else:
            # Get the number of neurons in the population
            n_pop = len(self.selected_df)
            # Get the total number of synapses and normalize them
            counts = self.recurrent_network_pop.groupby(['Source type'])['Target'].agg('count').sort_values(ascending=False)
            counts /= n_pop
            # Get the total weight per synapses and normalize them
            w_counts = self.recurrent_network_pop.groupby(['Source type'])['Weight'].agg('sum')
            w_counts /= n_pop
            path = os.path.join(self.images_path, f'{self.neuron_population}')
            os.makedirs(path, exist_ok=True)
            myplots.synapses_histogram(counts, y_label='# synapses', path=path)
            myplots.synapses_histogram(w_counts, y_label='Mean input weight [pA]', path=path) 
           
           
    def hVf_vs_dVf_analysis(self, path=''):
        mean_degrees = self.selected_df.groupby(['class'])[['k_out', 'k_in' , 'Input k_in', 'Bkg k_in', 'Total k_in']].mean().reset_index()
        sem_degrees = self.selected_df.groupby(['class'])[['k_out', 'k_in', 'Input k_in', 'Bkg k_in', 'Total k_in']].sem().reset_index()
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "degrees_table.txt"), "w") as f:
            for (i, row), (j, row2) in zip(mean_degrees.iterrows(), sem_degrees.iterrows()):
                f.write(" & ".join([x[:3] if type(x) == str else "%.1f"%x + r"$\pm$" + "%.1f"%row2[idx] for idx, x in enumerate(row.values)]) + " \\\\\n")
        
        mean_weighted_degrees = self.selected_df.groupby(['class'])[['Weighted k_out', 'Weighted k_in', 'Weighted Input k_in', 'Weighted Bkg k_in', 'Weighted Total k_in']].mean().reset_index()
        sem_weighted_degrees = self.selected_df.groupby(['class'])[['Weighted k_out', 'Weighted k_in', 'Weighted Input k_in', 'Weighted Bkg k_in', 'Weighted Total k_in']].sem().reset_index()
        with open(os.path.join(path, "weighted_degrees_table.txt"), "w") as f:
            for (i, row), (j, row2) in zip(mean_weighted_degrees.iterrows(), sem_weighted_degrees.iterrows()):
                f.write(" & ".join([x[:3] if type(x) == str else "%.1f"%x + r"$\pm$" + "%.1f"%row2[idx] for idx, x in enumerate(row.values)]) + " \\\\\n")
              
                
def main(flags):    
    synapses_analysis = SynapsesAnalysis(flags)
    synapses_analysis()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Define key flags')
    parser.add_argument('--gratings_orientation', type=int, choices=range(0, 360, 45), default=0)
    parser.add_argument('--gratings_frequency', type=int, default=2)
    parser.add_argument('--neurons', type=int, default=230924)
    parser.add_argument('--neuron_population', type=str, default='e23')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false')
    parser.set_defaults(reverse=False)

    flags = parser.parse_args()
    main(flags)  
    
    