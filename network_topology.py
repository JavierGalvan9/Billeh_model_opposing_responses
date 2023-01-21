#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:27:56 2021

@author: jgalvan
"""

import os
import sys
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "general_utils"))
import file_management
from other_utils import memory_tracer, timer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "billeh_model_utils"))
import other_billeh_utils
import load_sparse


class NetworkAnalysis:
    def __init__(self, flags):
        self.n_neurons = flags.neurons
        self.simulation_results = 'Predictive_coding_experiments/Simulation_results'
        self.directory = f'orien_{str(flags.gratings_orientation)}_freq_{str(flags.gratings_frequency)}_reverse_{str(flags.reverse)}_rec_{flags.neurons}'
        self.full_path = os.path.join(self.simulation_results, self.directory)
        self.network_results_path = f'Topological_analysis_{flags.neurons}'
        os.makedirs(self.network_results_path, exist_ok=True)
        self.full_data_path = os.path.join(self.full_path, 'Data', 'simulation_data.hdf5')
        # Load the simulation configuration attributes
        self.sim_metadata = {}
        with h5py.File(self.full_data_path, 'r') as f:
            dataset = f['Data']
            self.sim_metadata.update(dataset.attrs)
        self.data_dir = self.sim_metadata['data_dir']
        # Load the model network
        load_fn = load_sparse.cached_load_billeh
        self.input_population, self.network, self.bkg, _ = load_fn(self.sim_metadata['n_input'], self.n_neurons, self.sim_metadata['core_only'], 
                                                                   self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                                                  n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        self.true_pop_names = other_billeh_utils.pop_names(self.network, data_dir=self.data_dir)
    
    # @profile    
    def __call__(self):
        ### Get the network recurrent connectivity
        recurrent_network = self.create_network_df(self.network['synapses'])
        file_management.save_lzma(recurrent_network, 'recurrent_network', self.network_results_path)
        ### Get the network input connectivity
        input_network = self.create_network_df(self.input_population, source='LGN')
        file_management.save_lzma(input_network, 'input_network', self.network_results_path)
        ### Get the input background connectivity
        bkg_network = self.create_network_df(self.bkg, source='bkg')
        file_management.save_lzma(bkg_network, 'bkg_network', self.network_results_path)
        
        # Construct the recurrent and input digraph
        self.recurrent_graph = nx.DiGraph()
        self.recurrent_graph.add_weighted_edges_from(np.array(recurrent_network[['Source', 'Target', 'Weight']]))
        self.input_graph = nx.DiGraph()
        self.input_graph.add_weighted_edges_from(np.array(input_network[['Source', 'Target', 'Weight']]))
        self.bkg_graph = nx.DiGraph()
        self.bkg_graph.add_weighted_edges_from(np.array(bkg_network[['Source', 'Target', 'Weight']]))

        del recurrent_network, input_network, bkg_network

        n_edges = len(self.recurrent_graph.edges)
        self.shorted_pop_names = np.array([pop_name[:2] if pop_name[0]=='e' else pop_name for pop_name in self.true_pop_names])
        self.shorted_pop_names[self.shorted_pop_names=='e2'] = 'e23'
        # self.shorted_pop_names = np.array([pop_name[:2] if pop_name[1]!='2' else pop_name[:3] for pop_name in self.true_pop_names])
        self.network_info = self.degree_calculation(weighted=False)
        file_management.save_lzma(self.network_info, 'network_info', self.network_results_path)
        self.weighted_network_info = self.degree_calculation(weighted=True)
        file_management.save_lzma(self.weighted_network_info, 'weighted_network_info', self.network_results_path)
        
        path = os.path.join(self.network_results_path, 'Topological images', 'Degree distribution')
        self.degree_distribution_plot(self.network_info, self.weighted_network_info,
                                      filename='v1_degree_distribution.png', 
                                      path=path)    

        for neuron_type in set(self.shorted_pop_names):
            net = self.network_info.loc[self.network_info['Node type'] == neuron_type]
            wnet = self.weighted_network_info.loc[self.weighted_network_info['Node type'] == neuron_type]
            self.degree_distribution_plot(net, wnet,
                                          filename = neuron_type+'_degree_distribution.png', 
                                          path=path)
      
        metadata_path = os.path.join(self.network_results_path, 'topological_features.txt')
        with open(metadata_path, 'w') as out_file:
             out_file.write('Number of recurrent edges of V1 neurons: {n_edges}\n'.format(n_edges = n_edges))
             connection_prob = (sum(self.network_info['k_in']) + sum(self.network_info['k_out'])) /(self.n_neurons*(self.n_neurons - 1))
             out_file.write('Connection probability: {p}\n'.format(p = connection_prob))
            # print('Average shortest path length: ', nx.average_shortest_path_length(G, weight=None, method=None))

    def create_network_df(self, network_dict, source='recurrent'):#, core_only=True):
        synapses_weights = network_dict['weights']
        synapses_indices = network_dict['indices'] # 13828184
        target_network_ids = np.floor(synapses_indices[:, 0]/4).astype(dtype=np.int32)
        target_type_name = self.true_pop_names[target_network_ids]
        source_network_ids = synapses_indices[:,1]        
        if source == 'recurrent':
            source_type_name = self.true_pop_names[source_network_ids]
        elif source == 'LGN':
            # LGN units do only have non-zero out degree and v1 neurons just non-zero in-degree, which is the only thing we are interested in from LGN input
            source_type_name = ['LGN unit' for x in range(len(source_network_ids))]
        elif source == 'bkg':
            source_type_name = ['Bkg unit' for x in range(len(source_network_ids))]
        
        network_df = pd.DataFrame({'Target': target_network_ids, 'Target type': target_type_name,
                            'Source': source_network_ids, 'Source type': source_type_name,
                            'Weight': synapses_weights})
        # Check if any edge is repeates and add the weights in that case (according to the model construction that should not happen)
        self.network_df = network_df.groupby(['Target', 'Source', 'Target type', 'Source type'])['Weight'].agg('sum').to_frame(name = 'Weight').reset_index()
        return network_df
    
    def degree_calculation(self, weighted=False):
        if weighted:
            k_in=dict(self.recurrent_graph.in_degree(weight='weight'))
            k_out=dict(self.recurrent_graph.out_degree(weight='weight'))
            k_lgn_in=dict(self.input_graph.in_degree(weight='weight'))
            k_bkg_in=dict(self.bkg_graph.in_degree(weight='weight'))
        else:
            k_in=dict(self.recurrent_graph.in_degree())
            k_out=dict(self.recurrent_graph.out_degree())
            k_lgn_in=dict(self.input_graph.in_degree())
            k_bkg_in=dict(self.bkg_graph.in_degree())
            
        k_in = {k: v for k, v in sorted(k_in.items(), key=lambda item: item[0])}
        k_out = {k: v for k, v in sorted(k_out.items(), key=lambda item: item[0])}
        k_lgn_in = {k: v for k, v in sorted(k_lgn_in.items(), key=lambda item: item[0])}
        k_bkg_in = {k: v for k, v in sorted(k_bkg_in.items(), key=lambda item: item[0])}
        
        node_keys = np.fromiter(k_in.keys(), dtype=np.int32)
        k_in_degree = np.fromiter(k_in.values(), dtype=np.float32)
        k_out_degree = np.fromiter(k_out.values(), dtype=np.float32)
        
        lgn_node_keys = np.fromiter(k_lgn_in.keys(), dtype=np.int32)
        lgn_in_degree = np.fromiter(k_lgn_in.values(), dtype=np.float32)

        #Reshape the LGN input connections (some of them where missing due to a 0 lgn in degree)
        reshaped_lgn_in_degree = np.zeros((self.n_neurons), dtype=np.float32)
        v1_neurons_mask = lgn_node_keys<self.n_neurons
        #only v1 neurons are receiving inputs so not worry about sharing indices between v1 and LGN neurons
        reshaped_lgn_in_degree[lgn_node_keys[v1_neurons_mask]] = lgn_in_degree[v1_neurons_mask]
        bkg_in_degree = np.fromiter(k_bkg_in.values(), dtype=np.float32) 
        # it seems that the background node connects to all the v1 neurons, if it doesnt we should reshape as with the lgn_in_degree
        total_in_degree = k_in_degree + reshaped_lgn_in_degree + bkg_in_degree

        network_info = pd.DataFrame({'Node': node_keys, 'Node type': self.shorted_pop_names, 'k_in': k_in_degree, 
                                'k_out': k_out_degree, 'Input k_in': reshaped_lgn_in_degree, 'Bkg k_in': bkg_in_degree, 
                                'Total k_in': total_in_degree})
        return network_info
    
    def degree_distribution_plot(self, network_info, weighted_network_info, filename='degree_distribution.png', path=''):
        fig, axs = plt.subplots(2,2, sharex='col', sharey='col')
        for i, network_type, network_name in zip(range(2), [network_info, weighted_network_info], ['Non-weighted', 'Weighted']):
            max_degree = int(max([network_type['Total k_in'].max(), network_type['k_out'].max()]))
            min_degree = int(min([network_type['Total k_in'].min(), network_type['k_out'].min()]))
            binwidth = int((max_degree+abs(min_degree))/100)
            axs[0][i].hist(network_type['k_in'], bins=range(min_degree, max_degree + binwidth, binwidth), label='Recurrent $k_{in}$', color='b', alpha=0.6, zorder=1) 
            axs[0][i].hist(network_type['Total k_in'], bins=range(min_degree, max_degree + binwidth, binwidth), label='Total $k_{in}$', color='g', alpha=0.6, zorder=1) 
            axs[1][i].hist(network_type['k_out'], bins=range(min_degree, max_degree + binwidth, binwidth), label='Recurrent $k_{out}$', color='r', alpha=0.6)
            axs[0][i].set_xlim(min_degree, max_degree)
            axs[0][i].legend() 
            axs[0][i].set_title(network_name)
            axs[i][0].set_ylabel('# neurons')
            axs[1][i].set_xlabel('Degree')
            axs[1][i].set_xlim(min_degree, max_degree)
            axs[1][i].legend()
            
        os.makedirs(path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(path, filename), dpi=300, transparent=True)
        plt.close(fig)  
    

# @timer
# @memory_tracer  
def main(flags):
    network_analysis = NetworkAnalysis(flags)
    network_analysis()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define key flags')
    parser.add_argument('--gratings_orientation', type=int, choices=range(0, 360, 45), default=0)
    parser.add_argument('--gratings_frequency', type=int, default=2)
    parser.add_argument('--neurons', type=int, default=230924)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false')
    parser.set_defaults(reverse=False)
    
    flags = parser.parse_args()
    main(flags)  
