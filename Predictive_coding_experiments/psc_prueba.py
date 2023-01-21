#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:20:07 2021

@author: jgalvan
"""

import os
import absl
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from fnmatch import filter
import load_sparse
from data_management import load_lzma, save_lzma
mpl.style.use('default')
np.random.seed(3000)

data_path = os.path.join(
    'Simulation_results', 'orien_0_freq_6_reverse_False_rec5000', 'Data2')
original_data_path = os.path.join(
    'Simulation_results', 'orien_0_freq_6_reverse_False_rec5000', 'Data')
full_data_path = os.path.join(
    'Simulation_results', 'orien_0_freq_6_reverse_False_rec5000', 'Network_simulation_results', 'Input_current', 'e23_classification_base_keller.lzma')

n_neurons = 5000
first_simulation = 0
last_simulation = 1
n_simulations = 1
simulation_length = 2500

e23_df = load_lzma(full_data_path)
sel = e23_df.loc[e23_df['heatmap_neurons'] == True]
indices = sel['Tf index']
keller_type = sel['Keller type']

v = np.zeros((n_simulations*simulation_length, n_neurons), np.float32)
z = np.zeros((n_simulations*simulation_length, n_neurons), np.float32)
input_current = np.zeros((n_simulations*simulation_length, n_neurons), np.float32)
recurrent_current = np.zeros((len(sel), n_simulations*simulation_length, n_neurons), np.float32)
external_current = np.zeros((len(sel), n_simulations*simulation_length, 1), np.float32)

for i in range(first_simulation, last_simulation):
    vi = np.array(load_lzma(os.path.join(
        data_path, 'v_{n_neurons}_{i}.lzma'.format(n_neurons=n_neurons, i=i))))
    zi = np.array(load_lzma(os.path.join(
        data_path, 'z_{n_neurons}_{i}.lzma'.format(n_neurons=n_neurons, i=i)))) 
    input_currenti = np.array(load_lzma(os.path.join(
        data_path, 'input_current_{n_neurons}_{i}.lzma'.format(n_neurons=n_neurons, i=i)))) 
    # e23_dicti = pd.DataFrame(load_lzma(os.path.join(
    #     data_path, 'e23dict_{n_neurons}_{i}.lzma'.format(n_neurons=n_neurons, i=i))))
    
    v[(i-first_simulation)*simulation_length:(i+1-first_simulation)*simulation_length,:] = vi
    z[(i-first_simulation)*simulation_length:(i+1-first_simulation)*simulation_length,:] = zi
    input_current[(i-first_simulation)*simulation_length:(i+1-first_simulation)*simulation_length,:] = input_currenti
    # for idx, k in enumerate(indices):
    #     recurrent_current[idx, (i-first_simulation)*simulation_length:(i+1-first_simulation)*simulation_length,:] = e23_dicti[str(k)]['recurrent']
    #     external_current[idx, (i-first_simulation)*simulation_length:(i+1-first_simulation)*simulation_length,:] = e23_dicti[str(k)]['external']
        
    # del vi, zi, input_currenti, e23_dicti
# for i in range(first_simulation, last_simulation):
#     v0 = np.array(load_lzma(os.path.join(
#         original_data_path, 'v_{n_neurons}_{i}.lzma'.format(n_neurons=n_neurons, i=i))))
#     z0 = load_lzma(os.path.join(
#         original_data_path, 'z_{n_neurons}_{i}.lzma'.format(n_neurons=n_neurons, i=i)))
#     input_current0 = load_lzma(os.path.join(
#         original_data_path, 'input_current_{n_neurons}_{i}.lzma'.format(n_neurons=n_neurons, i=i)))    
    
# print("V y v0 son iguales, lo estoy haciendo bien :)")

fig = plt.figure()
colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
n_bin = 100  # Discretizes the interpolation into bins
cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
norm = mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=-0.25, vmax=0)
cs = plt.imshow(np.transpose(recurrent_current[0]), norm=norm,
                interpolation='none', aspect='auto', cmap=cm)
        

network = load_lzma(os.path.join(data_path,'network.lzma'))

def pop_names(network):
    data_dir = 'GLIF_network'
    path_to_csv = os.path.join(data_dir, 'network/v1_node_types.csv')
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    node_types = pd.read_csv(path_to_csv, sep=' ')
    node_h5 = h5py.File(path_to_h5, mode='r')
    node_type_id_to_pop_name = dict()
    for nid in np.unique(node_h5['nodes']['v1']['node_type_id']): 
        # if not np.unique all of the 230924 model neurons ids are considered, 
        # but nearly all of them are repeated since there are only 111 different indices
        ind_list = np.where(node_types.node_type_id == nid)[0]
        assert len(ind_list) == 1
        node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]
    true_pop_names = []  # it contains the pop_name of all the 230,924 neurons
    for nid in node_h5['nodes']['v1']['node_type_id']:
        true_pop_names.append(node_type_id_to_pop_name[nid])
     # Select population names of neurons in the present network (core)
    true_pop_names = np.array(true_pop_names)[network['tf_id_to_bmtk_id']]
    
    return true_pop_names

true_pop_names = pop_names(network)


target_tf_id = np.array(indices)
target_type_name = true_pop_names[target_tf_id]
time = np.arange(2500)
source_tf_id = np.arange(n_neurons)
source_type_name = true_pop_names[source_tf_id]

neuron_id = 740
fig, axs = plt.subplots(4, sharex=True)
axs[0].plot(v[:, neuron_id], color='r', ms=1,
             alpha=0.7, label='Membrane potential')
axs[0].set_ylabel('V_m [mV]')
axs[1].plot(input_current[:, neuron_id], color='r', ms=1,
             alpha=0.7, label='Input current')
axs[1].set_ylabel('In-current [pA]')
axs[2].plot(np.sum(recurrent_current[0], axis=1), color='r', ms=1,
             alpha=0.7, label='rec_current')
axs[2].set_ylabel('Rec-current [pA]')
axs[3].plot(np.sum(external_current[0], axis=1), color='b',
             ms=1, alpha=0.7, label='ext_current')
axs[3].set_yticks([0, 1])
axs[3].set_ylim(0, 1)
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel('Ext_current')

# # Create the MultiIndex from years, samples and patients.
# midx = pd.MultiIndex.from_product([target_tf_id, time, source_tf_id])


# b = np.sum(recurrent_current[0], axis=1)

d = recurrent_current[1]
df = pd.DataFrame(np.transpose(d))
df['Source'] = source_type_name
gdf = df.groupby(['Source']).sum()
e = gdf.sum(axis=1)
print(e)


p = np.sum(recurrent_current, axis=1)
df = pd.DataFrame(np.transpose(p))
df['Source'] = source_type_name
gdf = df.groupby(['Source']).sum()
gdf.columns = keller_type
e = gdf.sum(axis=1)
print(e)


# np.random.seed(1618033)
# #Set 3 axis labels/dims
# years = np.arange(5000) #Years
# samples = np.arange(0,30) #Samples
# patients = np.arange(5000) #Patients

# #Create random 3D array to simulate data from dims above
# # A_3D = np.random.random((years.size, samples.size, len(patients))) #(10, 20, 3)

# # Create the MultiIndex from years, samples and patients.
# midx = pd.MultiIndex.from_product([years, samples, patients])

# # Create sample data for each patient, and add the MultiIndex.
# patient_data = pd.DataFrame(np.random.randn(len(midx), 3), index = midx)


neurons_subset_label = ['dMM', 'unclassified', 'hMM']
for label in neurons_subset_label:
    print(label)
    neuron_tf_indices = e23_df.loc[(e23_df['heatmap_neurons'] == True) & (e23_df['Keller type'] == label), 'Tf index']
    
l = [x for x, _ in sorted(zip(neurons_subset_label, np.arange(3)), key=lambda element: (element[1]))]   





baseline_mean_df = e23_df.groupby(['Keller type'])['Keller baseline potential', 'Static firing rate', 'Keller baseline input_current recurrent', 
                                                   'Keller baseline input_current external', 'Keller baseline asc', 'Keller baseline input_current'].mean().reset_index()
baseline_sem_df = e23_df.groupby(['Keller type'])['Keller baseline potential', 'Static firing rate', 'Keller baseline input_current recurrent', 
                                                  'Keller baseline input_current external', 'Keller baseline asc', 'Keller baseline input_current'].sem().reset_index()
variations_mean_df = e23_df.groupby(['Keller type'])['Mean potential variations','Dynamic firing rate', 'Firing rate ratio', 'Modulation index',
                                                     'Mean input_current variations', 'Mean recurrent input_current variations', 
                                                     'Mean external input_current variations', 'Mean asc variations'].mean().reset_index()
variations_sem_df = e23_df.groupby(['Keller type'])['Mean potential variations','Dynamic firing rate', 'Firing rate ratio', 'Modulation index',
                                                     'Mean input_current variations', 'Mean recurrent input_current variations', 
                                                     'Mean external input_current variations', 'Mean asc variations'].sem().reset_index()    
baseline_mean_df['Static firing rate'] = baseline_mean_df['Static firing rate']*1000
baseline_sem_df['Static firing rate'] = baseline_sem_df['Static firing rate']*1000
variations_mean_df['Dynamic firing rate'] = variations_mean_df['Dynamic firing rate']*1000
variations_sem_df['Dynamic firing rate'] = variations_sem_df['Dynamic firing rate']*1000     
with open("baseline_latex_table.txt", "w") as f:
    for (i, row), (j, row2) in zip(baseline_mean_df.iterrows(), baseline_sem_df.iterrows()):
        f.write(" & ".join([x[:3] if type(x) == str else "%.2f"%x + r"$\pm$" + "%.2f"%row2[idx] for idx, x in enumerate(row.values)]) + " \\\\\n")
with open("variations_latex_table.txt", "w") as f:
    for (i, row), (j, row2) in zip(variations_mean_df.iterrows(), variations_sem_df.iterrows()):
        f.write(" & ".join([x[:3] if type(x) == str else "%.2f"%x + r"$\pm$" + "%.2f"%row2[idx] for idx, x in enumerate(row.values)]) + " \\\\\n")
