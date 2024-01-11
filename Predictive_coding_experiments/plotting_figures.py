# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:03:12 2022

@author: javig
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
from plotting_utils import DriftingGrating

    
 ########################## neurons_dynamic_classification.py ###############   

def plot_neuron_voltage_and_spikes_and_input_current(data, neuron_id, tf_id=None, 
                                                     mean_potential_variations=None, mean_input_current_variations=None,
                                                     baseline_potential=None, baseline_input_current=None,
                                                     voltage_threshold=None, input_current_threshold=None, 
                                                     stimuli_init_time=500, stimuli_end_time=1500,
                                                     fig_title='', path=''):
    # Plot the voltage, input_current and firing rate in a subplot layout
    variable_colors = ['g', 'r', 'b']
    variable_y_labels = [r'$V_m$ [mV]', r'$I_{syn}$ [pA]', 'Firing rate \n [Hz]']
    fig, axs = plt.subplots(3,1)
    for idx, variable in enumerate([data['v'], data['input_current'], data['firing_rate']]):
        n_simulations, simulation_length, n_neurons = variable.shape
        times = np.linspace(0, 2500, simulation_length)
        mean_variable = np.mean(variable[:,:, neuron_id], axis=0)
        color = variable_colors[idx]
        axs[idx].plot(times, mean_variable, color=color, ms=1, alpha=0.7)
        axs[idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5)
        axs[idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5)
        axs[idx].set_ylabel(variable_y_labels[idx], fontsize=12)
        # Include SEM if more than 1 simulation
        if n_simulations > 1:
            sem_variable = stats.sem(variable[:,:, neuron_id], axis=0)
            axs[idx].fill_between(times, mean_variable + sem_variable, 
                                mean_variable - sem_variable, 
                                alpha=0.3, color=color)
        if idx in [0,1]:
            axs[idx].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
            axs[idx].tick_params(axis='y', which='both', labelsize=10) 
        elif idx==2:
            axs[idx].tick_params(axis='both', which='both', labelsize=10) 
            axs[idx].set_xlabel('Time [ms]', fontsize=12)
    # Include baselines and classification thresholds if provided
    if baseline_potential is not None:
        axs[0].axhline(baseline_potential[neuron_id], linestyle='dotted', color='gray', linewidth=2, 
                       label='Mean pre-stimulus voltage: %.2f mV'% baseline_potential[neuron_id])
    if voltage_threshold is not None:
        axs[0].hlines(mean_potential_variations[neuron_id], 
                      stimuli_init_time, stimuli_end_time, linewidth=2, 
                      color='b', linestyles='-', label='Mean voltage', zorder=1000)  
        axs[0].axhline(-voltage_threshold, 0, simulation_length, linestyle='dotted', color='k', linewidth=2, zorder=1000, label='Classification threshold')
        axs[0].axhline(voltage_threshold, 0, simulation_length, linestyle='dotted', color='k', linewidth=2, zorder=1000)
    if baseline_input_current is not None:
        axs[1].axhline(baseline_input_current[neuron_id], linestyle='dotted', color='gray', linewidth=2, 
                       label='Mean pre-stimulus input_current: %.2f pA'% baseline_input_current[neuron_id])
    if input_current_threshold is not None:
        axs[1].hlines(mean_input_current_variations[neuron_id], 
                      stimuli_init_time, stimuli_end_time, linewidth=2, 
                      color='b', linestyles='-', label='Mean input current', zorder=1000)
        axs[1].axhline(-input_current_threshold, 0, simulation_length, linestyle='dotted', color='k', linewidth=2, zorder=1000, label='Classification threshold')
        axs[1].axhline(input_current_threshold, 0, simulation_length, linestyle='dotted', color='k', linewidth=2, zorder=1000)
    
    # axs[0].legend(loc='best')
    # axs[1].legend(loc='best')
    fig.suptitle(fig_title)
    # Save the figure
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    if tf_id is None:
        tf_id = neuron_id
    fig.savefig(os.path.join(path, f'neurons_{neuron_id}_index_{tf_id}.png'), dpi=300, transparent=True)
    plt.close(fig)
        
def plot_neuron_input_currents(data, neuron_id, 
                               tf_id=None, input_current_threshold=None,
                               stimuli_init_time=500, stimuli_end_time=1500, 
                               fig_title='', path=''):
    # Plot all the different current  source traces
    variable_colors = ['r', 'olive', 'dodgerblue']
    variable_labels = ['Total input current', 'Recurrent input current', 'Bottom up input current']
    fig = plt.figure()
    for idx, variable in enumerate([data['input_current'], data['recurrent_current'], data['bottom_up_current']]):
        n_simulations, simulation_length, n_neurons = variable.shape
        times = np.linspace(0, simulation_length, simulation_length)
        mean_variable = np.mean(variable[:,:, neuron_id], axis=0)
        color = variable_colors[idx]
        label = variable_labels[idx]
        plt.plot(times, mean_variable, color=color, ms=1, alpha=0.7, label=label)
        # Include SEM if more than 1 simulation
        if n_simulations > 1:
            sem_variable = stats.sem(variable[:,:, neuron_id], axis=0)
            plt.fill_between(times, mean_variable + sem_variable, 
                                mean_variable - sem_variable, 
                                alpha=0.3, color=color)
    # Include classification threshold if provided
    if input_current_threshold is not None:
        plt.axhline(-input_current_threshold, 0, simulation_length, linestyle='dotted', color='k', linewidth=2, zorder=1000, label='Classification threshold')
        plt.axhline(input_current_threshold, 0, simulation_length, linestyle='dotted', color='k', linewidth=2, zorder=1000)
    plt.ylabel('Input current [pA]', fontsize=12)
    plt.xlabel('Time [ms]', fontsize=12)            
    plt.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5)
    plt.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5)
    plt.axhline(0, linestyle='-', color='k', linewidth=1)
    plt.tick_params(axis='both', labelsize=10)
    # plt.legend()
    fig.suptitle(fig_title)
    # Save the figure
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    if tf_id is None:
        tf_id = neuron_id
    fig.savefig(os.path.join(path, f'neurons_{neuron_id}_index_{tf_id}_currents.png'), dpi=300, transparent=True)
    plt.close(fig)
    

def spike_effect_correction_plot(data, neuron_id, tf_id, pre_spike_gap=1, post_spike_gap=5, 
                                 stimuli_init_time=500, stimuli_end_time=1500, path=''):
    
    v, z = data['v'], data['z']
    n_simulations, simulation_length, n_neurons = v.shape
    v = v.reshape((n_simulations*simulation_length, n_neurons))
    z = z.reshape((n_simulations*simulation_length, n_neurons))
    v_corrected = np.copy(v)
    min_v = v_corrected[:,neuron_id].min()
    max_v = v_corrected[:,neuron_id].max()
    vs = v_corrected[:,neuron_id]
    zs = z[:,neuron_id].astype(dtype=bool)
    # Make a linear interpolation on the membrane voltage between pre_spike_gap and post_spike_gap
    for t_idx, spike in enumerate(zs[:-post_spike_gap]):
        if spike and t_idx >= pre_spike_gap:
            prev_value = vs[t_idx-pre_spike_gap]
            post_value = vs[t_idx+post_spike_gap]
            xp = [t_idx-pre_spike_gap, t_idx+post_spike_gap]
            fp = [prev_value, post_value]
            new_values = np.interp(np.arange(t_idx-pre_spike_gap+1, t_idx+post_spike_gap, 1), xp, fp)
            v_corrected[t_idx-pre_spike_gap+1:t_idx+post_spike_gap, neuron_id] = new_values
        
    fig, axs = plt.subplots(2, 1, sharex=True)
    times = np.linspace(0, simulation_length, simulation_length)
    # Original and corrected voltage
    axs[0].plot(times, v[:simulation_length, neuron_id], color='r', ms=1,
                alpha=0.7, label='Original')
    axs[0].plot(times, v_corrected[:simulation_length, neuron_id], color='r', ms=1,
                alpha=0.7, label=f'pre_{pre_spike_gap}_post_{post_spike_gap}')
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel('V [mV]')
    axs[0].set_ylim(min_v-5, max_v+10)
    # Spike train
    axs[1].plot(times, z[:int(simulation_length), neuron_id], color='b',
                 ms=1, alpha=0.7, label='Membrane potential')
    axs[1].set_yticks([0, 1])
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('Time [ms]')
    axs[1].set_ylabel('Spikes')
    # Plot stimulus init and end times
    for subplot in range(2):
        axs[subplot].axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1)
        axs[subplot].axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1)
    # Save the figure
    path = os.path.join(path, 'Voltage corrected samples')
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, f'pre_{pre_spike_gap}_post_{post_spike_gap}_idx_{tf_id}'), dpi=300, transparent=True)
    plt.close(fig)    
    
    
def firing_rate_heatmap(data, sampling_interval, frequency, normalize_firing_rates=True, 
                        exclude_NS_neurons=True, stimuli_init_time=500, stimuli_end_time=1500, reverse=False, path=''):   
    # Obtain the init and end times of the stimuli according to the firing rate sampling interval
    stimuli_init_time = int(stimuli_init_time/sampling_interval)
    stimuli_end_time = int(stimuli_end_time/sampling_interval)
    # Average over trials
    firing_rates = data['firing_rate']
    mean_firing_rates = np.mean(firing_rates, axis=0)
    simulation_length, n_neurons = mean_firing_rates.shape
    # Normalize the firing rate according to the maximum firing rate reached on the simulation
    if normalize_firing_rates:
        max_firing_rate = np.max(mean_firing_rates, axis=0)
        mean_firing_rates = np.divide(mean_firing_rates, max_firing_rate, 
                                      out=np.zeros_like(mean_firing_rates), where=max_firing_rate!=0)
        norm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        label = 'NFR'
    else:
        norm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=1.5, vmax=3)
        label = 'FR [Hz]'
    # Exclude from the representations the neurons without activity
    if exclude_NS_neurons:
        firing_neurons = np.any(mean_firing_rates!=0, axis=0)
        mean_firing_rates = mean_firing_rates[:, firing_neurons]
    else:
        firing_neurons = np.full(n_neurons, True)
        
    # Order the neurons according to their firing rate during the stimuli
    mean_variations = np.mean(mean_firing_rates[stimuli_init_time: stimuli_end_time, :], axis=0)
    sorted_indices = np.arange(mean_firing_rates.shape[1])
    sorted_indices = [x for _, x in sorted(zip(mean_variations, sorted_indices), key=lambda element: (element[0]), reverse=True)]   
    firing_rates_ordered = mean_firing_rates[:,sorted_indices]     
    # Create figure
    fig = plt.figure()
    grid = fig.subplots(nrows=2, ncols=2,
                   gridspec_kw={'width_ratios':(1,0.05), 'height_ratios':(1,0.1)}, sharex='col')
    firing_rates_ax, cax1 = grid[0]
    drifting_grating_ax, cax2 = grid[1]
    cax2.set_visible(False)
    firing_rates_ax.clear()
    drifting_grating_ax.clear()
    
    # Plot the firing rates heatmap
    cs = firing_rates_ax.imshow(np.transpose(firing_rates_ordered), norm=norm,
                                interpolation='none', aspect='auto', cmap='coolwarm')
    cbar = fig.colorbar(cs, cax=cax1, extend='neither')#, ticks=[0, 1, 2, 3])
    cbar.ax.set_title(label=label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    # cbar.ax.set_yticklabels(['0', '1', '2', '3'])
    firing_rates_ax.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    firing_rates_ax.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    firing_rates_ax.set_ylabel('Sorted neuron #', fontsize=12)
    firing_rates_ax.tick_params(axis='both',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False,
                                labelsize=10) 
    # Plot the stimuli scheme
    drifting_grating_plot = DriftingGrating(frequency=frequency, stimuli_init_time=stimuli_init_time, 
                                            stimuli_end_time=stimuli_end_time, reverse=reverse)
    drifting_grating_plot(drifting_grating_ax, 2500, stimulus_length=simulation_length)
    plt.subplots_adjust(wspace=0.07, hspace=0.07)
    drifting_grating_ax.set_xlim(firing_rates_ax.get_xlim())
    
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'firing_rates_heatmap_normalized_{normalize_firing_rates}_exclude_NS_neurons_{exclude_NS_neurons}.png'), dpi=300, transparent=True)
    plt.close(fig)

def baseline_histogram(data, path):
    # n_selected_neurons = len(data['baseline_input_current'])
    fig = plt.figure()
    # Input current
    ax1 = plt.subplot(4, 2, 1)
    ax1.hist(data['baseline_input_current'], label='Total', color='r', alpha=0.7)
    plt.setp(ax1.get_xticklabels(), visible=False)
    # ax1.set_ylim(0, int(n_selected_neurons/3))
    # Recurrent current
    ax2 = plt.subplot(4, 2, 3, sharex=ax1, sharey=ax1)
    ax2.hist(data['baseline_recurrent_current'], label='Recurrent', color='olive', alpha=0.7)
    plt.setp(ax2.get_xticklabels(), visible=False)
    # Bottom up current
    ax3 = plt.subplot(4, 2, 5, sharex=ax1, sharey=ax1)
    ax3.hist(data['baseline_bottom_up_current'], label='Bottom up', color='dodgerblue', alpha=0.7)
    plt.setp(ax3.get_xticklabels(), visible=False)
    # ASC_BKG current
    ax4 = plt.subplot(4, 2, 7, sharex=ax1, sharey=ax1)
    ax4.hist(data['baseline_asc_bkg'], label='ASC + bkg noise', color='gray', alpha=0.7)
    ax4.set_xlabel('Average input current [pA]', fontsize=9)
    for axs in [ax1, ax2, ax3, ax4]:
        axs.set_ylabel('# neurons', fontsize=9)
        axs.legend(fontsize=8)
    # Membrane potential
    ax5 = plt.subplot(1, 2, 2, sharey=ax1)
    ax5.hist(data['baseline_v'], color='b', alpha=0.7)
    ax5.set_xlabel('Average membrane potential [mV]', fontsize=9)
    ax5.tick_params(axis='both', labelsize=9)
    # fig.suptitle('Mean pre-stimulus analysis')
    # Save figure
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'baseline_distribution.png'), dpi=300, transparent=True)
    plt.close(fig) 
    
def firing_rate_histogram(firing_rate_ratio, fr_threshold, path=''):
    # Remove nans in the firing rate ratio
    firing_rate_ratio = firing_rate_ratio[np.logical_not(np.isnan(firing_rate_ratio))]
    # Plot the firing rate histogram
    fig = plt.figure()
    plt.hist(firing_rate_ratio, bins=np.logspace(np.log10(0.01),np.log10(50), 50))
    plt.axvline(1/fr_threshold, color='k', linestyle='--', linewidth=2, zorder=1000, label='Class threshold')
    plt.axvline(fr_threshold, color='k', linestyle='--', linewidth=2, zorder=1000)
    plt.axvspan(0.05, 0.5, alpha=0.2, color='#F06233')
    plt.axvspan(0.5, 2, alpha=0.2, color='#9CB0AE')
    plt.axvspan(2, 50, alpha=0.2, color='#33ABA2')
    plt.gca().set_xscale("log")
    plt.figtext(0.2, 0.8, 'hVf', fontsize=12, color='#F06233', fontweight='bold')
    plt.figtext(0.45, 0.8, 'unc', fontsize=12, color='#9CB0AE', fontweight='bold')
    plt.figtext(0.75, 0.8, 'dVf', fontsize=12, color='#33ABA2', fontweight='bold')
    plt.xlabel('Firing rate ratio')
    plt.ylabel('# neurons')
    plt.xlim(0.05, 50)
    plt.legend()
    fig.suptitle("Firing rate ratio histogram")
    # Save the figure
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'firing_rate_ratio.png'), dpi=300, transparent=True)
    plt.close(fig)

    
def modulation_index_histogram(modulation_index, path=''):
    # Remove nans in the modulation index
    modulation_index = modulation_index[np.logical_not(np.isnan(modulation_index))]
    # Plot the modulation index histogram
    fig = plt.figure()
    a = -np.logspace(np.log10(2), np.log10(0.1), 20)
    b = np.logspace(np.log10(0.1),np.log10(40), 30)
    log_array = np.concatenate((a, b))
    plt.hist(modulation_index, bins=log_array)
    plt.axvline(0, color='k', linestyle='--', linewidth=2, label='Class threshold') #threshold
    plt.axvspan(-1, 0, alpha=0.2, color='#F06233')
    plt.axvspan(0, 40, alpha=0.2, color='#33ABA2')
    # plt.gca().set_xscale("log")
    plt.xscale('symlog')
    plt.figtext(0.12, 0.8, 'hVf', fontsize=12, color='#F06233', fontweight='bold')
    plt.figtext(0.6, 0.8, 'dVf', fontsize=12, color='#33ABA2', fontweight='bold')
    plt.xlim(-1, 40)
    plt.xlabel('Modulation index')
    plt.ylabel('# neurons')
    plt.legend()
    fig.suptitle("Modulation index histogram")
    # Save the figure
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'modulation_index.png'), dpi=300, transparent=True)
    plt.close(fig)
    
def preferred_angle_distribution(selected_df, path=''):
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(9, 3), constrained_layout=True)
    class_labels = set(selected_df['class'])
    for idx, neu_class in enumerate(class_labels):
        class_angles = selected_df.loc[selected_df['class']==neu_class, 'preferred_angle']
        class_color = selected_df.loc[selected_df['class']==neu_class, 'color'].iloc[0]
        axs[idx].hist(class_angles, color=class_color, density=True, bins=40, label=neu_class)
        axs[idx].tick_params(axis='both', labelsize=12)
        axs[idx].set_xlabel(u'Preferred direction [\N{DEGREE SIGN}]', fontsize=12)
        axs[idx].legend(loc='upper right')
    axs[0].set_ylabel('Density of neurons', fontsize=12)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'preferred_angle_distribution.png'), dpi=300, transparent=True)
    plt.close(fig)
    
    
def input_current_classes_distributions(data, feature, pairs, pvalues, filename='', path=''):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    PROPS = {'boxprops':{'facecolor':'none', 'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},
            'capprops':{'color':'black'} }
    sns.boxplot(ax=ax, x=data['class'], y=data[feature], color='white', 
                width=.5, showfliers=False, order=['dVf', 'unclassified', 'hVf'], showmeans=True,
                meanprops={"markerfacecolor":"r", 
                           "markeredgecolor":"r",
                          "markersize":"6"}, 
                **PROPS)
    # Create an array with the keller colors
    colors = ['#33ABA2', '#9CB0AE', '#F06233']
    # Set your custom color palette
    kellerPalette = sns.set_palette(sns.color_palette(colors))
    sns.swarmplot(ax=ax, x=data['class'], y=data[feature], 
                  order=['dVf', 'unclassified', 'hVf'], size=0.7, 
                  palette=kellerPalette, zorder=0)
    # Add annotations
    annotator = Annotator(ax, pairs, x=data['class'], y=data['baseline_input_current'], 
                       order=['dVf', 'unclassified', 'hVf'], size=0.7, palette=kellerPalette)
    annotator.set_pvalues(pvalues)
    annotator.annotate()
    # Set figure configurations
    plt.ylabel('Baseline input current [pA]', fontsize=12)
    ax.set(xlabel=None)
    ax.tick_params(axis='both', which='both', labelsize=12) 
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, filename+'.png'), dpi=300, transparent=True)
    plt.close()
    
        
def plot_average_classes(data, selected_df, voltage_threshold=None, input_current_threshold=None,
                         stimuli_init_time=500, stimuli_end_time=1500, path=''):
    # Create a figure for each variable
    neurons_class = ['hVf', 'dVf', 'unclassified']
    variables = ['v', 'input_current', 'recurrent_current', 'bottom_up_current']
    labels = ['Membrane voltage response', 'Input current response', 'Recurrent current response', 'Bottom-up current response']
    for neuron_variable, label in zip(variables, labels):
        fig = plt.figure()
        for neu_class in neurons_class:
            class_stats = selected_df.loc[selected_df['class']==neu_class]
            class_indices = class_stats.index
            class_data = data[neuron_variable][:,:,class_indices]
            n_simulations, simulation_length, n_neurons = class_data.shape
            times = np.linspace(0, simulation_length, simulation_length)  
            mean_class = np.mean(class_data, axis=(0, 2))
            color = class_stats['color'].iloc[0]
            plt.plot(times, mean_class, color=color, label=neu_class)
            if n_simulations > 1:
                sem_class = (np.std(class_data, ddof=1, axis=(0, 2)) / np.sqrt(n_neurons))
                plt.fill_between(times, mean_class + sem_class, mean_class - sem_class,
                                 alpha=0.3, color=color)
        plt.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        plt.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        plt.hlines(0, 0, simulation_length, color='k', linewidths=1,zorder=10)
        # Include classification thresholds if provided
        if neuron_variable == 'v':
            if voltage_threshold is not None:
                plt.hlines(voltage_threshold, 0, simulation_length, linestyles='dotted', color='k', linewidths=2,zorder=10, label='Classification threshold')
                plt.hlines(-voltage_threshold, 0, simulation_length, linestyles='dotted', color='k', linewidths=2,zorder=10)
            plt.ylabel(f'{label} [mV]')
        elif neuron_variable == 'input_current':
            if input_current_threshold is not None:
                plt.hlines(input_current_threshold, 0, simulation_length, linestyles='dotted', color='k', linewidths=2,zorder=10, label='Classification threshold')
                plt.hlines(-input_current_threshold, 0, simulation_length, linestyles='dotted', color='k', linewidths=2,zorder=10)
            plt.ylabel(f'{label} [pA]')
            # plt.ylim(-2, 2)
        elif 'current' in neuron_variable:
            plt.ylabel(f'{label} [pA]')
            # plt.ylim(-2, 2)    
        plt.xlabel('Time [ms]')
        plt.tick_params(axis='both', labelsize=12)
        leg = plt.legend()
        neurons_color = ['#F06233', '#33ABA2', '#9CB0AE']
        for text, color in zip(leg.get_texts(), neurons_color):
            plt.setp(text, color=color)
        leg.set_zorder(102)
        # Save figure
        os.makedirs(path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(path, f'{neuron_variable}_response_per_class.png'), dpi=300, transparent=True)
        plt.close(fig)
        
def subplot_currents_average_classes(data, selected_df, input_current_threshold=None, 
                                     stimuli_init_time=500, stimuli_end_time=1500, path=''):
    # Create a figure with subplots for each current variable
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9, 3), constrained_layout=True)
    variables = ['recurrent_current', 'bottom_up_current', 'input_current']
    labels = ['Recurrent', 'Bottom up', 'Total']
    neurons_class = ['hVf', 'dVf', 'unclassified']
    for neuron_variable, label, idx in zip(variables, labels, range(len(variables))):
        for neu_class in neurons_class:
            class_stats = selected_df.loc[selected_df['class']==neu_class]
            class_indices = class_stats.index
            class_data = data[neuron_variable][:,:,class_indices]
            color = class_stats['color'].iloc[0]
            n_simulations, simulation_length, n_neurons = class_data.shape
            times = np.linspace(0, simulation_length, simulation_length) 
            mean_class = np.mean(class_data, axis=(0, 2))
            axs[idx].plot(times, mean_class, color=color, label=neu_class)
            if n_simulations > 1:
                sem_class = (np.std(class_data, ddof=1, axis=(0, 2)) / np.sqrt(n_neurons))
                axs[idx].fill_between(times, mean_class + sem_class, mean_class - sem_class,
                                      alpha=0.3, color=color)
        axs[idx].set_xlabel('Time [ms]', fontsize=8)
        axs[idx].tick_params(axis='both', labelsize=8)
        axs[idx].axhline(0, color='k', linewidth=1)
        axs[idx].axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        axs[idx].axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)        
        axs[idx].set_title(label, fontsize=9)
    
    leg = axs[0].legend(fontsize=7)
    neurons_color = ['#F06233', '#33ABA2', '#9CB0AE']
    for text, color in zip(leg.get_texts(), neurons_color):
        plt.setp(text, color=color)
    leg.set_zorder(102)
    axs[0].set_ylabel('Input current [pA]', fontsize=8)
    # Include classification threshold if provided
    if input_current_threshold is not None:
        threshold = axs[2].hlines(input_current_threshold, 0, simulation_length, 
                                  linestyles='dotted', color='k', linewidths=1,zorder=10, label='Classification threshold')
        axs[2].hlines(-input_current_threshold, 0, simulation_length, 
                      linestyles='dotted', color='k', linewidths=1,zorder=10)
        axs[2].legend([threshold],['Class threshold'], fontsize=7)   
    # Save figure
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'currents_average_per_class_subplot.png'), dpi=300, transparent=True)
    plt.close(fig)
    
    
def plot_current_per_class(data, selected_df, input_current_threshold=None, 
                           stimuli_init_time=500, stimuli_end_time=1500, path=''):
    # Create one plot for each current variable
    variables = ['input_current', 'recurrent_current', 'bottom_up_current']
    labels = ['Total', 'Recurrent', 'Bottom up']
    variables_colors = ['r', 'olive', 'dodgerblue']
    neurons_class = ['hVf', 'dVf', 'unclassified']
    for neu_class in neurons_class:
        class_stats = selected_df.loc[selected_df['class']==neu_class]
        class_indices = class_stats.index
        fig = plt.figure()
        for variable, label, color in zip(variables, labels, variables_colors):
            class_data = data[variable][:,:,class_indices]
            n_simulations, simulation_length, n_neurons = class_data.shape
            times = np.linspace(0, simulation_length, simulation_length)
            mean_class = np.mean(class_data, axis=(0, 2))
            plt.plot(times, mean_class, color=color, label=label, alpha=0.7)
            if n_simulations > 1:
                sem_class = (np.std(class_data, ddof=1, axis=(0, 2)) / np.sqrt(n_neurons))
                plt.fill_between(times, mean_class + sem_class, mean_class - sem_class,
                                 alpha=0.3, color=color)
        plt.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        plt.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        plt.hlines(0, 0, simulation_length, color='k', linewidths=1, zorder=10)
        # Include classification threshold if provided
        if input_current_threshold is not None:
            plt.hlines(input_current_threshold, 0, simulation_length, linestyles='dotted', color='k', linewidths=2,zorder=10, label='Classification threshold')
            plt.hlines(-input_current_threshold, 0, simulation_length, linestyles='dotted', color='k', linewidths=2,zorder=10)
        # plt.ylim(-2, 2)
        plt.ylabel('Input current [pA]')
        plt.xlabel('Time [ms]')
        plt.tick_params(axis='both', labelsize=12)
        leg = plt.legend()
        for text, color in zip(leg.get_texts(), variables_colors):
            plt.setp(text, color=color)
        leg.set_zorder(102)
        fig.suptitle(f'Average {neu_class} neuron current response')
        # Save figure
        os.makedirs(path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(path, f'average_{neu_class}_neuron_current_response.png'), dpi=300, transparent=True)
        plt.close(fig)
        
        
def subplot_current_per_class(data, selected_df, input_current_threshold=None, 
                              stimuli_init_time=500, stimuli_end_time=1500, path=''): 
    # Create subplots for each current variable
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(9, 3), constrained_layout=True)
    variables = ['input_current', 'recurrent_current', 'bottom_up_current']
    labels = ['Total', 'Recurrent', 'Bottom up']
    variables_colors = ['r', 'olive', 'dodgerblue']
    neurons_class = ['hVf', 'unclassified', 'dVf']
    for index, neu_class in enumerate(neurons_class):
        class_stats = selected_df.loc[selected_df['class']==neu_class]
        class_indices = class_stats.index
        for variable, label, color in zip(variables, labels, variables_colors):
            class_data = data[variable][:,:,class_indices]
            n_simulations, simulation_length, n_neurons = class_data.shape
            times = np.linspace(0, simulation_length, simulation_length)
            mean_class = np.mean(class_data, axis=(0, 2))
            axs[index].plot(times, mean_class, color=color,
                            label=label, alpha=0.7)
            if n_simulations > 1:
                sem_class = (np.std(class_data, ddof=1, axis=(0, 2)) / np.sqrt(n_neurons))
                axs[index].fill_between(times, mean_class + sem_class, mean_class - sem_class,
                                        alpha=0.3, color=color)
        axs[index].axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        axs[index].axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        axs[index].axhline(0, color='k', linewidth=1, zorder=10)
        if input_current_threshold is not None:
            axs[index].axhline(input_current_threshold, linestyle='dotted', color='k', linewidth=1,zorder=10, label='Class threshold')
            axs[index].axhline(-input_current_threshold, linestyle='dotted', color='k', linewidth=1,zorder=10)
        # axs[index].set_ylim(-2, 2)
        axs[index].set_xlabel('Time [ms]', fontsize=8)
        axs[index].set_title(neu_class, fontsize=9)
        axs[index].tick_params(axis='both', labelsize=8)
        leg = plt.legend(fontsize=7)
        for text, color in zip(leg.get_texts(), variables_colors):
            plt.setp(text, color=color)
        leg.set_zorder(102)
    axs[0].set_ylabel('Input current [pA]', fontsize=8)
    # Save figure
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'average_subplot_neuron_current_response.png'), dpi=300, transparent=True)
    plt.close(fig)
    
    
def firing_rate_per_class(data, selected_df, sampling_interval, normalize_firing_rates=True, 
                          exclude_NS_neurons=True, stimuli_init_time=500, stimuli_end_time=1500,
                          path=''):
    # Obtain the init and end times of the stimuli according to the firing rate sampling interval
    # stimuli_init_time = int(stimuli_init_time/sampling_interval)
    # stimuli_end_time = int(stimuli_end_time/sampling_interval)
    # Average over trials
    firing_rates = np.mean(data['firing_rate'], axis=0)
    simulation_length, n_neurons = firing_rates.shape
    # Normalize the firing rate according to the maximum firing rate reached on the simulation
    if normalize_firing_rates:
        max_firing_rate = np.max(firing_rates, axis=0)
        firing_rates = np.divide(firing_rates, max_firing_rate, 
                                 out=np.zeros_like(firing_rates), where=max_firing_rate!=0)
    # Exclude from the representations the neurons without activity
    if exclude_NS_neurons:
        firing_neurons = np.any(firing_rates!=0, axis=0)
        firing_rates = firing_rates[:, firing_neurons]
    else:
        firing_neurons = np.full(n_neurons, True)
    # Plot the population traces of the firing rates
    fig = plt.figure()
    ax = plt.subplot(111)
    simulation_length = firing_rates.shape[0]
    times = np.linspace(0, 2500, simulation_length)
    neurons_class = ['hVf', 'dVf', 'unclassified']
    for index, neu_class in enumerate(neurons_class):
        class_stats = selected_df.loc[np.logical_and(selected_df['class']==neu_class, firing_neurons)]
        class_indices = class_stats.index
        class_neurons = len(class_indices)
        class_color=class_stats['color'].iloc[0]
        class_mask = np.full(n_neurons, False)
        class_mask[class_indices] = True
        class_mask = class_mask[firing_neurons].astype(bool)
        # class_firing_rate = np.mean(firing_rates_mean[:, class_mask], axis=1)
        # class_firing_rate_sem = np.mean(firing_rates_sem[:, class_mask], axis=1)
        class_firing_rate = np.mean(firing_rates[:, class_mask], axis=1)
        class_firing_rate_sem = stats.sem(firing_rates[:, class_mask], axis=1)
        ax.plot(times, class_firing_rate, class_color, label=f'{neu_class}:{class_neurons}')        
        ax.fill_between(times, class_firing_rate + class_firing_rate_sem, class_firing_rate - class_firing_rate_sem,
                        alpha=0.3, color=class_color)
    ax.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    ax.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    ax.legend()
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Firing rate [Hz]')
    # Save figure
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'firing_rate_per_class.png'), dpi=300, transparent=True)
    plt.close(fig)
    
    
def voltage_heatmap(data, selected_df, stimuli_init_time=500, stimuli_end_time=1500, path=''):
    # Select the ordered neurons
    heatmap_df = selected_df.loc[np.logical_not(np.isnan(selected_df['heatmap_neurons']))]
    ordered_ids = [x for x,_ in sorted(zip(heatmap_df.index, heatmap_df['heatmap_neurons']), key=lambda element: (element[1]))]
    # Change dtype because plt.imshow inconsistency with float16
    v_selection = np.mean(data['v'][:, :, ordered_ids], axis=0).astype(float)
    simulation_length, n_neurons = v_selection.shape
    # Design the histogram with the style of Keller et al (2020)
    fig = plt.figure()
    ax = plt.subplot(111) 
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    # norm = mpl.colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=20)
    cs = plt.imshow(np.transpose(v_selection), norm=norm,
                    interpolation='none', aspect='auto', cmap=cm)
    cbar = fig.colorbar(cs, extend='both')
    cbar.set_label(label=r'$V_m$ $response$ $[mV]$', weight='bold')
    ax.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    ax.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Sorted neuron #')
    ax.set_xlabel('Time [ms]')
    ax.set_xticks(np.arange(0, 3000, 500))
    ax.set_xticklabels(np.arange(0, 3000, 500))
    # If a classification was made include some shading for the neurons of each class
    if 'class' in heatmap_df.columns:
        fig.patches.extend([plt.Rectangle((0.075, 0.613), 0.05, 0.267,
                                          fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])
        fig.patches.extend([plt.Rectangle((0.075, 0.113), 0.05, 0.24,
                                          fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.05, 0.82, 'hVf', rotation='vertical', color='#F06233', visible=True)
        fig.text(0.05, 0.13, 'dVf', rotation='vertical', color='#33ABA2', visible=True)
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, 'voltage_heatmap.png'), dpi=300, transparent=True)
        plt.close(fig)

        # Second figure
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(
            6.4, 12), sharex=True, constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0.5, hspace=-0.13, wspace=0)
        cs = axs[0].imshow(np.transpose(v_selection), norm=norm,
                        interpolation='none', aspect='auto', cmap=cm)
        cbar = fig.colorbar(cs, extend='both', ax=axs[0], location='right')
        cbar.set_label(label=r'$V_m$ $response$ $[mV]$', weight='bold', fontsize=18)
        axs[0].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=1)
        axs[0].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=1)
        axs[0].set_ylabel('Sorted neuron #', fontsize=16)
        fig.patches.extend([plt.Rectangle((0.06, 0.81), 0.046, 0.147,
                                          fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])
        fig.patches.extend([plt.Rectangle((0.06, 0.535), 0.046, 0.138,
                                          fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.025, 0.93, 'hVf', rotation='vertical',
                  color='#F06233', visible=True, fontsize=14)
        fig.text(0.025, 0.54, 'dVf', rotation='vertical',
                  color='#33ABA2', visible=True, fontsize=14)
        axs[0].tick_params(axis='both', which='major', labelsize=14)
        
        times = np.linspace(0, simulation_length, simulation_length)
        neurons_class = ['hVf', 'unclassified', 'dVf']
        for index, neu_class in enumerate(neurons_class):
            class_stats = selected_df.loc[selected_df['class']==neu_class]
            class_indices = class_stats.index
            class_color = class_stats['color'].iloc[0]
            class_data = np.mean(data['v'][:,:,class_indices], axis=0) #average over realizations
            class_mean = np.mean(class_data, axis=1) #average over neurons
            class_sem = stats.sem(class_data, axis=1)
            axs[1].plot(times, class_mean, color=class_color, label=neu_class)
            axs[1].fill_between(times, class_mean + class_sem, class_mean - class_sem,
                                alpha=0.3, color=class_color)
        
        axs[1].hlines(0, 0, simulation_length, linestyles='dotted', color='k', linewidths=1, zorder=1000)
        axs[1].vlines(stimuli_init_time, -10, 10, linestyle='dashed',
                      color='k', linewidth=1.5, alpha=1, zorder=1000)
        axs[1].vlines(stimuli_end_time, -10, 10, linestyle='dashed',
                      color='k', linewidth=1.5, alpha=1, zorder=1000)
        
        axs[1].set_ylim(-7.5, 7.5)
        axs[1].set_ylabel('Membrane potential [mV]', fontsize=16)
        axs[1].set_xlabel('Time [ms]', fontsize=16)
        
        axs[1].set_xticks(np.arange(0, 3000, 500))
        axs[1].set_xticklabels(np.arange(0, 3000, 500))
        axs[1].tick_params(axis='both', labelsize=16)
        
        leg = axs[1].legend(fontsize=16)
        for text, color in zip(leg.get_texts(), ['#F06233', '#33ABA2', '#9CB0AE']):
            plt.setp(text, color=color)
        
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, 'voltage_figure.png'), dpi=300, transparent=True)
        plt.close(fig)
    else:
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, 'voltage_heatmap.png'), dpi=300, transparent=True)
        plt.close(fig)
        
def currents_heatmap(data, variable_label, selected_df,  stimuli_init_time=500, 
                     stimuli_end_time=1500, path=''):
    # Select the ordered neurons
    heatmap_df = selected_df.loc[np.logical_not(np.isnan(selected_df['heatmap_neurons']))]
    ordered_ids = [x for x,_ in sorted(zip(heatmap_df.index, heatmap_df['heatmap_neurons']), key=lambda element: (element[1]))]
    data_selection = np.mean(data[variable_label][:, :, ordered_ids], axis=0).astype(float)
    simulation_length, n_neurons = data_selection.shape
                
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    cs = plt.imshow(np.transpose(data_selection), norm=norm,
                    interpolation='none', aspect='auto', cmap=cm)
    cbar = fig.colorbar(cs, extend='both')
    cbar.set_label(label=r'Input current response $[pA]$', weight='bold')
    ax.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    ax.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Sorted neuron #')
    ax.set_xlabel('Time [ms]')
    ax.set_xticks(np.arange(0, 3000, 500))
    ax.set_xticklabels(np.arange(0, 3000, 500))
    if 'class' in heatmap_df.columns:
        pos1 = ax.get_position() # get the original position 
        sample_row_height = pos1.height/len(heatmap_df)  
        selection = heatmap_df['class']
        n_dvf = list(selection.values).count('dVf')
        n_hvf = list(selection.values).count('hVf')
        fig.patches.extend([plt.Rectangle((pos1.x0-0.05, pos1.y1-n_hvf*sample_row_height), 0.05, n_hvf*sample_row_height,
                                          fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])
        fig.patches.extend([plt.Rectangle((pos1.x0-0.05, pos1.y0), 0.05, n_dvf*sample_row_height,
                                          fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.05, pos1.y1-2*sample_row_height, 'hVf', rotation='vertical', color='#F06233', visible=True)
        fig.text(0.05, pos1.y0, 'dVf', rotation='vertical', color='#33ABA2', visible=True)
            
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'{variable_label}_heatmap.png'), dpi=300, transparent=True)
    plt.close(fig)
 
def currents_figure(data, selected_df, input_current_threshold=None, 
                    stimuli_init_time=500, stimuli_end_time=1500, path=''):
    # Prepare the figure grid
    fig, grid = plt.subplots(nrows=2, ncols=4, figsize=(12,5),
                   gridspec_kw={'width_ratios':(1,1,1,0.05), 'height_ratios':(0.6,0.4)}, sharex='col')
    heatmaps_ax = grid[0]
    averages_ax = grid[1]
    averages_ax[3].set_visible(False)
    heatmaps_ax[0].sharey(heatmaps_ax[1])
    heatmaps_ax[1].sharey(heatmaps_ax[2])
    averages_ax[0].sharey(averages_ax[1])
    averages_ax[1].sharey(averages_ax[2])
    # Config heatmap colors
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    ### Heatmap
    heatmap_df = selected_df.loc[np.logical_not(np.isnan(selected_df['heatmap_neurons']))]
    ordered_ids = [x for x,_ in sorted(zip(heatmap_df.index, heatmap_df['heatmap_neurons']), key=lambda element: (element[1]))]
    current_variables = ['recurrent_current', 'bottom_up_current', 'input_current']
    for idx, current_variable in enumerate(current_variables):
        data_selection = np.mean(data[current_variable][:, :, ordered_ids], axis=0).astype(float)
        cs = heatmaps_ax[idx].imshow(np.transpose(data_selection), norm=norm,
                                     interpolation='none', aspect='auto', cmap=cm)
        heatmaps_ax[idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
        heatmaps_ax[idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
        heatmaps_ax[idx].tick_params(axis='both', bottom=False, labelbottom=False, labelsize=12)
        if idx != 0:
            heatmaps_ax[idx].set_yticklabels([])
            heatmaps_ax[idx].set_yticks([])
    heatmaps_ax[0].set_ylabel('Sorted neuron #', fontsize=12)
    cbar = fig.colorbar(cs, cax=heatmaps_ax[3], extend='both')
    cbar.set_label(label='Input current [pA]', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([-50, -25, 0, 25, 50])
    cbar.set_ticklabels([-50, -25, 0, 25, 50])
    ### Traces
    neurons_class = ['hVf', 'dVf', 'unclassified']
    for idx, neuron_variable in enumerate(current_variables):
        for neu_class in neurons_class:
            class_stats = selected_df.loc[selected_df['class']==neu_class]
            class_indices = class_stats.index
            class_data = data[neuron_variable][:,:,class_indices]
            n_simulations, simulation_length, n_neurons = class_data.shape
            mean_class = np.mean(class_data, axis=(0, 2))
            sem_class = (np.std(class_data, ddof=1, axis=(0, 2)) / np.sqrt(n_neurons))
            times = np.linspace(0, simulation_length, simulation_length)   
            averages_ax[idx].plot(times, mean_class, color=class_stats['color'].iloc[0], label=neu_class)
            averages_ax[idx].fill_between(times, mean_class + sem_class, mean_class - sem_class,
                                          alpha=0.3, color=class_stats['color'].iloc[0])
        averages_ax[idx].tick_params(axis='both', labelsize=12)
        averages_ax[idx].set_xticks(np.arange(0, 2500, 500))
        averages_ax[idx].set_xticklabels(np.arange(0, 2500, 500))
        averages_ax[idx].axhline(0, color='k', linewidth=1)
        averages_ax[idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1, zorder=10)
        averages_ax[idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1, zorder=10)        
        averages_ax[idx].set_xlabel('Time [ms]', fontsize=12)
        # averages_ax[idx].set_ylim(-1, 1)
        averages_ax[idx].set_xlim(0, 2500)
        if idx != 0:
            averages_ax[idx].set_yticklabels([])
            averages_ax[idx].set_yticks([])
    averages_ax[0].set_ylabel('Input current [pA]', fontsize=12)
    if input_current_threshold is not None:
        averages_ax[2].hlines(input_current_threshold, 0, simulation_length, 
                                          linestyles='dotted', color='k', linewidths=1,zorder=10, label='Classification threshold')
        averages_ax[2].hlines(-input_current_threshold, 0, simulation_length, 
                              linestyles='dotted', color='k', linewidths=1, zorder=10)
    averages_ax[1].get_shared_y_axes().join(averages_ax[1], averages_ax[0])
    averages_ax[2].get_shared_y_axes().join(averages_ax[2], averages_ax[0])
    
    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    pos1 = heatmaps_ax[0].get_position() # get the original position 
    sample_row_height = pos1.height/30  
    selection = heatmap_df['class']
    n_dvf = list(selection.values).count('dVf')
    n_hvf = list(selection.values).count('hVf')
    fig.patches.extend([plt.Rectangle((pos1.x0-0.025, pos1.y1-n_hvf*sample_row_height), 0.025, n_hvf*sample_row_height,
                                      fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.patches.extend([plt.Rectangle((pos1.x0-0.025, pos1.y0), 0.025, n_dvf*sample_row_height,
                                      fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.text(0.085, pos1.y1-2*sample_row_height, 'hVf', rotation='vertical', color='#F06233', visible=True, fontsize=10, weight='bold')
    fig.text(0.085, pos1.y0, 'dVf', rotation='vertical', color='#33ABA2', visible=True, fontsize=10, weight='bold')
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'composite_currents_heatmap.png'), dpi=300, transparent=True)
    plt.close(fig)
    
    
def currents_figure_with_stimulus(data, selected_df, variable_label, frequency, 
                                  stimuli_init_time=500, stimuli_end_time=1500,
                                  reverse=False, path=''):
    # Prepare the figure layout
    fig, grid = plt.subplots(nrows=2, ncols=2, figsize=(6,4),
                   gridspec_kw={'width_ratios':(1, 0.05), 'height_ratios':(0.85,0.15)}, sharex='col')
    heatmaps_ax, cax0 = grid[0]
    drifting_grating_ax, cax1 = grid[1]   
    cax1.set_visible(False)
    heatmaps_ax.clear()
    drifting_grating_ax.clear()
    # Heatmap
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    
    heatmap_df = selected_df.loc[np.logical_not(np.isnan(selected_df['heatmap_neurons']))]
    ordered_ids = [x for x,_ in sorted(zip(heatmap_df.index, heatmap_df['heatmap_neurons']), key=lambda element: (element[1]))]
    data_selection = np.mean(data[variable_label][:, :, ordered_ids], axis=0).astype(float)
    cs = heatmaps_ax.imshow(np.transpose(data_selection), norm=norm,
                            interpolation='none', aspect='auto', cmap=cm)
    heatmaps_ax.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    heatmaps_ax.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    heatmaps_ax.tick_params(axis='both', bottom=False, labelbottom=False, labelsize=10)
    heatmaps_ax.set_ylabel('Sorted neuron #', fontsize=10)
    cbar = fig.colorbar(cs, cax=cax0, extend='both')
    cbar.set_label(label='Input current [pA]', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels([-2, -1, 0, 1, 2])

    simulation_length = data_selection.shape[0]
    drifting_grating_plot = DriftingGrating(frequency=frequency, stimuli_init_time=stimuli_init_time, 
                                            stimuli_end_time=stimuli_end_time, reverse=reverse)
    drifting_grating_plot(drifting_grating_ax, simulation_length)
    
    plt.subplots_adjust(wspace=0.03, hspace=0.07, left=0.19)
    drifting_grating_ax.set_xlim(heatmaps_ax.get_xlim())
      
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'{variable_label}_heatmap.png'), dpi=300, transparent=True)
    plt.close(fig)


def firing_rate_figure(data, selected_df, sampling_interval, frequency, 
                       normalize_firing_rates=True,  exclude_NS_neurons=True, extra_neuron_pop=None, 
                       extra_firing_rates=None, stimuli_init_time=500, stimuli_end_time=1500, reverse=False, path=''):
    # Obtain the init and end times of the stimuli according to the firing rate sampling interval
    stimuli_init_time = int(stimuli_init_time/sampling_interval)
    stimuli_end_time = int(stimuli_end_time/sampling_interval)
    # Average over trials
    firing_rates = data['firing_rate']
    mean_firing_rates = np.mean(firing_rates, axis=0)
    simulation_length, n_neurons = mean_firing_rates.shape
    # Normalize the firing rate according to the maximum firing rate reached on the simulation
    if normalize_firing_rates:
        max_firing_rate = np.max(mean_firing_rates, axis=0)
        mean_firing_rates = np.divide(mean_firing_rates, max_firing_rate, 
                                      out=np.zeros_like(mean_firing_rates), where=max_firing_rate!=0)
        norm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        label = 'NFR'
    else:
        norm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=1.5, vmax=3)
        label = 'FR [Hz]'
    # Exclude from the representations the neurons without activity
    if exclude_NS_neurons:
        firing_neurons = np.any(mean_firing_rates!=0, axis=0)
        mean_firing_rates = mean_firing_rates[:, firing_neurons]
    else:
        firing_neurons = np.full(n_neurons, True)
        
    # Order the neurons according to their firing rate during the stimuli
    mean_variations = np.mean(mean_firing_rates[stimuli_init_time: stimuli_end_time, :], axis=0)
    sorted_indices = np.arange(mean_firing_rates.shape[1])
    sorted_indices = [x for _, x in sorted(zip(mean_variations, sorted_indices), key=lambda element: (element[0]), reverse=True)]   
    firing_rates_ordered = mean_firing_rates[:,sorted_indices]
    # Create the figure 
    fig, grid = plt.subplots(nrows=3, ncols=2, figsize=(4,8),
                   gridspec_kw={'width_ratios':(1,0.03), 'height_ratios':(1,0.5,0.1)}, sharex='col')
    firing_rates_ax, cax1 = grid[0]
    firing_rates_classes_ax, cax2 = grid[1]
    drifting_grating_ax, cax3 = grid[2]
    cax2.set_visible(False)
    cax3.set_visible(False)
    firing_rates_ax.clear()
    firing_rates_classes_ax.clear()
    drifting_grating_ax.clear()
    # Plot the firing rates heatmap
    cs = firing_rates_ax.imshow(np.transpose(firing_rates_ordered), norm=norm,
                                interpolation='none', aspect='auto', cmap='coolwarm')
    cbar = fig.colorbar(cs, cax=cax1, extend='neither') #, ticks=[0, 1, 2, 3])
    cbar.ax.set_title(label=label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    # cbar.ax.set_yticklabels(['0', '1', '2', '3'])
    firing_rates_ax.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    firing_rates_ax.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    firing_rates_ax.set_ylabel('Sorted neuron #', fontsize=12)
    firing_rates_ax.tick_params(axis='both',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False,
                                labelsize=10) 
    # Plot the population traces of the firing rates
    times = np.linspace(0, simulation_length, simulation_length)
    neurons_class = ['hVf', 'dVf']
    class_colors = ['#F06233', '#33ABA2']
    for neu_class, color in zip(neurons_class, class_colors):
        class_stats = selected_df.loc[np.logical_and(selected_df['class']==neu_class, firing_neurons)]
        class_indices = class_stats.index
        class_mask = np.full(n_neurons, False)
        class_mask[class_indices] = True
        class_mask = class_mask[firing_neurons].astype(bool)
        class_data = np.mean(mean_firing_rates[:,class_mask], axis=1)
        class_sem = stats.sem(mean_firing_rates[:,class_mask], axis=1)
        firing_rates_classes_ax.plot(times, class_data, color)
        firing_rates_classes_ax.fill_between(times, class_data + class_sem, 
                                             class_data - class_sem, 
                                             alpha=0.3, color=color)
    firing_rates_classes_ax.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8, zorder=10)
    firing_rates_classes_ax.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8, zorder=10)
    firing_rates_classes_ax.tick_params(axis='both',          # changes apply to the x-axis
                                        which='both',      # both major and minor ticks are affected
                                        bottom=False,      # ticks along the bottom edge are off
                                        top=False,         # ticks along the top edge are off
                                        labelbottom=False,
                                        labelsize=10) 
    if normalize_firing_rates:
        firing_rates_classes_ax.set_ylim(0, 1)
        firing_rates_classes_ax.set_ylabel(label, fontsize=12)
    else:
        firing_rates_classes_ax.set_ylabel(label, fontsize=12)
    # Include other populations firing rates in the plot    
    if extra_neuron_pop is not None:
        mean_extra_firing_rates = np.mean(extra_firing_rates, axis=0)
        if normalize_firing_rates:
            other_max_firing_rate = np.max(mean_extra_firing_rates, axis=0)
            mean_extra_firing_rates = np.divide(mean_extra_firing_rates, other_max_firing_rate, 
                                                out=np.zeros_like(mean_extra_firing_rates), where=other_max_firing_rate!=0)
            
        if exclude_NS_neurons:
            other_firing_neurons = np.any(mean_extra_firing_rates!=0, axis=0)
            mean_extra_firing_rates = mean_extra_firing_rates[:, other_firing_neurons]
        
        sem_extra_firing_rates = stats.sem(mean_extra_firing_rates, axis=1)
        mean_extra_firing_rates = np.mean(mean_extra_firing_rates, axis=1)
        firing_rates_classes_ax.plot(times, mean_extra_firing_rates, 'g')
        firing_rates_classes_ax.fill_between(times, mean_extra_firing_rates + sem_extra_firing_rates, 
                                             mean_extra_firing_rates - sem_extra_firing_rates, 
                                             alpha=0.3, color='g')

    # Plot the stimuli scheme
    drifting_grating_plot = DriftingGrating(frequency=frequency, stimuli_init_time=stimuli_init_time, 
                                            stimuli_end_time=stimuli_end_time, reverse=reverse)
    drifting_grating_plot(drifting_grating_ax, 2500, stimulus_length=simulation_length)
    plt.subplots_adjust(wspace=0.03, hspace=0.07, left=0.19)
    drifting_grating_ax.set_xlim(firing_rates_ax.get_xlim())
    
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'firing_rates_figure_normalized_{normalize_firing_rates}_exclude_NS_neurons_{exclude_NS_neurons}_other_pop_{extra_neuron_pop}.png'), dpi=300, transparent=True)
    plt.close(fig)
    
    
 ########################## dynamic_populations_comparison.py ###############   
def plot_average_population_comparison(input_current_1, input_current_2, 
                                       pop1_name='pop1', pop2_name='pop2',
                                       color1='#808BD0', color2='#92876B',
                                       stimuli_init_time=500, stimuli_end_time=1500,
                                       path=''):
    ### Represent the population traces
    n_simulations, simulation_length, n_neurons = input_current_1.shape
    fig = plt.figure(figsize=(6, 4))
    # Pop 1
    mean_1 = np.mean(input_current_1, axis=(0, 2))
    sem_1 = (np.std(input_current_1, ddof=1, axis=(0, 2)) /
                        np.sqrt(n_neurons))
    times = np.linspace(0, simulation_length, simulation_length)
    plt.plot(times, mean_1, color=color1,
                label=pop1_name)
    plt.fill_between(times, 
                      mean_1 + sem_1, 
                      mean_1 - sem_1, 
                      alpha=0.3, color=color1)
    # Pop 2   
    mean_2 = np.mean(input_current_2, axis=(0, 2))
    sem_2 = (np.std(input_current_2, ddof=1, axis=(0, 2)) /
                        np.sqrt(n_neurons))
    plt.plot(times, mean_2, color=color2,
                label=pop2_name)
    plt.fill_between(times, 
                      mean_2 + sem_2, 
                      mean_2 - sem_2, 
                      alpha=0.3, color=color2)
    
    plt.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    plt.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    plt.hlines(0, 0, simulation_length, color='k', linewidths=1,zorder=10)
    plt.ylabel('Input current response [pA]', fontsize=12)
    plt.xlabel('Time [ms]', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    leg = plt.legend()
    for text, color in zip(leg.get_texts(), [color1, color2]):
        plt.setp(text, color=color)
    leg.set_zorder(102)
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, f'input_current_comparison_{pop1_name}_{pop2_name}_trace.png'), dpi=300, transparent=True)
    plt.close(fig)
    

def populations_current_response_violin_hist(data, pairs, pvalues, colors, classifications_threshold=None, 
                                             path=''):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # Set your custom color palette
    violinPalette = sns.set_palette(sns.color_palette(colors))
    sns.violinplot(ax=ax, data=data, inner='quartile', palette=violinPalette, orient='v', zorder=10,
                  estimator='mean')
    # Set the pvalues diagram
    annotator = Annotator(ax, pairs, data=data, palette=violinPalette)
    annotator.set_pvalues(pvalues)
    annotator.annotate()
    # Set other plotting parameters
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
    if classifications_threshold is not None:
        plt.axhspan(classifications_threshold, 150, 
                    alpha=0.6, color='#33ABA2', zorder=1)
        plt.axhspan(-classifications_threshold, -50, 
                    alpha=0.6, color='#F06233', zorder=1)
    else:
        plt.axhspan(0, 60, 
                    alpha=0.6, color='#33ABA2', zorder=1)
        plt.axhspan(0, -60, 
                    alpha=0.6, color='#F06233', zorder=1)
    plt.ylabel('Input current variations [pA]', fontsize=12)
    plt.ylim(-60, 60)
    ax.tick_params(axis='both', which='both', labelsize=12) 
    fig.tight_layout()
    plt.savefig(os.path.join(path, 'populations_current_response_violin_hist.png'), dpi=300, transparent=True)
    plt.close(fig)
    

########################## dynamic_classification_comparison.py ###############   
def currents_comparison_figure(current_variables, frequencies, reverses, selected_df, 
                               stimuli_init_time=500, stimuli_end_time=1500, path=''):
    n_plots = len(current_variables)
    fig, grid = plt.subplots(nrows=2, ncols=n_plots+1, figsize=(3*n_plots,4),
                   gridspec_kw={'width_ratios':(1,)*n_plots + (0.05,), 'height_ratios':(0.85,0.15)}, sharex='col')
    heatmaps_ax = grid[0]
    drifting_grating_ax = grid[1]    
    # Share y axis in the stimulus illustrations
    # for idx in range(1, len(drifting_grating_ax)):
    #     drifting_grating_ax[idx].get_shared_y_axes().join(drifting_grating_ax[idx], drifting_grating_ax[0])

    drifting_grating_ax[-1].set_visible(False)    
    selected_df = selected_df.sort_values(by=['heatmap_neurons'])
    sorted_indices = selected_df.loc[np.logical_not(np.isnan(selected_df['heatmap_neurons'])), 'heatmap_neurons'].index
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    for idx, current_variable in enumerate(current_variables):
        neurons_sample = current_variable[:, :, sorted_indices].astype(float)
        neurons_sample = np.mean(neurons_sample, axis=0)
        simulation_length = neurons_sample.shape[0]
        cs = heatmaps_ax[idx].imshow(np.transpose(neurons_sample), norm=norm,
                                     interpolation='none', aspect='auto', cmap=cm)
        heatmaps_ax[idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
        heatmaps_ax[idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
        heatmaps_ax[idx].tick_params(axis='both', labelsize=8)            
        drifting_grating_plot = DriftingGrating(frequency=frequencies[idx], stimuli_init_time=stimuli_init_time, 
                                                stimuli_end_time=stimuli_end_time, reverse=reverses[idx])
        drifting_grating_plot(drifting_grating_ax[idx], simulation_length)
        drifting_grating_ax[idx].set_xlim(heatmaps_ax[idx].get_xlim())
    
        if idx != 0:
            y_axis = drifting_grating_ax[idx].axes.get_yaxis()
            y_label = y_axis.get_label()
            y_label.set_visible(False)
    
    for idx in range(1, n_plots):
        heatmaps_ax[idx].set_yticks([])
        heatmaps_ax[idx].set_yticklabels([])
        
    heatmaps_ax[0].set_ylabel('Sorted neuron #', fontsize=10)
    cbar = fig.colorbar(cs, cax=heatmaps_ax[-1], extend='both')
    cbar.set_label(label='Input current [pA]', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([-50, -25, 0, 25, 50])
    cbar.set_ticklabels([-50, -25, 0, 25, 50])
    plt.subplots_adjust(wspace=0.13, hspace=0.07)
    pos1 = heatmaps_ax[0].get_position() # get the original position 
    sample_row_height = pos1.height/30  
    selection = selected_df.loc[np.logical_not(np.isnan(selected_df['heatmap_neurons'])), 'class']
    n_dvf = list(selection.values).count('dVf')
    n_hvf = list(selection.values).count('hVf')
            
    fig.patches.extend([plt.Rectangle((pos1.x0-0.025, pos1.y1-n_hvf*sample_row_height), 0.025, n_hvf*sample_row_height,
                                      fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.patches.extend([plt.Rectangle((pos1.x0-0.025, pos1.y0), 0.025, n_dvf*sample_row_height,
                                      fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.text(0.085, pos1.y1-3*sample_row_height, 'hVf', rotation='vertical', color='#F06233', visible=True, fontsize=10, weight='bold')
    fig.text(0.085, pos1.y0+sample_row_height, 'dVf', rotation='vertical', color='#33ABA2', visible=True, fontsize=10, weight='bold')
        
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'currents_comparison.png'), dpi=300, transparent=True)
    plt.close(fig)
    
    
def matrices_comparison_figure(all_confussion_matrices, orientation, orientations, frequency, frequencies, path=''):
    n_plots = len(orientations)
    fig, grid = plt.subplots(nrows=1, ncols=n_plots+1, figsize=(3*n_plots, 3),
                   gridspec_kw={'width_ratios':(1,)*n_plots+ (0.05,)})
    matrices_axs = grid[:-1]
    cax = grid[-1]
    for idx, confussion_matrix, orientation2, frequency2 in zip(np.arange(n_plots), all_confussion_matrices, orientations, frequencies):
        sns.heatmap(confussion_matrix, vmin=0, vmax=1, annot=True, annot_kws={"size": 10}, fmt=".2f", yticklabels=True, cmap='binary', cbar_ax=cax, ax=matrices_axs[idx])
        matrices_axs[idx].set_xlabel(u'{orientation2}\N{DEGREE SIGN} - TF {frequency2} Hz'.format(orientation2=orientation2, frequency2=frequency2))
    matrices_axs[0].set_ylabel(u'{orientation}\N{DEGREE SIGN} - TF {frequency} Hz'.format(orientation=orientation, frequency=frequency))
    for idx in range(1, n_plots):
        matrices_axs[idx].set_yticks([])
        matrices_axs[idx].set_yticklabels([])
        
    plt.subplots_adjust(wspace=0.13, hspace=0.07, bottom=0.15)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'figure_matrices_comparison.png'), dpi=300, transparent=True)
    plt.close(fig)


########################## synapsis_analysis.py ###############   

def synapses_histogram(counts, y_label='', path=''):
    fig = plt.figure(figsize=(3, 5))
    key_record = {}
    # colors_exc = 'r'
    colors_inh = {'Htr3a':'b', 'Pvalb':'g', 'Sst':'y'}
    colors_exc = {'e2':'r', 'dV':'#33ABA2' , 'hV':'#F06233', 'un':'#9CB0AE'}
    inds = [x for _, x in sorted(zip(counts.keys(), range(len(counts.keys()))), key=lambda element: (element[0][1], element[0][0]))]   
    keys = counts.keys()[inds]
    counts = counts[inds]
    patches = []
    for key, count in zip(keys, counts):
        if key[:2] in ['dV', 'hV', 'un']:
            record_label = 'e2'
        else:
            record_label = key[:2]
        if record_label not in key_record.keys():
            key_record[record_label] = {}
            key_record[record_label]['record'] = count
            key_record[record_label]['index'] = 0
            if record_label == 'e2':
                exc_patch = mpatches.Patch(color='r', label='Exc')
                dVf_patch = mpatches.Patch(color='#33ABA2', label='dVf')
                hVf_patch = mpatches.Patch(color='#F06233', label='hVf')
                unc_patch = mpatches.Patch(color='#9CB0AE', label='unc')
                patches.append(exc_patch)
                patches.append(dVf_patch)
                patches.append(hVf_patch)
                patches.append(unc_patch)
                plt.bar('e23', count, width=0.8, align='center', color=colors_exc[key[:2]], label=key)
            elif record_label == 'i2':
                Htr3a_patch = mpatches.Patch(color='b', label='Htr3a')
                Pvalb_patch = mpatches.Patch(color='g', label='Pvalb')
                Sst_patch = mpatches.Patch(color='y', label='Sst')
                patches.append(Htr3a_patch)
                patches.append(Pvalb_patch)
                patches.append(Sst_patch)
                plt.bar('i23', count, width=0.8, align='center', color=colors_inh[key[3:]], label=key)
            elif key[1]!=2 and key[0]=='e':
                plt.bar(record_label, count, width=0.8, align='center', color='r', label=key)
            else:
                plt.bar(record_label, count, width=0.8, align='center', color=colors_inh[key[2:]], label=key)
        else:
            if key[:2] in ['e2', 'dV', 'hV', 'un']:
                key_record['e2']['index'] += 1
                plt.bar('e23', count, width=0.8, bottom=key_record['e2']['record'], align='center', color=colors_exc[key[:2]], label=key)
                key_record['e2']['record'] += count
            elif key[:2] == 'i2':
                key_record[key[:2]]['index'] += 1
                plt.bar('i23', count, width=0.8, bottom=key_record[key[:2]]['record'], align='center', color=colors_inh[key[3:]], label=key)        
                key_record[key[:2]]['record'] += count
            elif key[1]!=2 and key[0]=='e':
                key_record[key[:2]]['index'] += 1
                plt.bar(key[:2], count, width=0.8, bottom=key_record[key[:2]]['record'], align='center', color='r', label=key)
                key_record[key[:2]]['record'] += count
            else:
                key_record[key[:2]]['index'] += 1
                plt.bar(key[:2], count, width=0.8, bottom=key_record[key[:2]]['record'], align='center', color=colors_inh[key[2:]], label=key)
                key_record[key[:2]]['record'] += count
            
    plt.tick_params(axis='x', labelrotation=90)
    plt.ylabel(y_label)
    plt.legend(handles=patches)
    plt.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'{y_label}.png'), dpi=300, transparent=True)
    plt.close(fig)
    
def degree_distributions_per_class_plot(selected_df, n_bins=50, path=''):   
    max_degree = int(max([selected_df['Total k_in'].max(), selected_df['k_out'].max()]))
    min_degree = int(min([selected_df['Total k_in'].min(), selected_df['k_out'].min()]))
    binwidth = int((max_degree+abs(min_degree))/n_bins)
    max_weight = int(max([selected_df['Weighted Total k_in'].max(), selected_df['Weighted k_out'].max()]))
    min_weight = int(min([selected_df['Weighted Total k_in'].min(), selected_df['Weighted k_out'].min()]))
    binwidth_weight = int((max_weight+abs(min_weight))/n_bins)
    neuron_classes = set(selected_df['class'])
    for neu_class in neuron_classes:
        fig, axs = plt.subplots(2,2, sharex='col', sharey='col')
        reduced_selected_df = selected_df[selected_df['class']==neu_class]   
        axs[0][0].hist(reduced_selected_df['k_in'], density=True, bins=range(0, max_degree + binwidth, binwidth), label='Recurrent $k_{in}$', color='b', alpha=0.6) 
        axs[0][0].hist(reduced_selected_df['Total k_in'], density=True, bins=range(0, max_degree + binwidth, binwidth), label='Total $k_{in}$', color='g', alpha=0.6) 
        axs[1][0].hist(reduced_selected_df['k_out'], density=True, bins=range(0, max_degree + binwidth, binwidth), label='Recurrent $k_{out}$', color='r', alpha=0.6)
        axs[0][0].set_xlim(0, max_degree)
        axs[0][0].legend() 
        axs[0][0].set_title('Non-weighted')
        axs[0][0].set_ylabel('Probability density')
        axs[1][0].set_ylabel('Probability density')
        axs[1][0].set_xlabel('Degree')
        axs[1][0].legend()
        axs[0][1].hist(reduced_selected_df['Weighted k_in'], density=True, bins=range(min_weight, max_weight + binwidth_weight, binwidth_weight), label='Recurrent $k_{in}$', color='b', alpha=0.6) 
        axs[0][1].hist(reduced_selected_df['Weighted Total k_in'], density=True, bins=range(min_weight, max_weight + binwidth_weight, binwidth_weight), label='Total $k_{in}$', color='g', alpha=0.6) 
        axs[1][1].hist(reduced_selected_df['Weighted k_out'], density=True, bins=range(min_weight, max_weight + binwidth_weight, binwidth_weight), label='Recurrent $k_{out}$', color='r', alpha=0.6)
        axs[0][1].set_xlim(min_weight, max_weight)
        axs[0][1].legend() 
        axs[0][1].set_title('Weighted')
        axs[1][1].set_xlabel('Weight [pA]')
        axs[1][1].legend()
        
        fig.tight_layout()
        class_path = os.path.join(path, neu_class)
        os.makedirs(class_path, exist_ok=True)
        fig.savefig(os.path.join(class_path, 'degrees.png'), dpi=300, transparent=True)
        plt.close(fig)
        
        
def new_degree_distributions_classes_comparison_plot(selected_df, degree_key, weighted_degree_key, path=''):         
    selected_df = selected_df.loc[selected_df['class'] != 'unclassified']
    fig, axs = plt.subplots(1,2, figsize=(8, 4), sharex='col', sharey=True)
    my_pal = {"dVf": "#33ABA2", "hVf": "#F06233"}
    deg_ax = sns.histplot(data=selected_df, x=degree_key, hue='class', bins=50, stat='probability',
                 palette=my_pal, common_norm=False, legend=True, ax=axs[0])
    weigh_ax = sns.histplot(data=selected_df, x=weighted_degree_key, hue='class', bins=50, stat='probability', 
                 palette=my_pal, common_norm=False, legend=True, ax=axs[1])
    # axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Degree', fontsize=14)
    axs[0].set_ylabel('Probability', fontsize=14)
    axs[0].tick_params(axis='both', labelsize=12)
    # for legend text
    plt.setp(deg_ax.get_legend().get_texts(), fontsize='14')  
    deg_ax.legend(fontsize=14)

    axs[1].set_xlabel('Weight [pA]', fontsize=14)
    
    axs[1].set_ylabel('', fontsize=14)
    axs[1].tick_params(axis='both', labelsize=12)
    # axs[1].legend(loc='upper right')
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'new_hVf_vs_dVf_degree_distribution_'+degree_key+'.png'), dpi=300, transparent=True)
    plt.close(fig)
    
        
def degree_distributions_classes_comparison_plot(selected_df, degree_key, weighted_degree_key, path=''):         
    hVf_df = selected_df.loc[selected_df['class']=='hVf']
    dVf_df = selected_df.loc[selected_df['class']=='dVf']
    fig, axs = plt.subplots(1,2, figsize=(8, 4), sharex='col', sharey=False)
    # x = np.zeros((len()))
    axs[0].hist(dVf_df[degree_key], bins=50, density=True, label='dVf', color='#33ABA2', alpha=0.3, histtype='stepfilled') 
    axs[0].hist(hVf_df[degree_key], bins=50, density=True, label='hVf', color='#F06233', alpha=0.3, histtype='stepfilled') 
    axs[1].hist(dVf_df[weighted_degree_key], density=True, bins=50, label='dVf', color='#33ABA2', alpha=0.3, histtype='stepfilled') 
    axs[1].hist(hVf_df[weighted_degree_key], density=True, bins=50, label='hVf', color='#F06233', alpha=0.3, histtype='stepfilled') 
    # axs[0].set_xlim(0, self.max_degree)
    axs[0].legend(loc='upper right') 
    # axs[0].set_title(degree_key)
    # axs[1].set_title(weighted_degree_key)
    axs[0].set_ylabel('Probability density', fontsize=14)
    # axs[1].set_ylabel('# neurons', fontsize=14)
    axs[0].set_xlabel('Degree', fontsize=14)
    axs[1].set_xlabel('Weight [pA]', fontsize=14)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[1].legend(loc='upper right')
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'hVf_vs_dVf_degree_distribution_'+degree_key+'.png'), dpi=300, transparent=True)
    plt.close(fig)     
    
def classes_interconnection_matrix(selected_df, recurrent_network_pop, neurons_per_class, ax, feature='connection_probability'):
    neuron_classes = set(selected_df['class'])
    # isolate dvf and hVf neurons
    recurrent_network_pop = recurrent_network_pop.loc[(recurrent_network_pop['Target type']=='dVf')|
                                                    (recurrent_network_pop['Target type']=='hVf')]
    # Isolate the local recurrent network
    recurrent_network_pop = recurrent_network_pop.loc[(recurrent_network_pop['Source type']=='dVf')|
                                                    (recurrent_network_pop['Source type']=='hVf')]

    # print(recurrent_network_pop)
    if feature == 'connection_probability':
        label = 'Probability of connection (%)'
        connection_matrix = recurrent_network_pop.groupby(['Target type', 'Source type'])['Weight'].count().reset_index()
        for neu_class in neuron_classes:
            connection_matrix.loc[connection_matrix['Source type']==neu_class, 'Weight']/=neurons_per_class[neu_class]
        connection_matrix['Weight'] = connection_matrix['Weight']*100
    elif feature == 'in_degree':
        label = 'In degree'
        connection_matrix = recurrent_network_pop.groupby(['Target type', 'Source type'])['Weight'].count().reset_index()
    elif feature == 'weighted_in_degree':
        label = 'Weighted in degree (pA)'
        connection_matrix = recurrent_network_pop.groupby(['Target type', 'Source type'])['Weight'].sum().reset_index()
    elif feature == 'average_synaptic_weight':
        label = 'Average synaptic weight (pA)'
        connection_matrix = recurrent_network_pop.groupby(['Target type', 'Source type'])['Weight'].mean().reset_index()
    
    if feature != 'average_synaptic_weight':
        for neu_class in neuron_classes:    
            connection_matrix.loc[connection_matrix['Target type']==neu_class, 'Weight']/=neurons_per_class[neu_class]    

    connection_matrix = connection_matrix['Weight'].to_numpy().copy()
    # connection_matrix[1], connection_matrix[2] = connection_matrix[2], connection_matrix[1]
    # connection_matrix[6], connection_matrix[3] = connection_matrix[3], connection_matrix[6]
    # connection_matrix[8], connection_matrix[4] = connection_matrix[4], connection_matrix[8]
    # connection_matrix[7], connection_matrix[5] = connection_matrix[5], connection_matrix[7]
    
    # connection_matrix = connection_matrix.reshape((3,3))
    connection_matrix = connection_matrix.reshape((2,2))
    #m/=np.sum(m, axis=0)
    cm_df = pd.DataFrame(connection_matrix,
                         index = ['dVf', 'hVf'], 
                         columns = ['dVf', 'hVf'])

    # cm_df = pd.DataFrame(connection_matrix,
    #                      index = ['dVf', 'unc', 'hVf'], 
    #                      columns = ['dVf', 'unc', 'hVf'])

    sns.heatmap(cm_df, ax=ax, annot=True, fmt=".2f", yticklabels=True, cmap='binary', annot_kws={"fontsize":10},
                cbar_kws={'format': '%.2f', 'ticks': np.linspace(cm_df.values.min(), cm_df.values.max(), 5)})
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=10)
    ax.set_ylabel('Target neuron', fontweight='bold', fontsize=12)
    ax.set_xlabel('Source neuron', fontweight='bold', fontsize=12)
    ax.set_title(label, fontsize=12)
    plt.yticks(va="center")
    ax.tick_params(axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected   
                    labelsize=12)
   
def classes_connectivity_figure(recurrent_network_pop, weight=False, path=''):
    if weight:
        connection_matrix = recurrent_network_pop.groupby(['Target', 'Target type', 'Source type'])['Weight'].sum().reset_index()
        y_label = r'Total synaptic weight $[pA]$'
        fn = 'average_synaptic_weight_boxplot.png'
    else:
        connection_matrix = recurrent_network_pop.groupby(['Target', 'Target type', 'Source type'])['Weight'].count().reset_index()
        y_label = r'# of synapses'
        fn = 'number_synapses_boxplot.png'
        
    dvf_s = connection_matrix.loc[connection_matrix['Target type']=='dVf']
    hvf_s = connection_matrix.loc[connection_matrix['Target type']=='hVf']
    df1 = dvf_s[['Source type', 'Weight']].assign(Trial='dVf')
    df1.columns = ['Source type', 'Weight', 'Class']
    df2 = hvf_s[['Source type', 'Weight']].assign(Trial='hVf')
    df2.columns = ['Source type', 'Weight', 'Class']
    cdf = pd.concat([df1, df2])  
    true_order = ['i1Htr3a', 'dVf', 'hVf', 'unclassified', 
                      'i23Htr3a', 'i23Pvalb', 'i23Sst',
                      'e4', 'i4Htr3a', 'i4Pvalb', 'i4Sst', 
                      'e5', 'i5Htr3a', 'i5Pvalb', 'i5Sst',
                      'e6', 'i6Htr3a', 'i6Pvalb', 'i6Sst']
    source_types = [source for source in true_order if source in set(cdf['Source type'])]
            
    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    
    if weight:
        axes.set_yscale('symlog')

    # Define palette and boxplot configuration  
    my_pal = {"dVf": "#33ABA2", "hVf": "#F06233"}
    hue_plot_params = {
        'data': cdf,
        'x': 'Source type',
        'y': 'Weight',
        "order": source_types,
        "hue": "Class",
        "showfliers": False,
        "hue_order": ['dVf', 'hVf'],
        "palette": my_pal
    }
    
    sns.boxplot(ax=axes, **hue_plot_params)

    # Add Welch t-test annotations
    pairs = [((source_type, 'dVf'), (source_type, 'hVf')) for source_type in source_types]
    annotator = Annotator(axes, pairs, **hue_plot_params)
    annotator.configure(test='t-test_welch', loc='inside').apply_and_annotate()  # text_format is still simple

    axes.set_ylabel(y_label, fontsize=14)
    axes.xaxis.label.set_visible(False)
    axes.legend(loc='upper right', fontsize=14).set_visible(True)
    plt.setp(axes.get_xticklabels(), rotation=90)
    axes.tick_params(axis='x',          
                     which='both',     
                     labelsize=14) 
    axes.tick_params(axis='y',          
                     which='both',     
                     labelsize=12) 
    
    # If plotting synaptic weight, add a line at 0 pA and shades for excitatory and inhibitory regions
    if weight:
        x_min, x_max = axes.get_xlim()
        y_min, y_max = axes.get_ylim()
        axes.fill_between(x=[x_min, x_max], y1=0, y2=y_max, color='lightcoral', alpha=0.2, zorder=0)
        axes.fill_between(x=[x_min, x_max], y1=0, y2=y_min, color='lightblue', alpha=0.2, zorder=0)
        axes.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=0)

    plt.tight_layout()
    fig.savefig(os.path.join(path, fn), dpi=300, transparent=True)
    plt.close(fig)
    
    
######################### feature_selectivity_analysis.py #####################
    
def current_response_boxplots(input_current_response_df, classifications_df, path=''):
    # Isolate the perturbation responsive neurons
    pure_dVf_mask = np.all(classifications_df=='dVf', axis=1)
    pure_hVf_mask = np.all(classifications_df=='hVf', axis=1)
    perturbation_neurons_mask = np.logical_or(pure_dVf_mask, pure_hVf_mask)
    input_current_response_df = input_current_response_df.loc[perturbation_neurons_mask]
    # Plot the differential input current responses
    fig = plt.figure()
    ax = sns.swarmplot(data=input_current_response_df, size=1, palette="Set3", zorder=0)
    sns.boxplot(data=input_current_response_df, saturation=1, palette="Set3", orient='v', showfliers = False, ax=ax)
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .4))
    plt.ylabel('Current response [pA]', fontsize=12)
    plt.xlabel(u'Direction [\N{DEGREE SIGN}]', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'current_response_boxplots.png'), dpi=300, transparent=True)
    plt.close(fig) 

def differential_current_response_boxplots(input_current_response_df, classifications_df, path=''):
    # Isolate the perturbation responsive neurons
    pure_dVf_mask = np.all(classifications_df=='dVf', axis=1)
    pure_hVf_mask = np.all(classifications_df=='hVf', axis=1)
    perturbation_neurons_mask = np.logical_or(pure_dVf_mask, pure_hVf_mask)
    input_current_response_df = input_current_response_df.loc[perturbation_neurons_mask]
    # Take the absolute value of the deviation and normalize respect to the 0º direction
    input_current_response_df_norm = input_current_response_df.abs()
    input_current_response_df_norm = input_current_response_df_norm.sub(input_current_response_df_norm['0'], axis='rows')
    input_current_response_df_norm = input_current_response_df_norm.iloc[: , 1:]
    # Plot the differential input current responses
    fig = plt.figure()
    ax = sns.swarmplot(data=input_current_response_df_norm, size=1, palette="Set3", zorder=0)
    sns.boxplot(data=input_current_response_df_norm, saturation=1, palette="Set3", orient='v', showfliers = False, ax=ax)
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .4))
    plt.ylim(-20, 20)
    plt.ylabel('Differential current response [pA]', fontsize=12)
    plt.xlabel(u'Direction [\N{DEGREE SIGN}]', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'differential_current_response.png'), dpi=300, transparent=True)
    plt.close(fig) 

def mean_perturbation_responses_figure(currents_dict, stimuli_init_time=500, stimuli_end_time=1500, 
                                       path=''):
    # Plot the mean traces of dVf and hVf classes to different directions of the drifting gratings
    directions = np.array([0, 45, 90, 135])
    for neu_class in currents_dict.keys():
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        for idx, direction in enumerate(directions):
            currents_direction_mean = currents_dict[neu_class][str(direction)]['mean']
            currents_direction_sem = currents_dict[neu_class][str(direction)]['sem']
            
            reversed_direction = direction + 180
            currents_reversed_direction_mean = currents_dict[neu_class][str(reversed_direction)]['mean']
            currents_reversed_direction_sem = currents_dict[neu_class][str(reversed_direction)]['sem']
    
            simulation_lenght = len(currents_direction_mean)
            times = np.arange(0, simulation_lenght, 1)
            if idx==0:
                ax = axs[0, 0]
            elif idx==1:
                ax = axs[0, 1]
            elif idx==2:
                ax = axs[1, 0]
            elif idx==3:
                ax = axs[1, 1]
                
            ax.plot(times, currents_direction_mean, color='r', ms=1,
                    alpha=0.7, label=direction)
            ax.fill_between(times, 
                            currents_direction_mean + currents_direction_sem, 
                            currents_direction_mean - currents_direction_sem, 
                            alpha=0.3, color='r')
            ax.plot(times, currents_reversed_direction_mean, color='orange', ms=1,
                    alpha=0.7, label=reversed_direction)
            ax.fill_between(times, 
                            currents_reversed_direction_mean + currents_reversed_direction_sem, 
                            currents_reversed_direction_mean - currents_reversed_direction_sem, 
                            alpha=0.3, color='gray')
            ax.legend()
            ax.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5)
            ax.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5)
            for ax in axs.flat:
                ax.set(xlabel='Time [ms]', ylabel='Total input current [pA]')
                ax.label_outer()
              
        os.makedirs(path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(path, f'pure_{neu_class}_input_current_several_directions.png'), dpi=300, transparent=True)
        plt.close(fig)   
    
def mean_perturbation_responses_composite_figure(currents_dict, stimuli_init_time=500, 
                                                 stimuli_end_time=1500,path=''):
    # Plot the mean traces of dVf and hVf classes to different directions of the drifting gratings
    fig, axs = plt.subplots(2, 4, figsize=(12,5), sharex=True, sharey='row')
    directions = np.array([0, 45, 90, 135])
    for class_idx, neu_class in enumerate(currents_dict.keys()):
        if neu_class == 'hVf':
            class_color = '#F06233'
        else:
            class_color = '#33ABA2'
        for idx, direction in enumerate(directions):
            currents_direction_mean = currents_dict[neu_class][str(direction)]['mean']
            currents_direction_sem = currents_dict[neu_class][str(direction)]['sem']
            
            reversed_direction = direction + 180
            currents_reversed_direction_mean = currents_dict[neu_class][str(reversed_direction)]['mean']
            currents_reversed_direction_sem = currents_dict[neu_class][str(reversed_direction)]['sem']
    
            simulation_lenght = len(currents_direction_mean)
            times = np.arange(0, simulation_lenght, 1)
        
                            
            axs[class_idx, idx].plot(times, currents_direction_mean, color=class_color, ms=1,
                                     alpha=0.7, label=u'{direction}\N{DEGREE SIGN}'.format(direction=direction))
            axs[class_idx, idx].fill_between(times, 
                                            currents_direction_mean + currents_direction_sem, 
                                            currents_direction_mean - currents_direction_sem, 
                                            alpha=0.2, color=class_color)
            axs[class_idx, idx].plot(times, currents_reversed_direction_mean, color='k', ms=1,
                                     alpha=0.7, label=u'{direction}\N{DEGREE SIGN}'.format(direction=reversed_direction))
            axs[class_idx, idx].fill_between(times, 
                                            currents_reversed_direction_mean + currents_reversed_direction_sem, 
                                            currents_reversed_direction_mean - currents_reversed_direction_sem, 
                                            alpha=0.2, color='k')
            
            axs[class_idx, idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5)
            axs[class_idx, idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5)
            if class_idx == 0:
                axs[class_idx, idx].legend(loc='upper right')
            elif class_idx == 1:
                axs[class_idx, idx].legend(loc='lower right')
                axs[class_idx, idx].set_xlabel('Time [ms]')
            
    for ax in axs.flat:
        ax.set(xlabel='Time [ms]', ylabel='Total input current [pA]')
        ax.label_outer()
    axs[0, 0].set_ylabel('Total input current [pA]')
    axs[1, 0].set_ylabel('Total input current [pA]')
    plt.subplots_adjust(wspace=0.05, hspace=0.04)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'pure_class_neurons_input_current_response_several_directions.png'), dpi=300, transparent=True)
    plt.close(fig)   
    
    
def visual_preferred_direction(preferred_angles, path=''):    
    # Plot the preferred angle distribution
    fig = plt.figure()
    plt.hist(preferred_angles, bins=20) #, bins=np.arange(0, 360+45, 45))
    plt.ylabel('# neurons')
    plt.xlabel(u'Visual preferred angle [\N{DEGREE SIGN}]')
    # plt.grid(b=None)
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'preferred_angles_distribution.png'), dpi=300, transparent=True)
    plt.close(fig) 
    
    
def perturbation_preferred_direction(input_current_response_df, preferred_angles, path=''):
    ### Given the input_current response of the selected neurons for every direction,
    # determine the perturbation preferred direction
    input_current_response_df = input_current_response_df.abs()
    stimulus_directions = input_current_response_df.columns.astype(np.float32)
    den = (input_current_response_df*np.cos(stimulus_directions)).sum(axis=1)
    num = (input_current_response_df*np.sin(stimulus_directions)).sum(axis=1)
    # Calculate the preferred perturbation direction and restrict them to the [0, 360] span using mod operation
    preferred_pert_angles = np.degrees(np.arctan2(num, den))
    preferred_pert_angles = np.mod(preferred_pert_angles, 360)
    # Save a histogram of the preferred perturbation direction distribution
    fig = plt.figure()
    preferred_pert_angles.hist(bins=20)
    plt.ylabel('# neurons')
    plt.xlabel(u'Perturbation preferred direction [\N{DEGREE SIGN}]')
    plt.grid(b=None)
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'preferred_perturbation_direction_distribution.png'), dpi=300, transparent=True)
    plt.close(fig) 
    
    ### Find the distribution of the diferential between perturbation and preferred directions
    delta_dir = preferred_pert_angles.values - preferred_angles.values
    # Restrict the angles to the -180 to 180 range
    delta_dir = np.mod(delta_dir+180, 360)-180
    fig = plt.figure()
    plt.hist(delta_dir, bins=20)
    plt.ylabel('# neurons')
    plt.xlabel(u'$\Delta \theta$ [\N{DEGREE SIGN}]')
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'preferred_perturbation_direction_delta.png'), dpi=300, transparent=True)
    plt.close(fig)  
    

def preferred_temporal_frequency_histogram(preferred_temporal_frequency, classifications_df, 
                                           studied_temporal_frequencies, path=''):
    # Identify the preferred temporal frequencies for the dVf and hVf classes
    pure_dVf_mask = np.all(classifications_df=='dVf', axis=1).values
    pure_hVf_mask = np.all(classifications_df=='hVf', axis=1).values
    dVf_preferred_freq = preferred_temporal_frequency[pure_dVf_mask]
    hVf_preferred_freq = preferred_temporal_frequency[pure_hVf_mask]
    # Plot the distributions for both classes together
    fig = plt.figure()
    plt.hist([dVf_preferred_freq, hVf_preferred_freq], color=['#33ABA2', '#F06233'], 
             label = ['dVf', 'hVf'], alpha=0.5, density=True, 
             bins=studied_temporal_frequencies, align='mid')
    plt.ylabel('Density of neurons', fontsize=12)
    plt.xlabel('Preferred temporal frequency [Hz]', fontsize=12)
    plt.legend(loc='upper left')
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'preferred_temporal_frequencies.png'), dpi=300, transparent=True)
    plt.close(fig)

################ perturbation_responsive_neurons_analysis.py ##################
   
def plot_flash_responses_comparison(per_resp_dVf, dir_sel_dVf, per_resp_hVf, dir_sel_hVf, 
                                    stimuli_init_time=500, stimuli_end_time=1500,
                                    path=''):
    n_simulations, simulation_length, n_neurons = per_resp_dVf.shape
    fig = plt.figure(figsize=(6, 4))
    times = np.linspace(0, simulation_length, simulation_length)
    class_currents = [per_resp_dVf, dir_sel_dVf, per_resp_hVf, dir_sel_hVf]
    class_colors = ['#33ABA2', 'b', '#F06233', 'g']
    class_labels = ['Per. resp. dVf', 'Dir. sel. dVf', 'Per. resp. hVf', 'Dir. sel. hVf']
    
    for current, color, label in zip(class_currents, class_colors, class_labels):
        current = np.mean(current, axis=0)
        mean = np.mean(current, axis=1)
        sem = stats.sem(current, axis=1)
        plt.plot(times, mean, color=color,
                    label=label)
        plt.fill_between(times, mean + sem, mean - sem, alpha=0.3, color=color)
    
    plt.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    plt.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    plt.hlines(0, 0, simulation_length, color='k', linewidths=1,zorder=10)
    
    plt.ylabel('Input current response [pA]') 
    plt.xlabel('Time [ms]')
    plt.tick_params(axis='both', labelsize=10)
    leg = plt.legend()
    for text, color in zip(leg.get_texts(), class_colors):
        plt.setp(text, color=color)
    leg.set_zorder(102)
    
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'input_current_comparison_full_field_flash.png'), dpi=300, transparent=True)
    plt.close(fig)
    
    
################ classes_spatial_distribution_analysis.py ###################
    
def k_ripley_figure(pop_dict, radius, small_radius, k_random, k_random_small, 
                    k_random_sem, k_random_small_sem, path=''):
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    # Plot the k ripley functions for every class
    for idx, pop_name in enumerate(pop_dict.keys()):
        k = pop_dict[pop_name]['k_ripley']
        ax1.plot(radius, k, '.', markersize=3, color=pop_dict[pop_name]['color'], label=pop_name)
    # Plot the k ripley functions for the null distribution
    ax1.errorbar(radius, k_random, yerr=k_random_sem, fmt='k.', markersize=3, label='Null distribution')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_ylabel(r'$K(t)/A$')
    ax1.set_xlabel(r'$t ~~ [\mu m]$')
    # Make an inset plot
    left, bottom, width, height = [0.625,0.2,0.2,0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    # Include the k functions for the different classes
    for idx, pop_name in enumerate(pop_dict.keys()):
        k_small = pop_dict[pop_name]['k_ripley_small']
        ax2.plot(small_radius, k_small, '.', markersize=3, color=pop_dict[pop_name]['color'])
    # Include the k function for the null distribution
    ax2.errorbar(small_radius, k_random_small, yerr=k_random_small_sem, fmt='k.', markersize=3, label='Null distribution')
    ax2.set_ylabel(r'$K(t)/A$')
    ax2.set_xlabel(r'$t ~~ [\mu m]$')
    ax2.set_xticks([], minor=True)
    fig.savefig(os.path.join(path, 'k_ripley_fig.png'), transparent=True, dpi=300)
    plt.close(fig)

def radial_heigh_neurons_distribution(pop_dict, path=''):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    # Neuron classes
    for pop_name in pop_dict.keys():
        # Draw the density plot
        pop_color = pop_dict[pop_name]['color']
        sns.distplot(pop_dict[pop_name]['r'], hist = False, kde = True,
                     kde_kws = {'linewidth': 3},
                     label = pop_name,
                     ax=axes[0],
                     color = pop_color)
        sns.distplot(pop_dict[pop_name]['y'], hist = False, kde = True,
                     kde_kws = {'linewidth': 3},
                     label = pop_name,
                     ax=axes[1],
                    color = pop_color)
    axes[0].set_ylabel(r'Density of neurons')
    axes[0].set_xlabel(r'$Radius ~~ [\mu m]$')
    axes[0].legend()
    axes[1].set_ylabel(r'Density of neurons')
    axes[1].set_xlabel(r'$Height ~~ [\mu m]$')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'distances_distr.png'), transparent=True, dpi=300)
    plt.close(fig)

def neurons_column_visualization(pop_dict, core_column_radius, min_height, max_height, path=''):
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for pop_name in pop_dict.keys():
        pop_color = pop_dict[pop_name]['color']
        x = pop_dict[pop_name]['x']
        y = pop_dict[pop_name]['y']
        z = pop_dict[pop_name]['z']
        ax.scatter(x, z, y, alpha=0.5, marker='o', s=0.5, color=pop_color)
    # Include circules in the top and bottom of the cylinder
    x_circ = np.linspace(-core_column_radius, core_column_radius, 100)
    z_circ = np.sqrt(core_column_radius**2 - x_circ**2)
    R = np.linspace(0, core_column_radius, 100)
    u = np.linspace(0,  2*np.pi, 100)
    x = np.outer(R, np.cos(u))
    z = np.outer(R, np.sin(u))
    ax.plot(x_circ, z_circ, max_height, color='white', linewidth=1, alpha=0.5)
    ax.plot(x_circ, -z_circ, max_height, color='white', linewidth=1, alpha=0.5)
    ax.plot_surface(x, z, max_height*np.ones(x.shape), alpha=0.15, color='white', linewidth=0)    
    ax.plot(x_circ, z_circ, min_height, color='white', linewidth=1, alpha=0.5)
    ax.plot(x_circ, -z_circ, min_height, color='white', linewidth=1, alpha=0.5)
    ax.plot_surface(x, z, min_height*np.ones(x.shape), alpha=0.15, color='white', linewidth=0)    
    # Include a bar to illustrate the scale of the diagram
    z_scale = np.linspace(core_column_radius, 500, 100)
    x_scale = 450*np.ones(100)
    y_scale = -330*np.ones(100)
    ax.plot(x_scale, z_scale, y_scale, color='white', linewidth=2, alpha=1)    
    ax.text(670, 430, -330, r'100 $\mu m$', color='white', fontsize=8, horizontalalignment='center')
    plt.axis('off')
    plt.tight_layout()
    ax.view_init(elev=10, azim=0)
    plt.savefig(os.path.join(path, 'spatial_arrangement.png'), transparent=True, dpi=300)
    plt.close(fig)
    
    
    