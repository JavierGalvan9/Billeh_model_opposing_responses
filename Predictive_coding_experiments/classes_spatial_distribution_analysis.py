# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:22:13 2023

@author: UX325
"""

import os
import sys
import h5py
import argparse
import numpy as np
import plotting_figures as myplots
from numba import njit
from scipy import spatial, stats
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management
sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
import load_sparse


def make_tree(xs=None, ys=None, zs=None):
    # Create a spatial tree given the points of the neurons
    active_dimensions = [dimension for dimension in [xs,ys,zs] if dimension is not None]
    assert len(active_dimensions) > 0, "Must have at least 1-dimension to make tree"
    if len(active_dimensions)==1:
        points = np.c_[active_dimensions[0].ravel()]
    elif len(active_dimensions)==2:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel()]
    else:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel(), active_dimensions[2].ravel()]
    return spatial.cKDTree(points), len(active_dimensions)

@njit
def origin_distance(x=0, y=0, z=0):
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_overlap(points, bounding_size, score_radius):
    """
    We use this function to calculate the overlap area between the search circle
    in the cylinder and the cylinder.
    """
    d = origin_distance(x=points[:,0], z=points[:,1])
    vol = np.zeros(len(d))
    # Inner region
    inner_mask = ( d <= abs(score_radius-bounding_size) )
    vol = np.where(inner_mask, np.pi * score_radius**2, vol)
    # Outer region
    outer_mask = ( d >= score_radius+bounding_size )
    vol = np.where(outer_mask, 0, vol)
    # Boundary region
    intermediate_mask = np.logical_not(np.logical_or(inner_mask, outer_mask))
    r2, R2, d2 = score_radius**2, bounding_size**2, d**2
    r, R = score_radius, bounding_size
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    intermediate_vol = ( r2 * alpha + R2 * beta -
            0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))) 
    vol = np.where(intermediate_mask, intermediate_vol, vol)
                                                      
    assert np.all(vol >= 0), "Attempted to boundary correct a point not within the sample. Check sample_size and points."
    return vol

def calculate_ripley(radii, sample_size=400, xs=None, zs=None, boundary_correct=False, CSR_Normalise=False):
    results = []
    tree, dimensions = make_tree(xs=xs, zs=zs)
    if type(radii) is not list:
        radii = [radii]
    for radius in radii:
        score_vol = np.pi * radius**2
        bound_size = np.pi * sample_size**2
        points = np.zeros((len(xs), dimensions))
        for idx, coordinate in enumerate([xs, zs]):
            points[:, idx] = coordinate
        # Make boundary corrections
        if boundary_correct:
            vol = calculate_overlap(points, sample_size, radius)
            boundary_correction = vol/score_vol
            counts = tree.query_ball_point(points, radius, return_length=True).astype(np.float32)
            counts /= boundary_correction
            counts = np.sum(counts)
        else:
            counts = np.sum(tree.query_ball_point(points, radius, return_length=True))
        # Normalize according to the search radius        
        if CSR_Normalise:
            results.append((bound_size*counts/len(xs)**2) - score_vol)
        else:
            results.append(counts/len(xs)**2)
    if len(results)==1:
        return results[0]
    else:
        return results
    

class SpatialAnalysis:    
    def __init__(self, flags):
        self.orientation = flags.gratings_orientation
        self.frequency = flags.gratings_frequency
        self.reverse = flags.reverse
        self.n_neurons = flags.neurons
        self.neuron_population = flags.neuron_population
        self.core_column_radius = 400
        self.n_random_simulations = flags.n_random_simulation
        self.simulation_results = 'Simulation_results'
        self.directory = f'orien_{self.orientation}_freq_{self.frequency}_reverse_{self.reverse}_rec_{self.n_neurons}'
        self.full_path = os.path.join(self.simulation_results, self.directory)
        self.spatial_analysis_path = os.path.join(self.simulation_results, 'Spatial analysis', self.neuron_population)
        os.makedirs(self.spatial_analysis_path, exist_ok=True)
        # Load the simulation configuration attributes
        self.full_data_path = os.path.join(self.full_path, 'Data', 'simulation_data.hdf5')
        self.sim_metadata = {}
        with h5py.File(self.full_data_path, 'r') as f:
            dataset = f['Data']
            self.sim_metadata.update(dataset.attrs)
        self.data_dir = self.sim_metadata['data_dir'] 
        self.rd = np.random.RandomState(seed=self.sim_metadata['seed'])
    
    def __call__(self):
        # Load the network of the model
        load_fn = load_sparse.cached_load_billeh
        _, self.network, _, _ = load_fn(self.sim_metadata['n_input'], self.n_neurons, self.sim_metadata['core_only'], 
                                        self.data_dir, seed=self.sim_metadata['seed'], connected_selection=self.sim_metadata['connected_selection'], 
                                        n_output=self.sim_metadata['n_output'], neurons_per_output=self.sim_metadata['neurons_per_output'])
        # Load the population stats dataframe
        classification_path = os.path.join(self.full_path, self.neuron_population, 'classification_results')
        self.selected_df = file_management.load_lzma(os.path.join(classification_path, f'{self.neuron_population}_selected_df.lzma'))
        self.n_selected_neurons = len(self.selected_df)
        ### Calculate the K ripley value for the different classes
        pop_dict = {}
        population_names = ['dVf', 'unclassified', 'hVf']
        colors = ['#33ABA2', '#9CB0AE', '#F06233']
        min_height = []
        max_height = []
        radii = list(np.logspace(1, np.log10(self.core_column_radius), 100))
        radii_small = list(np.logspace(1, np.log10(25), 10))
        for idx, pop_name in enumerate(population_names):
            tf_indices = self.selected_df.loc[self.selected_df['class']==pop_name, 'Tf index']
            x = self.network['x'][tf_indices].astype(np.float32)
            y = self.network['y'][tf_indices].astype(np.float32)
            z = self.network['z'][tf_indices].astype(np.float32)
            r = np.sqrt(x**2 + z**2)
            min_height.append(np.min(y))
            max_height.append(np.max(y))
            # Calculate the K ripley function for the class
            k = calculate_ripley(radii, sample_size=self.core_column_radius, xs=x, zs=z, boundary_correct=True)
            k_small = calculate_ripley(radii_small, sample_size=self.core_column_radius, xs=x, zs=z, boundary_correct=True)
            pop_dict[pop_name] = {'indices': tf_indices,
                                  'x': x,
                                  'y': y,
                                  'z': z,
                                  'r': r,
                                  'k_ripley': k,
                                  'k_ripley_small': k_small,
                                  'color':colors[idx]}
        # Determine the height of the column
        self.min_height = np.min(min_height)
        self.max_height = np.max(max_height)
        
        ### Determine the K ripley function for a random sample of neurons     
        k_random_list = []
        k_random_small_list = []
        for realization in range(self.n_random_simulations):
            # Create a distribution of random neurons within the cylinder
            # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
            rs = self.core_column_radius * np.sqrt(self.rd.uniform(low=0, high=1, size=self.n_selected_neurons))
            alphas = 2 * np.pi * self.rd.uniform(low=0, high=1, size=self.n_selected_neurons)
            # Create cartesian coordinates
            xs = (rs * np.cos(alphas)).astype(np.float32)
            zs = (rs * np.sin(alphas)).astype(np.float32)
            # ys = np.random.uniform(self.min_height, self.max_height, size=self.n_selected_neurons)
            # Calculate the k ripley function for the random sample of neurons
            k_random = calculate_ripley(radii, sample_size=self.core_column_radius, xs=xs, zs=zs, boundary_correct=True)
            k_random_small = calculate_ripley(radii_small, sample_size=self.core_column_radius, xs=xs, zs=zs, boundary_correct=True)
            k_random_list.append(k_random)
            k_random_small_list.append(k_random_small)
        # Average over realizations and calculate the SEM   
        k_random = np.mean(k_random_list, axis=0)
        k_random_small = np.mean(k_random_small_list, axis=0)  
        k_random_sem = stats.sem(k_random_list, axis=0)
        k_random_small_sem = stats.sem(k_random_small_list, axis=0)
        
        ### K ripley figure
        myplots.k_ripley_figure(pop_dict, radii, radii_small, k_random, k_random_small, 
                                k_random_sem, k_random_small_sem, path=self.spatial_analysis_path)
        ### Height and radius distributions
        myplots.radial_heigh_neurons_distribution(pop_dict, path=self.spatial_analysis_path)
        ### Spatial arrangement within the column
        myplots.neurons_column_visualization(pop_dict, self.core_column_radius, 
                                             self.min_height, self.max_height, 
                                             path=self.spatial_analysis_path)
                
def main(flags):
     
    spatial_analysis = SpatialAnalysis(flags)
    spatial_analysis()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Define key flags')
    parser.add_argument('--gratings_orientation', type=int, choices=range(0, 360, 45), default=0)
    parser.add_argument('--gratings_frequency', type=int, default=2)
    parser.add_argument('--neurons', type=int, default=230924)
    parser.add_argument('--n_random_simulation', type=int, default=100)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false')
    parser.set_defaults(reverse=False)
    parser.add_argument('--neuron_population', type=str, default='e23')
    
    flags = parser.parse_args()
    main(flags)
    
    