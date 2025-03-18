import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import random
import yaml
from datetime import datetime
import time
import pandas as pd

import sys
import os
from utils import utils, data 
from non_linear import DAGNet, train

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation


def random_permutations(nvar, n_samples=3):
    selected_perms = []
    for _ in range(n_samples):
        perm = np.random.permutation(nvar)
        selected_perms.append(perm)
    return selected_perms

def inverse_permutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def apply_inverse_to_matrix(matrix, inverse_perm):
    return matrix[np.ix_(inverse_perm, inverse_perm)]

def check_specific_patterns(matrices):
    n = matrices[0].shape[0]  # Assuming all matrices are the same size
    consistent_pairs = []

    # Iterate over all pairs (i, j)
    for i in range(n):
        for j in range(n):
            if all(A[i, j] == 1 and A[j, i] == -1 for A in matrices):
                consistent_pairs.append((i, j, '1 and -1'))
            elif all(A[i, j] == -1 and A[j, i] == 1 for A in matrices):
                consistent_pairs.append((i, j, '-1 and 1'))
            elif all(A[i, j] == 0 and A[j, i] == 0 for A in matrices):
                consistent_pairs.append((i, j, '0 and 0'))
            elif all(A[i, j] == -1 and A[j, i] == -1 for A in matrices):
                consistent_pairs.append((i, j, '-1 and -1'))
            elif all(A[i, j] == 1 and A[j, i] == 1 for A in matrices):
                consistent_pairs.append((i, j, '1 and 1'))
    return consistent_pairs

def check_specific_patterns_2(matrices):
    n = matrices[0].shape[0]  # Assuming all matrices are the same size
    results = np.full((n, n), 5)  # Initialize with a default value that represents no pattern found
    
    # Pattern encoding
    pattern_to_number = {
        '1 and -1': 0,
        '-1 and 1': 1,
        '0 and 0': 2,
        '-1 and -1': 3,
        '1 and 1': 4
    }

    # Iterate over all pairs (i, j)
    for i in range(n):
        for j in range(n):
            for pattern, code in pattern_to_number.items():
                if all((A[i, j], A[j, i]) == tuple(map(int, pattern.split(' and '))) for A in matrices):
                    results[i, j] = code
                    break
    
    return results

def check_specific_patterns_reduced(matrices):
    n = matrices[0].shape[0]  # Assuming all matrices are the same size
    consistent_pairs = []

    forbidden_edges=[]
    sure_edges=[]
    
    # Iterate over all pairs (i, j)
    for i in range(n):
        for j in range(n):
            if all(A[i, j] == 1 and A[j, i] == -1 for A in matrices):
                consistent_pairs.append((i, j, '1 and -1'))
                sure_edges.append((i,j))
            elif all(A[i, j] == 0 and A[j, i] == 0 for A in matrices):
                consistent_pairs.append((i, j, '0 and 0'))
                forbidden_edges.append((i,j))
    return sure_edges, forbidden_edges

def visualize_patterns(pattern_matrix):
    # Define the color map
    cmap = plt.get_cmap('viridis', 6)  # 5 discrete colors
    fig, ax = plt.subplots(figsize=(8, 8))
    cbar = ax.imshow(pattern_matrix, cmap=cmap, aspect='equal')
    ax.grid(False)

    # Color bar settings
    cbar = fig.colorbar(cbar, ticks=np.arange(6), orientation='vertical')
    cbar.set_ticklabels(['1 and -1', '-1 and 1', '0 and 0', '-1 and -1', '1 and 1', 'None'])

    # Set axis labels
    ax.set_xlabel('Index j')
    ax.set_ylabel('Index i')
    ax.set_title('Heatmap of Matrix Patterns')

    # Axis tick settings
    ax.set_xticks(np.arange(len(pattern_matrix)))
    ax.set_yticks(np.arange(len(pattern_matrix)))
    ax.set_xticklabels(np.arange(len(pattern_matrix)))
    ax.set_yticklabels(np.arange(len(pattern_matrix)))

    plt.show()
    
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath('.')))  
    src_path = os.path.join(project_root, 'Graph-Misspecification/src')  

    print("Project root:", project_root)
    print("Source path:", src_path)

    sys.path.insert(0, src_path) 

    from utils import utils, data 

    with open('../configs/lagrangian_nl.yaml', 'r') as file:
        config = yaml.safe_load(file)

    torch.manual_seed(int(config['seed_data']))
    random.seed(int(config['seed_data']))
    np.random.seed(int(config['seed_data']))

    # Create output directory
    output_dir = config['output_dir']
    out_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(hash(tuple(config.items())))
    out_dir = os.path.join(output_dir, out_name)
    out_dir = os.path.join(output_dir, 'lagrangian_IHDP')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    dat= pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)
    dat_array = np.array(dat)

    t, y, y_cf = dat_array[:, 0], dat_array[:, 1][:, np.newaxis], dat_array[:, 2][:, np.newaxis]
    mu_0, mu_1, x = dat_array[:, 3][:, np.newaxis], dat_array[:, 4][:, np.newaxis], dat_array[:, 5:]
    index = [0, 1] + list(range(5, 30))
    data_cd = dat_array[:, index]
    
    nvars = data_cd.shape[1]
    n_samples = data_cd.shape[0]
    permutations_list = random_permutations(nvars, config['number_permutations'])

    results = []
    pattern_matrix_total = []
    for test in [config['test_PC']]:
        results = []
        for perm in permutations_list:
            print(f"permutation: {perm}, test: {test}")
            dat_array = np.array(data_cd)
            permuted_data = dat_array[:, perm]
            cg = pc(permuted_data, 0.05, test) 
            
            inverse_perm = inverse_permutation(perm)
            original_order_graph = apply_inverse_to_matrix(cg.G.graph, inverse_perm)
        
            results.append(original_order_graph)
            print(original_order_graph)

        pattern_matrix = check_specific_patterns_2(results)
        pattern_matrix_total.append(pattern_matrix)
        print("Patterns consistent across all matrices:", pattern_matrix)

    # Heatmap visualization
    #visualize_patterns(pattern_matrix)

    sure_edges, forbidden_edges = check_specific_patterns_reduced(results)
    xy = [0,1]
    
    torch.manual_seed(int(config['seed_lagrangian']))
    random.seed(int(config['seed_lagrangian']))
    np.random.seed(int(config['seed_lagrangian']))


    max_model = DAGNet(nvars, temp=config['init_temp'], temp_decay=config['temp_decay'],\
                                sure_edges=sure_edges, forbidden_edges=forbidden_edges)
    #min_model = copy.deepcopy(max_model)
    min_model = DAGNet(nvars, temp=config['init_temp'], temp_decay=config['temp_decay'],\
                                sure_edges=sure_edges, forbidden_edges=forbidden_edges)


    dataset = data.ObservationalDataset(dat_array)
    data_loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    start = time.time()
    # Train the model
    ate_max, A_max, valid_results_max = train(max_model, data_loader, xy, maxmin='max', sure_edges=sure_edges,\
                                                forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
    end = time.time()
    print(f"Time taken for max: {end - start} seconds")
    print(f'Maximum causal effect is {ate_max}')

    start = time.time()
    ate_min, A_min, valid_results_min = train(min_model, data_loader, xy, maxmin='min', sure_edges=sure_edges,\
                                                forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
    end = time.time()
    print(f"Time taken for max: {end - start} seconds")
    print(f'Maximum causal effect is {ate_min}')
    print(f"Time taken for min: {end - start} seconds")
    print(f"Total effect - ground truth: {np.mean(mu_0 - mu_1)}")


if __name__ == "__main__":
    main()