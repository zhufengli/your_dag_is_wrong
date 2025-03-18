import time
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset
import networkx as nx

import pickle as pkl
from generation.generate_data import dataset_generator

from datetime import datetime

def adj_mat_generator(_config, xy=None, prob_mat=None): 
    """
    Generate adjacency matrix, where by default targeted cause variable is the second to last position 
    and effect variable is the last variable (Inefficient algo). Can optionally pass x and y to it to 
    define the cause and effect variable positions, and whether they are connected or not.
    Args:
        _config['num_vars']: number of variables/nodes
        _config['xy_connected']: whether x and y are connected
        _config['xy_position']: where x and y are located (last, random, chosen)
        _config['graph_type']: default or erdos-renyi ('default', 'gnp')
        xy: indices of treatment x and outcome y. Default to last two nodes
        prob_mat: probability matrix (num_vars * num_vars) of how likely there will be an edges between two nodes
        providing a prob_mat overrides other parameters
    """
    num_vars = _config['n_var']
    max_time_seconds = 3
    if prob_mat is not None:
        assert prob_mat.size()[0] == prob_mat.size()[1], "provide a square probability matrix"
        assert prob_mat.size()[0] == num_vars, "number of variable does not match probability matrix"
        A = torch.bernoulli(prob_mat)
        while torch.trace(torch.exp(A) - torch.eye(num_vars)) != 0:
            A = torch.bernoulli(prob_mat)
        return A
    elif _config['graph_type'] == 'default':
        while True: 
            prob_mat = torch.triu(torch.rand((num_vars,num_vars)), diagonal=1)
            if xy is not None:
                assert len(xy) == 2, "Provide only two nodes for xy"
                assert xy[0] < num_vars and xy[1] < num_vars, "xy nodes are out of range"
                assert xy[0] < xy[1], "x should be less than y"
                if _config['xy_connected']: prob_mat[xy[0], xy[1]] = 1  
            elif _config['xy_position'] == 'last':
                if _config['xy_connected']: prob_mat[num_vars-2, num_vars-1] = 1
                          
            A = torch.bernoulli(prob_mat)
            start_time = time.time()
            while torch.trace(torch.exp(A) - torch.eye(num_vars)) != 0:
                A = torch.bernoulli(prob_mat)
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > max_time_seconds:
                    break
            if torch.trace(torch.exp(A) - torch.eye(num_vars)) == 0:
                break
        return A
    elif _config['graph_type'] == 'gnp':
        p = _config['edge_prob_gnp']
        G = nx.DiGraph()
        G.add_nodes_from(range(num_vars))

        # Add edges with probability p, ensuring acyclicity by only allowing edges from i to j where i < j
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                if np.random.rand() < p:
                    G.add_edge(i, j)
                
        if _config['xy_position'] == 'last':
            if _config['xy_connected']: 
                if not G.has_edge(num_vars-2, num_vars-1): G.add_edge(num_vars-2, num_vars-1)
            return torch.tensor(nx.to_numpy_array(G)), [num_vars-2, num_vars-1]
        elif _config['xy_position'] == 'random':
            xy = select_random_nodes(_config['n_var'])
            if _config['xy_connected']:
                if not G.has_edge(xy[0], xy[1]):
                    G.add_edge(xy[0], xy[1])
            return torch.tensor(nx.to_numpy_array(G)), [xy[0], xy[1]]
        elif _config['xy_position'] == 'chosen':
            raise ValueError("Too complicated, just choose 'last' or 'random' for erdos-renyi")
    

class ObservationalDataset(Dataset):
    def __init__(self, data):
        """
        Dataset to handle observational data.
        :param data: Observational data as a NumPy or torch array.
        """
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            self.data = data.float()
        else:
            raise TypeError("Data must be a NumPy or torch array.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def select_random_nodes(d):
    # Select x and y nodes such that x < y and they are in range (d/2,d)
    start = int(d / 2) - 1
    end = d  

    if end > start + 1:  
        x = random.randint(start, end - 2) 
        y = random.randint(x + 1, end - 1)  
        return [x, y]
    else:
        print("Not enough nodes to satisfy x < y, returning last two")
        return [d-2, d-1]  

def select_connected_nodes(A, d):
    start = d // 2 + 1
    candidates = [(i, j) for i in range(start, d) for j in range(i + 1, d) if A[i, j] == 1]

    if not candidates:
        print("No valid (x, y) pairs found. Selecetd last two.")
        xy = (d-1, d-2)
    else:
        xy = random.choice(candidates)
        print(f"Selected nodes (x, y): {xy}")        
    return xy


def get_sure_and_forbidden(A, sure_prob, forbidden_prob, xy):
    """
    Randomly sample sure edges and forbidden edges to avoid manually typing in all edges

    Args:
        A: the adjacency matrix
        sure_prob: the probability of an edge is sure edge
        forbidden_prob: the probability of an edge is forbidden edge
        xy: treatment x and outcome y
    """
    n = A.size()[0]
    sure_edges = set()
    forbidden_edges = set()
    x,y = xy

    for i in range(n):
        for j in range(n):
            if i==x and j==y:
                sure_edges.add((i, j))
                continue
            if A[i][j] == 1:
                if random.random() < sure_prob:
                    sure_edges.add((i, j))
            elif A[i][j] == 0:
                if random.random() < forbidden_prob:
                    forbidden_edges.add((i, j))

    return list(sure_edges), list(forbidden_edges)



def data_generator_linear(A, _config):
    """
    Generate data linearly with Gaussian noise from adjacency matrix
    
    Args:
        A: Odered adjacency matrix (from low to high)
        n_sample: number of sample to generate
        n_var: number of variable
        b: range of parameter beta [-b,b]
        r: range of root nodes [-r,r]
        n: range of gaussian noise [-n,n]
    """
    data = torch.empty(_config['n_var'], _config['n_sample'])
    param = torch.zeros(_config['n_var'], _config['n_var'])

    #Get the parent set
    parent_set = (A == 1).nonzero(as_tuple=True)

    for i in range(_config['n_var']):
        #generate root node
        if len((parent_set[1] == i).nonzero(as_tuple=False)) == 0:
            data[i] = (torch.rand(_config['n_sample'])- 0.5) * 2 * _config['r'] #To be decided
        #generate non root node
        else:
            parent = (parent_set[1] == i).nonzero(as_tuple=False).squeeze_(dim=1)
            parent_index = parent_set[0].gather(dim=0, index=parent) 
            beta = (torch.rand(len(parent_index))- 0.5) * 2 * _config['b']
            param[parent_index,i] = beta
            data[i] = torch.matmul(beta, data.index_select(dim = 0, index = parent_index)) \
                + (torch.rand(_config['n_sample'])- 0.5) * 2 * _config['noise'] 
    return torch.t(data), param

def check_sure_and_forbidden(A, edges, sure = True):
    """
    Check whether the predicted adjacency matrix contains sure edges and forbidden edges 
    """
    if sure:
        for i, j in edges:
            if A[i,j]!= 1:
                print(f'Adjacency matrix does not contain sure edges')
            break
    else:
        for i, j in edges:
            if A[i,j]!= 0:
                print(f'Adjacency matrix contains forbidden edges')
            break

def data_generator_non_linear(A, _config):
    """
    Generate data linearly with Gaussian noise from adjacency matrix
    
    Args:
        A: Odered adjacency matrix (from low to high)
        n_sample: number of sample to generate
        n_var: number of variable
        b: range of parameter beta [-b,b]
        r: range of root nodes [-r,r]
        n: range of gaussian noise [-n,n]
    """
    data = torch.empty(_config['n_var'], _config['n_sample'])
    param = torch.zeros(_config['n_var']*2, _config['n_var'])

    #Get the parent set
    parent_set = (A == 1).nonzero(as_tuple=True)

    for i in range(_config['n_var']):
        #generate root node
        if len((parent_set[1] == i).nonzero(as_tuple=False)) == 0:
            data[i] = (torch.rand(_config['n_sample'])- 0.5) * 2 * _config['r'] #To be decided
        #generate non root node
        else:
            #simple implementation, need to update
            #y = beta1*x + beta2*x^2 + beta3*xˆ3 + noise
            parent = (parent_set[1] == i).nonzero(as_tuple=False).squeeze_(dim=1)
            parent_index = parent_set[0].gather(dim=0, index=parent) 
            beta1 = (torch.rand(len(parent_index))- 0.5) * 2 * _config['b']
            beta2 = (torch.rand(len(parent_index))- 0.5) * 2 * _config['b']
            param[parent_index,i] = beta1
            param[parent_index + _config['n_var'],i] = beta2
            sliced_data = data.index_select(dim = 0, index = parent_index)
            data[i] = torch.matmul(beta1, data.index_select(dim = 0, index = parent_index)) \
                + torch.matmul(beta2, sliced_data*sliced_data) \
                + (torch.rand(_config['n_sample'])- 0.5) * 2 * _config['noise'] 
    return torch.t(data), param


def get_data_from_config(config):
    # Load the configuration from the specified file

    config['data_folder'] = os.path.abspath('../logs')
    config['output_dir'] = os.path.abspath('../logs')
    output_dir = os.path.abspath(config['output_dir'])

    out_name = 'temp'
    out_dir = os.path.join(output_dir, out_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Saving the yaml files
    config['n_var'] = config['nb_nodes']
    config['n_sample'] = config['nb_points']

    torch.manual_seed(int(config['seed_data']))
    random.seed(int(config['seed_data']))
    np.random.seed(int(config['seed_data']))

    generator = dataset_generator(config, skip_hash=True)
    generator.i_dataset = 0
    generator.generator = None

    generator.generate()

    data_path = generator.folder

    # A = torch.from_numpy(np.load(os.path.join(data_path, f'DAG{1}.npy')))
    dat = np.load(os.path.join(data_path, f'data{1}.npy'))
    dat = torch.from_numpy(dat.astype('float32'))
    # dataset = ObservationalDataset(dat)

    # with open(os.path.join(data_path, f'mechanism{1}.pkl'), 'rb') as f:
    #         mechanism = pkl.load(f)
        
    # Delete the saved data
    shutil.rmtree(data_path)
    return dat
