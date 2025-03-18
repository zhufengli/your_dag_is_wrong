import os
import random
from datetime import datetime
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils import utils
from utils import data
from generation.generate_data import dataset_generator
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation

import logging
from torchviz import make_dot
import yaml
import pickle as pkl
import sys
from sklearn.utils import resample

sys.path.append(os.path.abspath('./src/generation'))
import generation.dag_generator
import generation.causal_mechanisms as causal_mechanisms

############### Model Definitions ################
class DAGNet(nn.Module):
    def __init__(self, num_vars, temp=1.0, temp_decay=0.99, sure_edges=None, forbidden_edges=None):
        """
        Initialize the DAGNet model.
        :param num_vars: Number of observed variables (d).
        :param temp: Initial temperature for Gumbel-Softmax.
        :param temp_decay: Decay rate for the temperature.
        :param sure_edges: List of tuples (i, j) indicating edges that must be present (1).
        :param forbidden_edges: List of tuples (i, j) indicating edges that must be absent (0).
        """
        super(DAGNet, self).__init__()
        self.num_vars = num_vars
        self.temp = temp
        self.temp_decay = temp_decay
        self.mask, self.fixed_values = self.create_mask_and_fixed_values(num_vars, sure_edges, forbidden_edges)
        self.params = nn.Parameter(torch.randn(num_vars, num_vars) * self.mask + self.fixed_values)
    #TODO Softmax(self.params) --> h()
    @staticmethod
    def create_mask_and_fixed_values(num_vars, sure_edges, forbidden_edges):
        """
        Create a mask and fixed values tensor based on the given constraints.
        :param num_vars: Number of observed variables (d).
        :param sure_edges: List of tuples (i, j) for surely existing edges.
        :param forbidden_edges: List of tuples (i, j) for surely non-existing edges.
        :return: Tuple of (mask, fixed_values).
        """
        mask = torch.ones(num_vars, num_vars)
        fixed_values = torch.zeros(num_vars, num_vars)

        if sure_edges:
            for i, j in sure_edges:
                mask[i, j] = 0
                fixed_values[i, j] = 1

        if forbidden_edges:
            for i, j in forbidden_edges:
                mask[i, j] = 0
                fixed_values[i, j] = 0

        return mask, fixed_values

    def sample_binary(self):
        """
        Forward pass to sample a binary adjacency matrix from the parameters.
        """
        A = self.sample_adjacency(self.params, self.temp)
        A = A * self.mask + self.fixed_values  # mask masks out both forbidden and sure edges and fixed values replace the remaining known values

        # TODO: probably we want to be able to turn on and off annealing and have more control over it
        self.temp *= self.temp_decay  # Anneal temperature
        return A
    
    def sample_soft(self):
        """
        Forward pass to sample a soft adjacency matrix from the parameters.
        """
        A = self.sample_adjacency(self.params, self.temp, hard=False)
        A = A * self.mask + self.fixed_values  # mask masks out both forbidden and sure edges and fixed values replace the remaining known values
 
        # TODO: probably we want to be able to turn on and off annealing and have more control over it
        self.temp *= self.temp_decay  # Anneal temperature
        return A

    @staticmethod
    def sample_adjacency(params, temp, hard=True):
        """
        Sample a binary adjacency matrix from the continuous parameters using Gumbel-Softmax.
        :param params: Continuous parameters representing the adjacency matrix.
        :param temp: Temperature parameter for Gumbel-Softmax.
        :return: Binary adjacency matrix.
        """
        # Create a tensor of zeros with the same shape as params to represent the fixed logits for class 0
        zeros = torch.zeros_like(params)

        # Stack the params and zeros along a new dimension to create logits
        # The resulting shape will be [num_vars, num_vars, 2], where the last dimension represents logits for two classes
        logits = torch.stack([zeros, params], dim=-1)

        # Apply Gumbel-Softmax to the logits
        gumbel_softmax_sample = torch.nn.functional.gumbel_softmax(logits, tau=temp, hard=hard)

        # Select the class 1 probabilities (or samples in the case of hard=True)
        binary_adjacency = gumbel_softmax_sample[..., 1]

        return binary_adjacency
 
def check_all_indices_one(matrix, indices):
    # Check if all specified elements are 1
    return all(matrix[i][j] == 1 for i, j in indices)

def check_all_indices_zero(matrix, indices):
    # Check if all specified elements are 1
    return all(matrix[i][j] == 0 for i, j in indices)

def train(model, data_loader, xy, maxmin, sure_edges, forbidden_edges, _config, out_dir=None, visualize=True):
    """
    Training loop for the DAGNet model.
    :param model: The DAGNet model.
    :param data_loader: DataLoader for the dataset.
    :param xy: The indices of the causal effect to estimate.
    :param maxmin: Whether to maximize or minimize the query value.
    :param sure_edges: List of sure edges.
    :param forbidden_edges: List of forbidden edges.
    :param _config: Configuration dictionary for the optimization.
    """
    tmp_results = [] #For global minimum checking

    if out_dir is not None:
        writer = SummaryWriter(log_dir=out_dir)
    else:
        writer = SummaryWriter()

    optimizer = optim.Adam(model.parameters(), _config['lr_lag'])

    tau = _config['tau_init']
    lam = _config['lam_init']
    eta = _config['eta_init']
    
    curr_min = np.inf
    A_opt = None
    # Always loads the full dataset, but in a different order
    for batch in data_loader:
        for rnd in range(_config['n_rounds_lag']):
        # Find approximate solution of subproblem at fixed lambda
            for opt_stp in range(_config['opt_steps_lag']):
                A_bin = model.sample_binary()
                #A_soft = model.sample_soft()
                iter_idx = _config['opt_steps_lag'] * rnd + opt_stp
                query_value, dag_loss, constr = \
                    utils.compute_lagrangian(A_bin, _config['constr_type'], \
                                             batch, xy, lam, tau, maxmin, optimal= _config['optimal'])
                if(nx.is_directed_acyclic_graph(nx.DiGraph(A_bin.detach().numpy())) \
                    and check_all_indices_one(A_bin, sure_edges) and check_all_indices_zero(A_bin, forbidden_edges)):
                #if utils.h(A_bin) == 0:
                    tmp_results.append(query_value.item())
                    if(query_value < curr_min):
                        curr_min = query_value
                        A_opt = A_bin.clone().detach()
                loss = query_value + dag_loss
                optimizer.zero_grad()
                loss.backward()
                
                #backprop
                optimizer.step()
                iter_idx += 1
                
                if _config['save_all']:
                    #plot lambda, tau, and more at each step
                    writer.add_scalar(f'lagrangian/{maxmin}/loss', loss, iter_idx)
                    if maxmin == 'max':
                        writer.add_scalar(f'lagrangian/{maxmin}/query_value', -query_value, iter_idx)
                    else:
                        writer.add_scalar(f'lagrangian/{maxmin}/query_value', query_value, iter_idx)
                    writer.add_scalar(f'lagrangian/{maxmin}/h(A_bin)', utils.h(A_bin), iter_idx)
                    writer.add_scalar(f'lagrangian/{maxmin}/aug_Lag_penalty', dag_loss, iter_idx)
                    writer.add_scalar(f'lagrangian/{maxmin}/lambda',lam, iter_idx)
                    writer.add_scalar(f'lagrangian/{maxmin}/tau', tau, iter_idx)
                    writer.add_scalar(f'lagrangian/{maxmin}/-lam * constr', -lam * constr , iter_idx) #For better tracking the penalization
                    writer.add_scalar(f'lagrangian/{maxmin}/0.5*tau*constr**2',0.5 * tau * constr**2 , iter_idx) #For better tracking the penalization

            detached_constr = constr.clone().detach()
            #Update the contraints at the end of each round
            if torch.abs(detached_constr) < _config['slack']:
                lam = lam - tau * detached_constr
                eta = torch.max(torch.tensor([eta / tau ** 0.5, _config['eta_min']]))
            else:
                tau = torch.min(torch.tensor([_config['gamma'] * tau, _config['tau_max']]))
                eta = torch.max(torch.tensor([1 / tau ** 0.1, _config['eta_min']])) 

            
    if tmp_results!=[]:
        logging.info(f"Valid graphs found ({maxmin}) {len(tmp_results)}")
        valid_results = np.array(tmp_results)
        if maxmin == 'max':
            logging.info(f"Lagrangian: {maxmin} query value is : {(-valid_results).max()}")
            opt_val = (-valid_results).max()
            valid_results = -valid_results
        else:
            logging.info(f"Lagrangian: {maxmin} query value is : {valid_results.min()}")
            opt_val = valid_results.min()
        return opt_val, A_opt, valid_results
    else: 
        logging.info(f"Lagrangian: {maxmin} query value is : {np.nan}")
        return np.nan, np.nan, []

def bootstrap_resample(dataset, size):
    indices = np.random.choice(len(dataset), size=size, replace=True)
    return Subset(dataset, indices)
    

if __name__ == "__main__":
    # Load yaml file as configuration
    with open('configs/lagrangian.yaml', 'r') as file:
        config = yaml.safe_load(file)

    torch.manual_seed(int(config['seed_data']))
#    random.seed(int(config['seed_data']))
    np.random.seed(int(config['seed_data']))


    # Number of bootstrap samples
    n_bootstraps = 500


    # Create output directory
    output_dir = config['output_dir']
    out_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(hash(tuple(config.items())))
    out_dir = os.path.join(output_dir, out_name)
    #out_dir = os.path.join(output_dir, 'lagrangian_new_test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.info('Getting the ground truth adjacency matrix...')
    generator = dataset_generator(config)
    if config['nb_dag'] > 1:
        logging.warning("Multiple DAGs are not supported. Only the first DAG will be used.")
    generator.i_dataset = 0
    generator.generator = None
    logging.info("Generating the observational data...")
    generator.generate()
    logging.info(f"Data saved in {generator.folder}")
    data_path = generator.folder

    A_npy = np.load(os.path.join(data_path, f'DAG{1}.npy'))
    A = torch.from_numpy(A_npy)
    dat = np.load(os.path.join(data_path, f'data{1}.npy'))
    dataset = data.ObservationalDataset(dat)
    with open(os.path.join(data_path, f'mechanism{1}.pkl'), 'rb') as f:
        mechanism = pkl.load(f)
    
    # Delete the saved data
    shutil.rmtree(data_path)

    xy = utils.random_select_one_index(A)
    x,y = xy
    gt_causal_effect = utils.compute_linear_total_causal_effect(A, x, y, mechanism)
    print(f"Ground truth causal effect is {gt_causal_effect}")
    if A[xy] == 1:
        parents = (A[:, y] > 0).nonzero(as_tuple=True)[0].tolist()
        mech_index = parents.index(x)
        gt_dir_causal_effect = mechanism[y].coefflist[mech_index]
        print(f"Ground truth direct causal effect is {gt_dir_causal_effect}")
    else:
        print(f"x is not the direct parent of y, skip its direct causal effect", flush=True)

    sure_edges, forbidden_edges = data.get_sure_and_forbidden(A, sure_prob=config['sure_prob'],\
                                                            forbidden_prob=config['forbid_prob'], xy=xy)

    max_model = DAGNet(config['nb_nodes'], temp=config['init_temp'], temp_decay=config['temp_decay'],\
                            sure_edges=sure_edges, forbidden_edges=forbidden_edges)
    min_model = DAGNet(config['nb_nodes'], temp=config['init_temp'], temp_decay=config['temp_decay'],\
                            sure_edges=sure_edges, forbidden_edges=forbidden_edges)

    dataset = data.ObservationalDataset(dat)
    data_loader = DataLoader(dataset, batch_size=config['nb_points'], shuffle=True)

    start = time.time()
    # Train the model
    ate_max, A_max, valid_results_max = train(max_model, data_loader, xy, maxmin='max', sure_edges=sure_edges,\
                                            forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
    end = time.time()

    print(f"Time taken for max: {end - start} seconds")
    print(f'Maximum causal effect is {ate_max}')

    ###################################################################
    ####### Bootstrapping for computing causal query estimation 
    ###################################################################
    max_query_list = []

    for i in range(n_bootstraps):  
        bootstrap_sample = bootstrap_resample(dataset, len(dataset))
        loader = DataLoader(bootstrap_sample, batch_size=len(dataset))
        for data_resampled in loader:
            query_value, _ = utils.query(data_resampled, A_max, xy, optimal=config['optimal'])
        max_query_list.append(query_value)

    max_var = torch.std(torch.tensor(max_query_list))
    print(f"Estimation standard deviation for max: {max_var}")

    start = time.time()
    ate_min, A_min, valid_results_min = train(min_model, data_loader, xy, maxmin='min', sure_edges=sure_edges,\
                                            forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
    end = time.time()
    print(f"Time taken for min: {end - start} seconds")
    print(f'Minimum causal effect is {ate_min}')

    ###################################################################
    ####### Bootstrapping for computing causal query estimation 
    ###################################################################
    min_query_list = []
    
    for i in range(n_bootstraps):  
        bootstrap_sample = bootstrap_resample(dataset, len(dataset))
        loader = DataLoader(bootstrap_sample, batch_size=len(dataset))
        for data_resampled in loader:
            query_value, _ = utils.query(data_resampled, A_min, xy, optimal=config['optimal'])
        min_query_list.append(query_value)

    min_var = torch.std(torch.tensor(min_query_list))
    print(f"Estimation standard deviation for min: {min_var}")

    
