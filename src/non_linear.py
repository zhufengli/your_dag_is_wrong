import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import time
import logging
import shutil

import matplotlib.pyplot as plt
from generation.generate_data import dataset_generator

import networkx as nx
import numpy as np

from utils import utils
from utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import random

from torchviz import make_dot
import yaml
import pickle as pkl
import sys
import os

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
                x, y = xy[0], xy[1]
                if _config['optimal']: 
                    adjustmentset = (utils.get_optimal_adjustment(A_bin, xy) > 0).nonzero(as_tuple=True)[0].tolist()
                else:
                    adjustmentset = (utils.get_parent_adjustment(A_bin, xy) > 0).nonzero(as_tuple=True)[0].tolist()
                mlp_model = utils.MLP(len(adjustmentset) + 1) 
                inputs = torch.stack([batch[:, i] for i in adjustmentset + [x]], dim=1)
                target = batch[:, y]

                if (nx.is_directed_acyclic_graph(nx.DiGraph(A_bin.detach().numpy()))):
                    ##################################
                    #Adding this block to skip lagrangian optimization in case of failed MLP training
                    try:
                        utils.train_MLP(mlp_model, inputs, target, mlp_epoch=1000, patience=10, min_delta=0.1, validation = True)
                        query_value, dag_loss, constr = \
                            utils.compute_lagrangian(A_bin, _config['constr_type'], \
                                             batch, xy, lam, tau, maxmin, mlp_model=mlp_model, optimal= _config['optimal'])
                    except Exception:
                        loss = dag_loss
                        optimizer.zero_grad()
                        loss.backward()
                        #backprop
                        optimizer.step()
                        iter_idx += 1
                        continue
                    ##################################
                    iter_idx = _config['opt_steps_lag'] * rnd + opt_stp
                    if(nx.is_directed_acyclic_graph(nx.DiGraph(A_bin.detach().numpy())) \
                        and check_all_indices_one(A_bin, sure_edges) and check_all_indices_zero(A_bin, forbidden_edges)):
                    #if utils.h(A_bin) == 0:
                        tmp_results.append(query_value.item())
                        if(query_value.item() < curr_min):
                            curr_min = query_value.item()
                            A_opt = A_bin
                    loss = query_value + dag_loss
                    # Calculate gradients
                    optimizer.zero_grad()
                    loss.backward()

                    #backprop
                    optimizer.step()
                    iter_idx += 1

                    #plot lambda, tau, and more at each step
                    if _config['save_all']:
                        writer.add_scalar(f'{maxmin}/loss', loss, iter_idx)
                        writer.add_scalar(f'{maxmin}/query_value', query_value, iter_idx)
                        writer.add_scalar(f'{maxmin}/h(A)', constr, iter_idx)
                        writer.add_scalar(f'{maxmin}/Aug_Lag_penalty', dag_loss, iter_idx)
                        writer.add_scalar(f'{maxmin}/lambda',lam, iter_idx)
                        writer.add_scalar(f'{maxmin}/tau', tau, iter_idx)
                        writer.add_scalar(f'{maxmin}/-lam * constr', -lam * constr , iter_idx) #For better tracking the penalization
                        writer.add_scalar(f'{maxmin}/0.5*tau*constr**2',0.5 * tau * constr**2 , iter_idx) #For better tracking the penalization

                else:
                    query_value, dag_loss, constr = \
                            utils.compute_lagrangian(A_bin, _config['constr_type'], \
                                             batch, xy, lam, tau, maxmin, mlp_model = mlp_model, optimal= _config['optimal'])
                    iter_idx = _config['opt_steps_lag'] * rnd + opt_stp
                    loss = dag_loss
                    optimizer.zero_grad()
                    loss.backward()
                    #backprop
                    optimizer.step()
                    iter_idx += 1

                    #plot lambda, tau, and more at each step
                    writer.add_scalar(f'{maxmin}/loss', loss, iter_idx)
                    writer.add_scalar(f'{maxmin}/query_value', query_value, iter_idx)
                    writer.add_scalar(f'{maxmin}/h(A)', constr, iter_idx)
                    writer.add_scalar(f'{maxmin}/Aug_Lag_penalty', dag_loss, iter_idx)
                    writer.add_scalar(f'{maxmin}/lambda',lam, iter_idx)
                    writer.add_scalar(f'{maxmin}/tau', tau, iter_idx)
                    writer.add_scalar(f'{maxmin}/-lam * constr', -lam * constr , iter_idx) #For better tracking the penalization
                    writer.add_scalar(f'{maxmin}/0.5*tau*constr**2',0.5 * tau * constr**2 , iter_idx) #For better tracking the penalization
                    


            detached_constr = constr.clone().detach()
            #Update the contraints at the end of each round
            if detached_constr < _config['slack']:
                lam = lam - tau * detached_constr
                eta = torch.max(torch.tensor([eta / tau ** 0.5, _config['eta_min']]))
            else:
                tau = torch.min(torch.tensor([_config['gamma'] * tau, _config['tau_max']]))
                eta = torch.max(torch.tensor([1 / tau ** 0.1, _config['eta_min']])) 
            
            
    if tmp_results!=[]:
        valid_results = np.array(tmp_results)
        if maxmin == 'max':
            opt_val = (-valid_results).max()
            valid_results = -valid_results
        else:
            opt_val = valid_results.min()
    return opt_val, A_opt, valid_results


if __name__ == "__main__":
    # Load yaml file as configuration
    with open('./configs/lagrangian_nl.yaml', 'r') as file:
        config = yaml.safe_load(file)

    output_dir = os.path.abspath(config['output_dir'])
    # Create output directory
    out_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(hash(tuple(config.items())))
    out_dir = os.path.join(output_dir, out_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Set up logging COMMENT OUT LOCALLY WHEN RUNNING MULTIPLE RUNS

    # Saving the yaml files
    config['n_var'] = config['nb_nodes']
    config['n_sample'] = config['nb_points']
    logging.info('Saving config file to {out_dir}...')
    with open(os.path.join(out_dir, 'config.yaml'), 'w+') as ff:
        yaml.dump(config, ff)
    
    logging.info(f"Save all output to {out_dir}...")

    logging.info('Setting the seed for the data...')
    torch.manual_seed(int(config['seed_data']))
    #random.seed(int(config['seed_data']))
    np.random.seed(int(config['seed_data']))


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
    cause_values = [0, 1]
    utils.plot_graph(A, xy)
    
    expected_y, total_causal_effect = utils.compute_total_causal_effect(
    A_npy, dat.T, mechanism, x, y, cause_values, dat.shape[0]
    )
    
    print(f'Ground truth ATE is {total_causal_effect}')

    sure_edges, forbidden_edges = data.get_sure_and_forbidden(A, sure_prob=config['sure_prob'],\
                                                            forbidden_prob=config['forbid_prob'], xy=xy)

    max_model = DAGNet(config['n_var'], temp=config['init_temp'], temp_decay=config['temp_decay'],\
                            sure_edges=sure_edges, forbidden_edges=forbidden_edges)
    min_model = DAGNet(config['n_var'], temp=config['init_temp'], temp_decay=config['temp_decay'],\
                            sure_edges=sure_edges, forbidden_edges=forbidden_edges)
    
    data_loader = DataLoader(dataset, batch_size=config['n_sample'], shuffle=True)

    # Train the model
    start = time.time()
    ate_max, A_max, valid_results_max = train(max_model, data_loader, xy, maxmin='max',\
                                               sure_edges=sure_edges, forbidden_edges=forbidden_edges, _config=config)
    end = time.time()
    print(f"Time taken for max: {end - start} seconds")
    print(f'Maximum causal effect is {ate_max}')

    start = time.time()
    ate_min, A_min, valid_results_min = train(min_model, data_loader, xy, maxmin='min',\
                                               sure_edges=sure_edges, forbidden_edges=forbidden_edges, _config=config)
    end = time.time()
    print(f"Time taken for min: {end - start} seconds")
    print(f'Minimum causal effect is {ate_min}')
        
    



