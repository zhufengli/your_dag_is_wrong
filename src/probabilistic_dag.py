import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import gumbel_softmax
from utils.soft_sort import SoftSort_p1, gumbel_sinkhorn
from torch.autograd import Variable

import networkx as nx

from utils import utils
from utils import data
from torch.utils.tensorboard import SummaryWriter

from torchviz import make_dot
import logging
import time
import yaml


###Only for loading the dataset
import sys
import pickle as pkl
sys.path.append(os.path.abspath('./src/generation'))
import generation.dag_generator
import generation.causal_mechanisms as causal_mechanisms



# ------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ProbabilisticDAG(nn.Module):

    def __init__(self, n_nodes, temperature=1, temp_decay=0.9995, hard=True, order_type='sinkhorn', noise_factor=1.0, lr=1e-3, initial_adj=None, \
                 seed=0, sure_edges=None, forbidden_edges=None, ignore_sure=True):
        """Base Class for Probabilistic DAG Generator based on topological order sampling

        Args:
            n_nodes (int): Number of nodes
            temperature (float, optional): Temperature parameter for order sampling. Defaults to 0.5.
            hard (bool, optional): If True output hard DAG. Defaults to True.
            order_type (string, optional): Type of differentiable sorting. Defaults to 'sinkhorn'.
            noise_factor (float, optional): Noise factor for Sinkhorn sorting. Defaults to 1.0.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            seed (int, optional): Random seed. Defaults to 0.
            sure_edges (list, optional): List of sure edges. Defaults to None.
            forbidden_edges (list, optional): List of forbidden edges. Defaults to None.
            ignore_sure (bool, optional): ignore the sure_edges when sample an adjacency matrix
        """
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        
        self.n_nodes = n_nodes
        self.temperature = temperature
        self.hard = hard
        self.order_type = order_type
        self.sure_edges = sure_edges
        self.forbidden_edges = forbidden_edges
        self.temp_decay = temp_decay
        self.sure_mask, self.forbidden_mask = self.create_mask_and_fixed_values(n_nodes, sure_edges, forbidden_edges)
        self.ignore_sure = ignore_sure

        # Mask for ordering
        self.mask = torch.triu(torch.ones(self.n_nodes, self.n_nodes, device=device), 1)

        # define initial parameters
        if self.order_type == 'sinkhorn':
            self.noise_factor = noise_factor
            p = torch.zeros(n_nodes, n_nodes, requires_grad=True, device=device)
            self.perm_weights = torch.nn.Parameter(p)
        elif self.order_type == 'topk':
            p = torch.zeros(n_nodes, requires_grad=True, device=device)
            self.perm_weights = torch.nn.Parameter(p)
            self.sort = SoftSort_p1(hard=self.hard, tau=self.temperature)
        else:
            raise NotImplementedError
        e = torch.zeros(n_nodes, n_nodes, requires_grad=False, device=device)
        torch.nn.init.uniform_(e)

        if initial_adj is not None:
            initial_adj = initial_adj.to(device)
            zero_indices = (1 - initial_adj).bool()
            # set masked edges to zero probability
            e[zero_indices] = -300
            e.requires_grad = True
        torch.diagonal(e).fill_(-300)
        e.requires_grad = True
        self.edge_log_params = torch.nn.Parameter(e)
        if initial_adj is not None:
            self.edge_log_params.register_hook(lambda grad: grad * initial_adj.float())

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def sample_edges(self):
        p_log = F.logsigmoid(torch.stack((self.edge_log_params, -self.edge_log_params)))
        dag = gumbel_softmax(p_log, tau=self.temperature, hard=True, dim=0)[0]
        return dag

    def create_mask_and_fixed_values(self, num_vars, sure_edges, forbidden_edges):
        """
        Create a mask and fixed values tensor based on the given constraints.
        :param num_vars: Number of observed variables (d).
        :param sure_edges: List of tuples (i, j) for surely existing edges.
        :param forbidden_edges: List of tuples (i, j) for surely non-existing edges.
        :return: Tuple of (mask, fixed_values).
        """

        sure_mask = torch.zeros(num_vars, num_vars)
        for i, j in sure_edges:
            sure_mask[i,j] = 1

        forbidden_mask = torch.ones(num_vars, num_vars)
        for i, j in forbidden_edges:
             forbidden_mask[i, j] = 0

        return sure_mask, forbidden_mask 

    def sample_permutation(self):
        if self.order_type == 'sinkhorn':
            log_alpha = F.logsigmoid(self.perm_weights)
            P, _ = gumbel_sinkhorn(log_alpha, noise_factor=self.noise_factor, temp=self.temperature, hard=self.hard)
            P = P.squeeze().to(device)
        elif self.order_type == 'topk':
            logits = F.log_softmax(self.perm_weights, dim=0).view(1, -1)
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / 1
            P = self.sort(gumbels)
            P = P.squeeze()
        else:
            raise NotImplementedError
        return P

    def sample(self):
        dag_adj = self.sample_edges()
        while True:
            P = self.sample_permutation()
            P_inv = P.transpose(0, 1)
            if not self.ignore_sure:
                if torch.equal(torch.matmul(torch.matmul(P_inv, dag_adj), P) * self.sure_mask, self.sure_mask):
                    break
            else: break
        dag_adj = dag_adj * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        self.temperature *= self.temp_decay
        ### We can only remove edges, forbidden edges can be ensured
        dag_adj = dag_adj*self.forbidden_mask
        return dag_adj

    def log_prob(self, dag_adj):
        raise NotImplementedError

    def deterministic_permutation(self, hard=True):
        if self.order_type == 'sinkhorn':
            log_alpha = F.logsigmoid(self.perm_weights)
            P, _ = gumbel_sinkhorn(log_alpha, temp=self.temperature, hard=hard, noise_factor=0)
            P = P.squeeze().to(device)
        elif self.order_type == 'topk':
            sort = SoftSort_p1(hard=hard, tau=self.temperature)
            P = sort(self.perm_weights.detach().view(1, -1))
            P = P.squeeze()
        return P

    def get_threshold_mask(self, threshold):
        P = self.deterministic_permutation()
        P_inv = P.transpose(0, 1)
        dag = (torch.sigmoid(self.edge_log_params.detach()) > threshold).float()
        dag = dag * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        return dag

    def get_prob_mask(self):
        P = self.deterministic_permutation()
        P_inv = P.transpose(0, 1)
        e = torch.sigmoid(self.edge_log_params.detach())
        e = e * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        return e

    def print_parameters(self, prob=True):
        print('Permutation Weights')
        print(torch.sigmoid(self.perm_weights) if prob else self.perm_weights)
        print('Edge Probs')
        print(torch.sigmoid(self.edge_log_params) if prob else self.edge_log_params)



def check_all_indices_one(matrix, indices):
    # Check if all specified elements are 1
    return all(matrix[i][j] == 1 for i, j in indices)

def check_all_indices_zero(matrix, indices):
    # Check if all specified elements are 1
    return all(matrix[i][j] == 0 for i, j in indices)


def train(model, data_loader, xy, maxmin, sure_edges, forbidden_edges, _config, out_dir=None):
    """
    Training loop for the DAGNet model.
    :param model: The DAGNet model.
    :param data_loader: DataLoader for the dataset.
    :param optimizer: Optimizer used for training.
    :param epochs: Number of training epochs.
    """
    tmp_results = [] #For global minimum checking
    if out_dir is not None:
        writer = SummaryWriter(log_dir=out_dir)
    else:
        writer = SummaryWriter()

    model.train()
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    curr_min = np.inf
    A_opt = None
    # Always loads the full dataset, but in a different order
    for batch in data_loader:
        for rnd in range(_config['n_rounds_prob']):
        # Find approximate solution of subproblem at fixed lmbda
            for opt_stp in range(_config['opt_steps_prob']):
                A = model.sample()
                A[xy[0],xy[1]]=1 #Hard assignment of xy
                if not nx.is_directed_acyclic_graph(nx.DiGraph(A.detach().numpy())):
                    model.temperature = model.temperature/model.temp_decay
                    continue
                iter_idx = _config['opt_steps_prob'] * rnd + opt_stp

                query_value, _ = utils.query(batch, A, xy, optimal=_config['optimal'])
                if maxmin == 'max':
                    query_value *= -1

                A_copy = A.clone().detach()
                query_copy = query_value.clone().detach().item()
                if nx.is_directed_acyclic_graph(nx.DiGraph(A_copy.numpy())) \
                     and check_all_indices_zero(A_copy, forbidden_edges):
                    tmp_results.append(query_value.item())
                    if(query_copy < curr_min):
                        curr_min = query_copy
                        A_opt = A_copy


                loss = query_value
                if not loss.requires_grad:
                    model.temperature = model.temperature/model.temp_decay
                    continue
                # Calculate gradients
                model.optimizer.zero_grad()
                loss.backward()
                
                model.optimizer.step()
                iter_idx += 1
                
                if _config['save_all']:
                    #plot lambda, tau, and more at each step
                    writer.add_scalar(f'DP-DAG/{maxmin}/query_value', query_value.item(), iter_idx)
                    writer.add_scalar(f'lagrangian/{maxmin}/h(A)', utils.h(A), iter_idx)

            
        
    if tmp_results!=[]:
        valid_results = np.array(tmp_results)
        if maxmin == 'max':
            logging.info(f"DP-DAG: {maxmin} query value is : {(-valid_results).max()}")
            opt_val = (-valid_results).max()
            valid_results = -valid_results
        else:
            logging.info(f"DP-DAG: {maxmin} query value is : {valid_results.min()}")
            opt_val = valid_results.min()
        return opt_val, A_opt, valid_results
    else: 
        logging.info(f"DP-DAG: {maxmin} query value is : {np.nan}")
        return np.nan, np.nan, []

if __name__ == "__main__":
    # Load yaml file as configuration
    with open('configs/probabilistic_dag.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create output directory
    output_dir = config['output_dir']
    out_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(hash(tuple(config.items())))
    out_dir = os.path.join(output_dir, out_name)
    out_dir = os.path.join(output_dir, 'DP_DAG_test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_dir = "./data"
    directory_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    for d in directory_names:
        data_path = os.path.join(data_dir, d)
        unique_numbers = utils.get_unique_numbers(data_path)
        for n in unique_numbers:
            A = torch.from_numpy(np.load(os.path.join(data_path, f'DAG{n}.npy')))
            dat = np.load(os.path.join(data_path, f'data{n}.npy'))
            dataset = data.ObservationalDataset(dat)
            with open(os.path.join(data_path, f'mechanism{n}.pkl'), 'rb') as f:
                mechanism = pkl.load(f)

            xy =utils.random_select_one_index(A)
            utils.plot_graph(A, xy)
            x,y = xy
            parents = (A[:, y] > 0).nonzero(as_tuple=True)[0].tolist()
            mech_index = parents.index(x)
            gt_causal_effect = mechanism[y].coefflist[mech_index]
            print(f"Ground truth Causal effect is {gt_causal_effect}")
            sure_edges, forbidden_edges = data.get_sure_and_forbidden(A, sure_prob=config['sure_prob'],\
                                                                    forbidden_prob=config['forbid_prob'], xy=xy)

            max_model = ProbabilisticDAG(n_nodes=config['n_var'], order_type=config['order_type'], lr=config['lr_prob'],\
                                  seed=config['seed_prob'], sure_edges=sure_edges, forbidden_edges=forbidden_edges)
            min_model = ProbabilisticDAG(n_nodes=config['n_var'], order_type=config['order_type'], lr=config['lr_prob'],\
                                  seed=config['seed_prob'], sure_edges=sure_edges, forbidden_edges=forbidden_edges)
            dataset = data.ObservationalDataset(dat)
            data_loader = DataLoader(dataset, batch_size=config['n_sample'], shuffle=True)

             # optimize the model
            start = time.time()
            ate_max, A_max, _ = train(max_model, data_loader, xy, maxmin='max', sure_edges=sure_edges,\
                    forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
            end = time.time()
            print(f"Time taken for max: {end - start} seconds")

            # optimize the model
            start = time.time()
            ate_min, A_min, _ = train(min_model, data_loader, xy, maxmin='min', sure_edges=sure_edges,\
                forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
            end = time.time()
            print(f"Time taken for min: {end - start} seconds")

            print(f"Max ATE is {ate_max}")
            print(f"Min ATE is {ate_min}")

            #TODO REMOVE IF WANT TO DO FULL RUN
            break
        break
    




