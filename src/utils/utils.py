import os

import numpy as np
import math
import itertools
import yaml

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
from torch.utils.data import DataLoader, Subset
from utils import opt_adjustment
from utils import data

from utils.soft_sort import SoftSort_p1, gumbel_sinkhorn

import matplotlib.pyplot as plt
import networkx as nx
from itertools import permutations
import random
from itertools import combinations

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# generate all possible configurations from a config file
def generate_configurations(data, keys, current_config, results):
    # Base case: If all keys are processed, add the current configuration to results
    if not keys:
        results.append(current_config)
        return

    # Get the next key to process
    key = keys[0]

    # Get the value(s) for the current key
    values = data[key]

    # Check if the value is a list (multiple options) or a single value
    if isinstance(values, list):
        for option in values:
            # Add the current option to the current configuration
            updated_config = current_config.copy()
            updated_config[key] = option

            # Recursively generate configurations for the remaining keys
            generate_configurations(data, keys[1:], updated_config, results)
    else:
        # Single value case
        updated_config = current_config.copy()
        updated_config[key] = values

        # Recursively generate configurations for the remaining keys
        generate_configurations(data, keys[1:], updated_config, results)

def generate_all_configurations(config_file):
    # Load the YAML config file
    with open(config_file, 'r') as file:
        data = yaml.safe_load(file)

    # Get the keys from the config file
    keys = list(data.keys())

    # Initialize the results list
    results = []

    # Start generating configurations recursively
    generate_configurations(data, keys, {}, results)
    return results

######################## Graph Plotting ############################
def plot_graph(A, xy, out_dir=None, sure_edges=None, forbidden_edges=None):
    """
    Plot the graph.
    :param A: Adjacency matrix.
    :param xy: Tuple of two indices indicating treatment and target variable.
    :param out_dir: Output directory.
    """
    if(forbidden_edges==None and sure_edges==None):
        G = nx.from_numpy_array(A.numpy(), create_using=nx.DiGraph)
        color_map = []
        for node in G:
            if node == xy[0]:
                color_map.append('#FFC300')
            elif node == xy[1]: 
                color_map.append('#DAF7A6')   
            else:
                color_map.append('#7f8c8d')  
        nx.draw_networkx(G, node_color=color_map, with_labels = True)
        if out_dir is not None:
            plt.savefig(os.path.join(out_dir, 'ground_truth_graph.png'))
        else: plt.savefig('ground_truth_graph.png')
        plt.clf()
    else:
        G = nx.DiGraph()
        # for i in range(A.shape[0]):
        #     G.add_node(i)
        color_map = []
        for node in G:  
            if node == xy[0]:
                color_map.append('#FFC300')
            elif node == xy[1]: 
                color_map.append('#DAF7A6')   
            else:
                color_map.append('#7f8c8d')  
        n = A.shape[0]
        for i, j in itertools.product(range(n), range(n)):
            if i==j: continue
            elif((i,j) in sure_edges):
                G.add_edge(i, j, color='green', weight=3)
            elif((i,j) in forbidden_edges):
                G.add_edge(i, j, color='red', weight=3)
            else:
                if((j, i) not in sure_edges):
                    G.add_edge(i, j, color='black', weight=1)
        #pos = nx.spring_layout(G)
        #edges = G.edges()
        colors = nx.get_edge_attributes(G,'color').values()
        weights = nx.get_edge_attributes(G,'weight').values()
        nx.draw_networkx(G, node_color=color_map, edge_color=colors, width=list(weights), with_labels=True)
        if out_dir is not None:
            plt.savefig(os.path.join(out_dir, 'uncertainty_graph.png'))
        else: plt.savefig('uncertainty_graph.png')
        plt.clf()

######################## Causal Queries ############################
# Check if tensor is empty
def is_empty_tensor(tensor):
    return tensor.numel() == 0

def query(dat, A, xy, optimal):
    """
    Compute the query based on the adjacency matrix, data, and indices.
    :param data: Observational data.
    :param A: A d x d pytorch tensor representing the adjacency matrix.
    :param xy: Tuple of two indices.
    :return: The computed query value.
    """
    x, y = xy[0], xy[1]
    if optimal:
        adjustmentset = get_optimal_adjustment(A, xy)
    else:
        adjustmentset = get_parent_adjustment(A, xy)
    causal_effect = linear_causal_effect(dat, x, y, adjustmentset)
    return causal_effect, adjustmentset

def query_non_linear(data, A, xy, model, x_star = 1, optimal=False):
    """
    Compute the query based on the adjacency matrix, data, and indices.
    :param data: Observational data.
    :param A: A d x d pytorch tensor representing the adjacency matrix.
    :param xy: Tuple of two indices.
    :return: The computed query value.
    """
    x_1 = (torch.ones(data.shape[0]) * x_star).unsqueeze(1)
    x_0 = torch.zeros(data.shape[0]).unsqueeze(1)
    if optimal:
        s = get_optimal_adjustment(A, xy)
    else:
        s = get_parent_adjustment(A, xy)
    selection_mask = s.unsqueeze(0)  # Make it a row vector
    mask_list = []
    for idx in torch.nonzero(selection_mask):
        mult = torch.zeros(s.shape[0], s.shape[0])
        mult[idx[1], idx[1]] = 1
        if(selection_mask.dtype == torch.float64):
            mult = mult.double()
        mask_list.append(selection_mask @ mult)
    if len(mask_list) != 0:
        selection_mask = torch.cat(mask_list, dim=0)
        if(selection_mask.dtype == torch.float64):
            data = data.double()
        dat_parents = data @ selection_mask.t()  # Matrix multiplication to select parent columns
        # Concatenate the data
        inputs_star = torch.cat([dat_parents, x_1], dim=1)
        # Concatenate the data
        inputs_zeros = torch.cat([dat_parents, x_0], dim=1)
    else:
        inputs_star = x_1
        inputs_zeros = x_0

    model.eval()
    return torch.mean(model(inputs_star)-model(inputs_zeros))


###################### Effect estimation Functions ############################
def get_parent_adjustment(A, xy):
    """
    Select all parents of x according to matrix A and return them as a list of integers.
    :param A: A d x d pytorch tensor representing the adjacency matrix.
    :param xy: Tuple of two indices.
    :return: List of parent indices for the first entry in xy.
    """
    x, _ = xy[0], xy[1]
    parents = A[:, x]
    return parents

def blocks_backdoor_paths(G, x, y, adjustment_set):
    """
    Check if a given adjustment set blocks all backdoor paths from x to y in the graph G.
    
    Args:
    - G: A directed acyclic graph (DAG) as a networkx DiGraph object.
    - x: Treatment variable.
    - y: Outcome variable.
    - adjustment_set: A set of nodes to check.
    
    Returns:
    - True if the adjustment set blocks all backdoor paths; False otherwise.
    """
    for path in nx.all_simple_paths(G, source=x, target=y):
        if len(path) > 2 and path[1] != y:
            for node in path[1:-1]: 
                if node not in adjustment_set:
                    return False
    return True


def find_children_on_direct_path(graph, node_adj, treatment, effect):
        
    paths = list(nx.all_simple_paths(graph, source=treatment, target=effect))
        
    nodes_on_path = set()
    for path in paths:
        nodes_on_path.update(path)

    children = list(graph.successors(node_adj))

    children_on_path = [child for child in children if child in nodes_on_path]
    return children_on_path

def get_optimal_adjustment(A, xy):
    """
    Find the optimal adjustment set for estimating the causal effect of x on y.
    Handles cases where the graph might be cyclic by converting it to a DAG.
    
    :param A: Adjacency matrix representing the DAG (pytorch tensor)
    :param xy: Tuple of two indices.
    :return: Optimal adjustment set as a tensor of the same size as the parent adjustment tensor.
    """
    x, y = xy[0], xy[1]
    A_np = A.detach().numpy() if isinstance(A, torch.Tensor) else A
    G = nx.from_numpy_array(A_np, create_using=opt_adjustment.CausalGraph)

    new_adjustment_tensor = torch.zeros_like(A[:, x], requires_grad=True)
    
    try:
        if nx.is_directed_acyclic_graph(G):
        ### Optimal adjustment only works for DAG
            optimal_sets_list = list(G.optimal_adj_set(x, y, L=[], N=list(G.nodes())))
            if optimal_sets_list:
                for z1 in optimal_sets_list:
                    childrens_list = []
                    neighbors = find_children_on_direct_path(G, z1, x, y)
                    for n in neighbors:
                        childrens_list.append(n)
                    n_children = len(childrens_list)
                    
                    if n_children>0:
                        mask = torch.zeros_like(new_adjustment_tensor)
                        mask[z1] = 1/n_children  #equally divide gradient on all path from adjustment variale to variables on the directed path
                        for c in childrens_list:
                            new_adjustment_tensor = new_adjustment_tensor + mask * A[z1, c]
                return new_adjustment_tensor
            
            else:
                return new_adjustment_tensor
        else: 
        ### Otherwise we do the adjustment with parent adjustment
        ### TODO Check wether without adjustment is better option
            #return get_parent_adjustment(A, xy)
            return new_adjustment_tensor
        
    except nx.NetworkXError as e:
        # Return zero adjustment tensor if there's an error
        return new_adjustment_tensor

def linear_causal_effect(dat, x, y, s):
    """
    Compute an ordinary least squares regression from inputs to the target variable.
    :param data: Observational data.
    :param x: Index for the independent variable.
    :param y: Index for the dependent variable.
    :param s: List of indices for the parent variables.
    :return: Regression result.
    """
    # Stack the specified columns to create the input tensor with a bias term
    # Create a selection mask for the parents
    selection_mask = s.unsqueeze(0)  # Make it a row vector
    mask_list = []
    for idx in torch.nonzero(selection_mask):
        mult = torch.zeros(s.shape[0], s.shape[0])
        mult[idx[1], idx[1]] = 1
        if(selection_mask.dtype == torch.float64):
            mult = mult.double()
        mask_list.append(selection_mask @ mult)
    if len(mask_list) != 0:
        selection_mask = torch.cat(mask_list, dim=0)
        if(selection_mask.dtype == torch.float64):
            dat = dat.double()
        dat_parents = dat @ selection_mask.t()  # Matrix multiplication to select parent columns
        # Extract data for treatment and add a bias term
        dat_x = dat[:, x].unsqueeze(1)
        dat_bias = torch.ones(dat.shape[0], 1)  # Bias term
        # Concatenate the data
        dat_regressors = torch.cat([dat_parents, dat_bias, dat_x], dim=1)
    else:
        dat_x = dat[:, x].unsqueeze(1)
        dat_bias = torch.ones(dat.shape[0], 1)  # Bias term
        # Concatenate the data
        dat_regressors = torch.cat([dat_bias, dat_x], dim=1)
    # Data for the effect node y
    dat_y = dat[:, y].unsqueeze(1)
    beta = torch.linalg.lstsq(dat_regressors, dat_y).solution[-1]
    return beta

###################### Acyclicity constraints ############################
def h(A, acyc_constr='original', c=1):
    """
    DAGness fucnction.
    :acyc_constr: either original or stable
    :c: c=1 for directed edges and c=2 for bidirected edges
    """

    if acyc_constr == 'original':
        return torch.trace(torch.linalg.matrix_exp(A)) - A.shape[0]
    elif acyc_constr == 'alternate':
        d = A.shape[0]
        M = torch.eye(d) + c*A
        E = torch.matrix_power(M, d)
        return E.sum() - d
    elif acyc_constr == 'dogma':
        d = A.shape[0]
        s = torch.tensor(10000)
        try:
            # Compute the matrix I + εW
            M = s * torch.eye(d)- A * A
            log_det = -torch.linalg.slogdet(M)[1] + d * torch.log(s)
        except np.linalg.LinAlgError:
            # In case of numerical issues in determinant computation
            log_det = float('inf')
        return log_det



###################### Lagrangian computation ############################
def compute_lagrangian(A_bin, constr_type, data, xy, lam, tau, maxmin='max', mlp_model=None, optimal=False):
    """
    Compute the loss for the model.
    :param A: Adjacency matrix.
    :param data: Observational data.
    :param xy: Tuple of two indices indicating treatment and target variable.
    :param lam: Lagrange multiplier for DAG constraint.
    :param tau: Penalty for violating DAG constraint.
    :param maxmin: Whether to maximize or minimize the query.
    :param mlp_model: The model to use in the non-linear case.
    :return: Computed loss.
    """
    query_value = np.nan
    if mlp_model is None: # Linear case
        try:
            query_value, _ = query(data, A_bin, xy, optimal)
        except:
            print('error occured when computing query')
            #print(A_bin)
    else: # Non-linear case
        try:
            query_value = query_non_linear(data, A_bin, xy, mlp_model, optimal=optimal)
        except:
            print('error occured when computing query')
            #print(A_bin)

    if maxmin == 'max':
        query_value *= -1
    
    constr = h(A_bin)

    #Augmented Lagrangian method
    case1 = - lam * constr + 0.5 * tau * constr**2
    case2 = - 0.5 * lam**2 / tau

    if constr_type == 'equality':
        dag_loss = case1
    elif constr_type == 'inequality':
        if tau * constr <= lam:
            dag_loss = case1
        else:
            dag_loss = case2


    # Add other necessary loss components here
    return query_value, dag_loss, constr

############# MLP TRAINING FUNCTIONS ################
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(MLP, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Third fully connected layer
        self.fc3 = nn.Linear(hidden_size, 1)
        # Non-linear activation functions
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # Apply ReLU non-linearity
        x = self.fc2(x)
        x = self.relu(x) # Apply ReLU non-linearity
        x = self.fc3(x)
        return x
class BackwardCounter:
    """
    DEBUG ONLY
    """
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        torch.autograd.backward(*args, **kwargs)

def train_MLP(model, inputs, target, mlp_epoch=100, patience=10, min_delta=0.1, validation=False, val_ratio = 0.1):
    #TODO Need to pick a nice value for early stopping and learning rate, since we are using full batch training, maybe it's fine to have a larger lr
    optimizer_MLP = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    model.train()
    detached_inputs = inputs.detach()

    # Variables for early stopping
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    if validation:
        num_samples = len(inputs)
        num_val = int(val_ratio * num_samples)
        num_train = num_samples - num_val
        indices = torch.randperm(num_samples).tolist()
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        train_inputs = inputs[train_indices].detach()
        train_target = target[train_indices]
        val_inputs = inputs[val_indices].detach()
        val_targets = target[val_indices]

        for e in range(mlp_epoch):
            model.train()
            outputs = model(train_inputs)
            loss_MLP = criterion(torch.squeeze(outputs, 1), train_target)
            optimizer_MLP.zero_grad()
            loss_MLP.backward()
            optimizer_MLP.step()

            # Validation step
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_outputs = model(val_inputs)
                val_loss = criterion(torch.squeeze(val_outputs, 1), val_targets).item()

            # Early stopping based on validation loss
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict() 
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                #print(f'Early stopping at epoch {e+1}')
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        for param in model.parameters():
            param.requires_grad = False

        if e == (mlp_epoch-1):
            raise Exception("MLP model need more training epoch")

    else:
        for e in range(mlp_epoch):
            outputs = model(detached_inputs)
            loss_MLP = criterion(torch.squeeze(outputs, 1), target)
            optimizer_MLP.zero_grad()
            loss_MLP.backward()
            optimizer_MLP.step()

            # Early stopping
            if loss_MLP.item() < best_loss - min_delta:
                best_loss = loss_MLP.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                #print(f'Early stopping at epoch {e+1}')
                break

        #print(loss_MLP.item(), flush=True)

        for param in model.parameters():
            param.requires_grad = False

        if e == (mlp_epoch-1):
            raise Exception("MLP model does not converge")
        


###########COMPUTING GROUND TRUTH####################
import networkx as nx

def compute_linear_total_causal_effect(adj_matrix, start_node, end_node, mechanisms):
    G = nx.DiGraph()
    n = len(adj_matrix)
    
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)

    all_paths = list(nx.all_simple_paths(G, source=start_node, target=end_node))
    
    total_effect = 0.0

    for path in all_paths:
        path_effect = 1.0
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            parents = sorted([p for p in range(n) if adj_matrix[p][child] > 0])
            mech_index = parents.index(parent)
            path_effect *= mechanisms[child].coefflist[mech_index]
        
        total_effect += path_effect

    return total_effect

def compute_total_causal_effect(adj_matrix, observational_data, mechanisms, x_index, y_index, x_values, num_samples):
    """
    Compute the total causal effect from X to Y by performing interventions.
    """

    all_paths = find_all_directed_paths(adj_matrix, x_index, y_index)
    all_paths_nodes = get_nodes_on_paths(all_paths)

    expected_y = []

    for x_val in x_values:
        intervention = {x_index: x_val}

        data = simulate_intervention(
            adj_matrix, mechanisms, intervention, observational_data, all_paths_nodes, num_samples
        )

        y_mean = np.mean(data[y_index])
        expected_y.append(y_mean)

    total_causal_effect = expected_y[1] - expected_y[0]

    return expected_y, total_causal_effect

def get_trained_mlp(A, xy, data, optimal):
    """
    Train an MLP model for the given adjustment set."""
    x, y = xy
    if optimal: 
        adjustmentset = (get_optimal_adjustment(A, xy) > 0).nonzero(as_tuple=True)[0].tolist()
    else:
        adjustmentset = (get_parent_adjustment(A, xy) > 0).nonzero(as_tuple=True)[0].tolist()
    mlp_model = MLP(len(adjustmentset) + 1) 
    inputs = torch.stack([data[:, i] for i in adjustmentset + [x]], dim=1)
    target = data[:, y]
    train_MLP(mlp_model, inputs, target, mlp_epoch=1000, patience=10, min_delta=0.1, validation = True)
    return mlp_model

def find_all_directed_paths(adj_matrix, start_node, end_node):
    """
    Find all directed paths from start_node to end_node in a directed graph represented by an adjacency matrix.
    """
    G = nx.DiGraph(adj_matrix)
    all_paths = list(nx.all_simple_paths(G, source=start_node, target=end_node))
    return all_paths

def get_nodes_on_paths(all_paths):
    """
    Get a set of all unique nodes that are on any of the provided paths.
    """
    nodes_on_paths = set()
    for path in all_paths:
        nodes_on_paths.update(path)
    return nodes_on_paths

def simulate_intervention(adj_matrix, mechanisms, intervention, observational_data, all_paths_nodes, num_samples):
    """
    Simulate the causal model under a given intervention, recomputing only the nodes on the paths from X to Y.
    """
    n_nodes = adj_matrix.shape[0]
    data = observational_data.copy()

    # Create a subgraph containing only the nodes on the paths from X to Y
    subgraph_nodes = all_paths_nodes
    sub_adj_matrix = adj_matrix.copy()

    # Zero out edges not in the subgraph to prevent computation of unaffected nodes
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i not in subgraph_nodes or j not in subgraph_nodes:
                sub_adj_matrix[i, j] = 0

    # Perform topological sort on the subgraph
    G_sub = nx.DiGraph(sub_adj_matrix)
    causal_order_sub = list(nx.topological_sort(G_sub))

    # Apply interventions
    for node, value in intervention.items():
        data[node] = np.full((num_samples), value)

    # Simulate data for nodes on the paths, respecting causal order
    for node in causal_order_sub:
        if node in intervention:
            continue  # Skip intervened nodes
        parents = np.sort(np.where(sub_adj_matrix[:, node] > 0)[0])
        if len(parents) == 0:
            # Node with no parents in the subgraph, keep original data
            continue
        else:
            # Gather data from parent nodes
            causes = np.column_stack([data[parent] for parent in parents])
            data[node] = mechanisms[node](causes).reshape(num_samples)

    return data



    
        
########### PC ALGORITHM HELPERS #####################

def random_permutations(nvar, n_samples=3):
    # Generate all permutations of range(nvar)
    perms = itertools.permutations(range(nvar))
    
    # Select n_samples random permutations
    # This method avoids generating all permutations at once by randomly skipping some
    selected_perms = []
    sample_indices = sorted(random.sample(range(math.factorial(nvar)), n_samples))
    
    current_index = 0
    for i, perm in enumerate(perms):
        if i == sample_indices[current_index]:
            selected_perms.append(perm)
            current_index += 1
            if current_index >= n_samples:
                break

    return selected_perms

def inverse_permutation(perm):
    # Create an array to hold the position of each index
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def apply_inverse_to_matrix(matrix, inverse_perm):
    # Reorder both rows and columns of the matrix using the inverse permutation
    return matrix[np.ix_(inverse_perm, inverse_perm)]

def check_specific_patterns(matrices):
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

import re
def get_unique_numbers(directory_path):
    # Regular expression pattern to match numbers in filenames
    number_pattern = re.compile(r'\d+')
    
    # Set to store unique numbers
    unique_numbers = set()
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Find all numbers in the filename
        numbers = number_pattern.findall(filename)
        # Add each number to the set
        for number in numbers:
            unique_numbers.add(int(number))
    
    return unique_numbers

def random_select_one_index(A, direct_cause=True):
    #Auxilary function
    A = A.int()
    if direct_cause:
        ones_indices = torch.nonzero(A == 1, as_tuple=False)
        random_index = random.choice(ones_indices.tolist())
        return tuple(random_index)
    else:
        n = A.shape[0]
        R = A.clone().bool()
        for k in range(n):
            R = R | (R[:, k].unsqueeze(1) & R[k, :].unsqueeze(0))
        R = R.int()
        ones_indices = torch.nonzero(R == 1, as_tuple=False)
        random_index = random.choice(ones_indices.tolist())
        return tuple(random_index)

def bootstrap_resample(dataset, size):
    indices = np.random.choice(len(dataset), size=size, replace=True)
    return Subset(dataset, indices)

def get_std_deviation(config):
    xy = torch.from_numpy(config['xy'])
    mechanism = config['mechanism']
    dat = data.get_data_from_config(config)
    dataset = data.ObservationalDataset(dat)
    n_bootstraps = 50

    A = torch.from_numpy(config['lagrangian_opt_graphs'][0])
    query_list = []
    for i in range(n_bootstraps):  
        bootstrap_sample = bootstrap_resample(dataset, len(dataset))
        loader = DataLoader(bootstrap_sample, batch_size=len(dataset))
        for data_resampled in loader:
            if mechanism == 'linear':
                query_value, _ = query(data_resampled, A, xy, optimal=config['optimal'])
            else:
                mlp_model = get_trained_mlp(A, xy, data_resampled, config['optimal'])
                query_value = query_non_linear(data_resampled, A, xy, mlp_model, config['optimal'])
        query_list.append(query_value)
    std_dev_min = torch.std(torch.tensor(query_list))

    A = torch.from_numpy(config['lagrangian_opt_graphs'][1])
    query_list = []
    for i in range(n_bootstraps):  
        bootstrap_sample = bootstrap_resample(dataset, len(dataset))
        loader = DataLoader(bootstrap_sample, batch_size=len(dataset))
        for data_resampled in loader:
            if mechanism == 'linear':
                query_value, _ = query(data_resampled, A, xy, optimal=config['optimal'])
            else:
                mlp_model = get_trained_mlp(A, xy, data_resampled, config['optimal'])
                query_value = query_non_linear(data_resampled, A, xy, mlp_model, config['optimal'])
        query_list.append(query_value)
    std_dev_max = torch.std(torch.tensor(query_list))
    return [std_dev_min, std_dev_max]