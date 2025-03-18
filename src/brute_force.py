import yaml
import time
import random
import logging

import torch


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils import utils
from utils import data

class Brute_Force():
    def __init__(self, n_nodes, data, xy, sure_edges=None, forbidden_edges=None, \
                 ground_truth=None, optimal=True, linear=True):
        self.n = n_nodes
        self.sure_edges = sure_edges if sure_edges else []
        self.forbidden_edges = forbidden_edges if forbidden_edges else []
        self.curr_min = np.inf
        self.curr_max = -np.inf
        self.all_queries = []
        self.true_A = ground_truth
        self.data = data
        self.torch_data = torch.from_numpy(data).float()
        self.xy = xy
        self.A_min = None
        self.A_max = None
        self.time_limit_exceeded = False
        self.optimal = optimal
        self.linear = linear

        # Initialize a directed graph
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(n_nodes))

        # Add mandatory edges
        self.G.add_edges_from(self.sure_edges)

        # Start time
        self.start = time.time()

        # Iteratively generate DAGs
        self.generate_dags_iteratively()

    def is_acyclic(self, graph):
        # Leverage NetworkX's built-in function to check for cycles
        return nx.is_directed_acyclic_graph(graph)

    def process_graph(self, graph):
        if self.is_acyclic(graph):
            A = nx.to_numpy_array(graph, dtype=int)
            if self.true_A is not None and np.array_equal(A, self.true_A):
                logging.info("Found the ground truth DAG")
            # linear setting
            if self.linear:
                q = utils.query(torch.from_numpy(self.data).float(), \
                                torch.from_numpy(A).float(), self.xy, self.optimal)
                q = q[0]
            # nonlinear setting
            elif not self.linear:
                mlp_model = utils.get_trained_mlp(A, self.xy, self.torch_data, self.optimal)
                q = utils.query_non_linear(self.torch_data, torch.from_numpy(A).float(), \
                                           self.xy, mlp_model, self.optimal)
            self.all_queries.append(q)
            if q > self.curr_max:
                self.curr_max = q
                self.A_max = A
            if q < self.curr_min:
                self.curr_min = q
                self.A_min = A

    def generate_dags_iteratively(self):
        # Create a list of all possible edges except self-loops and forbidden edges
        all_edges = [(i, j) for i in range(self.n) for j in range(self.n) if i != j and (i, j) not in self.forbidden_edges]
        uncertain_edges = [edge for edge in all_edges if edge not in self.sure_edges]

        stack = [([], 0)]
        visited = set()

        while stack:
            curr_time = time.time()
            if self.time_limit_exceeded:
                return
            if curr_time - self.start > 10800:
                print('Time limit exceeded: stopping brute force')
                self.time_limit_exceeded = True
                return

            edges, pos = stack.pop()

            if pos >= len(uncertain_edges):
                graph = nx.DiGraph()
                graph.add_nodes_from(range(self.n))
                graph.add_edges_from(self.sure_edges)
                graph.add_edges_from(edges)
                if self.is_acyclic(graph):
                    self.process_graph(graph)
                continue

            u, v = uncertain_edges[pos]

            # Exclude the edge
            stack.append((edges, pos + 1))

            # Include the edge
            new_edges = edges + [(u, v)]
            graph = nx.DiGraph()
            graph.add_nodes_from(range(self.n))
            graph.add_edges_from(self.sure_edges)
            graph.add_edges_from(new_edges)
            if self.is_acyclic(graph):
                stack.append((new_edges, pos + 1))

if __name__ == "__main__":
    with open('configs/comparison.yaml', 'r') as file:
        config_data = yaml.safe_load(file)
    xy = (config_data['n_var'] - 2, config_data['n_var'] - 1)

    torch.manual_seed(int(config_data['seed_data']))
    random.seed(int(config_data['seed_data']))
    # Create a dataset and data loader
    A = data.adj_mat_generator(config_data)
    print(f"Ground truth adjacency matrix is {A}")
    G = nx.from_numpy_array(A.numpy(), create_using=nx.DiGraph)
    color_map = []
    for node in G:
        if node == xy[0]:
            color_map.append('#FFC300')
        elif node == xy[1]:
            color_map.append('#DAF7A6')
        else:
            color_map.append('#7f8c8d')
    nx.draw_networkx(G, node_color=color_map, with_labels=True)
    plt.savefig('graph.png')

    sure_edges, forbidden_edges = data.get_sure_and_forbidden(A, sure_prob=0.2, forbidden_prob=0.2, xy=xy)
    dat, param = data.data_generator_linear(A, config_data)
    print(f"Ground truth Causal effect is {param[xy]}")

    start = time.time()
    model = Brute_Force(config_data['n_var'], dat, xy, sure_edges, forbidden_edges, A)
    end = time.time()
    print(f"Time taken for Brute Force: {end - start} seconds")
    print(f'number of all possible graphs: {len(model.all_queries)}')

    print(f"Brute force: max query value is : {model.curr_max.item()}")
    print(f"Brute force: min query value is : {model.curr_min.item()}")
    assert model.curr_min <= model.curr_max, "Min query value is greater than max query value"
    assert model.curr_min == min(model.all_queries), "Min query value is not the minimum of all queries"
    assert model.curr_max == max(model.all_queries), "Max query value is not the maximum of all queries"
    assert param[xy] < model.curr_max, "Ground truth is greater than max query value"
    assert param[xy] > model.curr_min, "Ground truth is less than min query value"