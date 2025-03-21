""" DAG Generator.

Generates a dataset out of an acyclic FCM.
Author : Olivier Goudet and Diviyan Kalainathan
Modified by Philippe Brouillard, June 25th 2019

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""

import os
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import networkx as nx
from generation.causal_mechanisms import (LinearMechanism,
                               Polynomial_Mechanism,
                               SigmoidAM_Mechanism,
                               SigmoidMix_Mechanism,
                               GaussianProcessAdd_Mechanism,
                               GaussianProcessMix_Mechanism,
                               ANM_Mechanism,
                               NN_Mechanism,
                               NN_Mechanism_Add,
                               pnl_gp_mechanism,
                               pnl_mult_mechanism,
                               PostNonLinear_Mechanism,
                               Multimodal_X_Mechanism,
                               Multimodal_Circle_Mechanism,
                               Multimodal_ADN_Mechanism,
                               gmm_cause,
                               gaussian_cause,
                               variable_gaussian_cause,
                               uniform_cause,
                               uniform_cause_positive,
                               laplace_noise,
                               normal_noise,
                               uniform_noise,
                               nn_noise,
                               absolute_gaussian_noise,
                               variable_normal_noise,
                               NormalCause,
                               UniformCause)
import pickle

class DagGenerator:
    """Generate an DAG and data given a causal mechanism.

    Args:
        causal_mechanism (str): currently implemented mechanisms:
            ['linear', 'polynomial', 'sigmoid_add',
            'sigmoid_mix', 'gp_add', 'gp_mix', 'nn'].
        noise (str or function): type of noise to use in the generative process
            ('gaussian', 'uniform' or a custom noise function).
        noise_coeff (float): Proportion of noise in the mechanisms.
        initial_variable_generator (function): Function used to init variables
            of the graph, defaults to a Gaussian Mixture model.
        npoints (int): Number of data points to generate.
        nodes (int): Number of nodes in the graph to generate.
        prob_connection (int): Probability of connection between two nodes.
    """

    def __init__(self, causal_mechanism, noise='gaussian',
                 noise_coeff=.4,
                 cause=gaussian_cause,
                 npoints=500, nodes=8, prob_connection=0.4,
                 dag_type='erdos', rescale=False, f1=None, f2=None):
        super().__init__()
        self.mechanism = {'linear': LinearMechanism,
                          'polynomial': Polynomial_Mechanism,
                          'sigmoid_add': SigmoidAM_Mechanism,
                          'sigmoid_mix': SigmoidMix_Mechanism,
                          'gp_add': GaussianProcessAdd_Mechanism,
                          'gp_mix': GaussianProcessMix_Mechanism,
                          'anm': ANM_Mechanism,
                          'nn': NN_Mechanism,
                          'nn_add': NN_Mechanism_Add,
                          'pnl_gp_mechanism': pnl_gp_mechanism,
                          'pnl_mult_mechanism': pnl_mult_mechanism,
                          'x': Multimodal_X_Mechanism,
                          'circle': Multimodal_Circle_Mechanism,
                          'adn': Multimodal_ADN_Mechanism,
                          'post_nonlinear': PostNonLinear_Mechanism }[causal_mechanism]

        self.causal_mechanism = causal_mechanism
        self.positive = False
        self.rescale = rescale


        self.data = pd.DataFrame(None, columns=["V{}".format(i) for i in range(nodes)])
        self.nodes = nodes
        self.npoints = npoints

        try:
            self.initial_generator = {'gmm_cause': gmm_cause,
                                      'gaussian': gaussian_cause,
                                      'variable_gaussian': variable_gaussian_cause,
                                      'uniform': uniform_cause,
                                      'uniform_positive': uniform_cause_positive}[cause]
        except KeyError:
            print('This type of cause does not exist')
            self.initial_generator = cause

        try:
            self.noise = {'gaussian': normal_noise,
                          'variable_gaussian': variable_normal_noise,
                          'uniform': uniform_noise,
                          'laplace': laplace_noise,
                          'absolute_gaussian': absolute_gaussian_noise,
                          'nn': nn_noise(variable_normal_noise)}[noise]
        except KeyError:
            print('This type of noise does not exist')
            self.noise = noise

        self.noise_coeff = noise_coeff
        self.adjacency_matrix = np.zeros((nodes, nodes))
        self.prob_connection = prob_connection
        self.dag_type = dag_type
        self.cfunctions = None
        self.g = None
        self.f1 = f1
        self.f2 = f2

        # Constraints on PNL models to make sure they are identifiable
        if self.causal_mechanism == 'pnl_gp_mechanism':
            self.noise = laplace_noise
            print('Forcing noise to be laplacian')
        elif self.causal_mechanism == 'pnl_mult_mechanism':
            self.noise = absolute_gaussian_noise
            initial_variable_generator = uniform_cause_positive
            self.positive = True
            print('Forcing noise to be an absolute Gaussian and cause to be uniform with positive support')


    def estimate_expected_density(self, n=100):
        """ Estimate the expected density(number of edge per node) of a
            graph generation by generating multiple graphes

            Args:
                n: number of graph generated to estimate the density
        """
        estimated_density = 0.

        for i in range(n):
            self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
            self.generate_dag()
            estimated_density += np.sum(self.adjacency_matrix)

        return estimated_density/(n * self.nodes)


    def generate_dag(self):
        """ Create the structure of the graph """

        if self.dag_type == 'erdos':
            self.causal_order = np.random.permutation(np.arange(self.nodes))

            for i in range(self.nodes - 1):
                node = self.causal_order[i]
                possible_parents = self.causal_order[(i+1):]
                num_parents = np.random.binomial(n=self.nodes - i - 1,
                                                 p=self.prob_connection)
                parents = np.random.choice(possible_parents, size=num_parents,
                                           replace=False)
                self.adjacency_matrix[parents,node] = 1

        elif self.dag_type == 'default':
            for j in range(1, self.nodes):
                nb_parents = np.random.randint(0, min([self.parents_max, j])+1)
                for i in np.random.choice(range(0, j), nb_parents, replace=False):
                    self.adjacency_matrix[i, j] = 1

        try:
            self.g = nx.DiGraph(self.adjacency_matrix)
            assert not list(nx.simple_cycles(self.g))

        except AssertionError:
            # if verbose:
            #     print("Regenerating, graph non valid...")
            self.generate_dag()

        self.original_adjacency_matrix = np.copy(self.adjacency_matrix)


    def init_variables(self, fname_radical, i_dataset):
        """Redefine the causes, mechanisms and the structure of the graph,
        called by ``self.generate()`` if never called.

        Args:
            verbose (bool): Verbosity
        """
        self.generate_dag()

        # Mechanisms
        self.original_cfunctions = []
        for i in range(self.nodes):
            if sum(self.adjacency_matrix[:, i]):
                if self.mechanism is PostNonLinear_Mechanism:
                    self.original_cfunctions.append(self.mechanism(int(sum(self.adjacency_matrix[:, i])), self.npoints, self.noise, f1=self.f1, f2=self.f2, noise_coeff=self.noise_coeff))
                else:
                    self.original_cfunctions.append(self.mechanism(int(sum(self.adjacency_matrix[:, i])), self.npoints, self.noise, noise_coeff=self.noise_coeff))
            else:
                self.original_cfunctions.append(self.initial_generator)
        

        # = [self.mechanism(int(sum(self.adjacency_matrix[:, i])), self.npoints, self.noise, noise_coeff=self.noise_coeff) if sum(self.adjacency_matrix[:, i]) else self.initial_generator for i in range(self.nodes)]

        # increase nb_step in order to save the gaussian processes
        if self.causal_mechanism == 'gp_mix':
            for cfunction in self.original_cfunctions:
                if isinstance(cfunction, GaussianProcessMix_Mechanism):
                    cfunction.nb_step += 1
        self.cfunctions = self.original_cfunctions[:]

        #To have a better compatibility of different mechanism, we save the whole mechanism class instead of a single parameters. We can easily reproduce the ground truth treatment effect
        data_path = os.path.join(fname_radical, f'mechanism{i_dataset}.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(self.cfunctions, f)

    # def reinitialize(self):
    #     """ Reinitialize the adjacency matrix and the cfunctions """
    #     self.adjacency_matrix = np.copy(self.original_adjacency_matrix)
    #     # for parametric change:
    #     for i,c in enumerate(self.cfunctions):
    #         if isinstance(c, LinearMechanism) or isinstance(c, NN_Mechanism) or \
    #         isinstance(c, NN_Mechanism_Add) or isinstance(c, Multimodal_ADN_Mechanism) or \
    #         isinstance(c, Multimodal_X_Mechanism):
    #             c.reinit()
    #     self.cfunctions = self.original_cfunctions[:]

    def generate(self, npoints=None):
        """Generate data from an FCM defined in ``self.init_variables()``.

        Returns:
            tuple: (pandas.DataFrame, networkx.DiGraph), respectively the
            generated data and graph.
        """
        if npoints is None:
            npoints = self.npoints

        if self.cfunctions is None:
            self.init_variables()


        for i in nx.topological_sort(self.g):
            # Root cause

            if not sum(self.adjacency_matrix[:, i]):
                self.data['V{}'.format(i)] = self.cfunctions[i](npoints)
            # Generating causes
            else:
                self.data['V{}'.format(i)] = self.cfunctions[i](self.data.iloc[:, self.adjacency_matrix[:, i].nonzero()[0]].values)
            if self.rescale:
                self.data['V{}'.format(i)] = scale(self.data['V{}'.format(i)].values)
                if self.positive:
                    self.data['V{}'.format(i)] -= np.min(self.data['V{}'.format(i)].values) - 1e-8
        return self.data, self.g


    def to_csv(self, fname_radical, **kwargs):
        """
        Save the generated data to the csv format by default,
        in two separate files: data, and the adjacency matrix of the
        corresponding graph.

        Args:
            fname_radical (str): radical of the file names. Completed by
               ``_data.csv`` for the data file and ``_target.csv`` for the
               adjacency matrix of the generated graph.
            \**kwargs: Optional keyword arguments can be passed to pandas.
        """
        if self.data is not None:
            self.data.to_csv(fname_radical+'_data.csv', index=False, **kwargs)
            pd.DataFrame(self.adjacency_matrix).to_csv(fname_radical \
                                                       + '_target.csv',
                                                       index=False, **kwargs)

        else:
            raise ValueError("Graph has not yet been generated. \
                              Use self.generate() to do so.")


    def save_dag(self, fname_radical, i_dataset):
        dag_path = os.path.join(fname_radical, f'DAG{i_dataset}.npy')
        np.save(dag_path, self.adjacency_matrix)



    def save_data(self, fname_radical, i_dataset):
        data_path = os.path.join(fname_radical, f'data{i_dataset}.npy')
        np.save(data_path, self.data)


    def to_npy(self, fname_radical, i_dataset):
        """
        Save the generated data to the npy format,
        in two separate files: data and the adjacency matrix of the
        corresponding graph.

        Args:
            fname_radical (str): radical of the file names.
            i_dataset (int): i-th dataset
        """
        if self.data is not None:
            self.save_dag(fname_radical, i_dataset)
            self.save_data(fname_radical, i_dataset)
        else:
            raise ValueError("Graph has not yet been generated. \
                              Use self.generate() to do so.")