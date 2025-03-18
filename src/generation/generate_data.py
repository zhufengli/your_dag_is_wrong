import os
import sys
sys.path.append(os.path.abspath('./src/'))
import argparse
import csv
import yaml
import numpy as np
import generation.dag_generator as gen
from sklearn.preprocessing import StandardScaler

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


class dataset_generator:
    """ Generate datasets using dag_generator.py. `nb_dag` dags are sampled and
    then data are generated accordingly to the chosen parameters (e.g.
    mechanisms). Can generate dataset with 'hard stochastic' interventions """

    def __init__(self, config, verbose=False):
        """
        Generate a dataset containing interventions. The setting is similar to the
        one in the GIES paper (Characterization and greedy learning of interventional Markov
        equivalence classes of directed acyclic graphs). Save the lists of targets
        in a separate file.

        Args:
            mechanism (str): Type of mechanism use to generate the data
            (linear|polynomial|sigmoid_add|sigmoid_mix|gp_add|gp_mix|
            anm|nn|nn_add|pnl_gp_mechanism|pnl_mult_mechanism|post_nonlinear|x)
            cause (str): Distribution of initial causes
            (gmm_cause|gaussian|variable_gaussian|uniform|uniform_positive)
            noise (str): Distribution of noises
            (gaussian|variable_gaussian|uniform|laplace|absolute_gaussian|nn)
            noise_coeff (float): Noise coefficient

            nb_nodes (int): Number of nodes in each DAG
            prob_connection (int): Probability of connection between two nodes.
            nb_points (int): Number of points per interventions (thus the total =
                            nb_interventions * nb_points)
            rescale (bool): if True, rescale each variables
            suffix (str): Suffix that will be added at the end of the folder name

            nb_interventions (int): number of interventional settings
            obs_data (bool): if True, the first setting is generated without any interventions
            min_nb_target (int): minimal number of targets per setting
            max_nb_target (int): maximal number of targets per setting. For a fixed
                                 number of target, one can make min_nb_target==max_nb_target
            conservative (bool): if True, make sure that the intervention family is
                                 conservative: i.e. that all nodes have not been
                                 intervened in at least one setting.
            cover (bool): if True, make sure that all nodes have been
                                 intervened on at least in one setting.
            verbose (bool): if True, print messages to inform users
        """

        self.mechanism = config['mechanism']
        self.cause = config['initial_cause']
        self.noise = config['noise']
        self.noise_coeff = config['noise_coeff']
        self.nb_nodes = config['nb_nodes']
        self.prob_connection = config['prob_connection']
        self.i_dataset = 0
        self.nb_points = config['nb_points']
        if config['suffix']==None:
            self.suffix = self.mechanism
        else:
            self.suffix = config['suffix']
        self.rescale = config['rescale']
        self.config = config
        config_hash = str(hash(tuple(config.items())))
        self.filename = f'data_p{self.nb_nodes}_e{self.prob_connection}_n{self.nb_points}_{self.suffix}_{config_hash}'
        self.folder = os.path.join(config['data_folder'], self.filename)
        self.folder = os.path.abspath(self.folder)
        self.verbose = verbose
        self.generator = None

        # assert that the parameters
        self._checkup()
        self._create_folder()
        # Saving the yaml files
        with open(os.path.join(self.folder, 'config.yaml'), 'w+') as ff:
            yaml.dump(self.config, ff)

    def _checkup(self):
        possible_mechanisms = ["linear","polynomial","sigmoid_add","sigmoid_mix","gp_add","gp_mix",
                               "anm","nn","nn_add","pnl_gp_mechanism","pnl_mult_mechanism","post_nonlinear","x","circle","adn"]
        possible_causes = ["gmm_cause","gaussian","variable_gaussian","uniform","uniform_positive"]
        possible_noises = ["gaussian","variable_gaussian","uniform","laplace","absolute_gaussian","nn"]

        assert self.mechanism in possible_mechanisms, \
                f"mechanism doesn't exist. It has to be in {possible_mechanisms}"
        assert self.cause in possible_causes, \
                f"initial cause doesn't exist. It has to be in {possible_causes}"
        assert self.noise in possible_noises, \
                f"noise doesn't exist. It has to be in {possible_noises}"

    def _create_folder(self):
        """Create folders

        fname(str): path """

        try:
            os.mkdir(self.folder)
        except OSError:
            print(f"Cannot create the folder: {self.folder}")

    def _initialize_dag(self):
        if self.verbose:
            print(f'Sampling the DAG #{self.i_dataset}')
        self.generator = gen.DagGenerator(self.mechanism,
                                         noise=self.noise,
                                         noise_coeff=self.noise_coeff,
                                         cause=self.cause,
                                         npoints=self.nb_points,
                                         nodes=self.nb_nodes,
                                         prob_connection=self.prob_connection,
                                         rescale=self.rescale)

        self.generator.init_variables(self.folder, self.i_dataset+1)
        self.generator.save_dag(self.folder, self.i_dataset+1)

    def _save_data(self, i, data, regimes=None, mask=None):
        data_path = os.path.join(self.folder, f'data{i+1}.npy')
        np.save(data_path, data)

    def generate(self):
        if self.generator is None:
            self._initialize_dag()

        data, _ = self.generator.generate()

        if self.rescale:
            scaler = StandardScaler()
            scaler.fit_transform(data)
        self._save_data(self.i_dataset, data)


if __name__ == "__main__":
    # with open('configs/data_gen.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    all_configs = generate_all_configurations('configs/data_gen.yaml')
    tot = len(all_configs)
    vis = '=' * 10
    k = 1
    for config in all_configs:
        if config['initial_cause'] == 'variable_gaussian' or config['noise'] == 'variable_gaussian':
            if config['noise_coeff'] > 0.2:
                print('Skipping: noise_coeff too high for variable gaussian')
                continue

        print(f"{vis} {k}/{tot} {vis}")
        k += 1
        generator = dataset_generator(config)

        for i in range(config['nb_dag']):
            generator.i_dataset = i
            generator.generator = None

            print("Generating the observational data...")
            generator.generate()
            print("Data saved in ", generator.folder)

