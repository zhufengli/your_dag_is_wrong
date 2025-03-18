import os
import random
import shutil
from datetime import datetime
import time
import yaml
import logging
import argparse
import pickle as pkl


from generation.generate_data import dataset_generator
import generation.dag_generator as gen
from utils import utils
from utils import data
import lagrangian
import probabilistic_dag
import brute_force
import non_linear

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation




def run_single_comparison(config, output_dir):

    # Create output directory
    out_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(hash(tuple(config.items())))
    out_dir = os.path.join(output_dir, out_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Set up logging COMMENT OUT LOCALLY WHEN RUNNING MULTIPLE RUNS
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.basicConfig(filename=os.path.join(out_dir, 'logs'),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Saving the yaml files
    logging.info('Saving config file to {out_dir}...')
    with open(os.path.join(out_dir, 'config.yaml'), 'w+') as ff:
        yaml.dump(config, ff)

    logging.info(f"Save all output to {out_dir}...")

    logging.info('Setting the seed for the data...')
    torch.manual_seed(int(config['seed_data']))
    random.seed(int(config['seed_data']))
    np.random.seed(int(config['seed_data']))

    # 
    # if config['graph_type'] == 'gnp':
    #     A, xy = data.adj_mat_generator(config)
    # elif config['graph_type'] == 'default':
    #     if config['xy_position'] == 'last':
    #         A = data.adj_mat_generator(config)
    #         xy = [config['nb_nodes']-2, config['nb_nodes']-1]
    #     elif config['xy_position'] == 'random':
    #         xy = data.select_random_nodes(config['nb_nodes'])
    #         A = data.adj_mat_generator(config, xy)
    #     elif config['xy_position'] == 'chosen':
    #         if config['xy'] is not None:
    #             xy = config['xy']
    #             A = data.adj_mat_generator(config, xy)
    #         else: 
    #             logging.error('Please specify the treatment and outcome nodes in data_gen.yaml')
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

    A_npy = np.load(os.path.join(data_path, f'DAG{1}.npy')).astype('float32')
    A = torch.from_numpy(A_npy)
    dat = np.load(os.path.join(data_path, f'data{1}.npy'))
    torch_data = torch.from_numpy(dat.astype('float32'))
    dataset = data.ObservationalDataset(dat)
    with open(os.path.join(data_path, f'mechanism{1}.pkl'), 'rb') as f:
        mechanism = pkl.load(f)
    
    # Delete the saved data
    #TODO Put it back when finish testing
    #shutil.rmtree(data_path)

    xy =utils.random_select_one_index(A, direct_cause=False) ###When set direct_cause to false, 

    x,y = xy
    if config['mechanism'] == 'linear':
        gt_total_causal_effect = utils.compute_linear_total_causal_effect(A, x, y, mechanism)
        calc_gt_causal_effect = utils.query(torch_data, A, xy, config['optimal'])
        ground_truth_effect_calc = calc_gt_causal_effect[0]
    else:
        cause_values = [0, 1]
        expected_y, gt_total_causal_effect = utils.compute_total_causal_effect(
                            A_npy, dat.T, mechanism, x, y, cause_values, dat.shape[0])
        mlp_model = utils.get_trained_mlp(A, xy, torch_data, config['optimal'])
        calc_gt_causal_effect = utils.query_non_linear(torch_data, A, xy, mlp_model, optimal=config['optimal'])
        ground_truth_effect_calc = calc_gt_causal_effect
    
    logging.info(f"Ground truth Causal effect is {gt_total_causal_effect}")
    logging.info(f"Calculated ground truth Causal effect is {calc_gt_causal_effect}")

    logging.info('Getting the sure and forbidden edges...')

    if config['sure_and_forbidden_how'] == 'pc':
        if config["test"] == 'kci':
            test = kci
        elif config["test"] == 'fisher':
            test = fisherz
        else: 
            raise ValueError(f"No suitable test for PC {config['test']}")

        results = []
        permutations_list = []
        for _ in range(config["n_permutations"]):
            permutation = np.random.permutation(config["nb_nodes"])
            permutations_list.append(permutation)
            
        for perm in permutations_list:
            
            #print(f"permutation: {perm}, test: {config['test']}")
            dat_array = np.array(dat) 
            permuted_data = dat_array[:, perm]
            cg = pc(permuted_data, 0.1, test, show_progress=False) 
            
            inverse_perm = utils.inverse_permutation(perm)
            original_order_graph = utils.apply_inverse_to_matrix(cg.G.graph, inverse_perm)
            results.append(original_order_graph)
                   
        pattern_matrix = utils.check_specific_patterns(results)
        
        sure_edges, forbidden_edges = utils.check_specific_patterns_reduced(results) 
        
    elif config['sure_and_forbidden_how'] == 'random':
        sure_edges, forbidden_edges = data.get_sure_and_forbidden(A, sure_prob=config['sure_prob'],\
                                                               forbidden_prob=config['forbid_prob'], xy=xy)
    
    if config['nb_nodes'] < 15 and config['save_all']:
        logging.info(f'Ground truth adjacency matrix is {A}')
        logging.info(f'Plotting the ground truth graph...')
        utils.plot_graph(A, xy, out_dir)
        logging.info(f'Plotting the uncertainty graph...')
        utils.plot_graph(A, xy, out_dir, sure_edges, forbidden_edges)

    dataset = data.ObservationalDataset(dat)
    data_loader = DataLoader(dataset, batch_size=config['nb_points'], shuffle=True)

    logging.info(f"Initialize dictionary for results...")
    results = {
        ## Data properties
        "num_nodes": config['nb_nodes'],
        "num_edges_ground_truth": A.sum(),
        "num_edges_sure": len(sure_edges),
        "num_edges_forbidden": len(forbidden_edges),
        "ground_truth_graph": A.numpy(),
        "ground_truth_effect_calc": ground_truth_effect_calc.item(),
        "total_ground_truth_effect": gt_total_causal_effect,
        "xy": xy,
        ## Brute force
        "brute_force_time": np.nan,
        "brute_force_bounds": np.nan,
        "brute_force_opt_graphs": np.nan,
        "brute_force_num_graphs": np.nan,
        ## Probabilisic DAG
        "prob_time": np.nan,
        "prob_all_vals_min": np.nan,
        "prob_all_vals_max": np.nan,
        "prob_bounds": np.nan,
        "prob_opt_graphs": np.nan,
        ## lagrangian DAG
        "lagrangian_time": np.nan,
        "lagrangian_all_vals_min": np.nan,
        "lagrangian_all_vals_max": np.nan,
        "lagrangian_bounds": np.nan,
        "lagrangian_opt_graphs": np.nan,
        }
    # Saving an empty frame to start with to be able to do the cumulative analysis consistently
    result_path = os.path.join(out_dir, "results_global.npz")
    #np.savez(result_path, **results)

    # Run the DP-DAG
    if config['run_dp_dag'] and config['mechanism'] == 'linear':
        logging.info('Setting the seed for the probabilistic dag...')
        torch.manual_seed(int(config['seed_prob']))
        #random.seed(int(config['seed_prob']))
        np.random.seed(int(config['seed_prob']))

        logging.info('Initializing the probabilistic DAG method...')
        max_model_DP = probabilistic_dag.ProbabilisticDAG(n_nodes=config['nb_nodes'], order_type=config['order_type'], lr=config['lr_prob'],\
                                        seed=config['seed_prob'], sure_edges=sure_edges, forbidden_edges=forbidden_edges)
        min_model_DP = probabilistic_dag.ProbabilisticDAG(n_nodes=config['nb_nodes'], order_type=config['order_type'], lr=config['lr_prob'],\
                                        seed=config['seed_prob'], sure_edges=sure_edges, forbidden_edges=forbidden_edges)
        logging.info('Training the probabilistic DAG method...')
        start = time.time()
        ate_max_prob, A_max_prob, valid_results_max_prob = probabilistic_dag.train(max_model_DP, data_loader, xy, maxmin='max', sure_edges=sure_edges,\
                                                                                    forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
        ate_min_prob, A_min_prob, valid_results_min_prob = probabilistic_dag.train(min_model_DP, data_loader, xy, maxmin='min', sure_edges=sure_edges,\
                                                                                    forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
        end = time.time()
        logging.info(f"Time taken for DP-DAG: {end-start} seconds")
        results["prob_time"] = end-start
        results["prob_all_vals_min"] = np.array(valid_results_min_prob)
        results["prob_all_vals_max"] = np.array(valid_results_max_prob)
        results["prob_bounds"] = np.array([ate_min_prob, ate_max_prob])
        try:
            results["prob_opt_graphs"] = np.array([A_min_prob.numpy(), A_max_prob.numpy()])
        except AttributeError:
            results["prob_opt_graphs"] = np.array([np.nan, np.nan])
        np.savez(result_path, **results)
        logging.info(f"DP-DAG max:{ate_max_prob}")
        logging.info(f"DP-DAG min:{ate_min_prob}")

    # Run the Lagrangian
    if config['run_lagrangian']:
        logging.info('Setting the seed for the Lagrangian...')
        torch.manual_seed(int(config['seed_lagrangian']))
        #random.seed(int(config['seed_lagrangian']))
        np.random.seed(int(config['seed_lagrangian']))
        if config['mechanism'] == 'linear':
            logging.info('Initializing the Lagrangian method...')
            max_model_L = lagrangian.DAGNet(config['nb_nodes'], temp=config['init_temp'], temp_decay=config['temp_decay'],\
                                        sure_edges=sure_edges, forbidden_edges=forbidden_edges)
            min_model_L = lagrangian.DAGNet(config['nb_nodes'], temp=config['init_temp'], temp_decay=config['temp_decay'],\
                                        sure_edges=sure_edges, forbidden_edges=forbidden_edges)
            logging.info('Training the Lagrangian method...')
            start = time.time()
            ate_max_lag, A_max_lag, valid_results_max_lag = lagrangian.train(max_model_L, data_loader, xy, maxmin='max', sure_edges=sure_edges, \
                                                                            forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
            ate_min_lag, A_min_lag, valid_results_min_lag = lagrangian.train(min_model_L, data_loader, xy, maxmin='min', sure_edges=sure_edges, \
                                                                            forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
            end = time.time()
            logging.info(f"Time taken for Lagrangian: {end-start} seconds")
        else:
            logging.info('Initializing the Lagrangian method...')
            max_model_L = non_linear.DAGNet(config['nb_nodes'], temp=config['init_temp'], temp_decay=config['temp_decay'],\
                                        sure_edges=sure_edges, forbidden_edges=forbidden_edges)
            min_model_L = non_linear.DAGNet(config['nb_nodes'], temp=config['init_temp'], temp_decay=config['temp_decay'],\
                                        sure_edges=sure_edges, forbidden_edges=forbidden_edges)
            logging.info('Training the Lagrangian method...')
            start = time.time()
            ate_max_lag, A_max_lag, valid_results_max_lag = non_linear.train(max_model_L, data_loader, xy, maxmin='max', sure_edges=sure_edges, \
                                                                            forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
            ate_min_lag, A_min_lag, valid_results_min_lag = non_linear.train(min_model_L, data_loader, xy, maxmin='min', sure_edges=sure_edges, \
                                                                            forbidden_edges=forbidden_edges, _config=config, out_dir=out_dir)
            end = time.time()
            logging.info(f"Time taken for Lagrangian: {end-start} seconds")
        logging.info(f"lagrangian max is:{ate_max_lag}")
        logging.info(f"lagrangian min is:{ate_min_lag}")
        results["lagrangian_time"] = end-start
        results["lagrangian_all_vals_min"] = np.array(valid_results_min_lag)
        results["lagrangian_all_vals_max"] = np.array(valid_results_max_lag)
        results["lagrangian_bounds"] = np.array([ate_min_lag, ate_max_lag])
        try:
            results["lagrangian_opt_graphs"] = np.array([A_min_lag.detach().numpy(), A_max_lag.detach().numpy()])
        except AttributeError:
            results["lagrangian_opt_graphs"] = np.array([np.nan, np.nan])
        np.savez(result_path, **results)

    # Run Brute Force
    if config['run_brute_force'] and (config['nb_nodes'] < 10):
        logging.info('Running the Brute Force method...')
        start = time.time()
        model = brute_force.Brute_Force(config['nb_nodes'], dat, xy, sure_edges, forbidden_edges, A, config['optimal'])
        logging.info(f'number of all possible graphs: {len(model.all_queries)}')

        logging.info(f"Brute force: max query value is: {model.curr_max}")
        logging.info(f"Brute force: min query value is: {model.curr_min}")
        
        logging.info(f"Time taken for Brute Force: {end-start} seconds")
        results["brute_force_time"] = end-start
        results["brute_force_bounds"] = np.array([model.curr_min.item(), model.curr_max.item()])
        results["brute_force_opt_graphs"] = np.array([model.A_min, model.A_max])
        results["brute_force_num_graphs"] = len(model.all_queries)
        np.savez(result_path, **results)
    else:
        logging.info('Skipping Brute Force bounds due to too many variables...')

        np.savez(result_path, **results)

if __name__ == "__main__":
    # Load yaml file as configuration
    parser = argparse.ArgumentParser(description="Run a single comparison with the specified configuration.")
    parser.add_argument('--config', type=str, default='configs/comparison.yaml', \
                        help='Path to the configuration file. Default is default_config.yaml')
    parser.add_argument('--jobid', type=str, default='NA', \
                        help='jobid if this is a slurm job, helps to match to the error')
    args = parser.parse_args()

    # Load the configuration from the specified file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config['jobid'] = args.jobid

    output_dir = os.path.abspath(config['output_dir'])

    run_single_comparison(config, output_dir)