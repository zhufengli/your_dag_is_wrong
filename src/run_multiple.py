import os
import time
import yaml
from datetime import datetime
import functools, itertools, operator

from run_single_comparison import run_single_comparison
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    
# Function to run comparisons for each configuration
def run_multiple_comparisons():

    with open('configs/comparison.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Define the ranges for the configurations
    config_ranges = {
        "sure_prob": [0.1, 0.2, 0.4],
        "forbid_prob": [0.1, 0.2, 0.4],
        "n_var": [6, 7, 10],#, 20, 50, 100, 200, 400, 1000],
        #"noise": [0.3, 0.5, 0.7],
        #"seed_data": [17*i for i in range(2)],
        #"xy_position": ['random', 'last']
    }

    # Generate all combinations of configurations
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"multiple_{timestamp}"
    output_dir = os.path.join(os.path.abspath(config['output_dir']), dir_name)
    os.makedirs(output_dir)

    # Set up logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.basicConfig(filename=os.path.join(output_dir, 'logs'),
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

    keys, values = zip(*config_ranges.items())
    total_configs = functools.reduce(operator.mul, map(len, values), 1)
    vis = "=" * 10
    idx = 1
    for value_combination in itertools.product(*values):
        config_variation = dict(zip(keys, value_combination))
    
        # Update the data configuration with the new values
        config.update(config_variation)
        
        # Call the existing run_single_comparison function
        logging.info(f"{vis} {idx}/{total_configs} {vis}")
        idx += 1
        run_single_comparison(config, output_dir)


# Example usage
if __name__ == "__main__":
    run_multiple_comparisons()