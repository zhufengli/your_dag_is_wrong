# Code for Your Assumed DAG is Wrong and Here's How To Deal With It

Official code repository for the paper **"Your Assumed DAG is Wrong and Here's How To Deal With It"**, published in CLeaR (Causal Learning and Reasoning), 2025.  

We compute the maximum and minium values of a causal query that require as input an assumed ground truth causal graph in the setting where there is uncertainty about which causal graph is the correct one by specifying sure and forbidden edges. 
The maximum and minimum over the set of "plausible" or "allowed" causal graphs provide the range in which the true value could lie.

## Code Description
The main code is in the folder `src`. There are three methods currently implemented for the linear case
- `lagrangian.py`: Uses the lagrangian optimization-based method with a continuous acyclicity constraint
- `brute_force.py`: Just a brute force search over all methods (A baseline methods)
- `probabilistic_dag.py`: The DP-DAG method which is guaranteed to search only over DAGs
- `real_world_dataset.py`: Example script of lagrangian method running on IHDP dataset.

The `run_single_comparison.py` runs all of the three above methods for the linear case with `brute_force` running last. The results are stored in an .npz file for later analysis.

For the non-linear case, we only have `non_linear.py`, which uses the lagrangian-based method.

The script to run jobs on the cluster is `cluster_multiple.py`. Just edit that file, and then run `sbatch run.sh` to launch the script. Make sure to be in the correct conda environment, etc.

## Installation & Dependencies
To run the code, first install the required dependencies:

```bash
pip install -r requirements.txt
```

We recommend using a virtual environment:

```bash
python -m venv env
source env/bin/activate  
pip install -r requirements.txt
```

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@misc{padh2025assumeddagwrongheres,
      title={Your Assumed DAG is Wrong and Here's How To Deal With It}, 
      author={Kirtan Padh and Zhufeng Li and Cecilia Casolo and Niki Kilbertus},
      year={2025},
      eprint={2502.17030},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2502.17030}, 
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

