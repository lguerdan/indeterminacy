import json, sys, argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from dataclasses import asdict

from experiments import experiments as exp
from experiments.config import SelectionExperimentConfig

def save_experiment(results: pd.DataFrame, config: SelectionExperimentConfig, tag: str) -> None:
    """
    Save experiment results and configuration to disk.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing experiment results
    config : SelectionExperimentConfig
        Configuration dataclass used for the experiment
    tag : str
        Unique identifier for the experiment run
        
    Notes
    -----
    Saves:
        - results/{tag}.csv: Results DataFrame
        - results/{tag}.json: Experiment configuration
    """
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save results as CSV
    results_path = results_dir / f"{tag}.csv"
    results.to_csv(results_path, index=False)
    
    # Convert config dataclass to dictionary and save as JSON
    config_dict = asdict(config)
    config_path = results_dir / f"{tag}.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    print(f"Configuration saved to: {config_path}")

def run_experiment(tag):

    cfg = SelectionExperimentConfig(
        n_judges=50,
        n_trials=5,
        resp_set_configs=[(100, 2, 3), (100, 3, 5), (100, 3, 7), (100, 4, 7), (100, 4, 8), (100, 4, 10), 
                     (100, 4, 12), (100, 5, 8), (100, 5, 12), (100, 5, 20), (100, 3, 3), (100, 4, 4), 
                     (100, 5, 5), (100, 6, 6)],
        taus=[0.3, 0.5, 0.7],
        n_ratings_per_item=[1, 3, 5,10, 20, 50, 100],
        judge_eps_min = 0.02,
        judge_eps_max = 0.5,
        constrain_judge = [True],
        rating_process_configs = [{
                'human_order': 'decreasing',
                'judge_order': 'increasing', 
                'human_beta': 0,
                'judge_beta': 0, 
                'error_rate': 0,
                'skew': 0,
            }, {
                'human_order': 'decreasing',
                'judge_order': 'increasing', 
                'human_beta': 1,
                'judge_beta': 2,
                'error_rate': 0,
                'skew': 0
            }, {
                'human_order': 'increasing',
                'judge_order': 'decreasing', 
                'human_beta': 2,
                'judge_beta': 1,
                'error_rate': 0,
                'skew': 0
            }, {
                'human_order': 'decreasing',
                'judge_order': 'increasing', 
                'human_beta': 1,
                'judge_beta': 2,
                'error_rate': 0.3,
                'skew': 0
            }, {
                'human_order': 'increasing',
                'judge_order': 'decreasing', 
                'human_beta': 2,
                'judge_beta': 1,
                'error_rate': 0.3,
                'skew': 0
        }, {
                'human_order': 'decreasing',
                'judge_order': 'increasing', 
                'human_beta': 1,
                'judge_beta': 2,
                'error_rate': 0.3,
                'skew': 1
            }, {
                'human_order': 'increasing',
                'judge_order': 'decreasing', 
                'human_beta': 2,
                'judge_beta': 1,
                'error_rate': 0.3,
                'skew': 1
        }]
    )

    # Run experiment
    results = exp.run_fs_judge_selection_exp(cfg)
  
    save_experiment(results, cfg, tag)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, help='Experiment tag')
    args = parser.parse_args()
    
    run_experiment(tag=args.tag)