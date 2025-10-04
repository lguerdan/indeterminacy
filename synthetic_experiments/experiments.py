
import sys

sys.path.append('..')

import pandas as pd
import numpy as np
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Union, TypeVar, Dict, List
import copy
from scipy import stats
import random
from core import metrics
from core import utils
import matplotlib.pyplot as plt

from synthetic_experiments import dgp

def run_error_perturbation_experiment(cfg) -> pd.DataFrame:

    results = []
    
    for config in cfg.rating_process_configs:
        
        for error_rate in cfg.error_rates:

            human_order = config['human_order']
            human_beta = config['human_beta']
            judge_order = config['judge_order']
            judge_beta = config['judge_beta']
            skew = config['skew']

            print(f'running: human_order {human_order}, judge_order {judge_order}, human_beta {human_beta}, judge_beta {judge_beta}, error_rate {error_rate}, skew {skew}')
            
            for n_items, n_options, n_response_sets in cfg.resp_set_configs:
                print(f'task configuration: n_items={n_items}, n_options={n_options}, n_response_sets={n_response_sets}')
                
                for trial in range(cfg.n_trials):


                    # For each trial, sample the human rating distribution
                    hrd = dgp.sample_rating_distribution(
                        n_items=n_items,
                        n_options=n_options,
                        n_response_sets=n_response_sets,
                        beta=human_beta,
                        order=human_order,
                        skew=skew,
                        error_rate=error_rate
                    )
                    
                    error_corrupted_metrics = []
                    error_free_metrics = []
                    for _ in range(cfg.n_judges):

                        order = random.choice(['increasing', 'decreasing']) if cfg.rand_judge_order else judge_order
                        
                        # Sample a judge rating distribution
                        lrd = dgp.sample_judge_distribution(
                            n_items, n_options, n_response_sets,
                            cfg.judge_eps_min,
                            cfg.judge_eps_max,
                            hrd['theta'],
                            judge_beta,
                            order
                        )

                        # Compute metrics corrputed by rater error
                        metrics_obs = metrics.compute_performance_metrics(hrd, lrd, ml_distribution='obs', classification_j=[0], classification_tau=.5)

                        # Compute version of the forced choice distribution that is uncorrputed by rater error. 
                        hrd_star = hrd.copy()
                        hrd_star['O'] = np.matmul(hrd_star['F'], hrd_star['theta'][..., np.newaxis]).squeeze()
                        metrics_star = metrics.compute_performance_metrics(hrd_star, lrd, ml_distribution='star', classification_j=[0], classification_tau=.5)


                        error_corrupted_metrics.append(metrics_obs)
                        error_free_metrics.append(metrics_star)
                
                edf = pd.DataFrame(error_corrupted_metrics)
                df = pd.DataFrame(error_free_metrics)

                result = {}
                for metric in metrics_star.keys():
                    result[metric] = stats.spearmanr(edf[metric], df[metric]).statistic
                    
                result = {**result, **config}
                result['error_rate'] = error_rate
                result['trial'] = trial
                result['n_items'] = n_items
                result['n_options'] = n_options 
                result['n_response_sets'] = n_response_sets
                
                results.append(result)
                
    return pd.DataFrame(results)
                

def run_fs_judge_selection_exp(cfg) -> pd.DataFrame:
    """
    Run judge selection experiment with given configuration.
    
    Parameters
    ----------
    setup : ExperimentSetup
        Complete configuration for the experiment
        
    Returns
    -------
    pd.DataFrame
        Results of the experiment
    """

    # Initialize results dictionary with predefined columns
    results = {col: [] for col in [
        'target_measure', 'eval_measure', 'target_measure_category',
        'value', 'tau', 'rpi', 'trial', 'n_items', 'n_options', 'n_response_sets', 
        'human_order', 'judge_order', 'human_beta', 'judge_beta', 
        'skew', 'error_rate'
    ]}

    for config in cfg.rating_process_configs:

        human_order = config['human_order']
        human_beta = config['human_beta']
        judge_order = config['judge_order']
        judge_beta = config['judge_beta']
        skew = config['skew']
        error_rate = config['error_rate']

        print(f'running: human_order {human_order}, judge_order {judge_order}, human_beta {human_beta}, judge_beta {judge_beta}, error_rate {error_rate}, skew {skew}')

        for n_items, n_options, n_response_sets in cfg.resp_set_configs:

            print(f'task configuration: n_items={n_items}, n_options={n_options}, n_response_sets={n_response_sets}')

            for trial in range(cfg.n_trials):
                
                # For each trial, sample the human rating distribution
                hrd = dgp.sample_rating_distribution(
                    n_items=n_items,
                    n_options=n_options,
                    n_response_sets=n_response_sets,
                    beta=human_beta,
                    order=human_order,
                    skew=skew,
                    error_rate=error_rate
                )
            
                for rpi in cfg.n_ratings_per_item:
                    judge_results = []

                    # Sample a finite sample corpus
                    ratings = dgp.sample_ratings(hrd, rpi)

                    # Estimate human rating distribution 
                    hrd_hat = dgp.estimate_rating_distribution(
                        ratings, hrd['F'], hrd['F_prime'], hrd['E_prime'], hrd['rs_option_lookup']
                    )

                    for _ in range(cfg.n_judges):

                        # Sample a judge rating distribution
                        lrd = dgp.sample_judge_distribution(
                            n_items, n_options, n_response_sets,
                            cfg.judge_eps_min,
                            cfg.judge_eps_max,
                            hrd['theta'],
                            judge_beta,
                            judge_order
                        )

                        for tau in cfg.taus:

                            perf = metrics.compute_performance_metrics(
                                hrd_hat, lrd, classification_tau=tau, h_rs_downstream=hrd
                            )
                            perf['tau'] = tau
                            judge_results.append(perf)

                    jrdf = pd.DataFrame(judge_results)

                    # Update results
                    _update_results(
                        results, jrdf, cfg.taus, rpi, trial, n_items, n_options, n_response_sets, human_order, judge_order,
                        human_beta, judge_beta, skew, error_rate
                    )
            
    return pd.DataFrame(results)

def _update_results(results: dict, jrdf: pd.DataFrame, taus: np.ndarray, 
                   rpi: int, trial: int, n_items: int, n_options: int, n_response_sets: int, 
                   human_order: str, judge_order: str, human_beta: float,
                   judge_beta: float, skew: float, error_rate: float) -> None:
    """Helper function to update results dictionary."""
    for tm in metrics.TARGET_METRICS + metrics.DOWNSTREAM_METRICS:
        for dm in metrics.DOWNSTREAM_METRICS:
            for tau in taus:
                results['target_measure'].append(tm)
                results['target_measure_category'].append(
                    metrics.METRIC_TABLE[tm]['cat']
                )
                results['eval_measure'].append(dm)
                results['value'].append(
                    jrdf[jrdf['tau'] == tau]
                    .sort_values(by=tm)
                    .iloc[metrics.METRIC_TABLE[tm]['opt_ind']][dm]
                )
                results['tau'].append(tau)
                results['rpi'].append(rpi)
                results['trial'].append(trial)
                results['n_items'].append(n_items)
                results['n_options'].append(n_options)
                results['n_response_sets'].append(n_response_sets)
                results['human_order'].append(human_order)
                results['judge_order'].append(judge_order)
                results['human_beta'].append(human_beta)
                results['judge_beta'].append(judge_beta)
                results['skew'].append(skew)
                results['error_rate'].append(error_rate)

def run_metric_correlation_experiment(cfg):
    
    judge_results = []
    gtrial = 0

    for tau in cfg.taus:

        for constrain_judge in cfg.constrain_judge:

            for config in cfg.selection_configs.values():

                human_order = config['human_order']
                human_beta = config['human_beta']
                judge_order = config['judge_order']
                judge_beta = config['judge_beta']

                print(f'running tau: {tau} human_order: {human_order}, judge_order: {judge_order}, human_beta: {human_beta}, judge_beta: {judge_beta}')

                for n_items, n_options, n_response_sets in cfg.resp_set_configs:

                    for trial in range(cfg.n_trials):

                        hrd = dgp.sample_rating_distribution(
                            n_items=n_items,
                            n_options=n_options,
                            n_response_sets=n_response_sets,
                            beta=human_beta,
                            order=human_order,
                            error_rate=0,
                            skew=0
                        )

                        for _ in range(cfg.n_judges):
                            lrd = dgp.sample_judge_distribution(
                                n_items, n_options, n_response_sets,
                                cfg.judge_eps_min,
                                cfg.judge_eps_max,
                                None if not constrain_judge else hrd['theta'],
                                judge_beta,
                                judge_order
                            )

                            perf = metrics.compute_performance_metrics(
                                        hrd, lrd, classification_tau=tau
                                    )

                            perf['n_items'] = n_items
                            perf['n_options'] = n_options
                            perf['n_response_sets'] = n_response_sets
                            perf['human_order'] = human_order
                            perf['judge_order'] = judge_order
                            perf['human_beta'] = human_beta
                            perf['judge_beta'] = judge_beta
                            perf['tau'] = tau
                            perf['trial'] = gtrial
                            perf['constrain_judge'] = constrain_judge

                            judge_results.append(perf)
                        
                        gtrial += 1

    return pd.DataFrame(judge_results)



def finite_sample_human_rating_target_metric_exp(cfg):


    results = []

    results = {col: [] for col in [
        'consistency', 'bias', 'bias_mae',
        'bias_mse', 'tau', 'rpi', 'trial', 'n_items','n_options','n_response_sets'
    ]}


    for n_items, n_options, n_response_sets in cfg.resp_set_configs:

        for trial in range(cfg.n_trials):

            hrd = dgp.sample_rating_distribution(
                n_items=n_items,
                n_options=n_options,
                n_response_sets=n_response_sets,
                beta=0,
                order='increasing'
            )

            for rpi in cfg.rpis:

                # Sample a finite sample corpus
                ratings = dgp.sample_ratings(hrd, rpi)

                for tau in cfg.taus:

                    # Estimate human rating distribution
                    hrd_hat = dgp.estimate_rating_distribution(
                        ratings, hrd['F'], hrd['F_prime'], hrd['E_prime'], hrd['rs_option_lookup']
                    )

                    consistency, bias_mae, bias_mse, bias = metrics.compute_classification_metrics(
                        hrd_hat['omega'], hrd['omega'], j=[0], t=tau)

                    results['consistency'].append(consistency)
                    results['bias_mae'].append(bias_mae)
                    results['bias_mse'].append(bias_mse)
                    results['bias'].append(bias)
                    results['tau'].append(tau)
                    results['rpi'].append(rpi)
                    results['trial'].append(trial)
                    results['n_items'].append(n_items)
                    results['n_options'].append(n_options)
                    results['n_response_sets'].append(n_response_sets)


    return pd.DataFrame(results)


def elicitation_error_prevalence_exp(cfg):
    
    results = {col: [] for col in [
        'consistency', 'bias_mae', 'bias_mse', 'bias',  'n_items', 'error_rate', 'skew',
        'n_options', 'n_response_sets', 'trial', 'order', 'beta', 'tau'
    ]}
    
    for config in cfg.rating_process_configs:
        
        for error_rate in cfg.error_rates:
        
            human_order = config['human_order']
            human_beta = config['human_beta']
            skew = config['skew']
            error_rate = error_rate

            for (n_items, n_options, n_response_sets) in cfg.resp_set_configs:

                for trial in range(cfg.n_trials):

                    hrd = dgp.sample_rating_distribution(
                            n_items=n_items,
                            n_options=n_options,
                            n_response_sets=n_response_sets,
                            beta=human_beta,
                            order=human_order,
                            error_rate=error_rate,
                            skew=skew
                        )

                    for tau in cfg.taus:
                        consistency, bias_mae, bias_mse, bias = metrics.compute_classification_metrics(hrd['O'],hrd['omega'], j=[0], t=tau)

                        results['consistency'].append(consistency)
                        results['bias_mae'].append(bias_mae)
                        results['bias_mse'].append(bias_mse)
                        results['bias'].append(bias)
                        results['trial'].append(trial)
                        results['n_items'].append(n_items)
                        results['n_options'].append(n_options)
                        results['n_response_sets'].append(n_response_sets)
                        results['order'].append(human_order)
                        results['beta'].append(human_beta)
                        results['error_rate'].append(error_rate)
                        results['skew'].append(skew)
                        results['tau'].append(tau)


    return pd.DataFrame(results)
