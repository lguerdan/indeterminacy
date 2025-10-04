from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np


@dataclass
class EstimationExperimentConfig:
    """Configuration for finite sample judge selection experiment."""
    # Model parameters
    n_trials: int
    resp_set_configs: list
    taus: list
    rpis: list


@dataclass
class ElicitationPrevalenceExperimentConfig:
    """Configuration for finite sample judge selection experiment."""
    # Model parameters
    n_trials: int
    resp_set_configs: list
    taus: list
    selection_configs: dict

@dataclass
class CorrelationExperimentConfig:
    """Configuration for finite sample judge selection experiment."""
    # Model parameters
    n_judges: int
    n_trials: int
    resp_set_configs: list
    taus: list
    selection_configs: list
    judge_eps_min: float
    judge_eps_max: float
    constrain_judge: list

@dataclass
class SelectionExperimentConfig:
    """Configuration for finite sample judge selection experiment."""
    # Model parameters
    
    rating_process_configs: dict
    resp_set_configs: list
    taus: list
    n_ratings_per_item: list
        
    n_judges: int
    n_trials: int
    judge_eps_min: float
    judge_eps_max: float
    constrain_judge: list



@dataclass
class ErrorPerturbationExperimentConfig:
    """Configuration for finite sample judge selection experiment."""
    # Model parameters
    
    rating_process_configs: dict
    error_rates: list
    resp_set_configs: list
        
    n_judges: int
    n_trials: int
    judge_eps_min: float
    judge_eps_max: float

@dataclass
class ElicitationErrorPrevalenceExperimentConfig:
    rating_process_configs: dict
    error_rates: list
    resp_set_configs: list
    n_trials: int
    taus: float