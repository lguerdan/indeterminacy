import numpy as np
from itertools import combinations

def construct_rs_option_lookup(n_options):
    # Generate all subsets of rows
    all_rows = list(range(n_options))
    subset_combinations = []
    for subset_size in range(0, n_options + 1):  # Include empty subset
        subset_combinations.extend(combinations(all_rows, subset_size))
    
    # Convert subsets into binary rows
    binary_subsets = np.zeros((len(subset_combinations), n_options), dtype=int)
    for i, subset in enumerate(subset_combinations):
        binary_subsets[i, list(subset)] = 1
    
    return binary_subsets.T
