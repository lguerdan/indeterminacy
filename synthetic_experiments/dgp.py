import numpy as np
from typing import Dict, Optional

from core import utils
import helpers


def sample_F(n_items, n_options, n_response_sets, rs_option_lookup, beta, order):
    '''
        Returns an n_items x n_options x n_response_sets array.
        Satisfies the constraint that all FC responses are members of the response set.
    '''
    F = np.zeros((n_items, n_options, n_response_sets))

    for i in range(n_items):
        for m in range(n_response_sets):
            theta_nz = (rs_option_lookup[:,m] == 1)
            ns = (rs_option_lookup[:,m] == 1).sum()
            F[i,theta_nz,m] = sample_resp_set_probs(ns, beta=beta, order=order)
            
    assert np.allclose(F.sum(axis=1).sum(axis=0), n_items), 'Invalid forward translation matrix.'
            
    return F

def sample_F_prime(F, theta, O):
    '''
        Returns an n_items x n_options x n_response_sets array.
        Satisfies the constraint that all FC responses are members of the response set.
    '''
    n_items, n_options, n_response_sets = F.shape
    F_prime = np.zeros((n_items, n_response_sets, n_options))

    for i in range(n_items):
        # Apply Bayes grid-wise to back out inverse translation probabilities.
        py = np.clip(np.expand_dims(O[i], axis=1), 1e-10, None) # Clip O away from zero.
        F_prime[i] = ((F[i]*theta[i])/py).T
        
    assert np.allclose(F.sum(axis=1).sum(axis=0), n_items), 'Invalid inverse translation matrix.'
            
    return F_prime


def sample_rating_distribution(
    n_items: int,
    n_options: int,
    n_response_sets: int,
    theta: Optional[np.ndarray] = None,
    F: Optional[np.ndarray] = None,
    beta: float = 0,
    order: str = 'increasing',
    skew: float = 0,
    error_rate: float = 0
) -> Dict[str, np.ndarray]:
    
    # Validate input constraints
    assert n_response_sets >= n_options, 'Must be fewer options than response sets.'
    
    # Construct lookup table mapping options to response sets
    # Shape: (n_options, n_response_sets)
    rs_option_lookup = utils.construct_rs_option_lookup(n_options)[:, 1:n_response_sets+1]
    
    # Generate response set probabilities if not provided
    if theta is None:
        theta = np.random.dirichlet(np.ones(n_response_sets), size=n_items)
        
    E = sample_forward_error_matrix(
        n_items=n_items,
        n_options=n_options,
        n_response_sets=n_response_sets,
        error_rate=error_rate,
        skew=skew
    )
    
    # Apply error to theta to get distribution over observed response sets
    theta_observed = np.zeros_like(theta)
    for i in range(n_items):
        theta_observed[i] = np.dot(E[i], theta[i])
    
    
    # Generate mapping matrices if not provided
    if F is None:
        F = sample_F(n_items, n_options, n_response_sets, rs_option_lookup, beta, order)
    
    # Compute final rating probabilities
    O = np.matmul(F, theta_observed[..., np.newaxis]).squeeze()

    F_prime = sample_F_prime(F, theta_observed, O)
    
    # Generate reverse error matrix
    E_prime = sample_reverse_error_matrix(E, theta)
    
    # Compute expected ratings for each item
    omega = np.zeros_like(O)
    omega_obs = np.zeros_like(O)
    for i in range(n_items):
        omega[i] = (rs_option_lookup * theta[i]).sum(axis=1)
        omega_obs[i] = (rs_option_lookup * theta_observed[i]).sum(axis=1)
    
    # Package all components of the data generating process
    return {
        'F': F,                                 # Forward mapping matrix
        'F_prime': F_prime,                     # Inverse mapping matrix
        'theta': theta,                         # Response set probabilities
        'theta_observed': theta_observed,       # Observed resopnse set probabilities
        'O': O,                                 # Final rating probabilities
        'omega_obs': omega_obs,                 # Multi-label rating vector (with error)
        'omega': omega,                         # Multi-label rating vector (no error)
        'rs_option_lookup': rs_option_lookup,   # Option-response set lookup table,
        'E': E,                                 # Forward error matrix
        'E_prime': E_prime                      # Inverse error matrix
    }


def sample_forward_error_matrix(
    n_items: int,
    n_options: int,
    n_response_sets: int,
    error_rate: float,
    skew: float,
) -> np.ndarray:

    E = np.zeros((n_items, n_response_sets, n_response_sets))

    rs_option_lookup = utils.construct_rs_option_lookup(n_options)[:, 1:n_response_sets+1]

    for i in range(n_items):

        # Sample entries of E that are not errors
        E_i = (1-error_rate) * np.identity(n_response_sets)

        # Sample error terms, parameterized by the skew and error rate
        # Negative skew moves probability mass away from response sets containing the zero option. 
        # Positive resopnse sets move probability mass towards resopnse sets with the zero option.
        error = helpers.sample_skewed_stochastic_matrix(
            n_response_sets-1,
            skew=skew,
            error_rate=error_rate)

        # Construct overall matrix
        off_diagonal = ~np.eye(E_i.shape[0],dtype=bool)
        E_i[off_diagonal]  = error.flatten()

        # Map to original option ordering
        contains_zero = (rs_option_lookup[0, :] == 1).astype(int)
        ordered_indices = np.argsort(~contains_zero)  # ~ inverts so True (1) comes first
        inverse_order = np.argsort(ordered_indices)
        E_i_r = E_i[inverse_order][:, inverse_order]

        E[i,:,:] = E_i_r

    return E

def sample_reverse_error_matrix(E: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Generate inverse error matrices using Bayes' rule.
    
    Parameters
    ----------
    E : np.ndarray
        Forward error matrices of shape (n_items, n_response_sets, n_response_sets)
    theta : np.ndarray
        Response set probabilities of shape (n_items, n_response_sets)
        
    Returns
    -------
    np.ndarray
        Inverse error matrices of shape (n_items, n_response_sets, n_response_sets)
        Each matrix E'[i] has entries E'[i,v*,v] = P(S*^h = s_v* | S^h = s_v)
    """
    n_items, n_response_sets, _ = E.shape
    E_inv = np.zeros_like(E)
    
    for i in range(n_items):
        # P(S^h = s_v) = sum_v* P(S^h = s_v | S*^h = s_v*) P(S*^h = s_v*)
        p_s = np.dot(E[i], theta[i])
        
        # Apply Bayes rule: P(S*^h | S^h) = P(S^h | S*^h) P(S*^h) / P(S^h)
        for v in range(n_response_sets):
            if p_s[v] > 0:  # Avoid division by zero
                E_inv[i, :, v] = E[i, v, :] * theta[i] / p_s[v]
    
    return E_inv

def sample_resp_set_probs(j: int, beta: float, order: str, alpha: float = 0.05) -> np.ndarray:
    """
    Generate a set of response probabilities with exponential decay and optional noise.
    
    Parameters
    ----------
    j : int
        Number of response options to generate probabilities for
    beta : float
        Decay rate for the exponential function. Higher values create steeper probability drops
    order : str
        Ordering of probabilities, must be either 'decreasing' or 'increasing'
        - 'decreasing': highest probability first
        - 'increasing': lowest probability first
    alpha : float, optional (default=0.05)
        Standard deviation of the Gaussian noise added to probabilities
        
    Returns
    -------
    np.ndarray
        Array of length j containing normalized probabilities that sum to 1
    
    Notes
    -----
    1. Base probabilities are generated using an exponential decay: exp(-beta * x)
    2. Gaussian noise with std=alpha is added to introduce randomness
    3. Negative values are clipped to 0 before normalization
    4. Final probabilities are normalized to sum to 1
    """
    # Generate base probabilities using exponential decay
    probs = np.exp(-beta * np.arange(j))
    
    # Reverse array if decreasing order is requested
    if order == 'decreasing':
        probs = probs[::-1]
    
    # Add Gaussian noise
    noise = np.random.normal(0, alpha, j)
    noisy_probs = probs + noise
    
    # Ensure probabilities are non-negative
    noisy_probs = np.maximum(noisy_probs, 0)
    
    # Normalize to create valid probability distribution
    return noisy_probs / noisy_probs.sum()


def sample_ratings(hrd, ratings_per_item, coverage=1):

    n_items, n_options = hrd['O'].shape
    ratings = np.zeros((n_items, n_options), dtype=int)
    
    for i in range(n_items):
        ratings[i] = np.random.multinomial(ratings_per_item, hrd['O'][i])
    
    return ratings
    

def estimate_rating_distribution(ratings, F, F_prime, E_prime, rs_option_lookup):

    n_items, n_options = ratings.shape
    n_response_sets = rs_option_lookup.shape[1]
    
    theta_hat_obs = np.zeros((n_items, n_response_sets))
    theta_hat = np.zeros((n_items, n_response_sets))
    omega_hat = np.zeros((n_items, n_options))
    omega_hat_obs = np.zeros((n_items, n_options))
    
    # Normalize ratings to get probabilities
    ratings_per_item = ratings.sum(axis=1)[0]
    O_hat = ratings / ratings_per_item

    # First recover theta_observed using F_prime
    for i in range(n_items):
        theta_hat_obs[i] = np.matmul(F_prime[i], O_hat[i])

    # Then recover theta using E_inv
    for i in range(n_items):
        theta_hat[i] = np.dot(E_prime[i], theta_hat_obs[i])

    # Compute multi-label vector via observed response sets
    for i in range(n_items):
        omega_hat_obs[i] = (rs_option_lookup * theta_hat_obs[i]).sum(axis=1)
        omega_hat[i] = (rs_option_lookup * theta_hat[i]).sum(axis=1)

    return {
        'F': F,      
        'F_prime': F_prime,
        'E_prime': E_prime,
        'theta': theta_hat,   
        'theta_observed': theta_hat_obs,
        'O': O_hat,
        'omega': omega_hat,                   
        'omega_obs': omega_hat_obs,      
        'rs_option_lookup': rs_option_lookup
    }

def sample_judge_distribution(n_items, n_options, n_response_sets, eps_min, eps_max, hrd_theta, beta_llm, order):

    if hrd_theta is None:
        theta = None
    else:
        eps = np.random.uniform(eps_min, eps_max)
        noise = np.random.normal(0, eps, hrd_theta.shape)
        theta = np.apply_along_axis(helpers.project_simplex, 1, hrd_theta+noise)
        
    return sample_rating_distribution(
        n_items=n_items,
        n_options=n_options,
        n_response_sets=n_response_sets,
        theta=theta,
        beta=beta_llm,
        order=order,
        skew=0,        # LLMs have no measurement error, only randomness over response sets.
        error_rate=0,  # LLMs have no measurement error, only randomness over response sets.
    )

