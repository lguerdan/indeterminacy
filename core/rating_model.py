import numpy as np

from core.utils import construct_rs_option_lookup
from config import tasks

class RatingModel:
    def __init__(self):
        pass

    def construct_judge_rating_distribution(self, task_name, task_config, resp_table):

        # Insert extra option and response set to handle formatting errors
        n_options = task_config['n_options'] + 1
        n_response_sets = task_config['n_response_sets'] + 1

        # n_items x n_response sets
        theta = resp_table.sum(axis=1).mean(axis=2)

        # n_items x n_response sets
        O = resp_table.sum(axis=2).mean(axis=2)

        rs_option_lookup = construct_rs_option_lookup(n_options)[:, 1:n_response_sets+1]

        mu = np.matmul(theta, rs_option_lookup.T)

        # Compute delta
        counts = np.sum(resp_table, axis=3)
        response_totals = np.sum(counts, axis=1, keepdims=True)
        F = np.divide(counts, 
                    response_totals, 
                    out=np.zeros_like(counts, dtype=float),
                    where=response_totals != 0)


        # Compute delta reverse
        counts = np.sum(resp_table, axis=3) 
        option_totals = np.sum(counts, axis=2, keepdims=True)

        F_prime = np.divide(counts, 
                        option_totals, 
                        out=np.zeros_like(counts, dtype=float),
                        where=option_totals != 0)

        return {
            'F': F,        
            'F_prime': F_prime,
            'theta': theta,
            'O': O,            
            'omega': mu,
            'omega_obs': mu, 
            'rs_option_lookup': rs_option_lookup
        }

    def construct_human_rating_distribution(self, ratings, task_name, task_config, beta):

        n_items = ratings.shape[0]
        # Add extra option and response set to handle formatting errors
        n_options = task_config['n_options'] + 1
        n_response_sets = task_config['n_response_sets'] + 1
        
        O = tasks.get_task_forced_choice_distribution(
                task_name,
                ratings,
                n_items,
                n_options
            )
            
        F_prime = tasks.get_task_reverse_fc_translation(
            task_name, 
            n_items, 
            n_options, 
            n_response_sets,
            beta=beta
        )

        # Compute theta (distribution over response sets)
        theta = (O[:, :, None] * F_prime).sum(axis=1)

        # Compute response set table using lookup table
        rs_option_lookup = construct_rs_option_lookup(n_options)[:, 1:n_response_sets+1]
        mu = np.matmul(theta, rs_option_lookup.T)
        
        # Apply Bayes' to get forward F matrix
        numerator = F_prime * theta[:, None, :]
        denominator = O[:, :, None]
        epsilon = 1e-10
        
        # Apply Bayes' to get forward F matrix
        # F[i,k,v] = P(O=k|S=v) = P(S=v|O=k) * P(O=k) / P(S=v)
        F = np.zeros((n_items, n_options, n_response_sets))
        epsilon = 1e-10
        
        for i in range(n_items):
            for k in range(n_options):
                for v in range(n_response_sets):
                    if theta[i, v] > epsilon:  # Avoid division by zero
                        F[i, k, v] = F_prime[i, k, v] * O[i, k] / theta[i, v]
        
        return {
            'F': F,        
            'F_prime': F_prime,
            'theta': theta,
            'O': O,            
            'omega': mu,
            'omega_obs': mu, # Assume no rater error.
            'rs_option_lookup': rs_option_lookup
        }



    def sample_and_estimate_dgp(self, hrd, n_samples, estimation_approach='grand_mean'):
        """
        Sample data and estimate DGP components using specified approach.
        
        Parameters
        ----------
        hrd : dict
            Human rating distribution containing true values
        n_samples : int
            Number of samples to draw
        estimation_approach : str
            Either 'grand_mean' or 'item_specific'
        
        Returns
        -------
        dict
            Dictionary containing all estimated DGP components
        """
        # Estimate F and F_prime using specified approach
        F_hat, F_prime_hat = self.estimate_F_matrices(hrd, n_samples, estimation_approach)
        
        # Use true O
        O = hrd['O']
        
        # Estimate theta using matrix multiplication
        theta_hat = self.estimate_theta_via_matrix_mult(F_prime_hat, O)
        
        # Get response set lookup table
        rs_option_lookup = hrd['rs_option_lookup']
        
        # Compute item-specific mu (omega) using item-specific theta
        mu_hat = np.matmul(theta_hat, rs_option_lookup.T)
        
        return {
            'F': F_hat,        
            'F_prime': F_prime_hat,
            'theta': theta_hat,
            'O': O,            
            'omega': mu_hat,
            'omega_obs': mu_hat,  # Assume no rater error
            'rs_option_lookup': rs_option_lookup
        }


    def estimate_F_matrices(self, hrd, n_samples, estimation_approach='grand_mean'):
        """
        Estimate F and F_prime matrices using either grand mean or item-specific approach.
        
        Parameters
        ----------
        hrd : dict
            Human rating distribution containing true values
        n_samples : int
            Number of samples to draw
        estimation_approach : str
            Either 'grand_mean' or 'item_specific'
        
        Returns
        -------
        F_hat : np.ndarray
            Estimated F matrix
        F_prime_hat : np.ndarray
            Estimated F_prime matrix
        """
        n_items, n_options, n_response_sets = hrd['F'].shape
        
        # Sample data
        joint_counts = np.zeros((n_items, n_response_sets, n_options))
        option_counts = np.zeros((n_items, n_options))
        
        for i in range(n_items):
            for _ in range(n_samples):
                # Sample response set from true theta
                rs_idx = np.random.choice(n_response_sets, p=hrd['theta'][i])
                
                # Sample option given response set using true F
                option_probs = hrd['F'][i, :, rs_idx]
                option_idx = np.random.choice(n_options, p=option_probs)
                
                # Update counts
                joint_counts[i, rs_idx, option_idx] += 1
                option_counts[i, option_idx] += 1
        
        if estimation_approach == 'grand_mean':
            # Pool counts across all items
            joint_counts_global = joint_counts.sum(axis=0)
            option_counts_global = option_counts.sum(axis=0)
            rs_counts_global = joint_counts_global.sum(axis=1)
            
            # Estimate grand mean F_prime
            F_prime_grand = np.zeros((n_options, n_response_sets))
            for k in range(n_options):
                if option_counts_global[k] > 0:
                    F_prime_grand[k, :] = joint_counts_global[:, k] / option_counts_global[k]
            
            # Estimate grand mean F
            F_grand = np.zeros((n_options, n_response_sets))
            for v in range(n_response_sets):
                if rs_counts_global[v] > 0:
                    F_grand[:, v] = joint_counts_global[v, :] / rs_counts_global[v]
            
            # Broadcast to all items
            F_hat = np.broadcast_to(F_grand[np.newaxis, :, :], (n_items, n_options, n_response_sets)).copy()
            F_prime_hat = np.broadcast_to(F_prime_grand[np.newaxis, :, :], (n_items, n_options, n_response_sets)).copy()
            
        else:  # item_specific
            # Estimate F_prime for each item
            F_prime_hat = np.zeros((n_items, n_options, n_response_sets))
            for i in range(n_items):
                for k in range(n_options):
                    if option_counts[i, k] > 0:
                        F_prime_hat[i, k, :] = joint_counts[i, :, k] / option_counts[i, k]
            
            # Estimate F for each item
            F_hat = np.zeros((n_items, n_options, n_response_sets))
            rs_counts = joint_counts.sum(axis=2)
            
            for i in range(n_items):
                for v in range(n_response_sets):
                    if rs_counts[i, v] > 0:
                        F_hat[i, :, v] = joint_counts[i, v, :] / rs_counts[i, v]
        
        return F_hat, F_prime_hat

    def estimate_theta_via_matrix_mult(self, F_prime_hat, O):
        """
        Estimate theta using matrix multiplication: theta = F_prime^T @ O
        Note: We need to transpose F_prime to get the right dimensions
        
        Parameters
        ----------
        F_prime_hat : np.ndarray
            Estimated F_prime of shape (n_items, n_options, n_response_sets)
        O : np.ndarray
            Empirical forced choice distribution of shape (n_items, n_options)
        
        Returns
        -------
        theta_hat : np.ndarray
            Estimated theta of shape (n_items, n_response_sets)
        """
        n_items = O.shape[0]
        n_response_sets = F_prime_hat.shape[2]
        theta_hat = np.zeros((n_items, n_response_sets))
        
        for i in range(n_items):
            # Matrix multiplication: theta[i] = F_prime[i]^T @ O[i]
            # F_prime[i] has shape (n_options, n_response_sets)
            # O[i] has shape (n_options,)
            # Result has shape (n_response_sets,)
            theta_hat[i] = np.dot(F_prime_hat[i].T, O[i])
        
        return theta_hat



