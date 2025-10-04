import numpy as np


def project_simplex(v):
    """ 
        Efficiently project a vector onto the simplex.
        See: https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    """
    n = len(v)
    sorted_v = np.sort(v)[::-1]
    cssv = np.cumsum(sorted_v) - 1
    ind = np.arange(1, n+1)
    cond = sorted_v - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(v - theta, 0)


def sample_skewed_stochastic_matrix(n: int, skew: float = 0.0, error_rate: float = 0.0) -> np.ndarray:
    
    n_rows, n_cols = n, n+1

    matrix = np.zeros((n_rows, n_cols))
    indices = np.arange(n_rows)

    if error_rate == 0:
        return matrix

    if skew == 0:
        # No skew - uniform distribution
        matrix.fill((error_rate)/n_rows)
    else:
        for j in range(n_cols):
            if skew > 0:
                # Bias towards low indices
                probs = 1 / (1 + indices)
            else:
                # Bias towards high indices
                probs = 1 / (1 + indices[::-1])

            # Apply skew magnitude
            probs = probs ** abs(skew)

            # Normalize to sum to 1-error_rate
            matrix[:, j] = ((error_rate * probs) / probs.sum())

    return matrix