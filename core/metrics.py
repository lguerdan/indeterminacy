import numpy as np
from scipy.spatial import distance
from sklearn.metrics import cohen_kappa_score
import krippendorff

from core import utils


TARGET_METRICS = ['HR_h_h', 'Krippendorff_h_h', 'Fleiss_kappa_h_h', 'Cohen_kappa_h_h',
    'KL_lh_s_s', 'KL_hl_s_s', 'CE_lh_s_s', 'CE_hl_s_s', 'JSD_s_s', 'COV_h_hrs', 'COV_hat_h_hrs', 
    'MSE_srs_srs','MSE_hat_srs_srs', 'MSE_h_h', 'MSE_s_s']

DOWNSTREAM_METRICS = ['consistency', 'bias_mae', 'bias_mse', 'bias']


FC_RS_GROUPING = {
    'FC': ['HR_h_h', 'Krippendorff_h_h', 'Fleiss_kappa_h_h', 'Cohen_kappa_h_h',
    'KL_lh_s_s', 'KL_hl_s_s', 'CE_lh_s_s', 'CE_hl_s_s', 'JSD_s_s'],
    'RS': ['COV_h_hrs', 'MSE_srs_srs'],
}

TYPE_GROUPING = {
    'CAT_FC': ['HR_h_h', 'Krippendorff_h_h', 'Fleiss_kappa_h_h', 'Cohen_kappa_h_h'],
    'DIST_FC': ['KL_lh_s_s', 'KL_hl_s_s', 'CE_lh_s_s', 'CE_hl_s_s', 'JSD_s_s'],
    'CAT_RS': ['COV_h_hrs'],
    'DIST_RS': ['MSE_srs_srs']
}

METRIC_GROUPS = {
    'Categorical (Forced Choice)': ['HR_h_h', 'Krippendorff_h_h', 'Fleiss_kappa_h_h', 'Cohen_kappa_h_h'],
    'Distributional (Forced Choice)': ['KL_lh_s_s', 'KL_hl_s_s', 'CE_lh_s_s', 'CE_hl_s_s', 'JSD_s_s', 'MSE_s_s'],
    'Discrete (Multi-Label)': ['COV_h_hrs'],
    'Continuous (Multi-Label)': ['MSE_srs_srs']
}

METRIC_LOOKUP = {
    'CE_hl_s_s': 'Cross Entropy (h,j) (s/s)',
    'CE_lh_s_s': 'Cross Entropy (j,h) (s/s)',
    'KL_lh_s_s': 'KL-Divergence (j,h) (s/s)',
    'KL_hl_s_s': 'KL-Divergence (h,j) (s/s)',
    'JSD_s_s': 'JS-Divergence (s/s)',
    'HR_h_h': 'Hit Rate (h/h)',
    'Krippendorff_h_h': "Krippendorff's " + r"$\alpha$ (h/h)",
    'Fleiss_kappa_h_h': "Fleiss's "+ r"$\kappa$ (h/h)",
    'Cohen_kappa_h_h': "Cohen's "+ r"$\kappa$ (h/h)",
    'COV_h_hrs': r"Coverage: $F$ (h/hrs)",
    'COV_hat_h_hrs': r"Coverage: $\hat{F}$ (h/hrs)",
    'MSE_srs_srs': r"MSE: $F$ (srs/srs)",
    'MSE_hat_srs_srs': r"MSE: $\hat{F}$ (srs/srs)",
    'MSE_h_h': 'MSE (h/h)',
    'MSE_s_s': 'MSE (s/s)',
    'consistency': 'Consistency',
    'bias_mae': 'Bias (MAE)',
    'bias_mse': 'Bias (MSE)',
    'bias': 'Bias',
}

METRIC_LOOKUP_SHORT = {
    'CE_hl_s_s': 'CE(h,j) (s/s)',
    'CE_lh_s_s': 'CE(j,h) (s/s)',
    'KL_lh_s_s': 'KLD(j,h) (s/s)',
    'KL_hl_s_s': 'KLD(h,j) (s/s)',
    'JSD_s_s': 'JS-Divergence (s/s)',
    'HR_h_h': 'HR (h/h)',
    'Krippendorff_h_h': "Kripp. (h/h)",
    'Fleiss_kappa_h_h': "Fleiss's kappa (h/h)",
    'Cohen_kappa_h_h': "Cohen's "+ r"$\kappa$ (h/h)",
    'COV_h_hrs': 'Cov. (h/hrs) (F)',
    'COV_hat_h_hrs': 'Cov. (h/hrs) (Fhat)',
    'MSE_srs_srs': 'MSE (srs/srs) (F)',
    'MSE_hat_srs_srs': 'MSE (srs/srs) (Fhat)',
    'MSE_h_h': 'MSE (h/h)',
    'MSE_s_s': 'MSE (s/s)',
    'consistency': 'Consistency',
    'bias_mae': 'Bias (MAE)',
    'bias_mse': 'Bias (MSE)',
    'bias': 'Bias'
}


METRIC_TABLE = {
    'CE_hl_s_s': {
        'cat': 'Distributional',
        'opt_ind': 0
    },
    'CE_lh_s_s': {
        'cat': 'Distributional',
        'opt_ind': 0
    },
    'KL_lh_s_s': {
        'cat': 'Distributional',
        'opt_ind': 0
    },
    'KL_hl_s_s': {
        'cat': 'Distributional',
        'opt_ind': 0
    },
    'JSD_s_s': {
        'cat': 'Distributional',
        'opt_ind': 0
    },
    'HR_h_h': {
        'cat': 'Categorical',
        'opt_ind': -1
    },
    'Krippendorff_h_h': {
        'cat': 'Categorical',
        'opt_ind': -1
    },
    'Fleiss_kappa_h_h': {
        'cat': 'Categorical',
        'opt_ind': -1
    },
    'Cohen_kappa_h_h': {
        'cat': 'Categorical',
        'opt_ind': -1
    },
    'COV_h_hrs': {
        'cat': 'Categorical',
        'opt_ind': -1
    },
    'COV_hat_h_hrs': {
        'cat': 'Categorical',
        'opt_ind': -1
    },
    'MSE_srs_srs': {
        'cat': 'MSE',
        'opt_ind': 0
    },
    'MSE_hat_srs_srs': {
        'cat': 'MSE',
        'opt_ind': 0
    },
    'MSE_h_h': {
        'cat': 'MSE',
        'opt_ind': 0
    },
    'MSE_s_s': {
        'cat': 'MSE',
        'opt_ind': 0
    },
    'consistency': {
        'cat': 'downstream',
        'opt_ind': -1
    },
    'bias_mae': {
        'cat': 'downstream',
        'opt_ind': 0
    },
    'bias_mse': {
        'cat': 'downstream',
        'opt_ind': 0
    },
    'bias': {
        'cat': 'downstream',
        'opt_ind': 0
    }
}


def aggregate_responses(py_fc, py_ml, resp_space, tau=.5):

    if resp_space == 'h':
        return py_fc.argmax(axis=1)
    
    elif resp_space == 's':
        return py_fc

    elif resp_space == 'hrs':
        return (py_ml >= tau).astype(int)

    elif resp_space == 'srs':
        return py_ml


def compute_performance_metrics(h_rs, llm_rs, ml_distribution='obs', classification_j=[0], h_rs_downstream=None, h_rs_F_hat=None, classification_tau=.5):
    
    py_fc_llm, py_ml_llm_obs, py_ml_llm = llm_rs['O'], llm_rs['omega_obs'], llm_rs['omega']
    py_fc_rater, py_ml_rater_obs, py_ml_rater_star = h_rs['O'], h_rs['omega_obs'], h_rs['omega']

    # Rater error: Pick which distribution is used to compute multi-label human-judge agreement metrics.
    if ml_distribution == 'obs':
        py_ml_rater = py_ml_rater_obs
    
    elif ml_distribution == 'star':
        py_ml_rater = py_ml_rater_star


    y_llm_h = aggregate_responses(py_fc_llm, py_ml_llm_obs, resp_space='h')
    y_llm_s = aggregate_responses(py_fc_llm, py_ml_llm_obs, resp_space='s')
    y_llm_hrs = aggregate_responses(py_fc_llm, py_ml_llm_obs, resp_space='hrs', tau = classification_tau)
    y_llm_srs = aggregate_responses(py_fc_llm, py_ml_llm_obs, resp_space='srs')

    y_human_h = aggregate_responses(py_fc_rater, py_ml_rater, resp_space='h')
    y_human_s = aggregate_responses(py_fc_rater, py_ml_rater, resp_space='s')
    y_human_hrs = aggregate_responses(py_fc_rater, py_ml_rater, resp_space='hrs', tau = classification_tau)
    y_human_srs = aggregate_responses(py_fc_rater, py_ml_rater, resp_space='srs')

    one_hot_human, one_hot_llm = np.zeros_like(y_human_hrs), np.zeros_like(y_human_hrs)
    one_hot_llm[np.arange(one_hot_llm.shape[0]), y_llm_h] = 1
    one_hot_human[np.arange(one_hot_llm.shape[0]), y_human_h] = 1


    if h_rs_F_hat is not None:
        # If F is estimated, pass in the reconstructed multi-label vector for performance metrics
        py_fc_rater_hat, py_ml_rater_star_hat = h_rs_F_hat['O'], h_rs_F_hat['omega']
        y_human_srs_hat = aggregate_responses(py_fc_rater_hat, py_ml_rater_star_hat, resp_space='srs')
    else: 
        # Use the oracle values
        y_human_srs_hat = y_human_srs

    # If a downstream metric distribution is provided, use this for computing classification metrics.
    if h_rs_downstream is not None:
        py_ml_rater_star = h_rs_downstream['omega']

    consistency, bias_mae, bias_mse, bias = compute_classification_metrics(py_ml_llm, py_ml_rater_star, 
        j=classification_j, t=classification_tau)

    return {
        'HR_h_h': HR(y_human_h, y_llm_h),
        'Fleiss_kappa_h_h': fleiss_kappa(y_human_h, y_llm_h),
        'Cohen_kappa_h_h': cohen_kappa(y_human_h, y_llm_h),
        'Krippendorff_h_h': krippendorff_alpha(y_human_h, y_llm_h),
        'KL_lh_s_s': KL(y_llm_s, y_human_s),
        'KL_hl_s_s': KL(y_human_s, y_llm_s),
        'CE_lh_s_s': CE(y_llm_s, y_human_s),
        'CE_hl_s_s': CE(y_human_s, y_llm_s),
        'JSD_s_s': JSD(y_human_s, y_llm_s),
        'COV_h_hrs': coverage(y_llm_h, y_human_hrs),
        'COV_hat_h_hrs': coverage(y_llm_h, y_human_srs_hat),
        'MSE_srs_srs': MSE(y_llm_srs, y_human_srs),
        'MSE_hat_srs_srs': MSE(y_llm_srs, y_human_srs_hat),
        'MSE_h_h': MSE(one_hot_llm, one_hot_human),
        'MSE_s_s': MSE(y_llm_s, y_human_s),
        'consistency': consistency,
        'bias_mae': bias_mae,
        'bias_mse': bias_mse, 
        'bias': bias
    }

############## Categorical Forced-Choice Human--Judge Agreement Metrics ############
def HR(A, B):
    return (A == B).astype(int).mean()

def fleiss_kappa(A, B):
    """
    https://en.wikipedia.org/wiki/Fleiss%27_kappa
    Code From: https://gist.github.com/awni/4ed15dfcfd000bcc4fb0f4e4ee30a6a0
    """
    ratings = np.column_stack((A, B))
    N, R = ratings.shape
    NR =  N * R
    categories = set(ratings.ravel().tolist())
    P_example = -np.full(N, R)
    p_class = 0.0
    for c in categories:
        c_sum = np.sum(ratings == c, axis=1)
        P_example += c_sum**2
        p_class += (np.sum(c_sum) / float(NR)) ** 2
    P_example = np.sum(P_example) / float(NR * (R-1))
    k = (P_example - p_class) / (1 - p_class)
    return k

def krippendorff_alpha(A, B):
    data = [A.tolist(), B.tolist()]
    return krippendorff.alpha(data, level_of_measurement="nominal")

def cohen_kappa(A, B):
    return cohen_kappa_score(A, B)

#########################################################

############## Distributional Forced-Choice Human--Judge Agreement Metrics ############
def KL(A, B):
    """
    Computes the KL divergence D_KL(A || B) between two discrete probability distributions.
    
    Parameters:
    A (numpy.ndarray): Probability distribution A.
    B (numpy.ndarray): Probability distribution B.
    
    Returns:
    float: KL divergence value.
    """
    # Ensure no division by zero or log of zero by clipping B away from 0
    A = np.clip(A, 1e-10, None)  # Avoid undefined A log(A)
    B = np.clip(B, 1e-10, None)  # Avoid log(0) or division by zero
    
    return np.sum(A * np.log(A / B), axis=1).mean()

def CE(A, B):
    """
    Computes the cross-entropy H(A, B) between two distributions.
    
    Parameters:
    A (pd.Series or pd.DataFrame): True probability distribution.
    B (pd.Series or pd.DataFrame): Predicted probability distribution.
    
    Returns:
    float: Cross-entropy value.
    """
    # Ensure no log(0) by clipping B away from zero
    B = np.clip(B, 1e-10, None)
    
    # Compute cross-entropy
    return (-np.sum(A * np.log(B), axis=1)).mean()

def JSD(A, B):

    # Handle potential zeros by adding a small epsilon
    A_safe = np.where(A == 0, 1e-10, A)
    B_safe = np.where(B == 0, 1e-10, B)
    
    # Let scipy handle the normalization
    return distance.jensenshannon(A_safe, B_safe, axis=1).mean()

#########################################################


######### Hard Multi-Label "Response Set" Human--Judge Agreement Metrics #########
def coverage(A,B):
    return B[np.arange(A.shape[0]), A].mean()

#########################################################


######### Soft Multi-Label "Response Set" Human--Judge Agreement Metrics #########
def MSE(A, B):
    return ((A-B)**2).sum(axis=1).mean()

#########################################################


def compute_classification_metrics(py_mo_llm, py_mo_rater, j=[0], t=0.5):
    '''
    Compute classification metrics based on cumulative probability across specified options.
    
    Args:
        py_mo_llm: LLM predictions array
        py_mo_rater: Human rater predictions array
        j: Array of indices to sum for classification. Default [0]
        t: The probability threshold applied to the cumulative sum. Default 0.5
    '''
    
    # Sum probabilities across specified indices
    y_hat_llm = (py_mo_llm[:, j].sum(axis=1) > t)
    y_hat_human = (py_mo_rater[:, j].sum(axis=1) > t)

    consistency = (y_hat_llm == y_hat_human).mean()
    bias_mae = np.abs((y_hat_llm.mean() - y_hat_human.mean()))
    bias_mse = ((y_hat_llm.mean() - y_hat_human.mean()))**2
    bias = (y_hat_llm.mean() - y_hat_human.mean())

    return consistency, bias_mae, bias_mse, bias
