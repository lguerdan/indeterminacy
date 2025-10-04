import numpy as np


TASK_CONFIGS = {
    'civil_comments': {
        'path': 'civil_comments/stratified_subset.csv',
        'property': 'Toxicity',
        'n_options': 3,
        'n_response_sets': 7,
        'positive_categorization_options': [0,1],
        'prompt_fields': ['comment_text'],
        'valid_fc_tokens': ['A', 'B', 'C'],
        'valid_rs_tokens': ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC'],
        'ratings_per_item': '10+',
        'display': 'Civil Comments'
    },
    'mnli': {
        'path': 'chaosNLI_v1.0/chaosNLI_mnli_m.jsonl',
        'property': 'Linguistic Entailment',
        'n_options': 3,
        'n_response_sets': 7,
        'positive_categorization_options': [0],
        'prompt_fields': ['context', 'statement'],
        'valid_fc_tokens': ['A', 'B', 'C'],
        'valid_rs_tokens': ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC'],
        'ratings_per_item': '100',
        'display': 'MNLI'
    },
    'snli': {
        'path': 'chaosNLI_v1.0/chaosNLI_snli.jsonl',
        'property': 'Linguistic Entailment',
        'n_options': 3,
        'n_response_sets': 7,
        'positive_categorization_options': [0],
        'prompt_fields': ['context', 'statement'],
        'valid_fc_tokens': ['A', 'B', 'C'],
        'valid_rs_tokens': ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC'],
        'ratings_per_item': '100',
        'display': 'SNLI'
    },
    'alphanli': {
        'path': 'chaosNLI_v1.0/chaosNLI_alphanli.jsonl',
        'property': 'Linguistic Entailment',
        'n_options': 2,
        'n_response_sets': 3,
        'positive_categorization_options': [0],
        'prompt_fields': ['o_beginning', 'o_ending', 'H1', 'H2'],
        'valid_fc_tokens': ['A', 'B'],
        'valid_rs_tokens': ['A', 'B', 'AB'],
        'ratings_per_item': '100',
        'display': 'ALPHA NLI'
    },
    'summ_eval_relevance': {
        'path': 'SummEval/summ_eval_processed.csv',
        'property': 'Relevance',
        'n_options': 2,
        'n_response_sets': 3,
        'positive_categorization_options': [0],
        'prompt_fields': ['article', 'summary'],
        'valid_fc_tokens': ['A', 'B'],
        'valid_rs_tokens': ['A', 'B', 'AB'],
        'ratings_per_item': '8',
        'display': 'SummEval\n(Relevance)'
    },
    'summ_eval_coherence': {
        'path': 'SummEval/summ_eval_processed.csv',
        'property': 'Coherence',
        'n_options': 2,
        'n_response_sets': 3,
        'positive_categorization_options': [0],
        'prompt_fields': ['article', 'summary'],
        'valid_fc_tokens': ['A', 'B'],
        'valid_rs_tokens': ['A', 'B', 'AB'],
        'ratings_per_item': '8',
        'display': 'SummEval\n(Coherence)'
    },
    'summ_eval_consistency': {
        'path': 'SummEval/summ_eval_processed.csv',
        'property': 'Factuality',
        'n_options': 2,
        'n_response_sets': 3,
        'positive_categorization_options': [0],
        'prompt_fields': ['article', 'summary'],
        'valid_fc_tokens': ['A', 'B'],
        'valid_rs_tokens': ['A', 'B', 'AB'],
        'ratings_per_item': '8',
        'display': 'SummEval\n(Consistency)'
    },
    'summ_eval_fluency': {
        'path': 'SummEval/summ_eval_processed.csv',
        'property': 'Fluency',
        'n_options': 2,
        'n_response_sets': 3,
        'positive_categorization_options': [0],
        'prompt_fields': ['article', 'summary'],
        'valid_fc_tokens': ['A', 'B'],
        'valid_rs_tokens': ['A', 'B', 'AB'],
        'ratings_per_item': '8',
        'display': 'SummEval\n(Fluency)'
    },'qags': {
        'path': 'QAGS/article_ratings.csv',
        'property': 'Factuality',
        'n_options': 2,
        'n_response_sets': 3,
        'positive_categorization_options': [0],
        'prompt_fields': ['article', 'sentence'],
        'valid_fc_tokens': ['A', 'B'],
        'valid_rs_tokens': ['A', 'B', 'AB'],
        'ratings_per_item': '3',
        'display': 'QAGS'
    },  
    'topical_chat_uses_knowledge': {
        'path': 'TopicalChat/processed_ratings.csv',
        'property': 'Uses Knowledge',
        'n_options': 2,
        'n_response_sets': 3,
        'positive_categorization_options': [0],
        'prompt_fields': ['context', 'fact', 'response'],
        'valid_fc_tokens': ['A', 'B'],
        'valid_rs_tokens': ['A', 'B', 'AB'],
        'ratings_per_item': '3',
         'display': 'TopicalChat   \n Uses Knowledge'
    },
    'topical_chat_understandable': {
        'path': 'TopicalChat/processed_ratings.csv',
        'property': 'Understandable',
        'n_options': 2,
        'n_response_sets': 3,
        'positive_categorization_options': [0],
        'prompt_fields': ['context', 'fact', 'response'],
        'valid_fc_tokens': ['A', 'B'],
        'valid_rs_tokens': ['A', 'B', 'AB'],
        'ratings_per_item': '3',
        'display': 'TopicalChat   \n Understandable'
    }
}

def get_task_forced_choice_distribution(task_name, ratings, n_items, n_options):

    O = np.zeros((n_items, n_options))

    if task_name == 'civil_comments':
        O[:, 0] = ratings['n_very_toxic']        # Very Toxic
        O[:, 1] = ratings['n_toxic']             # Toxic
        O[:, 2] = ratings['n_no_or_uncertain']   # No or Uncertain
        
    elif task_name == 'alphanli':
        O[:, 0] = ratings['1'] # Hypothesis 1
        O[:, 1] = ratings['2'] # Hypothesis 2

    elif task_name == 'mnli': 
        O[:, 0] = ratings['e'] # Entailment
        O[:, 1] = ratings['n'] # Neutral
        O[:, 2] = ratings['c'] # Contradiction
        
    elif task_name == 'snli':
        O[:, 0] = ratings['e'] # Entailment
        O[:, 1] = ratings['n'] # Neutral
        O[:, 2] = ratings['c'] # Contradiction

    elif task_name == 'qags':
        O[:, 0] = ratings['yes'] # Factually consistent
        O[:, 1] = ratings['no'] # Factually inconsistent

    elif 'topical_chat' in task_name:

        if 'uses_knowledge' in task_name:
            O[:, 0] = ratings['uses_knowledge_yes'] 
            O[:, 1] = ratings['uses_knowledge_no']

        if 'understandable' in task_name:
            O[:, 0] = ratings['understandable_yes'] 
            O[:, 1] = ratings['understandable_no']


    elif 'summ_eval' in task_name:

        if 'relevance' in task_name:
            O[:, 0] = ratings['relevance_0'] # 1, 2 on Likert
            O[:, 1] = ratings['relevance_1'] # 3, 4, 5 on Likert

        if 'coherence' in task_name:
            O[:, 0] = ratings['coherence_0'] # 1, 2 on Likert
            O[:, 1] = ratings['coherence_1'] # 3, 4, 5 on Likert

        if 'consistency' in task_name:
            O[:, 0] = ratings['consistency_0'] # 1, 2 on Likert
            O[:, 1] = ratings['consistency_1'] # 3, 4, 5 on Likert
    
        if 'fluency' in task_name:
            O[:, 0] = ratings['fluency_0'] # 1, 2 on Likert
            O[:, 1] = ratings['fluency_1'] # 3, 4, 5 on Likert

    else:
        print(f'task {task_name} not defined')

    # Normalize O by row sums to get probabilities
    O = O / O.sum(axis=1, keepdims=True)

    return O


def get_task_reverse_fc_translation(task_name, n_items, n_options, n_response_sets, beta=1):

    F_prime_template = np.zeros((n_options, n_response_sets))
    
    if task_name == 'civil_comments':
        
        # Construct inverse delta matrix template
        F_prime_template[0, 0] = 1         # very toxic
        F_prime_template[1, 1] = 1         # toxic
        F_prime_template[2, 2] = 1-beta   # no/uncertain
        F_prime_template[2, 0] = beta     # no/uncertain
        F_prime_template[3, 7] = 1         # improperly formatted response
        
    elif task_name == 'mnli' or task_name == 'snli' :
        
        # Construct inverse delta matrix template
        F_prime_template[0, 0] = 1         # Entailment
        F_prime_template[1, 1] = 1         # Neutral
        F_prime_template[2, 2] = 1-beta   # Contradiction
        F_prime_template[2, 0] = beta     # no/uncertain
        F_prime_template[3, 7] = 1         # improperly formatted response
        
    elif task_name == 'alphanli':
        
        # Construct inverse delta matrix template
        F_prime_template[0, 0] = 1         # Hypothesis 1
        F_prime_template[1, 1] = 1-beta   # Hypothesis 2
        F_prime_template[1, 0] = beta     # Hypothesis 2
        F_prime_template[2, 3] = 1         # improperly formatted response


    # Binary classification tasks
    elif task_name == 'qags' or 'summ_eval' in task_name or 'topical_chat' in task_name:      
        
        # Construct inverse delta matrix template
        F_prime_template[0, 0] = 1         # Positive Class
        F_prime_template[1, 1] = 1-beta   # Negative Class
        F_prime_template[1, 0] = beta     # Pos|Neg
        F_prime_template[2, 3] = 1     # improperly formatted response

    
    else:
        print(f'task {task_name} not defined')

    # Replicate template for each item
    return np.tile(F_prime_template, (n_items, 1, 1))
        