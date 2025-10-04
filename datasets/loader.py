import pandas as pd
from pandas import json_normalize
from huggingface_hub import hf_hub_download

RANDOM_SEED = 42  # Used for item sub-sampling in experiments
REPO_ID = "lguerdan/indeterminacy-datasets"

def load_data(task_name, task_config, subset=True, directory='datasets', use_hf=True):
    """
    Load task data either from HuggingFace (default) or local directory.
    
    Parameters:
    - task_name: str, task identifier
    - task_config: dict, configuration from TASK_CONFIGS
    - subset: bool, whether to subsample to 200 items
    - directory: str, local directory path (only used if use_hf=False)
    - use_hf: bool, if True downloads from HuggingFace, else uses local path
    """

    if use_hf:

        if not subset:
            raise ValueError(
                "Full datasets are not available on HuggingFace due to space constraints. "
                "Please use subset=True, or set use_hf=False and provide local datasets. "
                "To obtain the full datasets, please contact the authors."
            )
        else:
            # Download from HuggingFace (cached after first download)
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=task_config['path'],
                repo_type="dataset"
            )
    
    else:
        # Use local path
        path = f"{directory}/{task_config['path']}"

    # Load based on task type
    if (task_name == 'civil_comments' or 
        task_name == 'qags' or 
        'summ_eval' in task_name or  
        'topical_chat' in task_name):
        df = pd.read_csv(path)
    
    elif task_name == 'alphanli':
        df = pd.read_json(path, lines=True)
        df_unpacked = json_normalize(df['label_counter'])
        df_examples = json_normalize(df['example'])
        df['1'] = df['label_counter'].apply(lambda x: x.get('1', 0))
        df['2'] = df['label_counter'].apply(lambda x: x.get('2', 0))
        df['o_beginning'] = df_examples['obs1']
        df['o_ending'] = df_examples['obs2']
        df['H1'] = df_examples['hyp1']
        df['H2'] = df_examples['hyp2']
        
    elif task_name == 'snli':
        df = pd.read_json(path, lines=True)
        df_unpacked = json_normalize(df['label_counter'])
        df_examples = json_normalize(df['example'])
        df['n'] = df['label_counter'].apply(lambda x: x.get('n', 0))
        df['e'] = df['label_counter'].apply(lambda x: x.get('e', 0))
        df['c'] = df['label_counter'].apply(lambda x: x.get('c', 0))
        df['context'] = df_examples['premise']
        df['statement'] = df_examples['hypothesis']
        
    elif task_name == 'mnli':
        df = pd.read_json(path, lines=True)
        df_unpacked = json_normalize(df['label_counter'])
        df_examples = json_normalize(df['example']) 
        df['n'] = df['label_counter'].apply(lambda x: x.get('n', 0))
        df['e'] = df['label_counter'].apply(lambda x: x.get('e', 0))
        df['c'] = df['label_counter'].apply(lambda x: x.get('c', 0))  
        df['context'] = df_examples['premise']
        df['statement'] = df_examples['hypothesis']

    if subset:
        df = df.sample(n=200, random_state=RANDOM_SEED).reset_index(drop=True)

    return df
