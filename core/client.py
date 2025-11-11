import os, time, json, random, glob
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

# API clients
import anthropic
from litellm import completion
from mistralai import Mistral
from openai import OpenAI
from huggingface_hub import snapshot_download


# Custom imports
from core.response_parser import ResponseParser
from core.rating_model import RatingModel
from config import keys, prompts
from project_datasets import loader


EXPERIMENTS_REPO_ID = "lguerdan/indeterminacy-experiments"

class ModelAPIClient:

    def __init__(self, task_configs, models, timeout_duration=5, temperature=1, max_tokens=5, n_samples=10, max_retries=3):
        """
        Initialize API client.
        
        Args:
            parser (ResponseParser, optional): Custom response parser for processing model outputs.
                If None, a default ResponseParser will be created.
        """
        self.rating_model = RatingModel()
        self.task_configs = task_configs
        self.models = models
        self.timeout_duration = timeout_duration
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_samples = n_samples
        self.max_retries = max_retries

    def load_run(self, results_path, run_tag, use_hf=False):
        """
        Load experimental run results either from HuggingFace or local directory.
        
        Parameters:
        -----------
        results_path : Path or str
            Base path where runs are stored locally (e.g., Path('../results/runs'))
        run_tag : str
            Name/tag of the specific run (e.g., 'main-run')
        use_hf : bool, default=False
            If True, downloads from HuggingFace first, then loads.
            If False, loads directly from local path.
        
        Returns:
        --------
        dict
            Dictionary containing all task results from the run
        """
        
        results_path = Path(results_path)
        
        if use_hf:
            # Download from HuggingFace
            print(f"Downloading run '{run_tag}' from HuggingFace...")
            local_path = snapshot_download(
                repo_id=EXPERIMENTS_REPO_ID,
                repo_type="dataset",
                allow_patterns=[f"runs/{run_tag}/**"], 
            )
            # Point to the downloaded location
            run_dir = Path(local_path) / "runs" / run_tag
            print(f"Downloaded to: {run_dir}")
        else:
            # Use local path directly
            run_dir = results_path / run_tag
            if not run_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        # Load task config
        config_path = run_dir / "task_config.json"
        with open(config_path, 'r') as f:
            task_configs = json.load(f)

        all_task_results = {}

        for task_name, task_config in task_configs.items():
            all_task_results[task_name] = {}
            all_task_results[task_name]['task_config'] = task_config

            # Load human ratings
            rating_path = run_dir / task_name / "ratings.csv"
            ratings = pd.read_csv(rating_path)
            all_task_results[task_name]['ratings'] = ratings

            # Load judge system results
            judge_results = {}
            mpaths = glob.glob(str(run_dir / task_name / "*.json"))
            for model in mpaths:
                
                with open(model, 'r') as f:
                    cached_data = json.load(f)

                    model_id = os.path.basename(model).split('.json')[0]
                
                    judge_results[model_id] = {
                        'model_info': cached_data['model_info'],
                        'resp_table': np.array(cached_data['resp_table']),
                        'p_judge_hat': {
                            k: np.array(v) for k, v in cached_data['p_judge_hat'].items()
                        }
                    }
            
            all_task_results[task_name]['judge_results'] = judge_results

        return all_task_results


    def run_tasks(self, run_tag, subset=True, mock=False, directory='datasets'):

        # Log task configuration
        Path(f"{run_tag}").mkdir(parents=True, exist_ok=True)
        config_path = f"{run_tag}/task_config.json"

        with open(config_path, 'w') as f:
            json.dump(self.task_configs, f)

        for task_name, task_config in self.task_configs.items():
            print(f'################ RUNNING {task_name} ###############')
            
            # Start timer for this task
            task_start_time = time.time()
            
            corpus = loader.load_data(
                task_name,
                task_config,
                subset=subset,
                directory=directory
            )
            
            self.judge_runner(
                task_name,
                task_config,
                self.models,
                corpus,
                run_tag,
                mock=mock
            )
            
            # Calculate elapsed time and print it
            task_elapsed_time = time.time() - task_start_time
            print(f'Task {task_name} completed in {task_elapsed_time:.2f} seconds ({task_elapsed_time/60:.2f} minutes)')
            
            print(f'#####################################################')
            print()


    @staticmethod
    def process_single_model(model, task_name, task_config, corpus, run_tag, n_samples, rating_model, mock=False):
        
        """Helper function to process a single model for multiprocessing"""
        
        model_id = f"{model['provider']}-{model['model']}"
        cache_path = f"{run_tag}/{task_name}/{model_id}.json"
        
        # Check if we have cached results
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                # Reconstruct the objects from cached data
                result = {
                    'model_id': model_id,
                    'model_info': cached_data['model_info'],
                    'resp_table': np.array(cached_data['resp_table']),
                    'p_judge_hat': {
                        k: np.array(v) for k, v in cached_data['p_judge_hat'].items()
                    }
                }
            print(f"Loaded cached results for {model_id}")
            return result
        
        print(f"Running {task_name}-{model_id}")
        
        # Create a temporary instance to handle the API calls

        temp_client = ModelAPIClient({}, [])
        rating_model = RatingModel()
        
        # If not cached, run the model
        resp_table = temp_client.score_corpus(
                        task_name,
                        task_config,
                        corpus,
                        model=model,
                        mock=mock
                    )
        p_judge_hat = rating_model.construct_judge_rating_distribution(
                        task_name,
                        task_config,
                        resp_table
                    )

        # Store results with model metadata
        model_results = {
            'model_info': {
                'provider': model['provider'],
                'model': model['model']
            },
            'resp_table': resp_table.tolist(),
            'p_judge_hat': {
                k: v.tolist() for k, v in p_judge_hat.items()
            }
        }

        # Save the judge run results
        with open(cache_path, 'w') as f:
            json.dump(model_results, f)

        result = {
            'model_id': model_id,
            'model_info': model_results['model_info'],
            'resp_table': resp_table,
            'p_judge_hat': p_judge_hat
        }
        print(f"Completed and cached results for {model_id}")
        return result


    def judge_runner(self, task_name, task_config, models, corpus, run_tag, mock):
        """
        Run models on corpus data across multiple tasks, with result caching.
        
        This function processes each task and model combination, generating response tables
        and decomposing judge ratings. Results are cached to disk to avoid redundant processing.
        
        Args:
            task_configs (dict): Dictionary mapping task names to their configuration parameters.
            models (list): List of dictionaries containing model configurations.
            corpus (DataFrame): DataFrame containing text data to be evaluated.
            path (str): Directory path for caching results.
            n_samples (int, optional): Number of samples to generate per input. Defaults to 10.
            
        Returns:
            dict: Dictionary of model results with response tables and decomposed judge ratings.
        """

        # Create cache directory if it doesn't exist
        Path(f"{run_tag}/{task_name}").mkdir(parents=True, exist_ok=True)

        judge_results = {}

        for model in models:
            model_id = f"{model['provider']}-{model['model']}"
            cache_path = f"{run_tag}/{task_name}/{model_id}.json"
            
            # Check if we have cached results
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    # Reconstruct the objects from cached data
                    judge_results[model_id] = {
                        'model_info': cached_data['model_info'],
                        'resp_table': np.array(cached_data['resp_table']),
                        'p_judge_hat': {
                            k: np.array(v) for k, v in cached_data['p_judge_hat'].items()
                        }
                    }
                print(f"Loaded cached results for {model_id}")
                continue
            
            print(f"running {task_name}-{model_id}")

            # If not cached, run the model
            resp_table = self.score_corpus(
                            task_name,
                            task_config,
                            corpus,
                            model=model,
                            mock=mock
                        )
            p_judge_hat = self.rating_model.construct_judge_rating_distribution(
                            task_name,
                            task_config,
                            resp_table
                        )

            # Store results with model metadata
            model_results = {
                'model_info': {
                    'provider': model['provider'],
                    'model': model['model']
                },
                'resp_table': resp_table.tolist(),
                'p_judge_hat': {
                    k: v.tolist() for k, v in p_judge_hat.items()
                }
            }

            # Save the judge run results
            with open(cache_path, 'w') as f:
                json.dump(model_results, f)


            judge_results[model_id] = {
                'model_info': model_results['model_info'],
                'resp_table': resp_table,
                'p_judge_hat': p_judge_hat
            }
            print(f"Completed and cached results for {model_id}")


        # Save the corpus of ratings
        ratings_path = f"{run_tag}/{task_name}/ratings.csv"
        corpus.to_csv(ratings_path, index=False)

        return judge_results


    def score_corpus(self, task_name, task_config, corpus, model, mock=False):
        """
        Score corpus items using both forced choice (FC) and response set (RS) prompts.
        
        For each item in the corpus, this function generates FC and RS responses using the
        specified model, and records the distribution of responses in a table.
        
        Args:
            task_name (str): Name of the task being evaluated.
            task_config (dict): Configuration parameters for the task.
            corpus (DataFrame): DataFrame containing items to score.
            model (dict): Dictionary containing model provider and name.
            mock (bool, optional): If True, return mock responses for testing. Defaults to False.
            
        Returns:
            np.ndarray: Response table of shape (n_items, n_options, n_response_sets).
        """

        n_items = corpus.shape[0]  # number of comments
        n_options = task_config['n_options']  # number of FC categories (A, B, C, _)
        n_response_sets = task_config['n_response_sets'] # number of RS categories (A, B, C, AB, AC, BC, ABC, _)

        # Account for null option and response set while constructing the table
        resp_table = np.zeros((n_items, n_options+1, n_response_sets+1, self.n_samples))

        for ix, item in tqdm(enumerate(corpus.itertuples()), total=n_items, desc="Processing items"):

            format_dict = {}
            for field in task_config['prompt_fields']:
                format_dict[field] = getattr(item, field)
            
            # Format prompts with the field values
            fc_prompt = prompts.PROMPTS[task_name]['FC'].format(**format_dict)
            rs_prompt = prompts.PROMPTS[task_name]['RS'].format(**format_dict)

            fc_response = self.score_item_with_api(
                                task_config=task_config,
                                prompt=fc_prompt,
                                provider=model['provider'],
                                model=model['model'],
                                fmt='FC',
                                mock=mock
                            )

            rs_response = self.score_item_with_api(
                                task_config=task_config,
                                prompt=rs_prompt,
                                provider=model['provider'],
                                model=model['model'],
                                fmt='RS',
                                mock=mock
                            )

            for trial, (fc_ix, rs_ix) in enumerate(zip(fc_response, rs_response)):
                resp_table[ix][fc_ix][rs_ix][trial] += 1
        
        return resp_table

    def score_item_with_api(self, task_config, prompt, provider, model, fmt, mock=False):
        """
        Call the specified API provider to generate and parse responses.
        
        This function handles API calls to different LLM providers (OpenAI, Mistral, Llama, Anthropic),
        processing the responses through the parser to extract structured response options.
        
        Args:
            prompt (str): Input prompt to send to the model.
            provider (str): API provider name ('openai', 'mistral', 'llama', or 'anthropic').
            model (str): Specific model name to use.
            fmt (str): Response format ('FC' or 'RS') for parsing.
            max_tokens (int, optional): Maximum tokens in response. Defaults to 5.
            mock (bool, optional): If True, return mock responses for testing. Defaults to False.
            
        Returns:
            list: List of parsed response options.
        """
        messages = [{"role": "user", "content": prompt}]

        parser = ResponseParser(task_config)

        if mock:
            # Generate random mock responses based on the format
            if fmt == 'FC':
                max_index = len(task_config['valid_fc_tokens']) - 1
                return [random.randint(0, max_index) for _ in range(self.n_samples)]
            else:  # RS format
                # For RS format, generate random indices between 0 and the number of valid response sets
                max_index = len(task_config['valid_rs_tokens']) - 1
                return [random.randint(0, max_index) for _ in range(self.n_samples)]
        
        # Initialize retry counter
        retry_count = 0
        
        while retry_count <= self.max_retries:
            
            # Sleep to avoid rate limit
            time.sleep(self.timeout_duration)

            try:
                if provider == 'deepseek':

                    responses = [completion(
                        model=f"{provider}/{model}",
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    ).choices[0].message.content for _ in range(self.n_samples)]

                elif provider == 'openai':

                    chat_response = completion(
                        model=f"{provider}/{model}",
                        messages=messages,
                        max_tokens=self.max_tokens,
                        n=self.n_samples,
                        temperature=self.temperature
                    )
                    responses = [chat_response.choices[i].message.content for i in range(self.n_samples)]
                
                elif provider == 'mistral':
                    
                    client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
                    chat_response = client.chat.complete(
                        model=model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        n=self.n_samples,
                        temperature=self.temperature
                    )
                    
                    responses = []
                    for i in range(self.n_samples):
                        message = chat_response.choices[i].message
                        print(chat_response.choices[i])
                        if isinstance(message.content, list):
                            final_answer = ""
                            for block in message.content:
                                if hasattr(block, 'type') and block.type == 'text':
                                    final_answer = getattr(block, 'text', '')
                                    break
                            print('final answer:', final_answer)
                            responses.append(final_answer)
                        else:
                            # Legacy string format
                            responses.append(message.content)

                elif provider == 'llama':

                    client = OpenAI(
                        api_key = os.environ["LLMANA_API_KEY"],
                        base_url="https://api.llama.com/compat/v1/",
                    )

                    responses = [client.chat.completions.create(
                        max_tokens=self.max_tokens,
                        model=model,
                        messages=messages,
                        temperature=self.temperature
                    ).choices[0].message.content for _ in range(self.n_samples)]

                elif provider == 'anthropic':
                    client = anthropic.Anthropic(
                        api_key=os.environ["ANTHROPIC_API_KEY"],
                    )
                    responses = []
                    for _ in range(self.n_samples):
                        chat_response = client.messages.create(
                            model=model,
                            max_tokens=self.max_tokens,
                            messages=messages,
                            temperature=self.temperature
                        )
                        responses.append(chat_response.content[0].text)

                else:
                    raise ValueError(f"Unsupported provider: {provider}")
                
                # If we reach here, the API call was successful
                return [
                    parser.parse_response(
                        responses[i], fmt
                    ) for i in range(min(len(responses), self.n_samples))
                ]
                
            except Exception as e:
                retry_count += 1
                print(f"Error calling {provider} API (attempt {retry_count}/{self.max_retries}): {str(e)}")
                
                # If we've reached the maximum number of retries, return an empty list
                if retry_count > self.max_retries:
                    print(f"Maximum retries ({self.max_retries}) reached for {provider} API. Giving up.")
                    return []
                


