import sys
import argparse
sys.path.append('..')

from config import tasks, models
from core.client import ModelAPIClient


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run model API client tasks with configurable parameters.'
    )
    
    parser.add_argument(
        '--run-tag',
        type=str,
        default='results/runs/replicate-main-exp',
        help='Tag for the run (default: runs/replicate-main-exp)'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        default='datasets',
        help='Directory containing datasets (default: datasets)'
    )
    
    
    parser.add_argument(
        '--mock',
        default=False,
        help='Run in mock mode (default: False)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of calls to each LLM per item (default: 10)'
    )
    
    return parser.parse_args()


def main():
    """Main function to execute the model API client tasks."""
    args = parse_args()
    
    client = ModelAPIClient(tasks.TASK_CONFIGS, models.MODELS)
    client.run_tasks(
        run_tag=args.run_tag,
        directory=args.directory,
        subset=True, # Required when using rating data subset stored on Hugging Face. 
        mock=args.mock,
        n_samples=args.n_samples
    )


if __name__ == '__main__':
    main()