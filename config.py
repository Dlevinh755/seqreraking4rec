import argparse
import os


RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'


parser = argparse.ArgumentParser(description='Configuration for the project.')


## Data filltering arguments
parser.add_argument('--dataset_code', type=str, default='beauty')
parser.add_argument('--min_rating', type=int, default=3)
parser.add_argument('--min_uc', type=int, default=20)   # min rating/user
parser.add_argument('--min_sc', type=int, default=20)   # min rating/item
parser.add_argument('--use_image', action='store_true', default=False, help='Filter out items without image')
parser.add_argument('--use_text', action='store_true', default=False, help='Filter out items without text')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--retrieval_epochs', type=int, default=3,
					help='Number of training epochs for Stage-1 retrieval models (e.g., neural LRURec).')
parser.add_argument('--batch_size_retrieval', type=int, default=128,
					help='Batch size for Stage-1 retrieval model training.')
parser.add_argument('--num_workers_retrieval', type=int, default=0,
					help='Number of DataLoader workers for Stage-1 retrieval model.')
parser.add_argument('--retrieval_patience', type=int, default=5,
					help='Early stopping patience (epochs without val improvement) for Stage-1 retrieval.')
arg = parser.parse_args()


