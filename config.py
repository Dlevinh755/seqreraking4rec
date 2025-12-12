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

# Generic Stage-1 retrieval training hyperparameters
parser.add_argument('--retrieval_epochs', type=int, default=3,
					help='Number of training epochs for Stage-1 retrieval models.')
parser.add_argument('--batch_size_retrieval', type=int, default=128,
					help='Batch size for Stage-1 retrieval model training.')
parser.add_argument('--num_workers_retrieval', type=int, default=0,
					help='Number of DataLoader workers for Stage-1 retrieval model.')
parser.add_argument('--retrieval_patience', type=int, default=5,
					help='Early stopping patience (epochs without val improvement) for Stage-1 retrieval.')

# MMGCN-specific hyperparameters
parser.add_argument('--mmgcn_lr', type=float, default=1e-3,
					help='Learning rate for MMGCN retriever.')
parser.add_argument('--mmgcn_reg_weight', type=float, default=1e-4,
					help='L2 regularization weight for MMGCN.')
parser.add_argument('--mmgcn_dim_x', type=int, default=64,
					help='Embedding dimension (dim_x) for MMGCN id embeddings.')
parser.add_argument('--mmgcn_num_layer', type=int, default=2,
					help='Number of GCN layers in MMGCN (currently effective up to 2).')
parser.add_argument('--mmgcn_aggr_mode', type=str, default='add',
					help='Aggregation mode in MMGCN GCN layers (e.g., add).')
parser.add_argument('--mmgcn_concate', type=int, default=0,
					help='Whether to concatenate id embedding with GCN output in MMGCN (0/1).')
parser.add_argument('--mmgcn_has_id', type=int, default=1,
					help='Whether MMGCN uses separate id embedding (0/1).')
arg = parser.parse_args()


