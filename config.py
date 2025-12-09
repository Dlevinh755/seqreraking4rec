import argparse
import os


RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'


parser = argparse.ArgumentParser(description='Configuration for the project.')


## Data filltering arguments
parser.add_argument('--dataset_code', type=str, default='beauty')
parser.add_argument('--min_rating', type=int, default=3)
parser.add_argument('--min_uc', type=int, default=22)   # min rating/user
parser.add_argument('--min_sc', type=int, default=22)   # min rating/item
parser.add_argument('--use_image', action='store_true', default=False, help='Filter out items without image')
parser.add_argument('--use_text', action='store_true', default=False, help='Filter out items without text')
parser.add_argument('--seed', type=int, default=42)
arg = parser.parse_args()


