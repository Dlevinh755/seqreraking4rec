#hhh%%writefile config.py
import argparse
import os


RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'


parser = argparse.ArgumentParser(description='Configuration for the project.')


## Data filltering arguments
parser.add_argument('--dataset_code', type=str, default='beauty')
parser.add_argument('--min_rating', type=int, default=3)    # minimum rating to consider positive
parser.add_argument('--min_uc', type=int, default=5)   # min rating/user
parser.add_argument('--min_sc', type=int, default=5)   # min rating/item
parser.add_argument('--use_image', action='store_true', default=False, help='Filter out items without image')
parser.add_argument('--use_text', action='store_true', default=False, help='Filter out items without text')
parser.add_argument('--generate_caption', action='store_true', default=False, 
                    help='Generate BLIP2 captions for images and save to CSV')
parser.add_argument('--generate_semantic_summary', action='store_true', default=False,
                    help='Generate Qwen3 VL semantic summaries for images and save to CSV')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--retrieval_epochs', type=int, default=100,
					help='Number of training epochs for Stage-1 retrieval models (e.g., neural LRURec).')
parser.add_argument('--batch_size_retrieval', type=int, default=512,
					help='Batch size for Stage-1 retrieval model training.')
parser.add_argument('--num_workers_retrieval', type=int, default=4,
					help='Number of DataLoader workers for Stage-1 retrieval model.')
parser.add_argument('--retrieval_patience', type=int, default=10,
					help='Early stopping patience (epochs without val improvement) for Stage-1 retrieval.')
parser.add_argument('--retrieval_lr', type=float, default=1e-4,
					help='Learning rate for all retrieval methods (default: 1e-3). Note: VBPR paper uses 5e-4, but we standardize to 1e-3 for fair comparison.')
parser.add_argument('--rerank_epochs', type=int, default=100,
					help='Number of training epochs for rerank models (e.g., BERT4Rec).')
parser.add_argument('--rerank_batch_size', type=int, default=8,
					help='Batch size for rerank model training.')
parser.add_argument('--rerank_lr', type=float, default=1e-4,
					help='Learning rate for rerank models (default: 1e-4).')
parser.add_argument('--rerank_patience', type=int, default=10,
					help='Early stopping patience (epochs without val improvement) for rerank models.')
parser.add_argument('--rerank_eval_candidates', type=int, default=20,
					help='Number of candidates to sample for reranker evaluation (default: 20). Used by all rerankers during validation.')
parser.add_argument('--qwen_max_candidates', type=int, default=None,
					help='Maximum number of candidates for Qwen reranker. If None, uses retrieval_top_k from pipeline config.')
parser.add_argument('--qwen3vl_mode', type=str, default='raw_image',
					choices=['raw_image', 'caption', 'semantic_summary', 'semantic_summary_small'],
					help='Prompt mode for Qwen3-VL reranker: raw_image, caption, semantic_summary, semantic_summary_small')
parser.add_argument('--semantic_summary_batch_size', type=int, default=4,
					help='Batch size for semantic summary generation (increase if GPU memory allows, recommended: 8-16 for T4)')
parser.add_argument('--semantic_summary_max_tokens', type=int, default=128,
					help='Maximum tokens for semantic summary generation (default: 64, reduced from 128 for speed)')
parser.add_argument('--use_quantization', action='store_true', default=True,
					help='Use 4-bit quantization for models (saves memory, may slightly reduce accuracy)')
parser.add_argument('--use_torch_compile', action='store_true', default=True,
					help='Use torch.compile() for faster inference (requires PyTorch 2.0+)')
parser.add_argument('--preload_all_images', action='store_true', default=True,
					help='Pre-load all images into memory before processing (faster but uses more RAM)')
# Script-specific arguments (not used by config, but added to avoid "unrecognized arguments" errors)
parser.add_argument('--retrieval_method', type=str, default=None, help='Retrieval method (used by train_retrieval.py)')
parser.add_argument('--retrieval_top_k', type=int, default=None, help='Number of candidates from Stage 1 (used by train_pipeline.py)')
parser.add_argument('--rerank_method', type=str, default=None, help='Rerank method (used by train_pipeline.py)')
parser.add_argument('--rerank_top_k', type=int, default=None, help='Number of final recommendations (used by train_pipeline.py)')
parser.add_argument('--metric_k', type=int, default=None, help='Cutoff for evaluation metrics (used by train_pipeline.py)')
parser.add_argument('--rerank_mode', type=str, default=None, help='Rerank mode (used by train_pipeline.py)')
parser.add_argument('--mode', type=str, default=None, help='Training mode (used by train_rerank_standalone.py)')
arg = parser.parse_args()


