#hhh%%writefile config.py
import argparse
import os


EXPERIMENT_ROOT = 'experiments'


parser = argparse.ArgumentParser(description='Configuration for the project.')

#=========================================================================
# Data preparation arguments
#=========================================================================
parser.add_argument('--data_path', type=str, default=None,
                    help='Path to data folder. If None, uses default "data" folder.')
parser.add_argument('--dataset_code', type=str, default='beauty')
parser.add_argument('--min_rating', type=int, default=4)    # minimum rating to consider positive
parser.add_argument('--min_uc', type=int, default=6)   # min rating/user
parser.add_argument('--min_sc', type=int, default=6)   # min rating/item
parser.add_argument('--use_image', action='store_true', default=False, help='Filter out items without image')
parser.add_argument('--use_text', action='store_true', default=False, help='Filter out items without text')
parser.add_argument('--generate_caption', action='store_true', default=False, 
                    help='Generate BLIP2 captions for images and save to CSV')
parser.add_argument('--generate_viu', action='store_true', default=False,
                    help='Generate Qwen3 VL VIU for images and save to CSV')
parser.add_argument('--viu_batch_size', type=int, default=8,
					help='Batch size for VIU generation (increase if GPU memory allows, recommended: 8-16 for T4)')
parser.add_argument('--viu_max_tokens', type=int, default=128,
					help='Maximum tokens for VIU generation (default: 64, reduced from 128 for speed)')
parser.add_argument('--use_quantization', action='store_true', default=True,
					help='Use 4-bit quantization for models (saves memory, may slightly reduce accuracy)')
parser.add_argument('--use_torch_compile', action='store_true', default=True,
					help='Use torch.compile() for faster inference (requires PyTorch 2.0+)')
parser.add_argument('--preload_all_images', action='store_true', default=True,
					help='Pre-load all images into memory before processing (faster but uses more RAM)')
parser.add_argument('--seed', type=int, default=42)


#===========================================================================
# Training reranking arguments
#===========================================================================
parser.add_argument('--rerank_epochs', type=int, default=1,
					help='Number of training epochs for rerank models (e.g., BERT4Rec).')
parser.add_argument('--rerank_batch_size', type=int, default=16,
					help='Batch size for rerank model training.')
parser.add_argument('--rerank_lr', type=float, default=5e-4,
					help='Learning rate for rerank models (default: 1e-4).')
parser.add_argument('--rerank_patience', type=int, default=2,
					help='Early stopping patience (epochs without val improvement) for rerank models.')
parser.add_argument('--rerank_eval_candidates', type=int, default=50,
					help='Number of candidates for reranker evaluation and data preparation (default: 20). Used for both validation and pre-generating candidates in data_prepare.py.')
parser.add_argument('--qwen_max_candidates', type=int, default=50,
					help='Maximum number of candidates for Qwen reranker during inference (default: 20). If None, uses retrieval_top_k from pipeline config.')
parser.add_argument('--qwen_max_seq_length', type=int, default=2048,
					help='Maximum sequence length for Qwen LLM models (default: 2048). Increase for longer prompts (e.g., 4096 for multimodal with images).')

parser.add_argument('--retrieval_eval_mode', type=str, default='full_ranking',
					choices=['full_ranking', 'candidate_list'],
					help='Evaluation mode for retrieval models: full_ranking (evaluate on all items) or candidate_list (evaluate only on pre-generated candidates, default: full_ranking).')
parser.add_argument('--qwen_mode', type=str, default='text_only',
					choices=['text_only', 'caption', 'VIU'],
					help='Prompt mode for Qwen reranker: text_only (description only), caption, VIU')
parser.add_argument('--qwen_model', type=str, default='qwen3-0.6b',
					help='Model for Qwen reranker. Can be: qwen3-0.6b, qwen3-2bvl, qwen3-1.7b, qwen3-4b, or any HuggingFace model name (e.g., Qwen/Qwen2.5-0.5B-Instruct)')
parser.add_argument('--qwen_max_history', type=int, default=5,
					help='Maximum number of items in user history to use for Qwen reranker prompts (default: 5). History will be truncated to the last N items if longer.')
parser.add_argument('--qwen_gradient_accumulation_steps', type=int, default=1,
					help='Gradient accumulation steps for Qwen LLM training (default: 2). Effective batch size = batch_size * gradient_accumulation_steps.')
parser.add_argument('--qwen_warmup_steps', type=int, default=10,
					help='Number of warmup steps for Qwen LLM training (default: 20).')
parser.add_argument('--qwen_lora_r', type=int, default=8,
					help='LoRA rank (r) for Qwen LLM fine-tuning (default: 8). Higher values = more parameters but better capacity.')
parser.add_argument('--qwen_lora_alpha', type=int, default=16,
					help='LoRA alpha for Qwen LLM fine-tuning (default: 16). Usually set to 2 * lora_r.')
parser.add_argument('--qwen_lora_dropout', type=float, default=0.05,
					help='LoRA dropout rate for Qwen LLM fine-tuning (default: 0.05).')
parser.add_argument('--qwen_verbose', type=int, default=1,
					choices=[0, 1, 2],
					help='Verbosity level for Qwen reranker: 0 (minimal), 1 (normal), 2 (verbose/debug, default: 1).')
parser.add_argument('--qwen_disable_progress_bar', action='store_true',
					help='Disable progress bars during Qwen reranker evaluation (default: False). Progress bars are shown by default.')
parser.add_argument('--qwen_temperature', type=float, default=1.0,
					help='Temperature for probability scaling in Qwen reranker (default: 1.0). Temperature < 1.0 makes distribution sharper, > 1.0 makes it smoother. Only affects probability extraction, not generation.')
parser.add_argument('--rerank_action', type=str, default='train',
					choices=['train', 'eval'],
					help='Action for rerank: train (train model) or eval (load pretrained model and evaluate only, default: train). When eval, pass model path to --qwen_model and Unsloth will automatically load the adapter.')

parser.add_argument('--max_text_length', type=int, default=256,
					help='Maximum text length in characters for item metadata (default: 512, range: 256-512). Text will be truncated from the end if longer.')

#===========================================================================
# Script-specific arguments (not used by config, but added to avoid "unrecognized arguments" errors)
#===========================================================================
parser.add_argument('--mode', type=str, default=None, help='Training mode (used by train_rerank_standalone.py)')
arg = parser.parse_args()

# Set RAW_DATASET_ROOT_FOLDER based on data_path argument
# If data_path is provided, use it; otherwise use default "data" folder
RAW_DATASET_ROOT_FOLDER = arg.data_path if arg.data_path is not None else 'data'


