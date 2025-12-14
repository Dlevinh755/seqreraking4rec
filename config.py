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

# VIP5 rerank (Stage-2) hyperparameters
# system
parser.add_argument('--vip5_distributed', action='store_true',
					help='Use distributed training for VIP5.')
parser.add_argument('--vip5_gpu', type=int, default=0,
					help='GPU index for single-GPU VIP5 runs.')
parser.add_argument('--vip5_output', type=str, default='snap',
					help='Output directory or prefix for VIP5 checkpoints/logs.')

# backbone/tokenizer
parser.add_argument('--vip5_backbone', type=str, default='t5-base',
					help='Backbone model name or checkpoint for VIP5 (e.g., t5-base or P5/VIP5 checkpoint).')
parser.add_argument('--vip5_tokenizer', type=str, default=None,
					help='Tokenizer name for VIP5; if None, trainer will use vip5_backbone.')
parser.add_argument('--vip5_max_text_length', type=int, default=512,
					help='Maximum text length for VIP5 inputs.')
parser.add_argument('--vip5_do_lower_case', action='store_true',
					help='Lowercase text before tokenization for VIP5.')

# visual settings (VIP5)
parser.add_argument('--vip5_image_feature_type', type=str, default='vitb32',
					help='Image feature backbone for VIP5 (vitb32/vitb16/vitl14/rn50/rn101).')
parser.add_argument('--vip5_image_feature_size_ratio', type=int, default=2,
					help='Number of visual tokens (size ratio) for VIP5.')
parser.add_argument('--vip5_use_vis_layer_norm', action='store_true',
					help='Whether to apply layer norm on visual features in VIP5.')

# adapter / parameter-efficient tuning
parser.add_argument('--vip5_use_adapter', action='store_true',
					help='Use adapters for VIP5 fine-tuning.')
parser.add_argument('--vip5_add_adapter_cross_attn', action='store_true',
					help='Add adapters to cross-attention blocks in VIP5.')
parser.add_argument('--vip5_use_lm_head_adapter', action='store_true',
					help='Use adapter on LM head in VIP5.')
parser.add_argument('--vip5_use_single_adapter', action='store_true',
					help='Use a single shared adapter instead of per-layer adapters in VIP5.')
parser.add_argument('--vip5_reduction_factor', type=int, default=16,
					help='Reduction factor for VIP5 adapter bottleneck.')
parser.add_argument('--vip5_track_z', action='store_true',
					help='Track adapter latent variables (z) in VIP5.')

# unfreeze options
parser.add_argument('--vip5_unfreeze_layer_norms', action='store_true',
					help='Unfreeze layer norm parameters in VIP5 backbone.')
parser.add_argument('--vip5_unfreeze_language_model', action='store_true',
					help='Unfreeze full language model weights in VIP5.')
parser.add_argument('--vip5_no_freeze_ln_statistics', action='store_true',
					help='If set, do NOT freeze LN statistics (overrides default freeze).')
parser.add_argument('--vip5_no_freeze_bn_statistics', action='store_true',
					help='If set, do NOT freeze BN statistics (overrides default freeze).')

# training
parser.add_argument('--vip5_dropout', type=float, default=0.1,
					help='Dropout rate for VIP5.')
parser.add_argument('--vip5_losses', type=str, default='rec',
					help='Loss configuration string for VIP5 (split into tasks list).')
parser.add_argument('--vip5_optim', type=str, default='adamw',
					help='Optimizer type for VIP5.')
parser.add_argument('--vip5_lr', type=float, default=1e-4,
					help='Learning rate for VIP5.')
parser.add_argument('--vip5_weight_decay', type=float, default=0.01,
					help='Weight decay for VIP5 optimizer.')
parser.add_argument('--vip5_adam_eps', type=float, default=1e-8,
					help='Adam epsilon for VIP5 optimizer.')
parser.add_argument('--vip5_gradient_accumulation_steps', type=int, default=1,
					help='Gradient accumulation steps for VIP5.')
parser.add_argument('--vip5_epoch', type=int, default=3,
					help='Number of training epochs for VIP5.')
parser.add_argument('--vip5_warmup_ratio', type=float, default=0.05,
					help='Warmup ratio for VIP5 learning rate scheduler.')
arg = parser.parse_args()


