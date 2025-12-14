"""Stage 2 training & evaluation script for VIP5 reranker.

Khung (skeleton) pipeline:
- Load retrieved.pkl từ Stage-1 (LRURec/MMGCN/VBPR).
- Khởi tạo VIP5Reranker với Args cấu hình.
- (TODO) Xây dựng dataset rerank từ retrieved.pkl + metadata (text/image).
- (TODO) Train VIP5 và đánh giá lại Recall/NDCG sau rerank.
"""

import pickle
from pathlib import Path
from typing import Dict, List

from pytorch_lightning import seed_everything

from config import arg, EXPERIMENT_ROOT
from rerank.registry import get_reranker_class
from rerank.methods.vip5 import Args as VIP5Args


RERANK_METHOD = "vip5"
# Stage-1 method dùng để lấy retrieved.pkl; có thể đổi giữa "lrurec", "mmgcn", "vbpr", ...
STAGE1_METHOD = "mmgcn"


def _load_retrieved(stage1_method: str) -> tuple[Path, Dict]:
    """Load retrieved.pkl của Stage-1 cho một method cụ thể."""
    retrieved_root = (
        Path(EXPERIMENT_ROOT)
        / "retrieval"
        / stage1_method
        / arg.dataset_code
        / f"seed{arg.seed}"
    )
    retrieved_path = retrieved_root / "retrieved.pkl"
    if not retrieved_path.is_file():
        raise FileNotFoundError(f"retrieved.pkl không tồn tại tại: {retrieved_path}")

    with retrieved_path.open("rb") as f:
        payload = pickle.load(f)
    return retrieved_path, payload


def main() -> None:
    # 1) Seed
    seed_everything(arg.seed)

    # 2) Load retrieved.pkl từ Stage-1
    retrieved_path, payload = _load_retrieved(STAGE1_METHOD)
    val_probs = payload.get("val_probs", [])
    val_labels = payload.get("val_labels", [])
    test_probs = payload.get("test_probs", [])
    test_labels = payload.get("test_labels", [])

    print("=" * 80)
    print(f"Loaded Stage-1 retrieved from: {retrieved_path}")
    print(
        f"VAL users: {len(val_probs)}, TEST users: {len(test_probs)}, "
        f"Dataset: {arg.dataset_code}"
    )
    print("=" * 80)

    # 3) Khởi tạo VIP5Reranker với cấu hình mặc định (có thể chỉnh trong VIP5Args)
    RerankerCls = get_reranker_class(RERANK_METHOD)
    vip5_args = VIP5Args(
        distributed=arg.vip5_distributed,
        gpu=arg.vip5_gpu,
        output=arg.vip5_output,
        backbone=arg.vip5_backbone,
        tokenizer=arg.vip5_tokenizer,
        max_text_length=arg.vip5_max_text_length,
        do_lower_case=arg.vip5_do_lower_case,
        image_feature_type=arg.vip5_image_feature_type,
        image_feature_size_ratio=arg.vip5_image_feature_size_ratio,
        use_vis_layer_norm=arg.vip5_use_vis_layer_norm,
        use_adapter=arg.vip5_use_adapter,
        add_adapter_cross_attn=arg.vip5_add_adapter_cross_attn,
        use_lm_head_adapter=arg.vip5_use_lm_head_adapter,
        use_single_adapter=arg.vip5_use_single_adapter,
        reduction_factor=arg.vip5_reduction_factor,
        track_z=arg.vip5_track_z,
        unfreeze_layer_norms=arg.vip5_unfreeze_layer_norms,
        unfreeze_language_model=arg.vip5_unfreeze_language_model,
        freeze_ln_statistics=not arg.vip5_no_freeze_ln_statistics,
        freeze_bn_statistics=not arg.vip5_no_freeze_bn_statistics,
        dropout=arg.vip5_dropout,
        losses=arg.vip5_losses,
        optim=arg.vip5_optim,
        lr=arg.vip5_lr,
        weight_decay=arg.vip5_weight_decay,
        adam_eps=arg.vip5_adam_eps,
        gradient_accumulation_steps=arg.vip5_gradient_accumulation_steps,
        epoch=arg.vip5_epoch,
        warmup_ratio=arg.vip5_warmup_ratio,
    )
    reranker = RerankerCls(top_k=50, args=vip5_args)  # type: ignore[arg-type]

    # 4) (SKELETON) Fit reranker
    # Hiện tại chưa có dataset rerank chi tiết, nên truyền train_data rỗng
    # và payload qua kwargs để thuận tiện implement sau.
    train_data: Dict[int, List[int]] = {}
    reranker.fit(train_data, retrieved_payload=payload)
    print(f"Initialized and (pseudo-)fitted {reranker.get_name()}.")

    # TODO:
    # - Xây dựng dataset rerank từ val/test + metadata (text/image).
    # - Cài đặt training loop VIP5 (optimizer, scheduler, losses, ...).
    # - Đánh giá lại metric sau khi rerank.


if __name__ == "__main__":
    main()
