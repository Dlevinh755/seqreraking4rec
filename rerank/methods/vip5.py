from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch

from rerank.base import BaseReranker


import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import re
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from pprint import pformat
from transformers.models.t5.modeling_t5 import T5LayerNorm
from rerank.model import vip5 as modeling_vip5
from adapters import (
    AdapterController,
    OutputParallelAdapterLayer,
    AdapterConfig,
)

proj_dir = Path(__file__).resolve().parent.parent

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

image_feature_dim_dict = {
    'vitb32': 512,
    'vitb16': 512,
    'vitl14': 768,
    'rn50': 1024,
    'rn101': 512
}


class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        # Nếu muốn giảm log của transformers, có thể chỉnh level trực tiếp
        if not self.verbose:
            logging.getLogger("transformers").setLevel(logging.ERROR)

    def create_config(self):
        from transformers import T5Config

        if 't5' in self.args.backbone:
            config_class = T5Config
        else:
            return None

        config = config_class.from_pretrained(self.args.backbone)

        args = self.args
        
        for k, v in vars(args).items():
            setattr(config, k, v)

        config.feat_dim = image_feature_dim_dict[args.image_feature_type]
        config.n_vis_tokens = args.image_feature_size_ratio
        config.use_vis_layer_norm = args.use_vis_layer_norm
        config.reduction_factor = args.reduction_factor
        
        config.use_adapter = args.use_adapter
        config.add_adapter_cross_attn = args.add_adapter_cross_attn
        config.use_lm_head_adapter = args.use_lm_head_adapter
        config.use_single_adapter = args.use_single_adapter
        config.unfreeze_layer_norms = args.unfreeze_layer_norms
        config.unfreeze_language_model = args.unfreeze_language_model
        
        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout

        config.losses = args.losses
        
        tasks = re.split("[, ]+", args.losses) # tranform to list
        
        if args.use_adapter:
            CONFIG_CLASS = AdapterConfig
            
            config.adapter_config = CONFIG_CLASS()
            config.adapter_config.tasks = tasks
            config.adapter_config.d_model = config.d_model # for adapter
            config.adapter_config.use_single_adapter = args.use_single_adapter
            config.adapter_config.reduction_factor = args.reduction_factor
            config.adapter_config.track_z = args.track_z
        else:
            config.adapter_config = None

        return config

    def create_model(self, model_class, config=None, **kwargs):
        print(f'Building Model at GPU {self.args.gpu}')

        model_name = self.args.backbone

        model = model_class.from_pretrained(
            model_name,
            config=config,
            **kwargs
        )
        return model
    
    def print_trainable_params_percentage(self, model):
        orig_param_size = sum(p.numel() for p in model.parameters())

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainable_size = count_parameters(model)

        percentage = trainable_size / orig_param_size * 100

        print(f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})")

        return percentage
    
    def freeze_whole_model(self):
        for n, p in self.model.named_parameters():
            p.requires_grad = False
    
    def partial_eval(self):
        # the purpose is to fix some of the norm statistics
        model = self.model.module if self.args.distributed else self.model

        def LM_LN_eval(model):
            for name, sub_module in model.named_modules():
                if "adapter" in name: # skip all adapters modules
                    continue
                if isinstance(sub_module, (modeling_vip5.T5Stack, modeling_vip5.JointEncoder)):
                    # print(f"Change {name} to eval mode...")
                    sub_module.eval()

        def only_LN_eval(model):
            for name, sub_module in model.named_modules():
                if "adapter" in name: # skip all adapters modules
                    continue
                if "visual_embedding" in name: # skip trainable parameters
                    continue
                if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                    # print(f"Change {name} to eval mode...")
                    sub_module.eval() # freeze the LN statistics and dropout

        def only_BN_eval(model):
            for name, sub_module in model.named_modules():
                if "adapter" in name: # skip all adapters modules
                    continue
                if isinstance(sub_module, (nn.BatchNorm2d)):
                    # print(f"Change {name} to eval mode...")
                    sub_module.eval() # freeze the LN statistics and dropout

        if self.args.freeze_ln_statistics:
            only_LN_eval(model)

        if self.args.freeze_bn_statistics:
            only_BN_eval(model)

    def unfreeze_parameters(self):
        targets = ["visual_embedding"]
        # unfreeze the parameters in targets anyway
        for n, p in self.model.named_parameters():
            if any(t in n for t in targets):
                p.requires_grad = True
                print(f"{n} is trainable...")

        if self.args.unfreeze_language_model:
            targets = ["lm_head", "shared"]
            for n, p in self.model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")
            for name, sub_module in self.model.named_modules():
                if isinstance(sub_module, (modeling_vip5.T5Stack, modeling_vip5.JointEncoder)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

        for name, sub_module in self.model.named_modules():
            if self.args.unfreeze_layer_norms:
                if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_adapter:
                if isinstance(sub_module, (AdapterController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_lm_head_adapter:
                if isinstance(sub_module, (OutputParallelAdapterLayer)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True     
    
    def create_tokenizer(self, **kwargs):
        from transformers import T5Tokenizer

        # Nếu dùng tokenizer kiểu P5/VIP5 thì dùng lớp P5Tokenizer trong file này,
        # ngược lại fallback về T5Tokenizer chuẩn.
        if self.args.tokenizer and 'p5' in str(self.args.tokenizer).lower():
            tokenizer_class = P5Tokenizer
        else:
            tokenizer_class = T5Tokenizer

        tokenizer_name = self.args.backbone

        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name,
            max_length=self.args.max_text_length,
            do_lower_case=self.args.do_lower_case,
            **kwargs,
        )

        return tokenizer
    
    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        from torch.optim import AdamW
        from transformers.optimization import get_linear_schedule_with_warmup

        no_decay = ["bias", "LayerNorm.weight"]

        if 'adamw' in self.args.optim:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optim = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_eps)
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optim = self.args.optimizer(optimizer_grouped_parameters, self.args.lr)

        batch_per_epoch = len(self.train_loader)
        t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epoch
        warmup_ratio = self.args.warmup_ratio
        warmup_iters = int(t_total * warmup_ratio)
        
        if self.verbose:
            print("Batch per epoch: %d" % batch_per_epoch)
            print("Total Iters: %d" % t_total)
            print('Warmup ratio:', warmup_ratio)
            print("Warm up Iters: %d" % warmup_iters)

        lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path):
        # Đơn giản hoá: dùng trực tiếp torch.load thay vì utils.load_state_dict
        state_dict = torch.load(ckpt_path, map_location='cpu')
        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self):

        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.model.apply(init_bert_weights)
        self.model.init_weights()

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))

    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)
        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)


@dataclass
class Args:
    # system
    distributed: bool = False
    gpu: int = 0
    output: str = "snap"

    # backbone/tokenizer
    backbone: str = "t5-base"          # hoặc checkpoint P5/VIP5 của repo
    tokenizer: str | None = None       # nếu None -> trainer sẽ set = backbone
    max_text_length: int = 512
    do_lower_case: bool = False

    # visual settings (VIP5)
    image_feature_type: str = "vitb32"     # must be one of: vitb32/vitb16/vitl14/rn50/rn101
    image_feature_size_ratio: int = 2      # số “visual tokens”
    use_vis_layer_norm: bool = True

    # adapter / parameter-efficient tuning
    use_adapter: bool = True
    add_adapter_cross_attn: bool = True
    use_lm_head_adapter: bool = False
    use_single_adapter: bool = False
    reduction_factor: int = 16
    track_z: bool = False

    # unfreeze options
    unfreeze_layer_norms: bool = False
    unfreeze_language_model: bool = False
    freeze_ln_statistics: bool = True
    freeze_bn_statistics: bool = True

    # training
    dropout: float = 0.1
    losses: str = "rec"                 # trainer_base split losses -> tasks list
    optim: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    adam_eps: float = 1e-8
    gradient_accumulation_steps: int = 1
    epoch: int = 3
    warmup_ratio: float = 0.05
    # behavior toggles
    append_label_if_missing: bool = False
    zero_metrics_if_missing: bool = False


class VIP5Trainer(TrainerBase):
    """Trainer cụ thể cho VIP5 dựa trên TrainerBase.

    Giả định model VIP5 trong `modeling_vip5` nhận batch ở dạng kwargs
    và trả về một dict hoặc object có thuộc tính `loss`.
    DataLoader (train/val/test) sẽ được truyền từ ngoài vào.
    """

    def __init__(self, args: Args, train_loader=None, val_loader=None, test_loader=None, train: bool = True) -> None:
        super().__init__(args, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, train=train)
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.tokenizer = None

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def _compute_loss_from_outputs(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, dict):
            loss = outputs.get("loss", None)
        else:
            loss = getattr(outputs, "loss", None)
        if loss is None:
            raise ValueError("Model output không chứa trường 'loss'. Hãy chỉnh lại forward của VIP5.")
        return loss

    def _evaluate_val_loss(self) -> float:
        if self.val_loader is None or self.model is None:
            return float("inf")

        return self._evaluate_loader(self.val_loader, name="val")

    def _evaluate_loader(self, loader, name: str = "val") -> float:
        """Generic evaluation over a DataLoader that computes loss, Recall@K and NDCG@K.

        Returns average loss (or inf if not computable).
        Prints metrics when `self.verbose`.
        """
        if loader is None or self.model is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        steps = 0
        from evaluation.metrics import recall_at_k, ndcg_at_k

        recalls = []
        ndcgs = []
        K = 10

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch_to_device(batch)

                # compute validation loss if model provides valid_step
                try:
                    outputs = self.model.valid_step(batch)
                    loss = self._compute_loss_from_outputs(outputs)
                    total_loss += float(loss.item())
                except Exception:
                    # skip loss if valid_step not implemented
                    pass

                # compute ranking metrics by scoring each candidate via the model
                bsize = batch["input_ids"].size(0)
                for i in range(bsize):
                    cands = batch.get("candidates", [])[i]
                    if not cands:
                        continue
                    label = int(batch.get("label_id", -1)[i].item()) if batch.get("label_id", None) is not None else -1

                    enc_input = batch["input_ids"][i].unsqueeze(0)
                    enc_attn = batch["attention_mask"][i].unsqueeze(0)

                    scores = []
                    for j, _ in enumerate(cands):
                        letter = chr(ord("A") + j)
                        try:
                            dec = self.tokenizer(letter, return_tensors="pt")
                            dec_input = dec["input_ids"].to(self.device)
                        except Exception:
                            dec_input = torch.tensor(self.tokenizer.encode(letter)).unsqueeze(0).to(self.device)

                        try:
                            out = self.model.forward(input_ids=enc_input.to(self.device), attention_mask=enc_attn.to(self.device), labels=dec_input)
                            if isinstance(out, dict):
                                l = out.get("loss", None)
                            else:
                                l = getattr(out, "loss", None)
                            if l is None:
                                l = out[0] if isinstance(out, (list, tuple)) else None
                            score = -float(l.item()) if l is not None else 0.0
                        except Exception:
                            score = 0.0
                        scores.append(score)

                    ranked = [c for _, c in sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)]
                    if label != -1:
                        recalls.append(recall_at_k(ranked, [label], K))
                        ndcgs.append(ndcg_at_k(ranked, [label], K))
                    else:
                        zero_flag = getattr(self.args, 'zero_metrics_if_missing', False)
                        try:
                            from config import arg as global_arg
                            zero_flag = zero_flag or getattr(global_arg, 'vip5_zero_metrics_if_missing', False)
                        except Exception:
                            pass
                        if zero_flag:
                            recalls.append(0.0)
                            ndcgs.append(0.0)

                steps += 1

        avg_loss = total_loss / max(1, steps)
        if recalls:
            avg_rec = float(sum(recalls) / len(recalls))
            avg_ndcg = float(sum(ndcgs) / len(ndcgs))
        else:
            avg_rec = 0.0
            avg_ndcg = 0.0

        if self.verbose:
            print(f"           {name}_loss   = {avg_loss:.4f}")
            print(f"           Recall@{K} = {avg_rec:.4f}, NDCG@{K} = {avg_ndcg:.4f}")

        return avg_loss

    def train(self) -> None:
        if self.train_loader is None:
            raise ValueError("VIP5Trainer.train yêu cầu train_loader khác None.")

        # 1) Build config, tokenizer, model
        config = self.create_config()
        self.tokenizer = self.create_tokenizer()

        # Giả định modeling_vip5 có lớp VIP5Tuning; nếu khác, bạn có thể đổi tại đây
        self.model = self.create_model(modeling_vip5.VIP5Tuning, config=config)
        self.model.to(self.device)

        # 2) Đóng băng / mở băng tham số theo chiến lược VIP5
        self.freeze_whole_model()
        self.partial_eval()
        self.unfreeze_parameters()

        # 3) Optimizer + scheduler
        optim, scheduler = self.create_optimizer_and_scheduler()

        best_val_loss = float("inf")
        global_step = 0

        for epoch in range(self.args.epoch):
            self.model.train()
            # initialize epoch loss counter
            epoch_loss = 0.0
            # Show progress bar in terminal when verbose
            if self.verbose:
                loader_iter = tqdm(self.train_loader, desc=f"[Epoch {epoch+1}/{self.args.epoch}]", total=len(self.train_loader))
            else:
                loader_iter = self.train_loader
            for step, batch in enumerate(loader_iter):
                # VIP5Tuning.train_step trả về dict có khóa 'loss'
                outputs = self.model.train_step(batch)
                loss = self._compute_loss_from_outputs(outputs)

                loss.backward()
                optim.step()
                scheduler.step()
                optim.zero_grad()

                epoch_loss += float(loss.item())
                global_step += 1

            avg_train_loss = epoch_loss / max(1, len(self.train_loader))

            if self.verbose:
                print(f"[Epoch {epoch+1}/{self.args.epoch}] train_loss = {avg_train_loss:.4f}")

            # 4) Evaluate trên validation nếu có
            val_loss = self._evaluate_val_loss()
            if self.verbose and val_loss < float("inf"):
                print(f"           val_loss   = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # lưu checkpoint tốt nhất
                self.save("vip5_best")

        # After training, if test_loader present and a best checkpoint exists, evaluate on test set
        try:
            if self.test_loader is not None:
                # load best checkpoint
                best_path = os.path.join(self.args.output, "vip5_best")
                if os.path.isfile(best_path + ".pth"):
                    if self.verbose:
                        print('Loading best checkpoint for test evaluation...')
                    self.load(best_path)
                    # evaluate on test set and print metrics
                    self._evaluate_loader(self.test_loader, name="test")
                else:
                    if self.verbose:
                        print('No best checkpoint found to evaluate on test set.')
        except Exception as e:
            print('Error during post-train test evaluation:', e)


class VIP5Reranker(BaseReranker):
    """Khung (skeleton) cho VIP5 reranker.

    Hiện tại chỉ khởi tạo cấu hình và device, phần model/backbone
    sẽ được bổ sung sau. `fit` chỉ đánh dấu đã được gọi để có thể
    debug pipeline end-to-end.
    """

    def __init__(self, top_k: int = 50, args: Args | None = None) -> None:
        super().__init__(top_k=top_k)
        self.args = args or Args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.tokenizer = None

    def fit(self, train_data: Dict[int, List[int]], **kwargs: Any) -> None:
        """Train VIP5 reranker.

        `train_data` hiện để trống (Dict[int, List[int]]) chỉ để khớp interface.
        Các thông tin cần thiết (retrieved candidates, text, image features, ...)
        có thể truyền thêm qua **kwargs trong bước implement sau.
        """

        # Lấy DataLoader nếu người dùng truyền vào
        train_loader = kwargs.get("train_loader", None)
        val_loader = kwargs.get("val_loader", None)
        test_loader = kwargs.get("test_loader", None)

        if train_loader is None:
            # Chưa có dữ liệu train cụ thể, chỉ đánh dấu đã init để pipeline chạy được
            print("[VIP5Reranker] train_loader=None, bỏ qua bước train (skeleton).")
            self.is_fitted = True
            return

        trainer = VIP5Trainer(self.args, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, train=True)
        trainer.train()

        # Lưu lại model/tokenizer đã train để dùng cho rerank sau này
        self.model = trainer.model
        self.tokenizer = trainer.tokenizer
        self.is_fitted = True
        

    def rerank(self, user_id: int, candidates: List[int], **kwargs: Any) -> List[Tuple[int, float]]:
        """Rerank danh sách candidates.

        Hiện tại skeleton chỉ giữ nguyên thứ tự đầu vào và gán score
        dựa trên vị trí để pipeline có thể chạy end-to-end.
        """
        self._validate_fitted()
        if not candidates:
            return []
        # Giữ nguyên thứ tự, score giảm dần theo index
        return [
            (item_id, float(len(candidates) - idx))
            for idx, item_id in enumerate(candidates[: self.top_k])
        ]

def build_vip5_training_sample(
    args: Args,
    seq: List[int],
    candidates: List[int],
    label: int,
    text_dict: Dict[int, Dict[str, Any]],
    tokenizer,
    task: str = "rec",
    max_title_len: int = 32,
) -> Dict[str, Any]:
    """Xây dựng sample train cho VIP5 tương tự seq_to_token_ids của LLM.

    - seq: lịch sử user (danh sách item id theo thứ tự thời gian).
    - candidates: danh sách item id ứng viên.
    - label: item id đúng (phải nằm trong candidates).
    - text_dict: map item_id -> meta dict with 'title', 'text', etc.
    - tokenizer: tokenizer T5/P5 đã load sẵn.

    Hàm trả về dict gồm input cho encoder (input_ids, attention_mask)
    và target_ids cho decoder (chuỗi ký tự A/B/C... ứng với item đúng).

    Phần ghép thêm whole_word_ids, category_ids, vis_feats sẽ được
    xử lý ở bước collate/batch builder cho VIP5.
    """

    def truncate_title(title: str) -> str:
        try:
            tokens = tokenizer.tokenize(title)[:max_title_len]
            return tokenizer.convert_tokens_to_string(tokens)
        except Exception as e:
            print(f"Error tokenizing title: {repr(title)}, error: {e}")
            return title[:max_title_len]  # fallback to simple truncation

    # Chuẩn bị text lịch sử người dùng
    history_parts: List[str] = []
    for idx, item_id in enumerate(seq):
        item_meta = text_dict.get(item_id, {})
        title = item_meta.get('title', '')
        if not title:
            continue
        history_parts.append(f"({idx + 1}) " + truncate_title(title))
    seq_t = " \n ".join(history_parts) if history_parts else "(no history)"

    # Chuẩn bị text cho candidate pool
    candidate_parts: List[str] = []
    for idx, item_id in enumerate(candidates):
        item_meta = text_dict.get(item_id, {})
        title = item_meta.get('title', '')
        letter = chr(ord("A") + idx)
        candidate_parts.append(f"({letter}) " + truncate_title(title))
    can_t = " \n ".join(candidate_parts)

    # Ký tự output tương ứng vị trí của label trong candidates
    if label not in candidates and label != -1:
        # Behavior controlled by args / global config: either append label or raise
        append_flag = getattr(args, 'append_label_if_missing', False)
        # fall back to global config if args doesn't have the flag
        try:
            from config import arg as global_arg
            append_flag = append_flag or getattr(global_arg, 'vip5_append_label_if_missing', False)
        except Exception:
            pass

        if append_flag:
            print(f"Warning: label {label} not in candidates {candidates}, appending label to candidates.")
            candidates = list(candidates) + [label]
        else:
            raise ValueError("Label item_id must exist in candidates for VIP5 sample.")
    
    if label == -1:
        output_letter = 'A'  # dummy
    else:
        output_letter = chr(ord("A") + candidates.index(label))

    # Câu lệnh hướng dẫn tương tự LLMRec nhưng có thể thay đổi sau
    system_prompt = (
        "Given user history in chronological order, "
        "recommend an item from the candidate pool with its index letter."
    )
    input_text = f"User history: {seq_t}; \n Candidate pool: {can_t}"

    # Encoder input cho VIP5 (nguồn)
    encoder_text = system_prompt + "\n" + input_text
    enc = tokenizer(
        encoder_text,
        truncation=True,
        max_length=args.max_text_length,
        padding=False,
        return_tensors=None,
    )

    # Decoder target: chỉ cần ký tự A/B/C... biểu diễn lựa chọn đúng
    dec = tokenizer(
        output_letter,
        truncation=True,
        max_length=8,
        padding=False,
        return_tensors=None,
    )

    sample: Dict[str, Any] = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        # VIP5Tuning.train_step mong đợi trường "target_ids" cho decoder labels
        "target_ids": dec["input_ids"],
        # candidates list và label id để collate_fn ghép visual features
        "candidates": candidates,
        "label_id": int(label),
        "task": task,
        # loss_weights sẽ được đưa về tensor [B] ở collate_fn, mặc định = 1.0
        "loss_weights": 1.0,
    }

    return sample


from transformers import T5Tokenizer, PreTrainedTokenizer
import re
import sentencepiece as spm

# The special tokens of T5Tokenizer is hard-coded with <extra_id_{}>
# I create another class P5Tokenizer extending it to add <user_id_{}> & <item_id_{}>

class P5Tokenizer(T5Tokenizer):
    def __init__(
        self, 
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        user_extra_ids=0,
        item_extra_ids=0,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        if user_extra_ids > 0:
            additional_special_tokens.extend(["<user_id_{}>".format(i) for i in range(user_extra_ids)])
        
        if item_extra_ids > 0:
            additional_special_tokens.extend(["<item_id_{}>".format(i) for i in range(item_extra_ids)])

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._user_extra_ids = user_extra_ids
        self._item_extra_ids = item_extra_ids

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self._extra_ids + self._user_extra_ids + self._item_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._user_extra_ids - self._item_extra_ids
        elif "<user_id_" in token:
            match = re.match(r"<user_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._item_extra_ids
        elif "<item_id_" in token:
            match = re.match(r"<item_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            if index > self.sp_model.get_piece_size() + self._extra_ids + self._user_extra_ids - 1:
                token = "<item_id_{}>".format(self.vocab_size - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<user_id_{}>".format(self.vocab_size - self._item_extra_ids - 1 - index)
            else:
                token = "<extra_id_{}>".format(self.vocab_size - self._user_extra_ids - self._item_extra_ids - 1 - index)
        return token