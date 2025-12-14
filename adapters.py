"""Adapter & PEFT modules used by VIP5.

Phiên bản này mở rộng bản tối giản trước đó, dựa theo đoạn
code bạn gửi: bao gồm AdapterConfig, AdapterController, Adapter
chuẩn Houlsby, OutputAdapter, cùng một số lớp tiện ích/hyper-net
để tương thích tối đa với VIP5.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers.activations import get_activation


# ====================================================================
# Cấu hình adapter cơ bản
# ====================================================================


@dataclass
class AdapterConfig:
    """Cấu hình adapter chuẩn Houlsby cho T5/VIP5.

    Được đơn giản hoá nhưng vẫn giữ các tham số chính:
    - reduction_factor: tỉ lệ bottleneck
    - non_linearity: hàm kích hoạt
    - add_layer_norm_before/after: LN trước/sau adapter
    - use_single_adapter: 1 adapter cho mọi task hay mỗi task 1 adapter
    - tasks: danh sách task name
    - track_z: lưu lại hidden giữa để debug
    """

    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = False
    non_linearity: str = "gelu_new"
    reduction_factor: int = 16
    weight_init_range: float = 1e-2

    # dimension hidden của backbone (set từ config.d_model bên VIP5)
    d_model: int = 768

    # danh sách task và cấu hình multi-task
    tasks: Optional[List[str]] = None
    use_single_adapter: bool = False

    # debug / phân tích
    track_z: bool = False


# ====================================================================
# Các lớp tiện ích cho adapter
# ====================================================================


class Activations(nn.Module):
    """Wrapper đơn giản quanh get_activation của HF."""

    def __init__(self, activation_type: str) -> None:
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.f(x)


def init_linear_layer(linear_layer: nn.Linear, std: float = 1e-2) -> None:
    """Khởi tạo Linear như trong paper adapter."""

    nn.init.normal_(linear_layer.weight, std=std)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim: int, output_dim: int, std: float = 1e-2) -> nn.Linear:
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


# ====================================================================
# Adapter cơ bản & Adapter cho output head
# ====================================================================


class Adapter(nn.Module):
    """Adapter chuẩn: down -> activation -> up."""

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        self.reduction_factor = config.reduction_factor
        self.down_sample_size = max(1, self.input_dim // self.reduction_factor)

        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = linear_layer(self.input_dim, self.down_sample_size, std=config.weight_init_range)
        self.up_sampler = linear_layer(self.down_sample_size, self.input_dim, std=config.weight_init_range)
        self.track_z = config.track_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.down_sampler(x)
        z = self.activation(z)
        if self.track_z:
            self.z = z  # type: ignore[attr-defined]
        output = self.up_sampler(z)
        return output


class OutputAdapter(nn.Module):
    """Adapter cho head output (lm_head), có output_dim riêng."""

    def __init__(self, config: AdapterConfig, output_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        self.reduction_factor = config.reduction_factor
        self.down_sample_size = max(1, self.input_dim // self.reduction_factor)

        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = linear_layer(self.input_dim, self.down_sample_size, std=config.weight_init_range)
        self.up_sampler = linear_layer(self.down_sample_size, output_dim, std=config.weight_init_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output

    def resize_up_sampler(self, resized_size: int) -> None:
        self.up_sampler = linear_layer(self.down_sample_size, resized_size, std=self.config.weight_init_range)


# ====================================================================
# Controller & Layer wrapper
# ====================================================================


class AdapterController(nn.Module):
    """Giữ nhiều adapter cho các task khác nhau và chọn đúng adapter theo task."""

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.tasks: List[str] = list(config.tasks or [])
        self.use_single_adapter = config.use_single_adapter
        self.adapters = nn.ModuleDict()
        self._construct_adapters(self.tasks)

    def _construct_adapters(self, tasks: List[str]) -> None:
        if self.use_single_adapter and tasks:
            shared_adapter = Adapter(self.config)
            for t in tasks:
                self.adapters[t] = shared_adapter
        else:
            for t in tasks:
                self.adapters[t] = Adapter(self.config)

    def get_adapter(self, task: str) -> Adapter:
        if task not in self.adapters:
            # nếu task mới xuất hiện, tạo adapter mới cho task đó
            self.adapters[task] = Adapter(self.config)
        return self.adapters[task]

    def forward(self, inputs: torch.Tensor, task: str) -> torch.Tensor:  # type: ignore[override]
        adapter = self.get_adapter(task)
        z = inputs
        outputs = adapter(z)
        # residual connection
        outputs = outputs + inputs
        return outputs


class AdapterLayer(nn.Module):
    """Adapter kèm LayerNorm trước/sau nếu cần."""

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.adapter = Adapter(config)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter

        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.d_model)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = self.adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs


class OutputParallelAdapterLayer(nn.Module):
    """Adapter đặt song song với output head (ví dụ lm_head)."""

    def __init__(self, config: AdapterConfig, output_dim: int):
        super().__init__()
        self.adapter = OutputAdapter(config, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.adapter(inputs)

    def resize_output_dim(self, resized_size: int) -> None:
        self.adapter.resize_up_sampler(resized_size)


# ====================================================================
# (Tuỳ chọn) Các cấu hình mở rộng cho meta/hyper adapters
# ====================================================================


@dataclass
class MetaAdapterConfig(AdapterConfig):
    """Cấu hình meta-adapter (dựa trên đoạn code tham khảo, rút gọn)."""

    task_embedding_dim: int = 512
    hidden_dim: int = 128
    projected_task_embedding_dim: int = 64
    task_hidden_dim: int = 128
    train_task_embeddings: bool = False
    non_linearity: str = "gelu_new"
    # cho hyper-net
    unique_hyper_net: bool = True
    unique_hyper_net_layer_norm: bool = True
    efficient_unique_hyper_net: bool = False
    task_to_embeddings: Optional[Dict[str, str]] = None
    input_dim: int = 768


@dataclass
class CompactorConfig(AdapterConfig):
    """Cấu hình cho Hyper-complex adapters (rút gọn)."""

    hidden_dim: int = 128
    intrinsic_dim: int = 100
    hypercomplex_adapters: bool = True
    hypercomplex_division: int = 4
    train_task_adapters: bool = True
    learn_phm: bool = True
    hypercomplex_nonlinearity: str = "glorot-uniform"
    shared_phm_rule: bool = True
    factorized_phm: bool = True
    shared_W_phm: bool = False
    factorized_phm_rule: bool = False
    phm_c_init: str = "normal"
    phm_rank: int = 1
    phm_init_range: float = 1e-4


@dataclass
class LRAdapterConfig(AdapterConfig):
    """Cấu hình cho Low-rank adapters (rút gọn)."""

    hidden_dim: int = 128
    intrinsic_dim: int = 100
    hypercomplex_adapters: bool = False
    hypercomplex_division: int = 4
    train_task_adapters: bool = True
    learn_phm: bool = True
    hypercomplex_nonlinearity: str = "glorot-uniform"
    shared_phm_rule: bool = True
    factorized_phm: bool = True
    shared_W_phm: bool = False
    factorized_phm_rule: bool = False
    phm_c_init: str = "normal"
    phm_rank: int = 1
    phm_init_range: float = 1e-4
    # Low-rank adapters.
    low_rank_adapters: bool = True
    low_rank_w_init: str = "glorot-uniform"
    low_rank_rank: int = 1


# Mapping tên cấu hình -> class, dùng nếu sau này cần AutoAdapterConfig
ADAPTER_CONFIG_MAPPING: Dict[str, type] = {
    "adapter": AdapterConfig,
    "meta_adapter": MetaAdapterConfig,
    "compactor": CompactorConfig,
    "lr_adapter": LRAdapterConfig,
}


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class để tạo đúng config theo tên."""

    @classmethod
    def get(cls, config_name: str) -> AdapterConfig:
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}".format(
                config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())
            )
        )

