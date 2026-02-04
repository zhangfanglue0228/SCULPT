# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
import torch.nn.functional as F
import math
import logging
import shutil
from pprint import pprint

from utils import load_state_dict, LossMeter, set_global_logging_level, FusedOptimizer
# import wandb
from pprint import pformat
from transformers.models.t5.modeling_t5 import T5LayerNorm
import modeling_bart
import modeling_t5
from adapters import (
    AdapterLayer, 
    AdapterController,
    OutputParallelAdapterLayer,
    MetaAdapterConfig,
    AdapterConfig,
    CompactorConfig,
    LRAdapterConfig,
    TaskEmbeddingController,
    AdapterLayersHyperNetController,
    AdapterLayersOneHyperNetController
)

from prompt import EncoderPromptConfig, DecoderPromptConfig, PromptController
from lora import LoraConfig, LoRALayer

from vis_encoder import CLIPResNetEncoder
from clip.model import VisualAdapter

import math

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

class DoraLinear_simple(nn.Module):
    def __init__(self, m: torch.nn.Linear, lora_r= 1, lora_dropout = 0.0, lora_s = 1.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = m.in_features
        self.out_features = m.out_features
        self.original_weight_matrix = m.weight.detach()
        self.weight_m = nn.Parameter(torch.empty((self.out_features, 1), **factory_kwargs),requires_grad=True)
        self.weight_v = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs),requires_grad=False)
        if m.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs), requires_grad = True)
        else:
            self.register_parameter('bias', None)
        ### init weight_m and weight_v and bias
        with torch.no_grad():
            m = nn.utils.weight_norm(m, dim=0)
            copy_weight_m = m.weight_g.detach()
            copy_weight_v = m.weight_v.detach()
            self.weight_m.copy_(copy_weight_m)
            self.weight_v.copy_(copy_weight_v)
            if m.bias is not None:
                copy_bias = m.bias.detach()
                self.bias.copy_(copy_bias)


        self.lora = LoRALayer(r=lora_r,lora_alpha=lora_r, lora_dropout=0.1,merge_weights=False)
        self.lora_A = nn.Parameter(m.weight.new_zeros((self.lora.r, self.in_features)))
        self.lora_B = nn.Parameter(m.weight.new_zeros((self.out_features, self.lora.r)))
        self.scaling = lora_s  ## don't know if this is really needed as tining scaling is essentially the same as tuning learning rate

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.merged = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        new_weight_v = self.weight_v + (self.lora_A.T @ self.lora_B.T).T.detach() * self.scaling
        weight = ( self.weight_m / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * (self.weight_v + (self.lora_A.T @ self.lora_B.T).T * self.scaling)

        return nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, lora_dim={}, lora_scale={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.lora.r, self.scaling
        )

class DoraLinear(nn.Module):
    def __init__(self, m: torch.nn.Linear, lora_r= 1, lora_dropout = 0.0, lora_s = 1.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = m.in_features
        self.out_features = m.out_features
        self.original_weight_matrix = m.weight.detach()
        self.weight_m = nn.Parameter(torch.empty((self.out_features, 1), **factory_kwargs),requires_grad=True)
        self.weight_v = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs),requires_grad=False)
        if m.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs), requires_grad = True)
        else:
            self.register_parameter('bias', None)
        ### init weight_m and weight_v and bias
        with torch.no_grad():
            m = nn.utils.weight_norm(m, dim=0)
            copy_weight_m = m.weight_g.detach()
            copy_weight_v = m.weight_v.detach()
            self.weight_m.copy_(copy_weight_m)
            self.weight_v.copy_(copy_weight_v)
            if m.bias is not None:
                copy_bias = m.bias.detach()
                self.bias.copy_(copy_bias)


        self.lora = LoRALayer(r=lora_r,lora_alpha=lora_r, lora_dropout=0.1,merge_weights=False)
        self.lora_A = nn.Parameter(m.weight.new_zeros((self.lora.r, self.in_features)))
        self.lora_B = nn.Parameter(m.weight.new_zeros((self.out_features, self.lora.r)))
        self.scaling = lora_s  ## don't know if this is really needed as tining scaling is essentially the same as tuning learning rate

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.merged = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        new_weight_v = self.weight_v + (self.lora_A.T @ self.lora_B.T).T * self.scaling
        weight = ( self.weight_m / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * (self.weight_v + (self.lora_A.T @ self.lora_B.T).T * self.scaling)

        return nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, lora_dim={}, lora_scale={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.lora.r, self.scaling
        )

class SVDLoraLinear(nn.Module):
    def __init__(self, m: torch.nn.Linear, lora_r= 1, lora_dropout = 0.0, lora_s = 1.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = m.in_features
        self.out_features = m.out_features
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs), requires_grad=False)
        self.original_weight_matrix = m.weight.detach()
        self.lora = LoRALayer(r=lora_r,lora_alpha=lora_r, lora_dropout=0.1, merge_weights=False)
        self.lora_A = nn.Parameter(m.weight.new_zeros((self.lora.r, self.in_features)), requires_grad=True)
        self.lora_sigma = nn.Parameter(m.weight.new_zeros((self.lora.r, 1)), requires_grad=True)
        self.lora_B = nn.Parameter(m.weight.new_zeros((self.out_features, self.lora.r)), requires_grad=True)
        self.scaling = lora_s  ## don't know if this is really needed as tining scaling is essentially the same as tuning learning rate
        if m.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs), requires_grad = True)
        else:
            self.register_parameter('bias', None)

        
        ### init weight_m and weight_v and bias
        with torch.no_grad():
            u, s, v = torch.linalg.svd(m.weight.detach().to(dtype=torch.float32), full_matrices=False)
            u = u[:, :self.lora.r]
            s = s[:self.lora.r]
            v = v[:self.lora.r, :]
            # s = torch.diag(s)

            copy_weight_lora_A = v.detach()
            copy_weight_lora_sigma = s.unsqueeze(1).detach()
            copy_weight_lora_B = u.detach()

            # print(self.lora_sigma.shape)
            # print(copy_weight_lora_sigma.shape)

            self.lora_A.data.copy_(copy_weight_lora_A)
            self.lora_sigma.copy_(copy_weight_lora_sigma)
            self.lora_B.data.copy_(copy_weight_lora_B)

            # del u, s, v

            if m.bias is not None:
                copy_bias = m.bias.detach()
                self.bias.copy_(copy_bias)

            weight_low = (self.lora_B * self.lora_sigma.view(-1) @ self.lora_A) * self.scaling

            new_weight = self.original_weight_matrix - weight_low.to(self.original_weight_matrix.device)
            self.weight.data.copy_(new_weight.detach())
        
        self.merged = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        weight = self.weight + (self.lora_A.T * self.lora_sigma.view(-1) @ self.lora_B.T).T * self.scaling

        return nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, lora_dim={}, lora_scale={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.lora.r, self.scaling
        )
    
class SVDDoraLinear(nn.Module):
    def __init__(self, m: torch.nn.Linear, lora_r= 1, lora_dropout = 0.0, lora_s = 1.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = m.in_features
        self.out_features = m.out_features
        self.original_weight_matrix = m.weight.detach()
        self.weight_m = nn.Parameter(torch.empty((self.out_features, 1), **factory_kwargs),requires_grad=True)
        self.weight_v = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs),requires_grad=False)
        if m.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs), requires_grad = True)
        else:
            self.register_parameter('bias', None)
        ### init weight_m and weight_v and bias
        with torch.no_grad():
            m = nn.utils.weight_norm(m, dim=0)
            copy_weight_m = m.weight_g.detach()
            copy_weight_v = m.weight_v.detach()
            self.weight_m.copy_(copy_weight_m)
            self.weight_v.copy_(copy_weight_v)
            
            u, s, v = torch.linalg.svd(copy_weight_m, full_matrices=False)
            u = u[:, :lora_r]
            s = s[:lora_r]
            v = v[:lora_r, :]
            s = torch.tensor(s, dtype=m.weight.dtype)

            if m.bias is not None:
                copy_bias = m.bias.detach()
                self.bias.copy_(copy_bias)


            self.lora = LoRALayer(r=lora_r,lora_alpha=lora_r, lora_dropout=0.1, merge_weights=False)
            self.lora_A = nn.Parameter(m.weight.new_zeros((self.lora.r, self.in_features)))
            self.lora_sigma = nn.Parameter(s)
            self.lora_B = nn.Parameter(m.weight.new_zeros((self.out_features, self.lora.r)))

            self.svd_V = nn.Linear(self.in_features, self.lora.r, bias=False, device=self.lora_A.device)
            self.svd_sigma = nn.Linear(1, self.lora.r, bias=False, device=self.lora_A.device)
            self.svd_U = nn.Linear(self.lora.r, self.out_features, bias=False, device=self.lora_A.device)

            # self.svd_V = nn.Parameter(m.weight.new_zeros((self.lora.r, self.in_features)))
            # self.svd_sigma = nn.Parameter(s)
            # self.svd_U = nn.Parameter(m.weight.new_zeros((self.out_features, self.lora.r)))

            self.lora_A.data.copy_(v.detach())
            # self.svd_V.data.copy_(v.detach())
            self.lora_B.data.copy_(u.detach())
            # self.svd_U.data.copy_(v.detach())

            self.svd_V.weight.data.copy_(v.detach())
            self.svd_sigma.weight.data.copy_(s.unsqueeze(1).detach())
            self.svd_U.weight.data.copy_(u.detach())

            print(u.device, copy_weight_m.device, self.svd_V.weight.device)

            del u, s, v


        self.lora = LoRALayer(r=lora_r,lora_alpha=lora_r, lora_dropout=0.1,merge_weights=False)
        self.lora_A = nn.Parameter(m.weight.new_zeros((self.lora.r, self.in_features)))
        self.lora_B = nn.Parameter(m.weight.new_zeros((self.out_features, self.lora.r)))
        self.scaling = lora_s  ## don't know if this is really needed as tining scaling is essentially the same as tuning learning rate

        self.merged = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # new_weight_v = self.weight_v + ((self.lora_A.T * self.lora_sigma.view(-1) @ self.lora_B.T) - (self.svd_V.T * self.svd_sigma.view(-1) @ self.svd_U.T)).T * self.scaling
        new_weight_v = self.weight_v + ((self.lora_A.T * self.lora_sigma.view(-1) @ self.lora_B.T) - (self.svd_V.weight.T * self.svd_sigma.weight.view(-1) @ self.svd_U.weight.T)).T * self.scaling
        weight = ( self.weight_m / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * (self.weight_v + (((self.lora_A.T * self.lora_sigma.view(-1) @ self.lora_B.T) - (self.svd_V.weight.T * self.svd_sigma.weight.view(-1) @ self.svd_U.weight.T))).T * self.scaling)

        return nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, lora_dim={}, lora_scale={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.lora.r, self.scaling
        )

class SCULPTLinear(nn.Module):
    """
    SCULPT Linear Layer for VL-T5.
    Replaces a standard nn.Linear layer.
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        r: int, 
        init_r_multiplier: int = 2, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.0,
        device=None, 
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        # 1. 复制基础层属性
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # 冻结的预训练权重 W0
        self.weight = nn.Parameter(
            base_layer.weight.detach().clone(), 
            requires_grad=False
        )
        if base_layer.bias is not None:
            self.bias = nn.Parameter(
                base_layer.bias.detach().clone(), 
                requires_grad=True # SCULPT 通常允许 Bias 训练，可根据 args 配置调整
            )
        else:
            self.register_parameter('bias', None)

        # 2. SCULPT 参数
        self.r = r  # 目标秩 (Final Target Rank)
        self.r_init = r * init_r_multiplier # 初始搜索秩
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.merged = False
        
        # 3. SVD 初始化 (核心逻辑)
        # 对 W0 进行 SVD 分解: W0 = U * S * Vh
        with torch.no_grad():
            w_float = self.weight.data.float()
            u, s, vh = torch.linalg.svd(w_float, full_matrices=False)
            
            # 截断
            u_init = u[:, :self.r_init]        # (out, r_init)
            # s_init = s[:self.r_init]         # (r_init,) -> 不直接使用，我们初始化 sigma 为 0
            vh_init = vh[:self.r_init, :]      # (r_init, in)

        # 4. 定义可训练参数 (Trainable Branch)
        # lora_A (V^T): (r_init, in)
        self.lora_A = nn.Linear(self.in_features, self.r_init, bias=False)
        # lora_B (U): (out, r_init)
        self.lora_B = nn.Linear(self.r_init, self.out_features, bias=False)
        # lora_sigma: (r_init, 1) 对角矩阵向量化
        self.lora_sigma = nn.Linear(1, self.r_init, bias=False)

        # 5. 参数初始化
        with torch.no_grad():
            self.lora_A.weight.data.copy_(vh_init)
            self.lora_B.weight.data.copy_(u_init)
            nn.init.zeros_(self.lora_sigma.weight)

        # 6. 注册 Mask Buffer (不参与梯度更新)
        self.register_buffer("lora_mask", torch.ones(self.r_init))
        
        # 移动到正确设备
        self.to(device)

    def get_importance_score(self):
        """Calculates importance score: Score = |s| * |grad(s)|"""
        if self.lora_sigma.weight.grad is None:
            return torch.zeros_like(self.lora_sigma.weight.view(-1))
        
        s = self.lora_sigma.weight.view(-1)
        grad = self.lora_sigma.weight.grad.view(-1)
        score = torch.abs(s) * torch.abs(grad)
        return score

    def update_mask(self, threshold):
        """Updates mask based on global threshold."""
        with torch.no_grad():
            scores = self.get_importance_score()
            new_mask = (scores >= threshold).float()
            self.lora_mask.copy_(new_mask)

    def get_orthogonal_loss(self):
        """||P_down^T P_down - I||^2 + ||P_up P_up^T - I||^2"""
        # lora_B is U (out, r)
        U = self.lora_B.weight
        # lora_A is V^T (r, in). V is A^T (in, r)
        V = self.lora_A.weight.T 

        def orth_loss_matrix(M):
            MtM = torch.mm(M.t(), M)
            I = torch.eye(MtM.size(0), device=M.device)
            return torch.norm(MtM - I, p='fro') ** 2

        loss_U = orth_loss_matrix(U)
        loss_V = orth_loss_matrix(V)
        return loss_U + loss_V

    def get_l1_norm(self):
        """Calculate L1 norm of trainable singular values."""
        return torch.sum(torch.abs(self.lora_sigma.weight))

    def forward(self, x: torch.Tensor):
        # 1. Base Forward (Frozen W0)
        # F.linear uses (input, weight.T) convention implies weight is (out, in)
        result = F.linear(x, self.weight, bias=self.bias)
        
        if self.merged:
            return result

        # 2. SCULPT Delta Forward
        # Y = W0*x + scaling * (U * (S * Mask) * V^T) * x
        
        x_dropped = self.lora_dropout(x.to(self.lora_A.weight.dtype))
        
        # Effective Sigma
        sigma_eff = self.lora_sigma.weight.view(-1) * self.lora_mask
        
        # Compute Path
        # Step 1: x @ V (x @ A.T)
        step1 = self.lora_A(x_dropped) 
        # Step 2: Multiply Sigma
        step2 = step1 * sigma_eff 
        # Step 3: @ U.T (B(step2))
        term_train = self.lora_B(step2)
        
        delta_output = term_train * self.scaling
        
        return result + delta_output

class SculptScheduler:
    """
    Global Pruning Scheduler for SCULPT.
    """
    def __init__(self, model: nn.Module, args):
        self.model = model
        self.args = args
        
        # 参数从 args 读取
        self.t_start = getattr(args, 'sculpt_t_start', 100)
        self.t_end = getattr(args, 'sculpt_t_end', 1000)
        self.pruning_freq = getattr(args, 'sculpt_pruning_freq', 10)
        self.r_target = getattr(args, 'sculpt_r', 8)
        self.r_init = self.r_target * getattr(args, 'sculpt_init_r_multiplier', 2)
        
        # 收集所有 SCULPT 层
        self.sculpt_layers = self._get_sculpt_layers(model)
        self.num_layers = len(self.sculpt_layers)
        
        self.total_rank_init = self.num_layers * self.r_init
        self.total_rank_final = self.num_layers * self.r_target
        
        print(f"[SCULPT] Scheduler Initialized. Layers: {self.num_layers}, "
              f"Rank Budget: {self.total_rank_init} -> {self.total_rank_final}")

    def _get_sculpt_layers(self, model):
        layers = []
        for m in model.modules():
            if isinstance(m, SCULPTLinear):
                layers.append(m)
        return layers

    def calculate_budget(self, global_step):
        if global_step < self.t_start:
            return self.total_rank_init
        if global_step >= self.t_end:
            return self.total_rank_final
        
        # Cubic Decay
        progress = (global_step - self.t_start) / (self.t_end - self.t_start)
        decay_factor = (1 - progress) ** 3
        current_budget = self.total_rank_final + (self.total_rank_init - self.total_rank_final) * decay_factor
        return int(current_budget)

    def step(self, global_step):
        # 频率检查
        if global_step % self.pruning_freq != 0:
            return None
        if global_step < self.t_start or global_step > self.t_end:
            return None

        current_budget = self.calculate_budget(global_step)
        
        # 收集全局分数
        all_scores_list = []
        for layer in self.sculpt_layers:
            # 移动到 CPU 计算分位点
            score = layer.get_importance_score().detach().to(device="cpu", dtype=torch.float32)
            all_scores_list.append(score)
        
        if not all_scores_list:
            return None
            
        global_scores = torch.cat(all_scores_list)
        if global_scores.sum() == 0:
            return None
            
        # 确定阈值
        k = min(current_budget, global_scores.numel())
        k = max(k, 1)
        sorted_scores, _ = torch.sort(global_scores, descending=True)
        threshold = sorted_scores[k - 1].item()
        
        # 更新 Mask
        total_kept = 0
        for layer in self.sculpt_layers:
            layer.update_mask(threshold)
            total_kept += layer.lora_mask.sum().item()
            
        metrics = {
            "sculpt_budget": current_budget,
            "sculpt_threshold": threshold,
            "sculpt_kept_ranks": total_kept,
            "sculpt_sparsity": 1.0 - (total_kept / self.total_rank_init)
        }
        return metrics

class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.verbose = True
        if self.args.distributed or self.args.deepspeed:
            if self.args.gpu != 0:
                self.verbose = False

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        self.include_vis_encoder = self.args.feature_type.startswith("raw")
        self.deepspeed = args.deepspeed

    def create_config(self):
        from transformers import T5Config, BartConfig

        if 't5' in self.args.backbone:
            config_class = T5Config
        elif 'bart' in self.args.backbone:
            config_class = BartConfig
        else:
            return None

        config = config_class.from_pretrained(self.args.backbone)

        args = self.args


        for k, v in vars(args).items():
            setattr(config, k, v)

        config.feat_dim = args.feat_dim
        config.pos_dim = args.pos_dim
        config.n_images = 2
        config.n_boxes = args.n_boxes
        config.expand_vis_embedding = args.expand_vis_embedding
        config.n_image_tokens = args.n_image_tokens
        config.vis_use_transformer = args.vis_use_transformer
        config.downsample = args.downsample
        config.oneddownsample = args.oneddownsample
        config.sparse_sample = args.sparse_sample

        config.use_vis_order_embedding = args.use_vis_order_embedding
        config.additional_visual_embedding_layers = args.additional_visual_embedding_layers
        config.mid_dim = args.mid_dim
        config.reduction_factor = args.reduction_factor

        config.vis_pooling_output = args.vis_pooling_output

        config.use_lm_head_adapter = args.use_lm_head_adapter

        config.use_hyperformer = args.use_hyperformer
        config.use_adapter = args.use_adapter
        config.use_compacter = args.use_compacter
        config.use_lradapter = args.use_lradapter

        config.add_adapter_cross_attn = args.add_adapter_cross_attn

        tasks = re.split("[, ]+", args.tasks) # tranform to list

        if args.use_hyperformer or args.use_adapter or args.use_compacter or args.use_lradapter:
            
            assert config.use_hyperformer + config.use_adapter + config.use_compacter + config.use_lradapter <= 1, "You can only at most one kind of adapters."

            if args.use_hyperformer:
                CONFIG_CLASS = MetaAdapterConfig
            elif args.use_adapter:
                CONFIG_CLASS = AdapterConfig
            elif args.use_compacter:
                CONFIG_CLASS = CompactorConfig
            elif args.use_lradapter:
                CONFIG_CLASS = LRAdapterConfig

            config.adapter_config = CONFIG_CLASS()
            config.adapter_config.tasks = tasks
            config.adapter_config.input_dim = config.d_model # for hyperformer
            config.adapter_config.d_model = config.d_model # for adapter and compactor
            config.adapter_config.unique_hyper_net = args.unique_hyper_net
            config.adapter_config.efficient_unique_hyper_net = args.efficient_unique_hyper_net
            config.adapter_config.use_single_adapter = args.use_single_adapter
            config.adapter_config.hypercomplex_division = args.hypercomplex_division
            config.adapter_config.phm_rank = args.phm_rank
            config.adapter_config.shared_phm_rule = args.shared_phm_rule
            config.adapter_config.factorized_phm = args.factorized_phm
            config.adapter_config.low_rank_rank = args.low_rank_rank
            config.adapter_config.phm_init_range = args.phm_init_range

            config.adapter_config.share_down_sampler = args.share_down_sampler
            config.adapter_config.share_up_sampler = args.share_up_sampler
            config.adapter_config.reduction_factor = args.reduction_factor
            config.adapter_config.shared_phm_rule_over_tasks = args.shared_phm_rule_over_tasks

            config.adapter_config.add_layer_norm_before_adapter = args.add_layer_norm_before_adapter
            config.adapter_config.add_layer_norm_after_adapter = args.add_layer_norm_after_adapter

            config.adapter_config.track_z = args.track_z

            if args.projected_task_embedding_dim != -1:
                config.adapter_config.projected_task_embedding_dim = args.projected_task_embedding_dim
        else:
            config.adapter_config = None

        # for prompt        
        if args.encoder_prompt_len > 0:
            config.encoder_prompt_config = EncoderPromptConfig()
            config.encoder_prompt_config.prompt_len = args.encoder_prompt_len
            config.encoder_prompt_config.tasks = tasks
            config.encoder_prompt_config.use_single_prompt = args.use_single_prompt
            config.encoder_prompt_config.mid_dim = args.mid_dim
        else:
            config.encoder_prompt_config = None

        if args.decoder_prompt_len > 0:
            config.decoder_prompt_config = DecoderPromptConfig()
            config.decoder_prompt_config.prompt_len = args.decoder_prompt_len
            config.decoder_prompt_config.tasks = tasks
            config.decoder_prompt_config.use_single_prompt = args.use_single_prompt
            config.decoder_prompt_config.mid_dim = args.mid_dim
        else:
            config.decoder_prompt_config = None

        # for lora
        if args.use_lora:
            config.lora_config = LoraConfig()
            config.lora_config.lora_dim = args.lora_dim
            config.lora_config.lora_alpha = args.lora_alpha
            config.lora_config.tasks = tasks
            config.lora_config.use_single_lora = args.use_single_lora
        
        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout

        config.use_vis_layer_norm = args.use_vis_layer_norm
        config.individual_vis_layer_norm = args.individual_vis_layer_norm
        config.losses = args.losses

        config.share_vis_lang_layer_norm = args.share_vis_lang_layer_norm
        config.classifier = args.classifier

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
                if isinstance(sub_module, (modeling_bart.JointEncoder, modeling_bart.BartDecoder, modeling_t5.T5Stack, modeling_t5.JointEncoder)):
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
                if isinstance(sub_module, (modeling_bart.JointEncoder, modeling_bart.BartDecoder, modeling_t5.T5Stack, modeling_t5.JointEncoder)):
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        print(f"{param_name} is trainable...")
                        param.requires_grad = True

        if self.args.unfreeze_lm_head:
            targets = ["lm_head", "shared"] # shared and lm_head share the same weight
            for n, p in self.model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")

        if self.args.use_lora:
            # targets = ["lora", "bias"]
            lora_targets = ["layers"]
            lora_lora_target = ["v_proj", "q_proj"]
            for n, p in self.model.named_parameters():
                # if any(t in n for t in targets):
                if any(t in n for t in lora_targets) and isinstance(p,nn.Linear):
                    if any(t in n for t in lora_lora_target):
                        p.requires_grad = True
                        print(f"{n} is trainable...")

        if self.args.use_dora:
            print("apply dora tuning")

            if self.args.lora_settings:
                dora_targets = ["layers"]
                dora_lora_target = ["v_proj", "q_proj"]
                it=[(name,m) for name,m in self.model.named_modules()]
                module_dict={}
                for n, p in it:
                    module_dict[n]=p
                    idx=n.rfind('.')
                    if idx==-1:
                        idx=0
                    father_name=n[:idx]
                    if father_name in module_dict:
                        father_module=module_dict[father_name]
                    else:
                        raise RuntimeError(f"father module {father_name} not found")
                    
                    if any(t in n for t in dora_targets) and isinstance(p,nn.Linear):
                        if any(t in n for t in dora_lora_target):
                            
                            if self.args.dora_simple:
                                print("apply dora simple instead")
                                replace_m = DoraLinear_simple(m = p, lora_r= self.args.lora_dim, lora_dropout= 0.0, device=self.model.device)
                            else:
                                replace_m = DoraLinear(m = p, lora_r= self.args.lora_dim, lora_dropout= 0.0, device=self.model.device)
                            setattr(father_module,n[idx+1:],replace_m)
                            replace_m.weight_m.requires_grad = True
                            replace_m.weight_v.requires_grad = False
                            replace_m.bias.requires_grad = True
                            del p
        
        if self.args.use_svdlora:
            print("apply svdlora tuning")

            if self.args.lora_settings:
                dora_targets = ["layers"]
                dora_lora_target = ["v_proj", "q_proj"]
                it=[(name,m) for name,m in self.model.named_modules()]
                module_dict={}
                for n, p in it:
                    module_dict[n]=p
                    idx=n.rfind('.')
                    if idx==-1:
                        idx=0
                    father_name=n[:idx]
                    if father_name in module_dict:
                        father_module=module_dict[father_name]
                    else:
                        raise RuntimeError(f"father module {father_name} not found")
                    
                    if any(t in n for t in dora_targets) and isinstance(p,nn.Linear):
                        if any(t in n for t in dora_lora_target):
                            
                            if self.args.dora_simple:
                                print("apply dora simple instead")
                                replace_m = DoraLinear_simple(m = p, lora_r= self.args.lora_dim, lora_dropout= 0.0, device=self.model.device)
                            else:
                                replace_m = SVDLoraLinear(m = p, lora_r= self.args.lora_dim, lora_dropout= 0.0, device=self.model.device)
                            setattr(father_module,n[idx+1:],replace_m)
                            replace_m.weight.requires_grad = False
                            # replace_m.weight_m.requires_grad = True
                            # replace_m.weight_v.requires_grad = False
                            replace_m.lora_A.requires_grad = True
                            replace_m.lora_sigma.requires_grad = True
                            replace_m.lora_B.requires_grad = True
                            replace_m.bias.requires_grad = True
                            del p
        
        if self.args.use_svddora:
            print("apply svddora tuning")

            if self.args.lora_settings:
                dora_targets = ["layers"]
                dora_lora_target = ["v_proj", "q_proj"]
                it=[(name,m) for name,m in self.model.named_modules()]
                module_dict={}
                for n, p in it:
                    module_dict[n]=p
                    idx=n.rfind('.')
                    if idx==-1:
                        idx=0
                    father_name=n[:idx]
                    if father_name in module_dict:
                        father_module=module_dict[father_name]
                    else:
                        raise RuntimeError(f"father module {father_name} not found")
                    
                    if any(t in n for t in dora_targets) and isinstance(p,nn.Linear):
                        if any(t in n for t in dora_lora_target):
                            
                            if self.args.dora_simple:
                                print("apply dora simple instead")
                                replace_m = DoraLinear_simple(m = p, lora_r= self.args.lora_dim, lora_dropout= 0.0, device=self.model.device)
                            else:
                                replace_m = SVDDoraLinear(m = p, lora_r= self.args.lora_dim, lora_dropout= 0.0, device=self.model.device)
                            setattr(father_module,n[idx+1:],replace_m)
                            replace_m.weight_m.requires_grad = True
                            replace_m.weight_v.requires_grad = False
                            replace_m.svd_U.weight.requires_grad = False
                            replace_m.svd_sigma.weight.requires_grad = False
                            replace_m.svd_V.weight.requires_grad = False
                            replace_m.bias.requires_grad = True
                            del p
        
        if self.args.use_sculpt:
            print("Applying SCULPT tuning...")
            sculpt_r = getattr(self.args, 'sculpt_r', 8)
            init_r_multiplier = getattr(self.args, 'sculpt_init_r_multiplier', 2)
            lora_alpha = getattr(self.args, 'lora_alpha', 16) # 复用 lora_alpha 或 args.sculpt_alpha
            lora_dropout = getattr(self.args, 'dropout', 0.1)

            if self.args.lora_settings: # 假设沿用 lora_settings 决定替换哪些层
                target_modules_list = ["layers"] # T5/BART 通常是 layers
                target_suffixes = ["v_proj", "q_proj"] # 目标层后缀

                it = [(name, m) for name, m in self.model.named_modules()]
                module_dict = {name: m for name, m in it}

                for n, p in it:
                    # 检查是否为目标层
                    if not isinstance(p, nn.Linear):
                        continue
                    
                    is_target = any(t in n for t in target_modules_list) and \
                                any(n.endswith(suffix) for suffix in target_suffixes)
                    
                    if is_target:
                        # 找到父模块进行替换
                        idx = n.rfind('.')
                        father_name = n[:idx] if idx != -1 else ""
                        child_name = n[idx+1:]
                        
                        if father_name in module_dict:
                            father_module = module_dict[father_name]
                        else:
                            continue # Should not happen

                        print(f"Replacing {n} with SCULPTLinear (r={sculpt_r}, r_init={sculpt_r*init_r_multiplier})")
                        
                        # 实例化 SCULPTLinear
                        replace_m = SCULPTLinear(
                            base_layer=p,
                            r=sculpt_r,
                            init_r_multiplier=init_r_multiplier,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            device=p.weight.device,
                            dtype=p.weight.dtype
                        )
                        
                        # 替换
                        setattr(father_module, child_name, replace_m)
                        
                        replace_m.weight.requires_grad = False       # 冻结 W0
                        replace_m.lora_A.weight.requires_grad = True # 训练 P_up
                        replace_m.lora_B.weight.requires_grad = True # 训练 P_down
                        replace_m.lora_sigma.weight.requires_grad = True # 训练 Sigma
                        if replace_m.bias is not None:
                            replace_m.bias.requires_grad = True      # 训练 Bias
                        # 释放原层显存
                        del p

        if self.args.unfreeze_bias:
            targets = ["bias"]
            # unfreeze the parameters in targets anyway
            for n, p in self.model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...({p.numel()})")


        if self.args.unfreeze_encoder_layer_norms:
            target1 = "encoder."
            target2 = "layer_norm"
            target3 = "layernorm"
            # unfreeze the parameters in targets anyway
            for n, p in self.model.named_parameters():
                # if any(t in n for t in targets):
                if target1 in n and (target2 in n or target3 in n):
                    p.requires_grad = True
                    print(f"{n} is trainable...({p.numel()})")

        if self.args.unfreeze_decoder_layer_norms:
            target1 = "decoder."
            target2 = "layer_norm"
            target3 = "layernorm"
            # unfreeze the parameters in targets anyway
            for n, p in self.model.named_parameters():
                # if any(t in n for t in targets):
                if target1 in n and (target2 in n or target3 in n):
                    p.requires_grad = True
                    print(f"{n} is trainable...({p.numel()})")



        for name, sub_module in self.model.named_modules():
            if self.args.decoder_prompt_len > 0 or self.args.encoder_prompt_len > 0:
                if isinstance(sub_module, (PromptController)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_vis_encoder:
                if isinstance(sub_module, (CLIPResNetEncoder)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_vis_last_layer:
                if "visual.layer4" in name and "visual.layer4." not in name:
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_vis_adapter:
                if isinstance(sub_module, (VisualAdapter)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_layer_norms:
                if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_batch_norms:
                if isinstance(sub_module, (nn.BatchNorm2d)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_adapter or self.args.use_compacter or self.args.use_lradapter:
                if isinstance(sub_module, (AdapterController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_lm_head_adapter:
                if isinstance(sub_module, (OutputParallelAdapterLayer)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_hyperformer:
                if isinstance(sub_module, (TaskEmbeddingController, AdapterLayersHyperNetController, AdapterLayersOneHyperNetController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

        print(self.model)
            
    def create_tokenizer(self, **kwargs):
        from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
        from tokenization import VLT5Tokenizer, VLT5TokenizerFast

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        elif 'bart' in self.args.tokenizer:
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast

        tokenizer_name = self.args.backbone

        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name,
            max_length=self.args.max_text_length,
            do_lower_case=self.args.do_lower_case,
            **kwargs
            )

        return tokenizer

    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        from transformers.optimization import AdamW, get_linear_schedule_with_warmup

        no_decay = ["bias", "LayerNorm.weight"]

        if 'adamw' in self.args.optim:

            if self.args.use_separate_optimizer_for_visual:
                
                # transformer's parameters
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if ( (not any(nd in n for nd in no_decay)) and ("vis_encoder" not in n) ) ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.lr,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if ( (any(nd in n for nd in no_decay)) and ("vis_encoder" not in n ))],
                        "weight_decay": 0.0,
                        "lr": self.args.lr,
                    },
                ]
                
                visn_model = self.model.vis_encoder
                if self.args.use_adam_for_visual:

                    vis_optimizer_grouped_parameters = [
                        {
                            "params": [p for n, p in visn_model.named_parameters() if not any(nd in n for nd in no_decay)],
                            "weight_decay": self.args.vis_weight_decay,
                            "lr": self.args.vis_lr,
                        },
                        {
                            "params": [p for n, p in visn_model.named_parameters() if any(nd in n for nd in no_decay)],
                            "weight_decay": 0.0,
                            "lr": self.args.vis_lr,
                        },
                    ]
                    optim = AdamW(
                        optimizer_grouped_parameters + vis_optimizer_grouped_parameters,
                        lr=self.args.lr,
                        # betas=(0.9, 0.98),
                        eps=self.args.adam_eps
                    )
                else:
                    optim = AdamW(
                        optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_eps
                    )
                    vis_optim = torch.optim.SGD(
                        visn_model.parameters(), 
                        self.args.vis_lr,
                        momentum=0,
                        weight_decay=self.args.vis_weight_decay
                    )

                    optim = FusedOptimizer([optim, vis_optim])

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
                optim = AdamW(optimizer_grouped_parameters,
                            lr=self.args.lr, eps=self.args.adam_eps)

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

            # if self.include_vis_encoder:
            #     trainable_parameters = trainable_parameters + list(self.vis_encoder.parameters())

            optim = self.args.optimizer(optimizer_grouped_parameters, self.args.lr)

        batch_per_epoch = len(self.train_loader)
        t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
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
        state_dict = load_state_dict(ckpt_path, 'cpu')

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self):

        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
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

        if self.deepspeed:
            self.model.save_checkpoint(self.args.output, name)
        else:
            torch.save(self.model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))

    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)
