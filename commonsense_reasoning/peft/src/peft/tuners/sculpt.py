import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class SCULPT_Config(PeftConfig):
    """
    Configuration for SCULPT.
    """
    # 基础 LoRA 参数
    r: int = field(default=8, metadata={"help": "Target Rank (Final Rank) for SCULPT"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha scaling factor"})
    lora_dropout: float = field(default=None, metadata={"help": "Dropout probability"})
    
    # SCULPT 参数
    # 初始化参数
    init_r_multiplier: int = field(
        default=2, 
        metadata={"help": "Multiplier for initial rank. r_init = r * init_r_multiplier"}
    )
    orth_reg_weight: float = field(
        default=0.01,
        metadata={"help": "Weight for orthogonal regularization loss"}
    )
    lasso_reg_weight: float = field(
        default=0.001,
        metadata={"help": "Weight for L1 (Lasso) regularization loss on singular values"}
    )
    # 剪枝调度参数
    t_start: int = field(
        default=100, 
        metadata={"help": "Step to start pruning (Warmup steps). Before this, mask stays all-ones."}
    )
    t_end: int = field(
        default=1000, 
        metadata={"help": "Step to end pruning (Final tuning steps). After this, mask is frozen."}
    )
    pruning_freq: int = field(
        default=10, 
        metadata={"help": "Frequency of mask updates (delta T). Update mask every `pruning_freq` steps."}
    )
    
    # 兼容性参数
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the SCULPT model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from SCULPT layers to be set as trainable."}
    )

    def __post_init__(self):
        self.peft_type = "SCULPT"


class SCULPT_Model(torch.nn.Module):
    """
    SCULPT Model Wrapper.
    Wraps the base model and replaces specified Linear layers with SCULPTLinear layers.
    """
    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self.sculpt_layers = []
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use SCULPT with 8-bit quantization, please install the `bitsandbytes` package."
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        
        # 计算初始秩 r_init
        r_init = self.peft_config.r * self.peft_config.init_r_multiplier
        
        kwargs = {
            "r": self.peft_config.r, # Target Rank
            "r_init": r_init,        # Initial Search Rank
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        print("8 bit lora")
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear):
                    # Standard Linear Layer Replacement
                    new_module = SCULPTLinear(target.in_features, target.out_features, bias=bias, **kwargs)
                    self.sculpt_layers.append(new_module)
                else:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
                print(f"{key} replaced with SCULPTLinear (r_target={kwargs['r']}, r_init={kwargs['r_init']})")

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
        
    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight # Keep original W0

        # Perform SVD on the Pretrained Weight W0
        with torch.no_grad():
            # W0 shape: (out_features, in_features)
            # U: (out, out), S: (min,), Vh: (in, in)
            weight_for_svd = new_module.weight.detach().to(dtype=torch.float32)
            if new_module.fan_in_fan_out:
                weight_for_svd = weight_for_svd.T
            
            u, s, vh = torch.linalg.svd(weight_for_svd, full_matrices=False)
            
            # Truncate to r_init
            r_init = new_module.r_init
            u = u[:, :r_init]        # (out, r_init) -> lora_B (Down projection)
            s = s[:r_init]           # (r_init,)
            vh = vh[:r_init, :]      # (r_init, in)  -> lora_A (Up projection)

            # Initialize Trainable Branch
            new_module.lora_B.weight.data.copy_(u)
            new_module.lora_sigma.weight.data.copy_(s.unsqueeze(1)) # (r, 1) to act as vector
            new_module.lora_A.weight.data.copy_(vh)

            weight_low = torch.mm(u * s, vh)
            weight_low = weight_low * new_module.scaling
            weight_low = transpose(weight_low, new_module.fan_in_fan_out)
            new_module.weight.data -= weight_low.to(new_module.weight.device)

        # Clean up
        del u, s, vh, weight_low, weight_for_svd
        # torch.cuda.empty_cache()

        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, SCULPTLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError   


class SCULPTLayer:
    def __init__(
        self,
        r: int,
        r_init: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r           # Final Target Rank
        self.r_init = r_init # Initial Search Rank
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

        self.is_sculpt_layer = True


class SCULPTLinear(nn.Linear, SCULPTLayer):
    """
    The Core Layer for SCULPT.
    Implements: Y = W0*x + (Delta_Train - Delta_Fixed)*x
    Where Delta_Train is masked dynamically.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        r_init: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SCULPTLayer.__init__(self, r=r, r_init=r_init, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        
        # --- Trainable Branch (P_down, s, P_up) ---
        if r_init > 0:
            # lora_A: Right Projection (V^T), shape (r_init, in_features)
            self.lora_A = nn.Linear(in_features, r_init, bias=False)
            # lora_B: Left Projection (U), shape (out_features, r_init)
            self.lora_B = nn.Linear(r_init, out_features, bias=False)
            # lora_sigma: Singular Values vector, shape (r_init, 1) for easy broadcasting
            self.lora_sigma = nn.Linear(1, r_init, bias=False)
            
            self.scaling = self.lora_alpha / self.r # Scaling factor
            
            # Freeze base weight
            self.weight.requires_grad = False
            # --- Mask Buffer ---
            # Registered buffer: saved in state_dict, moves to device, but no grad
            self.register_buffer("mask", torch.ones(r_init))

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A.weight)
            nn.init.zeros_(self.lora_B.weight)
            nn.init.zeros_(self.lora_sigma.weight)

    def train(self, mode: bool=True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_sigma.train(mode)
        self.lora_B.train(mode)

    def get_importance_score(self):
        """
        Calculates importance score: Score = |s| * |grad(s)|
        """
        if self.lora_sigma.weight.grad is None:
            # Handle cases where grad is not yet available (e.g. before first backward)
            return torch.zeros_like(self.lora_sigma.weight.view(-1))
            
        s = self.lora_sigma.weight.view(-1)
        grad = self.lora_sigma.weight.grad.view(-1)
        score = torch.abs(s) * torch.abs(grad)
        return score

    def update_mask(self, threshold):
        """
        Updates mask based on global threshold.
        """
        with torch.no_grad():
            scores = self.get_importance_score()
            # If score >= threshold, keep (1), else prune (0)
            new_mask = (scores >= threshold).float()
            self.mask.copy_(new_mask)

    def get_orthogonal_loss(self):
        """
        Calculate orthogonality loss for U and V.
        ||P_down^T P_down - I||^2 + ||P_up P_up^T - I||^2
        """
        # lora_B is P_down (out, r)
        U = self.lora_B.weight
        # lora_A is P_up (r, in) -> effectively V^T
        V = self.lora_A.weight.T # (in, r)

        def orth_loss_matrix(M):
            # M is (N, r)
            MtM = torch.mm(M.t(), M)
            I = torch.eye(MtM.size(0), device=M.device)
            return torch.norm(MtM - I, p='fro') ** 2

        loss_U = orth_loss_matrix(U)
        loss_V = orth_loss_matrix(V)
        return loss_U + loss_V

    def get_l1_norm(self):
        """
        Calculate L1 norm of trainable singular values.
        """
        return torch.sum(torch.abs(self.lora_sigma.weight))

    def merge(self):
        """
        Physical pruning at inference time.
        Removes pruned ranks and merges delta into base weight.
        """
        if self.merged:
            return

        with torch.no_grad():
            # 1. Identify kept indices
            kept_indices = torch.nonzero(self.mask.view(-1)).squeeze()
            if kept_indices.numel() == 0:
                print("Warning: All ranks pruned in a layer!")
                # Just merge fixed subtraction if anything
                pass
            
            # 2. Reconstruct Trainable Delta (Pruned)
            # Use index_select to physically reduce dimensions
            s_kept = self.lora_sigma.weight.view(-1)[kept_indices]
            u_kept = torch.index_select(self.lora_B.weight, 1, kept_indices) # (out, r_kept)
            v_kept = torch.index_select(self.lora_A.weight, 0, kept_indices) # (r_kept, in)

            # Delta Train = U_kept * S_kept * V_kept
            # Note: v_kept is already V^T shape from Linear definition
            delta_train = (u_kept * s_kept) @ v_kept

            # 3. Apply Scaling
            delta_W = delta_train * self.scaling
            
            # 4. Merge into Base Weight
            # Handle fan_in_fan_out (if weight is transposed)
            if self.fan_in_fan_out:
                delta_W = delta_W.T
            
            self.weight.data += delta_W.to(self.weight.device)
            self.merged = True
            
            # 6. Memory Cleanup (Optional but good practice)
            # We can delete lora parameters now, or keep them to avoid errors if called again
            # For this implementation, we set flag `merged` and subsequent forwards use `self.weight`
            print(f"SCULPT Layer Merged. Final Rank: {kept_indices.numel()}")

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        elif self.r_init > 0 and not self.merged:
            # 1. Base Forward
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
            # 2. SCULPT Delta Forward
            # Dropout on input
            x_dropped = self.lora_dropout(x.to(self.lora_A.weight.dtype))
            
            # --- Trainable Branch ---
            # sigma_effective = sigma * mask
            sigma_eff = self.lora_sigma.weight.view(-1) * self.mask
  
            # Compute:
            step1 = self.lora_A(x_dropped) # x @ A.T
            step2 = step1 * sigma_eff      # Broadcast mult
            term_train = self.lora_B(step2) # (x @ A.T @ S) @ B.T

            # Combine
            delta_output = term_train * self.scaling
            result += delta_output

        else:
            # Merged or no-adapter case
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result
    

class MergedLinear(nn.Linear, SCULPTLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        raise NotImplementedError
    

if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, SCULPTLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            Wdecompose: bool = False,
            **kwargs,
        ):
            raise NotImplementedError

    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, SCULPTLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            **kwargs,
        ):
            raise NotImplementedError

