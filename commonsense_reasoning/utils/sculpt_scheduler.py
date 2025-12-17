import math
import torch
import torch.nn as nn
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class SculptScheduler:
    """
    Global Pruning Scheduler for SCULPT.
    Controls the dynamic rank allocation across all SCULPT layers using a cubic schedule.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the scheduler.
        
        Args:
            model (nn.Module): The model containing SCULPTLinear layers.
                               Must have a valid `peft_config` attribute.
        """
        self.model = model
        self.config = model.peft_config
        
        # Load schedule parameters from config
        self.t_start = self.config.t_start
        self.t_end = self.config.t_end
        self.pruning_freq = self.config.pruning_freq
        
        # Count SCULPT layers and calculate total rank bounds
        self.sculpt_layers = self._get_sculpt_layers(model)
        self.num_layers = len(self.sculpt_layers)
        
        if self.num_layers == 0:
            logger.warning("SculptScheduler initialized but no SCULPTLinear layers found in model.")
        
        # Calculate Rank Budget bounds
        # Note: All layers share the same r and r_init from config
        self.r_target_per_layer = self.config.r
        self.r_init_per_layer = self.config.r * self.config.init_r_multiplier
        
        self.total_rank_init = self.num_layers * self.r_init_per_layer
        self.total_rank_final = self.num_layers * self.r_target_per_layer
        
        logger.info(
            f"SculptScheduler Initialized:\n"
            f"  - Num Layers: {self.num_layers}\n"
            f"  - Initial Total Rank: {self.total_rank_init} (Per layer: {self.r_init_per_layer})\n"
            f"  - Final Total Rank: {self.total_rank_final} (Per layer: {self.r_target_per_layer})\n"
            f"  - Schedule: Start={self.t_start}, End={self.t_end}, Freq={self.pruning_freq}"
        )

    def _get_sculpt_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        Helper to extract all SCULPTLinear layers from the model.
        Using string check to avoid circular imports if strictly decoupled.
        """
        if hasattr(model, "sculpt_layers"):
            return list(model.sculpt_layers)
        
        # 处理可能的 DDP/Model Parallel 包装情况
        if hasattr(model, "module") and hasattr(model.module, "sculpt_layers"):
            return list(model.module.sculpt_layers)
        
        layers = []
        for module in model.modules():
            # Check class name or attribute to identify SCULPTLinear
            if getattr(module, "is_sculpt_layer", False):
                layers.append(module)
        return layers

    def calculate_budget(self, global_step: int) -> int:
        """
        Calculates the total rank budget B(t) using a Cubic Schedule.
        
        Formula:
            B(t) = B_final + (B_init - B_final) * (1 - (t - t_start) / (t_end - t_start))^3
        """
        # 1. Warmup Phase: Keep full initial rank
        if global_step < self.t_start:
            return self.total_rank_init
        
        # 2. Final Phase: Fixed at target rank
        if global_step >= self.t_end:
            return self.total_rank_final
        
        # 3. Pruning Phase: Cubic Decay
        progress = (global_step - self.t_start) / (self.t_end - self.t_start)
        decay_factor = (1 - progress) ** 3
        
        current_budget = self.total_rank_final + (self.total_rank_init - self.total_rank_final) * decay_factor
        return int(current_budget)

    def step(self, global_step: int):
        """
        Executes the pruning step if the conditions are met.
        Should be called after loss.backward() but before optimizer.step().
        
        Args:
            global_step (int): Current training step.
        """
        # Check frequency
        if global_step % self.pruning_freq != 0:
            return

        # Check range (Optionally allows one final update at t_end to clamp exact budget)
        if global_step < self.t_start or global_step > self.t_end:
            return

        # 1. Calculate Target Budget B(t)
        current_budget = self.calculate_budget(global_step)
        
        # 2. Collect Importance Scores from ALL layers
        all_scores_list = []
        for layer in self.sculpt_layers:
            # We move scores to CPU to ensure we can concatenate them 
            # regardless of which GPU the layer is on.
            score = layer.get_importance_score().detach().to(device="cpu", dtype=torch.float32)
            all_scores_list.append(score)
        
        if not all_scores_list:
            return

        # Concatenate all scores into one massive vector
        global_scores = torch.cat(all_scores_list)

        if global_scores.sum() == 0:
            logger.warning(
                f"[SCULPT Warning] Step {global_step}: Detected all-zero importance scores. "
                "Gradients might be missing or zero. Pruning is skipped to ensure safety."
            )
            return
        
        # 3. Determine Global Threshold
        # We need to keep top `current_budget` elements.
        # So we find the k-th largest value, where k = current_budget.
        
        # Safety check: budget cannot exceed number of available ranks (though unlikely with cubic logic)
        total_available_ranks = global_scores.numel()
        k = min(current_budget, total_available_ranks)
        k = max(k, 1) # Keep at least 1 rank globally to avoid NaN
        
        # torch.kthvalue finds the k-th smallest, so for k-th largest we can sort descending
        # Sorting is safer and easier to debug than topk/kthvalue for arbitrary k on CPU
        sorted_scores, _ = torch.sort(global_scores, descending=True)
        threshold = sorted_scores[k - 1].item() # Index is k-1
        
        # 4. Update Masks
        # Broadcast the scalar threshold to all layers
        # layer.update_mask handles the device movement internally
        pruned_count = 0
        total_kept = 0
        
        for layer in self.sculpt_layers:
            layer.update_mask(threshold)
            # Optional: gather stats
            total_kept += layer.mask.sum().item()
        
        metrics = {
            "sculpt/budget": current_budget,
            "sculpt/threshold": threshold,
            "sculpt/sparsity": 1.0 - (total_kept / self.total_rank_init),
            "sculpt/kept_ranks": total_kept
        }

        # Logging
        if global_step % (self.pruning_freq * 10) == 0: # Log less frequently to avoid spam
            current_sparsity = 1.0 - (total_kept / self.total_rank_init)
            logger.info(
                f"[SCULPT Step {global_step}] "
                f"Budget: {current_budget} | Kept: {int(total_kept)} | "
                f"Threshold: {threshold:.6f} | Global Sparsity: {current_sparsity:.2%}"
            )
        
        return metrics