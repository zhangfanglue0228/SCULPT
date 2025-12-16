import torch
import logging
from transformers import Trainer
from .sculpt_scheduler import SculptScheduler

logger = logging.getLogger(__name__)

class SculptTrainer(Trainer):
    """
    Custom Trainer for SCULPT.
    
    Responsibilities:
    1. Integrate Orthogonal Regularization and L1 (Lasso) Regularization into the loss function.
    2. Manage the SCULPT Pruning Scheduler execution within the training loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize the Pruning Scheduler
        # We pass the model explicitly. If the model is wrapped (e.g., DDP), 
        # SculptScheduler handles finding the underlying layers.
        self.sculpt_scheduler = SculptScheduler(self.model)
        
        # Cache loss weights from config to avoid repeated attribute access
        if hasattr(self.model, "peft_config"):
            self.orth_reg_weight = self.model.peft_config.orth_reg_weight
            self.lasso_reg_weight = self.model.peft_config.lasso_reg_weight
        else:
            # Fallback or strict error. For safety, we log warning and default to 0.
            logger.warning("SCULPT config not found in model. Regularization losses will be disabled.")
            self.orth_reg_weight = 0.0
            self.lasso_reg_weight = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overridden to add SCULPT regularization terms.
        Loss = Task_Loss + lambda_orth * Orth_Loss + lambda_lasso * L1_Loss
        """
        # 1. Compute original task loss (CrossEntropy, etc.)
        original_return = super().compute_loss(model, inputs, return_outputs=return_outputs)
        
        if return_outputs:
            loss, outputs = original_return
        else:
            loss = original_return

        # 2. Compute Regularization Losses
        # We iterate over the cached layers in scheduler to save performance (no re-scanning model)
        orth_loss = 0.0
        lasso_loss = 0.0
        
        # Check if we need to compute reg loss (optimization)
        if self.orth_reg_weight > 0 or self.lasso_reg_weight > 0:
            for layer in self.sculpt_scheduler.sculpt_layers:
                # Accumulate Orthogonal Loss
                if self.orth_reg_weight > 0:
                    orth_loss += layer.get_orthogonal_loss()
                
                # Accumulate Lasso (L1) Loss on singular values
                if self.lasso_reg_weight > 0:
                    lasso_loss += layer.get_l1_norm()
            
            # 3. Combine Losses
            # Scale by their respective lambdas
            total_reg_loss = (self.orth_reg_weight * orth_loss) + (self.lasso_reg_weight * lasso_loss)
            loss += total_reg_loss

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs
        
        Inject the SCULPT scheduler step to update masks based on the gradients.
        """
        # 1. 执行父类
        loss = super().training_step(model, inputs)
        
        # 2. SCULPT 剪枝（仅在梯度同步时执行）
        if self.accelerator.sync_gradients:
            self.sculpt_scheduler.step(self.state.global_step)
        
        # 3. 返回 loss
        return loss