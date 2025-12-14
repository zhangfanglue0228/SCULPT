import torch

from transformers import Trainer

class svdloraTrainer(Trainer):
    def __init__(self, lambda_oc=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_oc = lambda_oc
    def compute_loss(self, model, inputs, return_outputs=False):
        original_return = super().compute_loss(model, inputs, return_outputs)
        if return_outputs:
            (original_loss, outputs) = original_return
        else:
            original_loss = original_return

        reg_loss = 0.0
        for module in model.modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                U = module.lora_B.weight
                V = module.lora_A.weight
            
                # Calculate the orthogonal loss of U^T U
                UtU = torch.mm(U.t(), U)
                I_U = torch.eye(UtU.size(0), device=UtU.device)
                reg_U = torch.linalg.matrix_norm(UtU - I_U, ord='fro') ** 2  # Frobenius norm squared
                
                # Calculate the orthogonal loss of V^T V
                VtV = torch.mm(V, V.t())
                I_V = torch.eye(VtV.size(0), device=VtV.device)
                reg_V = torch.linalg.matrix_norm(VtV - I_V, ord='fro') ** 2

                reg_loss += (reg_U + reg_V)
        
        loss = original_loss + self.lambda_oc * reg_loss

        return (loss, outputs) if return_outputs else loss