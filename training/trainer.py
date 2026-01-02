import torch
from tqdm import tqdm
import os
from .utils import save_checkpoint, estimate_loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config 

    def train(self):
        self.model.train()
        
        # Define precision type
        ctx = torch.autocast(device_type=self.device, dtype=torch.bfloat16)

        pbar = tqdm(self.train_loader, desc="Training")
        
        self.iter_num = getattr(self, 'iter_num', 0) 

        for x, y in pbar:
            self.iter_num += 1
            iter_num = self.iter_num 
    
            x, y = x.to(self.device), y.to(self.device)
            
            # 1. Forward pass with Mixed Precision
            with ctx:
                logits, loss = self.model(x, targets=y)
            
            # 2. Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # 3. Logging
            if iter_num % self.config.log_interval == 0:
                print(f"Step {iter_num}: Loss {loss.item():.4f}")

            # 4. Evaluation & Checkpointing
            if iter_num > 0 and iter_num % self.config.eval_interval == 0:
                val_loss = estimate_loss(self.model, self.val_loader, self.device)
                print(f"\nValidation Loss: {val_loss:.4f}")

            # 5. Checkpointing: Save model every 1000 steps
            if iter_num > 0 and iter_num % self.config.save_interval == 0:
                # Convert ModelArgs to a dictionary before saving
                model_args_dict = self.model.args.__dict__
                save_checkpoint(self.model, self.optimizer, iter_num, "checkpoints/", model_args_dict)