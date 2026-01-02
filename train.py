import torch
import os
import glob
from torch.utils.data import DataLoader
from model import GPT, ModelArgs
from training import Trainer, PretokenizedDataset, configure_optimizers

# =============================================================================
# Configuration
# =============================================================================
class TrainConfig:
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4         
    
    # Hyperparameters
    batch_size = 64         
    max_seq_len = 256       
    learning_rate = 5e-4    
    max_iters = 5000        
    
    # Logging
    log_interval = 10
    eval_interval = 200
    save_interval = 1000
    out_dir = "checkpoints"

def get_latest_checkpoint(dir_path):
    """Finds the checkpoint with the highest iteration number."""
    files = glob.glob(f"{dir_path}/ckpt_*.pt")
    if not files:
        return None
    
    # Extract version number and sort
    latest_file = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_file

# =============================================================================
# Main Training Script
# =============================================================================
def main():
    print(f"🔥 Training on Device: {TrainConfig.device.upper()}")
    
    # 1. Load Data
    train_ds = PretokenizedDataset("data/processed/train.bin", max_seq_len=TrainConfig.max_seq_len)
    val_ds = PretokenizedDataset("data/processed/validation.bin", max_seq_len=TrainConfig.max_seq_len)

    train_loader = DataLoader(
        train_ds, 
        batch_size=TrainConfig.batch_size, 
        shuffle=True, 
        num_workers=TrainConfig.num_workers, 
        pin_memory=True 
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=TrainConfig.batch_size, 
        shuffle=False, 
        num_workers=TrainConfig.num_workers, 
        pin_memory=True
    )

    # 2. Initialize Model
    args = ModelArgs(
        dim=256,            
        n_layers=8,         
        n_heads=8, 
        vocab_size=10000,
        max_seq_len=TrainConfig.max_seq_len,
        dropout=0.0
    )
    
    model = GPT(args)
    model.to(TrainConfig.device)

    print(f"   Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # 3. Optimizer
    optimizer = configure_optimizers(
        model, 
        weight_decay=0.1, 
        learning_rate=TrainConfig.learning_rate, 
        betas=(0.9, 0.95), 
        device_type=TrainConfig.device
    )
    
    # --- AUTO-RESUME LOGIC ---
    start_iter = 0
    latest_ckpt = get_latest_checkpoint(TrainConfig.out_dir)
    
    if latest_ckpt:
        print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=TrainConfig.device)
        
        # Load model weights
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        model.load_state_dict(state_dict)
        
        # Load optimizer state (momentum, etc.)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Set starting iteration
        start_iter = checkpoint['iter_num'] + 1
        print(f"   Resuming at step {start_iter}")
    else:
        print("🆕 No checkpoint found. Starting from scratch.")

    # 4. Start Training
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        TrainConfig.device, 
        TrainConfig()
    )
    
    trainer.iter_num = start_iter 
    
    trainer.train()

if __name__ == "__main__":
    import os
    os.makedirs(TrainConfig.out_dir, exist_ok=True)
    torch.set_float32_matmul_precision('high')
    
    main()