
import torch
import argparse
import glob
from torch.utils.data import DataLoader
from model import GPT, ModelArgs
from training import PretokenizedDataset
import torch.nn.functional as F
import math
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
class EvalConfig:
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4

    # Hyperparameters
    batch_size = 64
    max_seq_len = 256
    eval_iters = 100 

    # Data
    val_data_path = "data/processed/validation.bin"
    out_dir = "checkpoints"


def get_latest_checkpoint(dir_path):
    """Finds the checkpoint with the highest iteration number."""
    files = glob.glob(f"{dir_path}/ckpt_*.pt")
    if not files:
        return None
    latest_file = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_file


def load_model(checkpoint_path, device):
    """Loads a model from a checkpoint."""
    print(f"⏳ Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    args = ModelArgs(
        dim=256,
        n_layers=8,
        n_heads=8,
        vocab_size=10000,
        max_seq_len=EvalConfig.max_seq_len,
        dropout=0.0
    )
    model = GPT(args)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
    return model


@torch.no_grad()
def evaluate(model, data_loader, device, eval_iters):
    """Evaluates the model on a subset of the data loader."""
    print(f"Evaluating for {eval_iters} iterations...")
    losses = []
    
    # Create a progress bar
    pbar = tqdm(data_loader, total=eval_iters, desc="Evaluating")

    for i, (x, y) in enumerate(pbar):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())

    if not losses:
        print("Warning: No batches were evaluated. Check eval_iters and data loader.")
        return float('nan'), float('nan')

    avg_loss = torch.tensor(losses).mean().item()
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT model.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint file. If not provided, the latest checkpoint is used.")
    args = parser.parse_args()

    print(f"🔥 Evaluating on Device: {EvalConfig.device.upper()}")

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print("No checkpoint specified. Searching for the latest one...")
        checkpoint_path = get_latest_checkpoint(EvalConfig.out_dir)

    if not checkpoint_path:
        print("❌ No checkpoint found. Please train a model first or specify a checkpoint path.")
        return

    # Load model
    model = load_model(checkpoint_path, EvalConfig.device)

    # Load validation data
    val_ds = PretokenizedDataset(EvalConfig.val_data_path, max_seq_len=EvalConfig.max_seq_len)
    val_loader = DataLoader(
        val_ds,
        batch_size=EvalConfig.batch_size,
        shuffle=False,
        num_workers=EvalConfig.num_workers,
        pin_memory=True
    )

    # Run evaluation
    avg_loss, perplexity = evaluate(model, val_loader, EvalConfig.device, EvalConfig.eval_iters)

    print("\n" + "="*30)
    print("✨ Evaluation Results ✨")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Average Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("="*30)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
