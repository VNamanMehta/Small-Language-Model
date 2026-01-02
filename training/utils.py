import torch
import os

def save_checkpoint(model, optimizer, iter_num, out_dir, model_args):
    """Saves the model state to a file."""
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'model_args': model_args,
    }
    torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
    print(f"Saved checkpoint to {out_dir}")

@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters=50):
    """
    Estimates loss efficiently by looking at a random sample of batches
    instead of the entire dataset (which takes too long).
    """
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= eval_iters: break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train() # Switch back to train mode
    return sum(losses) / len(losses)