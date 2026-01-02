import torch.nn as nn
from torch.optim import AdamW

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """
    Configures the optimizer, separating parameters that should be decayed (weights)
    from those that shouldn't (biases, norms, embeddings).
    """
    # 1. Filter parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    valid_param_ids = set(id(p) for p in param_dict.values())
    
    decay_params = []
    nodecay_params = []
    seen_ids = set()
    
    # Whitelist / Blacklist
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    # 2. Iterate modules to determine decay policy
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            
            if id(p) not in valid_param_ids:
                continue
            
            if id(p) in seen_ids:
                continue
            
            seen_ids.add(id(p))

            if pn.endswith('bias'):
                # All biases -> No decay
                nodecay_params.append(p)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # Linear weights -> Decay
                decay_params.append(p)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # Embeddings / Norms -> No decay
                nodecay_params.append(p)
            elif p.dim() == 1:
                # Heuristic: Any 1D parameter (like RMSNorm scale) -> No decay
                nodecay_params.append(p)
            else:
                # Default: Decay everything else
                decay_params.append(p)

    # 3. Create the optimizer
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    use_fused = (device_type == 'cuda')
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)

    return optimizer