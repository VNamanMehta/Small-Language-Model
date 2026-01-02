import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelArgs
from .transformer import TransformerBlock
from .model_utils import RMSNorm, precompute_freqs_cis

class GPT(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        self.tok_embeddings.weight = self.output.weight # Weight tying
        
        self.freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, 
            self.args.max_seq_len * 2
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[:T].to(x.device)

        for layer in self.layers:
            x = layer(x, freqs_cis)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @classmethod
    def from_default(cls):
        return cls(ModelArgs())