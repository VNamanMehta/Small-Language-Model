from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 256            # Embedding dimension
    n_layers: int = 8         # Number of Transformer blocks
    n_heads: int = 8          # Number of Query heads
    n_kv_heads: Optional[int] = None # Number of Key/Value heads (for GQA)
    vocab_size: int = 10000   # Vocabulary size
    multiple_of: int = 32     # SwiGLU hidden layer size multiple
    max_seq_len: int = 512    # Context window
    dropout: float = 0.0      # Dropout probability