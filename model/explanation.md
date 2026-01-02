# Model Directory Explanation

## Overview
The `model` directory contains the core neural network architecture for the small language model. It implements a GPT/Llama-style transformer architecture with attention mechanisms.

## Files and Their Purpose

### 1. **config.py**
**Purpose**: Centralized configuration management for the model.

**Why It's Required**: 
- Defines hyperparameters (embedding dimensions, number of layers, attention heads, etc.)
- Ensures consistency across training, evaluation, and inference
- Makes it easy to experiment with different model sizes without modifying multiple files

**Key Concepts**:
- `vocab_size`: Number of unique tokens in tokenizer vocabulary
- `block_size`: Maximum sequence length the model can process
- `n_layer`: Number of transformer blocks (depth)
- `n_head`: Number of attention heads (for multi-head attention)
- `n_embd`: Embedding dimension size

---

### 2. **attention.py**
**Purpose**: Implements the self-attention mechanism and multi-head attention.

**Why It's Required**:
- Self-attention is the core mechanism that allows the model to weigh relationships between tokens
- Without attention, the model cannot learn context and dependencies between words

**Theory**:
- **Self-Attention**: Computes query (Q), key (K), and value (V) transformations
- **Math**: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`
- **Why softmax**: Converts attention scores to probability distribution (sum to 1)
- **Why divide by √d_k**: Prevents attention scores from growing too large, stabilizing gradients
- **Masking**: Prevents future tokens from influencing past tokens (causal masking) in language modeling

**Multi-Head Attention**:
- Splits embedding into multiple "heads" to learn different types of relationships
- Allows parallel learning of different attention patterns

---

### 3. **transformer.py**
**Purpose**: Implements the transformer block (attention + feedforward layers).

**Why It's Required**:
- Stacks attention and feedforward layers to create deep representations
- Each transformer block processes and refines token representations

**Components**:
- **Layer Normalization**: Normalizes inputs before each sub-layer for training stability
- **Attention Layer**: Computes relationships between tokens
- **Feedforward Network**: Two linear layers with ReLU activation for non-linearity
- **Residual Connections**: `output = sublayer(x) + x` - helps gradients flow during backpropagation

**Why Residual Connections?**
- Prevents vanishing gradient problem in deep networks
- Allows information to bypass layers without modification

---

### 4. **gpt.py**
**Purpose**: The main GPT model class that orchestrates all components.

**Why It's Required**:
- Combines token embeddings, positional embeddings, transformer blocks, and output layers
- Provides the complete forward pass for training and inference

**Key Components**:
- **Token Embedding**: Converts token IDs to dense vectors
- **Positional Embedding**: Adds position information so the model knows token order
- **Transformer Blocks**: Multiple stacked transformer layers
- **Language Modeling Head**: Linear layer that outputs logits for next token prediction

**Why Positional Embedding?**
- Self-attention is permutation-invariant (doesn't care about order)
- Positional embeddings encode absolute positions so the model understands sequence order

---

### 5. **model_utils.py**
**Purpose**: Utility functions for model operations.

**Why It's Required**:
- Provides helper functions like model initialization, weight setup, and inference utilities
- Reduces code duplication across the codebase
- Handles model device management (CPU/GPU)

**Common Utilities**:
- Weight initialization strategies (Xavier, normal distribution)
- Model freezing/unfreezing for fine-tuning
- Checkpoint loading and saving

---

## Architecture Summary

```
Input Tokens (token_ids)
    ↓
Token Embedding + Positional Embedding
    ↓
[Transformer Block × n_layer]
  ├─ Multi-Head Self-Attention
  ├─ Layer Norm + Residual Connection
  ├─ Feedforward Network
  └─ Layer Norm + Residual Connection
    ↓
Output Linear Layer (Language Modeling Head)
    ↓
Logits → Probabilities (softmax) → Next Token
```

---

## Why This Architecture?

1. **Attention Mechanism**: Learns long-range dependencies efficiently
2. **Multi-Head Attention**: Captures diverse relationship patterns
3. **Transformer Blocks**: Enables deep stacking with residual connections
4. **Layer Normalization**: Stabilizes training and accelerates convergence
5. **Positional Embeddings**: Preserves sequence order information

This design allows the model to understand context and generate coherent text.