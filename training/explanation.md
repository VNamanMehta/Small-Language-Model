# Training Directory Explanation

## Overview
The `training` directory contains all components needed to train the language model, including dataset handling, optimization, and training loop orchestration.

## Files and Their Purpose

### 1. **trainer.py**
**Purpose**: Main training loop orchestrator.

**Why It's Required**:
- Handles the complete training pipeline (forward pass, loss computation, backward pass, optimization)
- Manages training state, checkpoints, and logging
- Coordinates interactions between model, optimizer, and dataset

**Key Responsibilities**:
- **Forward Pass**: Feeds batches through the model
- **Loss Computation**: Calculates cross-entropy loss between predictions and targets
- **Backward Pass**: Computes gradients via backpropagation
- **Optimization**: Updates model weights based on gradients
- **Checkpointing**: Saves model state at intervals for recovery and evaluation

**Why These Steps Matter**:
- Forward pass generates predictions
- Loss measures how wrong predictions are
- Backward pass computes gradient direction
- Optimization moves weights in direction that reduces loss

---

### 2. **optimizer.py**
**Purpose**: Implements optimization algorithms for updating model weights.

**Why It's Required**:
- Raw gradient descent is inefficient and often gets stuck in local minima
- Advanced optimizers adapt learning rates and accumulate momentum

**Common Optimizers Used**:

**SGD (Stochastic Gradient Descent)**
- Simple: `weight = weight - learning_rate × gradient`
- Problem: Can oscillate wildly with high learning rates

**Adam (Adaptive Moment Estimation)**
- Maintains exponential moving averages of gradients (first moment) and squared gradients (second moment)
- Formula: `weight = weight - learning_rate × m̂ / (√v̂ + ε)`
- **Why Adam is Popular**:
  - Adapts learning rate per parameter
  - Handles sparse gradients well
  - Combines momentum (first moment) and adaptive learning (second moment)
  - Converges faster than SGD

**AdamW (Adam with Weight Decay)**
- Adds weight decay (L2 regularization) properly
- Prevents overfitting by penalizing large weights

**Key Concepts**:
- **Learning Rate**: Controls step size for weight updates
- **Momentum**: Accumulates gradients to avoid oscillation
- **Adaptive Learning Rate**: Different parameters get different effective learning rates
- **Gradient Clipping**: Prevents exploding gradients in deep networks

---

### 3. **dataset.py**
**Purpose**: Handles data loading and batching for training.

**Why It's Required**:
- Transforms raw text into token sequences the model can process
- Creates proper train/validation splits
- Generates batches for efficient training

**Key Components**:
- **Tokenization**: Converts text → token IDs using trained tokenizer
- **Batching**: Groups sequences together for parallel processing
- **Sequence Creation**: Generates input-target pairs for language modeling
  - Input: tokens 0-999
  - Target: tokens 1-1000 (shifted by 1)
  - Model predicts next token given previous tokens

**Why Shifting by 1?**
- Creates supervised learning pairs
- For sequence `[a, b, c, d]`, creates:
  - Input `[a, b, c]` → Target `b`
  - Input `[a, b]` → Target `c`
  - etc.

**Data Loading Strategy**:
- Uses PyTorch DataLoader for efficient batching
- `num_workers`: Parallel data loading on multiple CPU cores
- `pin_memory`: Speeds up GPU data transfer on CUDA

---

### 4. **utils.py**
**Purpose**: Utility functions for training support.

**Why It's Required**:
- Reduces code duplication
- Provides common training operations

**Common Utilities**:
- **Seed Setting**: Ensures reproducibility across runs
- **Learning Rate Scheduling**: Adjusts learning rate during training
  - Warm-up: Gradually increase LR to prevent instability
  - Decay: Reduce LR over time for fine-tuning
- **Metrics Computation**: Calculates loss, perplexity, accuracy
- **Device Management**: Handles CPU/GPU switching
- **Logging**: Formats and saves training metrics

---

### 5. **__init__.py**
**Purpose**: Package initialization file.

**Why It's Required**:
- Marks the directory as a Python package
- Allows importing from training module: `from training import trainer`
- Can expose key classes/functions at package level

---

## Training Process Flow

```
Load Data
    ↓
Create DataLoader (batches)
    ↓
Initialize Model and Optimizer
    ↓
[For each epoch]:
  ├─ [For each batch]:
  │  ├─ Forward Pass: model(batch_input)
  │  ├─ Compute Loss: CrossEntropyLoss(predictions, targets)
  │  ├─ Backward Pass: loss.backward()
  │  ├─ Gradient Clipping: (prevent explosion)
  │  ├─ Optimizer Step: optimizer.step()
  │  └─ Optimizer Zero: optimizer.zero_grad()
  │
  ├─ Validation (measure generalization)
  ├─ Log Metrics
  └─ Save Checkpoint
    ↓
Training Complete
```

---

## Key Training Concepts

### 1. **Loss Function: Cross-Entropy**
- Measures difference between predicted probability distribution and true next token
- Formula: `L = -Σ(true_label × log(predicted_probability))`
- **Why Cross-Entropy?** Well-suited for classification tasks, numerically stable

### 2. **Backpropagation**
- Computes gradients by chain rule through all layers
- Gradient tells us which direction reduces loss
- Optimizer uses gradients to update weights

### 3. **Overfitting Prevention**
- **Validation Split**: Evaluate on unseen data during training
- **Early Stopping**: Stop if validation loss increases
- **Weight Decay**: Penalize large weights
- **Dropout**: Randomly disable neurons during training

### 4. **Batch Normalization Alternatives**
- **Layer Normalization**: Better for transformers, normalizes across features
- Stabilizes training and enables higher learning rates

---

## Why This Training Setup?

1. **Modular Design**: Easy to swap optimizers or datasets
2. **Efficient Data Loading**: Parallel batching accelerates training
3. **Adaptive Optimization**: Adam handles diverse gradient landscapes
4. **Checkpoint System**: Recover from interruptions; select best model
5. **Validation Monitoring**: Detect overfitting early

The training directory provides everything needed to efficiently teach the model to predict the next token!