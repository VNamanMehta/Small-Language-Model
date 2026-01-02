import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPT, ModelArgs

# =============================================================================
# Configuration
# =============================================================================
CHECKPOINT_PATH = "checkpoints/ckpt_36000.pt"
TOKENIZER_PATH = "data/tokenizer/tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generation settings
PROMPT = "Once upon a time, there was a huge dragon"
MAX_NEW_TOKENS = 200   # How much to generate
TEMPERATURE = 0.8      # Higher = Creative/Random, Lower = Logical/Safe
TOP_K = 200            # Restrict to top K probable tokens

def load_model(checkpoint_path):
    """
    Loads the model weights from a checkpoint.
    NOTE: We must instantiate the model with the SAME args used in training.
    """
    print(f"⏳ Loading model from {checkpoint_path}...")
    
    # 1. Define the Architecture (must match training)
    args = ModelArgs(
        dim=256,
        n_layers=8,
        n_heads=8,
        vocab_size=10000,
        max_seq_len=256,
        dropout=0.0
    )
    
    # 2. Initialize Model
    model = GPT(args)
    model.to(DEVICE)
    
    # 3. Load State Dict
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    
    # The trainer saved the model under the key 'model'
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval() 
    print("✅ Model loaded successfully.")
    
    return model

def generate(model, tokenizer, prompt, max_new_tokens, temperature=1.0, top_k=None):
    """
    The Core Generation Loop.
    """
    # 1. Encode the prompt
    print(f"\n📝 Prompt: {prompt}")
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded.ids, dtype=torch.long, device=DEVICE).unsqueeze(0) # (1, T)

    print("🤖 Generating...")
    # 2. Auto-regressive Loop
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.args.max_seq_len:]
        
        with torch.no_grad(): 
            # Forward pass
            logits, _ = model(idx_cond)
            
            logits = logits[:, -1, :] # (B, Vocab_Size)
            
            # --- Sampling Logic ---
            
            # A. Apply Temperature
            # High temp (e.g., 1.2) flattens distribution -> More random
            # Low temp (e.g., 0.1) sharpens distribution -> More predictable
            logits = logits / temperature
            
            # B. Apply Top-K Filtering
            if top_k is not None:
                # Keep only the top K probabilities, mask the rest as -infinity
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # C. Convert to Probabilities
            probs = F.softmax(logits, dim=-1)
            
            # D. Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

    # 3. Decode and Print
    generated_text = tokenizer.decode(idx[0].tolist())
    return generated_text

def main():
    # 1. Load Tokenizer
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA not available. Generation will be slow.")
        
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # 2. Load Model
    try:
        model = load_model(CHECKPOINT_PATH)
    except FileNotFoundError:
        print(f"❌ Checkpoint not found at {CHECKPOINT_PATH}")
        print("   Please run train.py first to generate a checkpoint.")
        return

    # 3. Generate
    output = generate(
        model, 
        tokenizer, 
        PROMPT, 
        MAX_NEW_TOKENS, 
        TEMPERATURE, 
        TOP_K
    )
    
    print("-" * 50)
    print(output)
    print("-" * 50)

if __name__ == "__main__":
    main()