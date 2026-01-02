import os
import multiprocessing
import numpy as np
from datasets import load_from_disk, DatasetDict
from tokenizers import Tokenizer
from tqdm import tqdm
from typing import cast

# ==========================================
# Configuration & Constants
# ==========================================
RAW_DATA_PATH = "data/raw"               # Where the Arrow dataset is saved 
PROCESSED_DATA_PATH = "data/processed"   # Where we will save the binary files
TOKENIZER_PATH = "data/tokenizer/tokenizer.json" 
NUM_PROC = multiprocessing.cpu_count()   # Use ALL available CPU cores

def write_shard(args):
    """
    Worker function: Writes a specific chunk of token IDs to the shared binary file.
    This runs simultaneously on multiple cores.
    """
    shard_idx, shard_ds, filename, start_idx = args
    
    # 1. Calculate the exact size of this worker's chunk
    shard_len = sum(len(x) for x in shard_ds["ids"])
    
    # 2. Open the "Sniper Scope" on the shared file
    # - filename: The massive binary file created by the main process.
    # - mode='r+': Read/Write mode. Crucially, this DOES NOT wipe the file.
    # - offset: The specific byte position where this worker starts writing.
    #   We multiply start_idx by 2 because we use uint16 (2 bytes per number).
    arr = np.memmap(
        filename, 
        dtype=np.uint16, 
        mode='r+', 
        shape=(shard_len,), 
        offset=start_idx * 2 
    )
    
    # 3. The Writing Loop
    # We write data relative to our own window (arr).
    # arr[0] corresponds to the global file position 'start_idx'.
    idx = 0
    for example in shard_ds:
        ids = example["ids"]
        arr[idx : idx + len(ids)] = ids
        idx += len(ids)
    
    # 4. Flush to disk to ensure data is saved
    arr.flush()
    
    return shard_len

def main():
    print(f"--- Starting Parallel Data Preparation ({NUM_PROC} Cores) ---")
    
    # Load resources
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ds = cast(DatasetDict, load_from_disk(RAW_DATA_PATH))

    # ======================================================
    # PHASE 1: Parallel Tokenization
    # ======================================================
    def process_batch(examples):
        # Encode a batch of text into IDs
        batch_outputs = tokenizer.encode_batch(examples["text"])
        return {"ids": [out.ids for out in batch_outputs]}

    print(f"🚀 Tokenizing input text...")
    # ds.map is the standard for high-performance processing.
    # It splits the data and runs 'process_batch' on all cores.
    tokenized_ds = ds.map(
        process_batch,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=["text"], # Drop raw text to save RAM
        desc="Tokenizing"
    )

    # ======================================================
    # PHASE 2: Parallel Binary Packing
    # ======================================================
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    for split in ["train", "validation"]:
        print(f"\n📦 Processing split: {split}")
        
        # A. Pre-calculate Total Size
        # We map over the dataset just to get lengths (very fast).
        # We need the total count to create the empty file first.
        lengths = tokenized_ds[split].map(
            lambda x: {"len": len(x["ids"])},
            num_proc=NUM_PROC,
            remove_columns=["ids"], 
            keep_in_memory=True
        )["len"]
        
        total_len = sum(lengths)
        filename = os.path.join(PROCESSED_DATA_PATH, f"{split}.bin")
        
        # B. Allocate the Empty File
        # mode='w+' creates a new file (or overwrites) and fills it with 50GB of zeros.
        print(f"   Allocating file: {filename} ({total_len / 1e6:.1f}M tokens)")
        fp = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(total_len,))
        fp.flush()
        del fp  # Close the file handle immediately so workers can open it
        
        # C. Prepare Tasks for Workers
        # We divide the dataset into N shards (one for each core).
        print("   Calculating worker offsets...")
        shard_size = len(tokenized_ds[split]) // NUM_PROC
        tasks = []
        offset_in_file = 0
        
        for i in range(NUM_PROC):
            # 1. Determine which slice of stories this worker processes
            start = i * shard_size
            end = (i + 1) * shard_size if i < NUM_PROC - 1 else len(tokenized_ds[split])
            shard = tokenized_ds[split].select(range(start, end))
            
            # 2. Calculate where this slice begins in the binary file
            # The offset is the sum of all lengths processed by previous workers.
            shard_tokens_count = sum(lengths[start:end])
            
            # 3. Create the task tuple
            tasks.append((i, shard, filename, offset_in_file))
            
            # Update offset for the next worker
            offset_in_file += shard_tokens_count

        # D. Execute Parallel Writing
        # We use a Pool to launch all workers at once.
        print(f"   Writing to disk with {NUM_PROC} streams...")
        with multiprocessing.Pool(NUM_PROC) as pool:
            for _ in tqdm(pool.imap_unordered(write_shard, tasks), total=NUM_PROC):
                pass

    print("\n✅ Data preparation complete!")

if __name__ == "__main__":
    main()

"""
Example Flow: How it prevents data corruption
Imagine we have 4 Stories and 2 CPU Cores.

Story 1: 10 tokens
Story 2: 20 tokens
Story 3: 15 tokens
Story 4: 25 tokens
Total: 70 tokens

Step 1: Allocation (Main Process)
The main process sums the lengths (70) and creates a train.bin file on the hard drive with 70 empty slots (zeros).

Step 2: Task Assignment
The script calculates the "Starting Line" for each core:
Core A (handling Story 1 & 2):
Starts at Index 0.
Needs to write 30 tokens (10 + 20).
Core B (handling Story 3 & 4):
Starts at Index 30 (because Core A took the first 30 slots).
Needs to write 40 tokens (15 + 25).

Step 3: Parallel Execution 
Core A opens the file with offset=0. It sees a window from 0-30. It fills it with Story 1 and 2.
Core B opens the file with offset=60 (30 tokens x 2 bytes). It sees a window starting at slot 30. It cannot see or touch Core A's data. 
It fills its window with Story 3 and 4.
Because they have strictly defined, non-overlapping windows, they can write at full speed simultaneously without corrupting the file.
"""