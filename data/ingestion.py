# data/ingestion.py
import os
from datasets import load_dataset, DatasetDict
from typing import cast

# Configuration
DATASET_NAME = "roneneldan/TinyStories"
RAW_DATA_PATH = "data/raw"

def main():
    print(f"📥 Downloading {DATASET_NAME} from Hugging Face Hub...")

    ds = cast(DatasetDict, load_dataset(DATASET_NAME))
    
    print(f"💾 Saving dataset to local disk: {RAW_DATA_PATH}...")
    # This saves it as Apache Arrow files (efficient, memory-mapped).
    # Subsequent loads from this path will be instant.
    ds.save_to_disk(RAW_DATA_PATH)
    
    print("✅ Ingestion complete.")
    print(f"   Train size: {len(ds['train'])}")
    print(f"   Validation size: {len(ds['validation'])}")

if __name__ == "__main__":
    main()