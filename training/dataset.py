import torch
import numpy as np
from torch.utils.data import Dataset

class PretokenizedDataset(Dataset):
    def __init__(self, bin_path: str, max_seq_len: int):
        self.max_seq_len = max_seq_len
        # Load the binary file as a memory-mapped array (doesn't consume RAM)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
    def __len__(self):

        return len(self.data) - self.max_seq_len - 1

    def __getitem__(self, idx):
        # Grab a chunk of data
        chunk = self.data[idx : idx + self.max_seq_len + 1]
        
        # Convert to Tensor (int64 is required for PyTorch embedding indices)
        chunk = torch.from_numpy(chunk.astype(np.int64))
        
        # Input (x) and Target (y)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y