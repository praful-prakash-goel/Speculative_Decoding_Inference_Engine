import torch
import os
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")

context_length = 1024
batch_size = 64

train_data = None
val_data = None

def _init_data():
    """Helper to load data only when needed"""
    
    global train_data, val_data
    
    # If data is not available exit the system
    if not os.path.exists(TRAIN_BIN) or not os.path.exists(VAL_BIN):
        print("\n" + "="*50)
        print("ERROR: Dataset not found!")
        print(f"Looking for: {TRAIN_BIN}")
        print("="*50)
        print("Please run the data preparation script first:")
        print("  python data/prepare_data.py")
        print("="*50 + "\n")
        sys.exit(1)
    
    if train_data is None:
        train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
        val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')

def get_batch(split="train"):
    # Ensure data is loaded
    if train_data is None:
        _init_data()
    
    # get the data according to the split
    data = train_data if split == "train" else val_data
    # fetch random indices
    idxs = torch.randint(len(data) - context_length - 1, size=(batch_size,))
    
    # src ids (input)
    x = torch.stack([
        torch.from_numpy((data[i : i + context_length]).astype(np.int64))
        for i in idxs
    ])
    
    # tgt ids (shift by 1)
    y = torch.stack([
        torch.from_numpy((data[i + 1 : i + 1 + context_length]).astype(np.int64))
        for i in idxs
    ])
    
    return x, y