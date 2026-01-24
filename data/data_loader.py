import torch
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")

context_length = 64
batch_size = 32

train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')


def get_batch(split="train"):
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