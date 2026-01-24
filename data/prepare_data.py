import os
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
TRAIN_BIN = os.path.join(OUT_DIR, "train.bin")
VAL_BIN = os.path.join(OUT_DIR, "val.bin")

TRAIN_TOKENS = 500_000_000
VAL_TOKENS = 5_000_000

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def write_tokens_streaming(split_name, out_path, max_tokens):
    ds = load_dataset("openwebtext", split='train', streaming=True)
    ds = ds.shuffle(buffer_size=10_000, seed=1337)
    
    # buffer to store tokens and write at once
    buffer = []
    total = 0
    
    with open(out_path, "wb") as f:
        for ex in ds:
            text = ex['text']
            # tokenize the text using pre-trained gpt2 tokenizer
            ids = tokenizer.encode(text)
            
            # skip empty/tiny sequences
            if len(ids) < 10:
                continue
                
            buffer.extend(ids + [tokenizer.eos_token_id])
            if len(buffer) >= 1_000_000:
                # write at once
                arr = np.array(buffer[:1_000_000], dtype=np.uint16)
                f.write(arr.tobytes())
                total += len(buffer)
                buffer = buffer[1_000_000:]
                
                print(f"{split_name}: wrote {total}/{max_tokens}", end='\r')
                
                if total >= max_tokens:
                    break
    
    print(f"\nDone {split_name}. Wrote {total} tokens to {out_path}")
    
if __name__ == '__main__':
    write_tokens_streaming("val", VAL_BIN, VAL_TOKENS)
    write_tokens_streaming("train", TRAIN_BIN, TRAIN_TOKENS)