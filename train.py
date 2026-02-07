import torch
from data.data_loader import get_batch
from model.model_architecture import build_model
from model.config import MAIN_MODEL_CONFIG, DRAFT_MODEL_CONFIG
from inference.generate import generate
import os
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = os.environ.get("MODEL_NAME", "main").lower()
base_dir = "saved_models"
os.makedirs(base_dir, exist_ok=True)

checkpoint_path = os.path.join(base_dir, f"{model_name}_model.pt")
config_path = os.path.join(base_dir, f"{model_name}_config.json")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# phase 1 params for draft model
max_iters = 20_000
warmup_steps = 1_000
eval_iters = 20
eval_interval = 1_000
accumulation_steps = 16
base_lr = 3e-4
weight_decay = 0.1

@torch.no_grad()
def estimate_loss(model):
    '''
    Calculate train and validation loss
    
    Args:    
        model: Model which is to be evaluated
    '''
    
    output = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        
        for iter in range(eval_iters):
            # Sample a batch
            x, y = get_batch(split=split)
            x = x.to(device)
            y = y.to(device)
            
            # Calculate loss
            _, loss = model(x, y)
            losses[iter] = loss.item()
            
        output[split] = losses.mean()
        
    model.train()
    return output

def get_lr(step, base_lr, warmup_steps, total_steps):
    '''
    Learning rate warmup with cosine decay
    
    Args:    
        step: Current step in which the model is
        base_lr: Base learning rate set at the starting of the training
        warmup_steps: Number of steps to warmup the learning rate
        total_steps: Total steps for which the model has to be trained
    '''
    
    # Warmup the learning rate
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    # Cosine decay
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
def train_model():
    print("Using:", torch.cuda.get_device_name(0))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    # Build model
    if model_name.lower() == 'main':
        model_config = MAIN_MODEL_CONFIG
        # save config to json
        MAIN_MODEL_CONFIG.save_to_json(config_path)
    elif model_name.lower() == 'draft':
        model_config = DRAFT_MODEL_CONFIG
        # save config to json
        DRAFT_MODEL_CONFIG.save_to_json(config_path)
    else:
        print(">> Only two models are available to train: 'main' and 'draft'. Please enter a valid model name")
        return
    print(f">> Config saved to artifact: {config_path}")
        
    model = build_model(device=device, config=model_config)
    print(f">> Training {model_name.lower()} model: {sum(p.numel() for p in model.parameters())/1e6}M Parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # If checkpoint is stored, then load it else start fresh
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['val_loss']
        start_iter = checkpoint['iter'] + 1
        print(f"Restored model from checkpoint at step: {checkpoint['iter']}")
    else:
        start_iter = 0
        best_val_loss = float('inf')
    
    end_iter = start_iter + max_iters
    max_updates = max_iters // accumulation_steps
    
    # Training loop
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for iter in range(start_iter, end_iter):
        if iter % eval_interval == 0 or iter+1 == end_iter:
            # Evaluate after fixed interval
            losses = estimate_loss(model)
            train_loss, val_loss = losses['train'], losses['val']
            train_perplexity, val_perplexity = torch.exp(train_loss), torch.exp(val_loss)
            
            print(f">> Step {iter} - Train Loss: {train_loss}, Train PPL: {train_perplexity}, Val Loss: {val_loss}, Val PPL: {val_perplexity}")
            
            # If current val_loss is less than best_val_loss, then store the checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "iter": iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at step {iter} - train_loss : {train_loss}, val_loss : {val_loss}")
            
        if iter % 10_000 == 0 or iter + 1 == end_iter:
            # Test inference of the model after every 10k micro iterations
            print("\n", "=="*50, sep="")
            prompt = "In the future, artificial intelligence will"
            generate(prompt=prompt, model=model)
            print("=="*50, "\n", sep="")
            
        # Fetch a training batch
        x, y = get_batch("train")
        x = x.to(device)
        y = y.to(device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
            
        # Normalize the loss
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights only after every accumulation steps
        if (iter + 1) % accumulation_steps == 0 or (iter + 1) == end_iter:
            # Gradient clipping to prevent exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update lr
            lr_now = get_lr(step=optimizer_step, base_lr=base_lr, warmup_steps=warmup_steps, total_steps=max_updates)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_now
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
        
    
if __name__ == '__main__':
    train_model()