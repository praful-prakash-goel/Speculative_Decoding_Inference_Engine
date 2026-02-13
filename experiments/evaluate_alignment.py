import torch
from data.data_loader import get_batch
from inference.generate import get_model
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Computing alignment on: {DEVICE.upper()}\n")

@torch.no_grad()
def compute_agreement(main_model, draft_model, device=DEVICE, n_batches=20):
    '''
    Calculates the agreement rate between draft and mian model
    
    Args:
        main_model: The main model
        draft_model: The draft model
        device: Device to use for token generation
        n_batches: Number of batches over which the agreement rate should be calculated
    '''
    
    total = 0
    matches = 0
    
    for _ in range(n_batches):
        x, _ = get_batch("val")
        x = x.to(device=device)
        
        # Forward both models
        main_logits, _ = main_model(x, use_cache=False)
        draft_logits, _ = draft_model(x, use_cache=False)
        
        # Take last token prediction
        main_next = torch.argmax(main_logits[:, -1, :], dim=-1)
        draft_next = torch.argmax(draft_logits[:, -1, :], dim=-1)
        
        matches += (main_next == draft_next).sum().item()
        total += main_next.numel()
    
    return matches / total
    
        
if __name__ == '__main__':
    # CLI Arguments
    parser = argparse.ArgumentParser("Evaluate alignment between draft model and main model")
    
    parser.add_argument(
        "--draft_model", type=str, default="draft_medium",
        choices=["draft_small", "draft_medium"], help="Select draft model for evaluating alignment"
    )
    args = parser.parse_args()
    
    draft_model_name = args.draft_model
    
    print("----- Running the benchmarks -----\n")
    
    # Load the models
    print(">> Loading Main Model...")
    main_model = get_model(model_name="main")
    print("\n>> Loading Draft Model...")
    draft_model = get_model(model_name=draft_model_name)
    
    if main_model and draft_model:
        print(f"\n>> Computing alignment score...")
        # Calculate alignment score
        alignment_score = compute_agreement(main_model=main_model, draft_model=draft_model)
        
        print(f">> Alignment between main model and {draft_model_name} model: {alignment_score*100}%")