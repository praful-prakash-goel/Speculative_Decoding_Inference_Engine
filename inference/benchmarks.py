import torch
import time
from .generate import get_model
from data.prepare_data import tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Benchmarking on: {DEVICE.upper()}\n")

# Defining the prompts to use for benchmarking
PROMPTS = [
    "The capital of France is",
    "Artificial Intelligence works by",
    "The history of the Roman Empire is vast and"
]

def calculate_tps(generate_func, model_name, device=DEVICE, prompts=PROMPTS):
    '''
    Calculates average tokens per second for the given model
    
    Args:
        generate_func: Function to call for token generation
        model_name: Name of the model being used
        device: Device to use for token generation
        prompts: List of prompts on which the benchmarks will be calculated
    '''
    
    timings = []
    tokens_generated = []
    
    # Warmup the model
    for _ in range(2):
        input_ids = torch.tensor([tokenizer.encode("Warmup")], dtype=torch.long, device=device)
        _ = generate_func(input_ids, max_new_tokens=100)

    # Actual Test
    print(f"\n>> Running benchmark for {model_name}...")
    for p in prompts:
        input_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        
        # Start timer
        if device == "cuda": torch.cuda.synchronize()
        start_time = time.time()
        
        output_ids = generate_func(input_ids, max_new_tokens=100)
        
        # Stop timer
        if device == "cuda": torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate time taken and number of tokens generated
        duration = end_time - start_time
        total_len = output_ids.shape[1]
        input_len = input_ids.shape[1]
        generated_tokens = total_len - input_len
        
        timings.append(duration)
        tokens_generated.append(generated_tokens)
    
    # Calculate avg tps
    total_time = sum(timings)
    total_tokens = sum(tokens_generated)
    avg_tps = total_tokens / total_time
    
    return avg_tps
        
if __name__ == '__main__':
    print("----- Running the benchmarks -----\n")
    
    # Load the models
    print(">> Loading Main Model...")
    main_model = get_model(model_name="main")
    print(">> Loading Draft Model...")
    draft_model = get_model(model_name="draft")
    
    if main_model and draft_model:
        # Calculate avg tps for main model
        tps_main_without_cache = calculate_tps(generate_func=main_model.generate, model_name="main")
        tps_main_with_cache = calculate_tps(generate_func=main_model.generate_with_cache, model_name="main")

        # Calculate avg tps for draft model
        tps_draft_without_cache = calculate_tps(generate_func=draft_model.generate, model_name="draft")
        tps_draft_with_cache = calculate_tps(generate_func=draft_model.generate_with_cache, model_name="draft")

        
        print(f"\n>> Average tokens per second for main model, without cache: {tps_main_without_cache}, with_cache: {tps_main_with_cache}")
        print(f">> Average tokens per second for draft model, without cache: {tps_draft_without_cache}, with_cache: {tps_main_with_cache}")
    else:
        print(f">> Error while loading models.")