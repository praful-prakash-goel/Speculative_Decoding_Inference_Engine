import torch
import time
from .generate import get_model
from data.prepare_data import tokenizer
from .speculative_engine import generate_speculative
from functools import partial

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Benchmarking on: {DEVICE.upper()}\n")

# Defining the prompts to use for benchmarking
PROMPTS = [
    "The capital of France is",
    "Artificial Intelligence works by",
    "The history of the Roman Empire is vast and"
]

def calculate_tps(generate_func, method_name, use_cache, device=DEVICE, prompts=PROMPTS, reset_callback=None):
    '''
    Calculates average tokens per second for the given model
    
    Args:
        generate_func: Function to call for token generation
        method_anme: Name of the method for which tps is being calculated
        use_cache: Boolean variable to determine whether to use cache or not
        device: Device to use for token generation
        prompts: List of prompts on which the benchmarks will be calculated
        reset_callback: if use_cache is True, then call reset_callback() to reset cache before generation
    '''
    
    timings = []
    tokens_generated = []
    
    # Warmup the model
    for _ in range(2):
        input_ids = torch.tensor([tokenizer.encode("Warmup")], dtype=torch.long, device=device)
        if reset_callback is not None:
            reset_callback()
            
        _ = generate_func(input_ids, max_new_tokens=512, use_cache=use_cache)

    # Actual Test
    print(f"\n>> Running benchmark for {method_name}...")
    for p in prompts:
        input_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        
        if reset_callback is not None:
            reset_callback()
            
        # Start timer
        if device == "cuda": torch.cuda.synchronize()
        start_time = time.time()
        
        output_ids = generate_func(input_ids, max_new_tokens=512, use_cache=use_cache)
        
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
    print("\n>> Loading Draft Model...")
    draft_model = get_model(model_name="draft")
    
    def reset_main():
        for block in main_model.blocks:
            block.sa_heads.reset_cache()
        
    def reset_draft():
        for block in draft_model.blocks:
            block.sa_heads.reset_cache()
    
    if main_model and draft_model:
        # Calculate avg tps for main model
        tps_main_without_cache = calculate_tps(generate_func=main_model.generate, method_name="main without cache", use_cache=False)
        tps_main_with_cache = calculate_tps(generate_func=main_model.generate, method_name="main with cache", use_cache=True, reset_callback=reset_main)

        # Calculate avg tps for draft model
        tps_draft_without_cache = calculate_tps(generate_func=draft_model.generate, method_name="draft without cache", use_cache=False)
        tps_draft_with_cache = calculate_tps(generate_func=draft_model.generate, method_name="draft with cache", use_cache=True, reset_callback=reset_draft)
        
        # Calculate avg tps for speculative decoding
        generate_func = partial(
            generate_speculative,
            main_model,
            draft_model,
            gamma=5
        )
        tps_speculative_without_cache = calculate_tps(generate_func=generate_func, method_name="speculative without cache", use_cache=False)
        tps_speculative_with_cache = calculate_tps(generate_func=generate_func, method_name="speculative with cache", use_cache=True, reset_callback=reset_draft)
        
        print(f"\n>> Average tokens per second for main model, without cache: {tps_main_without_cache}, with_cache: {tps_main_with_cache}")
        print(f">> Average tokens per second for draft model, without cache: {tps_draft_without_cache}, with_cache: {tps_draft_with_cache}")
        print(f">> Average tokens per second for speculative engine, without cache: {tps_speculative_without_cache}, with_cache: {tps_speculative_with_cache}")

        speedup_without_cache = tps_speculative_without_cache / tps_main_without_cache
        speedup_with_cache = tps_speculative_with_cache / tps_main_with_cache
        print(f"\nSpeedup, without_cache: {speedup_without_cache}, with_cache: {speedup_with_cache}")
    else:
        print(f">> Error while loading models.")