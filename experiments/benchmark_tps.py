import torch
import time
import pandas as pd
from inference.generate import get_model
from data.prepare_data import tokenizer
from inference.speculative_engine import generate_speculative
from functools import partial
import argparse
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "benchmarks.csv")
STRESS_PATH = os.path.join(BASE_DIR, "stress_test.csv")

# Defining the prompts to use for benchmarking
PROMPTS = [
    "The capital of France is",
    "Artificial Intelligence works by",
    "The history of the Roman Empire is vast and"
]

def calculate_tps(generate_func, max_new_tokens, use_cache, method_name=None, device=DEVICE, prompts=PROMPTS, reset_callback=None, verbose=False):
    '''
    Calculates average tokens per second for the given model
    
    Args:
        generate_func: Function to call for token generation
        max_new_tokens: Maximum number of tokens to generate
        use_cache: Boolean variable to determine whether to use cache or not
        method_anme: Name of the method for which tps is being calculated
        device: Device to use for token generation
        prompts: List of prompts on which the benchmarks will be calculated
        reset_callback: if use_cache is True, then call reset_callback() to reset cache before generation
        verbose: if True, then print debugging statements
    '''
    
    timings = []
    tokens_generated = []
    acceptance_list = []
    mean_accepted_list = []
    
    # Warmup the model
    for _ in range(2):
        input_ids = torch.tensor([tokenizer.encode("Warmup")], dtype=torch.long, device=device)
        if reset_callback is not None:
            reset_callback()
        
        with torch.no_grad():
            _ = generate_func(input_ids, max_new_tokens=max_new_tokens, use_cache=use_cache)

    # Actual Test
    if verbose and method_name:
        print(f"\n>> Running benchmark for {method_name}...")
        
    for p in prompts:
        input_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        
        if reset_callback is not None:
            reset_callback()
            
        # Start timer
        if device == "cuda": torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = generate_func(input_ids, max_new_tokens=max_new_tokens, use_cache=use_cache)
        
        if isinstance(output, tuple):
            output_ids = output[0]
            acceptance_list.append(output[1])
            mean_accepted_list.append(output[2])
        else:
            output_ids = output
            
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
    
    # calculate avg acceptance and mean accepted for speculative engine
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0
    if any(a is not None for a in acceptance_list):
        avg_acceptance = sum(acceptance_list) / len(acceptance_list)
        avg_mean_accepted = sum(mean_accepted_list) / len(mean_accepted_list)
    else:
        avg_acceptance = None
        avg_mean_accepted = None
    
    return avg_tps, avg_acceptance, avg_mean_accepted
        
if __name__ == '__main__':
    print(f">> Benchmarking on: {DEVICE.upper()}\n")
    # CLI Arguments
    parser = argparse.ArgumentParser("Evaluate alignment between draft model and main model")

    parser.add_argument(
        "--gamma", type=int, default=5, help="Number of draft tokens to speculate per step"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate"
    )
    args = parser.parse_args()
    
    gamma = args.gamma
    max_new_tokens = args.max_new_tokens
    
    print("----- Running the benchmarks -----\n")
    
    # Load the models
    print(">> Loading Main Model...")
    main_model = get_model(model_name="main")
    print("\n>> Loading Draft Small Model...")
    draft_small_model = get_model(model_name="draft_small")
    print("\n>> Loading Draft Medium Model...")
    draft_medium_model = get_model(model_name="draft_medium")
    
    # Callback to reset cache
    def reset_main():
        for block in main_model.blocks:
            block.sa_heads.reset_cache()
        
    def reset_draft():
        for block in draft_small_model.blocks:
            block.sa_heads.reset_cache()
        
        for block in draft_medium_model.blocks:
            block.sa_heads.reset_cache()
        
    def reset_both():
        reset_draft()
        reset_main()
    
    if all([main_model, draft_small_model, draft_medium_model]):
        draft_models = {
            'small': draft_small_model,
            'medium': draft_medium_model
        }
        results = []
        
        main_model.eval()
        draft_small_model.eval()
        draft_medium_model.eval()
        
        # Calculate avg tps for main model
        tps_main_without_cache, _, _ = calculate_tps(generate_func=main_model.generate, max_new_tokens=max_new_tokens, method_name="main without cache", use_cache=False)
        tps_main_with_cache, _, _ = calculate_tps(generate_func=main_model.generate, max_new_tokens=max_new_tokens, method_name="main with cache", use_cache=True, reset_callback=reset_main)

        # Calculate avg tps for draft small model
        tps_draft_small_without_cache, _, _ = calculate_tps(generate_func=draft_small_model.generate, max_new_tokens=max_new_tokens, method_name="draft small without cache", use_cache=False)
        tps_draft_small_with_cache, _, _ = calculate_tps(generate_func=draft_small_model.generate, max_new_tokens=max_new_tokens, method_name="draft small with cache", use_cache=True, reset_callback=reset_draft)
        
        # Calculate avg tps for draft medium model
        tps_draft_medium_without_cache, _, _ = calculate_tps(generate_func=draft_medium_model.generate, max_new_tokens=max_new_tokens, method_name="draft medium without cache", use_cache=False)
        tps_draft_medium_with_cache, _, _ = calculate_tps(generate_func=draft_medium_model.generate, max_new_tokens=max_new_tokens, method_name="draft medium with cache", use_cache=True, reset_callback=reset_draft)
        
        print("\n===== BASELINE TPS =====")
        header = f"{'Model':<15} {'No Cache':>12} {'Cache':>12}"
        print(header)
        print(f"-" * len(header))
        
        print(f"{'Main':<15} {tps_main_without_cache:>12.2f} {tps_main_with_cache:>12.2f}")
        print(f"{'Draft Small':<15} {tps_draft_small_without_cache:>12.2f} {tps_draft_small_with_cache:>12.2f}")
        print(f"{'Draft Medium':<15} {tps_draft_medium_without_cache:>12.2f} {tps_draft_medium_with_cache:>12.2f}")
        
        baseline_tps = {
            "Main": (None, tps_main_without_cache, tps_main_with_cache),
            "Draft small": ("small", tps_draft_small_without_cache, tps_draft_small_with_cache),
            "Draft medium": ("medium", tps_draft_medium_without_cache, tps_draft_medium_with_cache),
        }

        # Store results for dataframe
        for method, (draft, tps_without, tps_with) in baseline_tps.items():
            results.append({
                "method": method,
                "draft": draft,
                "gamma": None,
                "cache": False,
                "tps": tps_without,
                "speedup":None,
                "acceptance":None,
                "mean_accepted":None
            })
            
            results.append({
                "method": method,
                "draft": draft,
                "gamma": None,
                "cache": True,
                "tps": tps_with,
                "speedup":None,
                "acceptance":None,
                "mean_accepted":None
            })
            
        # Calculate avg tps and speedup for speculative decoding
        print(f"\n===== Speculative (gamma = {gamma}) =====")
        header = f"{'Draft':<15} {'NoCache TPS':>12} {'Cache TPS':>12} {'NoCache x':>12} {'Cache x':>12}"
        print(header)
        print("-" * len(header))
        for draft_name, draft_model in draft_models.items():
            generate_func = partial(
                generate_speculative,
                main_model,
                draft_model,
                gamma=gamma,
                return_stats=True
            )
            
            tps_speculative_without_cache, _, _ = calculate_tps(generate_func=generate_func, max_new_tokens=max_new_tokens, method_name="speculative without cache", use_cache=False)
            tps_speculative_with_cache, _, _ = calculate_tps(generate_func=generate_func, max_new_tokens=max_new_tokens, method_name="speculative with cache", use_cache=True, reset_callback=reset_both)
            
            speedup_without_cache = tps_speculative_without_cache / tps_main_without_cache
            speedup_with_cache = tps_speculative_with_cache / tps_main_with_cache
            
            print(
                f"{draft_name:<15} "
                f"{tps_speculative_without_cache:>12.2f} "
                f"{tps_speculative_with_cache:>12.2f} "
                f"{speedup_without_cache:>12.2f} "
                f"{speedup_with_cache:>12.2f}"
            )
        
        # Perform gamma sweep
        print(f"\n>> Performing gamma sweep for benchmarking...")
        gamma_values = [1, 2, 3, 5, 7, 10]
        for gamma in gamma_values:
            print(f">> Gamma: {gamma}")
            for draft_name, draft_model in draft_models.items():
                generate_func = partial(
                    generate_speculative,
                    main_model,
                    draft_model,
                    gamma=gamma,
                    return_stats=True
                )
                
                tps_without, acceptance_without, mean_accepted_without = calculate_tps(generate_func=generate_func, max_new_tokens=max_new_tokens, method_name="speculative without cache", use_cache=False)
                tps_with, acceptance_with, mean_accepted_with = calculate_tps(generate_func=generate_func, max_new_tokens=max_new_tokens, method_name="speculative with cache", use_cache=True, reset_callback=reset_both)
                
                speedup_without = tps_without / tps_main_without_cache
                speedup_with = tps_with / tps_main_with_cache
                
                # Store results for dataframe
                results.append({
                    "method": "speculative",
                    "draft": draft_name,
                    "gamma": gamma,
                    "cache": False,
                    "tps": tps_without,
                    "speedup": speedup_without,
                    "acceptance": acceptance_without,
                    "mean_accepted": mean_accepted_without
                })
                
                results.append({
                    "method": "speculative",
                    "draft": draft_name,
                    "gamma": gamma,
                    "cache": True,
                    "tps": tps_with,
                    "speedup": speedup_with,
                    "acceptance": acceptance_with,
                    "mean_accepted": mean_accepted_with
                })
        
        print(f"\n>> Performing stress test for different context lengths...")
        stress_results = []
        
        for context_length in [16, 64, 128, 256, 512]:
            print(f">> Context Length: {context_length}")
            
            tps_main_cache, _, _ = calculate_tps(generate_func=main_model.generate, max_new_tokens=context_length, use_cache=True, reset_callback=reset_main)
            stress_results.append({
                'context_length': context_length,
                "configuration": "Main Baseline (With Cache)",
                "tps": tps_main_cache
            })
            
            generate_func = partial(
                generate_speculative,
                main_model,
                draft_medium_model,
                gamma=5
            )
            
            tps_speculative_without, _, _ = calculate_tps(generate_func=generate_func, max_new_tokens=context_length, use_cache=False)
            stress_results.append({
                "context_length": context_length,
                "configuration": "Speculative Medium (Without Cache)",
                "tps": tps_speculative_without
            })
            
            tps_speculative_with, _, _ = calculate_tps(generate_func=generate_func, max_new_tokens=context_length, use_cache=True, reset_callback=reset_both)
            stress_results.append({
                "context_length": context_length,
                "configuration": "Speculative Medium (With Cache)",
                "tps": tps_speculative_with
            })
            
        df = pd.DataFrame(results)
        df.to_csv(SAVE_PATH)
        print(f"\n>> Benchmarks result saved at {SAVE_PATH}")
        
        stress_df = pd.DataFrame(stress_results)
        stress_df.to_csv(STRESS_PATH)
        print(f"\n>> Stress Test result saved at {STRESS_PATH}")
        
    else:
        print(f"\n>> Error while loading models.")