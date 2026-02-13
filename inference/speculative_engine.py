import torch
from .generate import get_model
from data.prepare_data import tokenizer
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CACHE = os.environ.get("USE_CACHE", "True").lower()

def generate_speculative(main_model, draft_model, input_ids, max_new_tokens=512, device=DEVICE, gamma=5, use_cache=False, return_stats=False):
    # print(f"\n>> Input ids : {input_ids}")
    if use_cache:
        # reset cache
        for block in draft_model.blocks:
            block.sa_heads.reset_cache()
        
    
    speculated_ids = draft_model.generate(
        input_ids,
        max_new_tokens=gamma,
        use_cache=use_cache,
        repetition_penalty=1.5
    )
    
    tokens_generated = 0
    total_accepted_tokens = 0
    draft_generated_tokens = 0
    total_steps = 0
    while tokens_generated < max_new_tokens:
        # print(f"\n>> Speculated ids: {tokenizer.decode(speculated_ids[0].tolist())}")
        input_len = input_ids.shape[1]
        draft_tokens = speculated_ids[:, input_len:]
        
        # print(f"DEBUG: Main Model Input Length: {speculated_ids.shape[1]}")
        # print(f"DEBUG: Main Model Input Text: {tokenizer.decode(speculated_ids[0])[-50:]}") # Print last 50 chars
        target_logits, _ = main_model(speculated_ids, use_cache=False)
        
        start_idx = input_len - 1
        end_idx = input_len + gamma - 1
        
        for i in range(start_idx, end_idx + 1):
            step_logits = target_logits[:, i, :]
            current_history = speculated_ids[:, :i+1]
            
            target_logits[:, i, :] = main_model.apply_repetition_penalty(
                step_logits,
                current_history,
                repetition_penalty=1.5
            )
        
        verification_logits = target_logits[:, start_idx : end_idx, :]
        main_tokens = torch.argmax(verification_logits, dim=-1)
        
        # print(f"\n>> Draft tokens: {tokenizer.decode(draft_tokens[0].tolist())}")
        # print(f"\n>> Main tokens: {tokenizer.decode(main_tokens[0].tolist())}")
        accepted_tokens = 0
        for i in range(gamma):
            draft_token = draft_tokens[0, i]
            main_token = main_tokens[0, i]
            
            if draft_token == main_token:
                accepted_tokens += 1
            else:
                break
        
        correction_token = torch.argmax(target_logits[:, input_len + accepted_tokens - 1, :], dim=-1)
        # print(f"\nCorrect token : {tokenizer.decode(correction_token)}")
        valid_draft = draft_tokens[:, :accepted_tokens]
        # print(f"\nValid draft : {tokenizer.decode(valid_draft[0].tolist())}")
        
        new_tokens = torch.cat([valid_draft, correction_token.unsqueeze(0)], dim=1)
        # print(f"\nnew tokens : {tokenizer.decode(new_tokens[0].tolist())}")
        input_ids = torch.cat([input_ids, new_tokens], dim=1)
        # print(f"\ninput ids : {tokenizer.decode(input_ids[0].tolist())}")
        
        tokens_generated += (accepted_tokens + 1)
        total_accepted_tokens += accepted_tokens
        draft_generated_tokens += gamma
        total_steps += 1
        
        if use_cache:
            valid_len = input_len + accepted_tokens
            for block in draft_model.blocks:
                block.sa_heads.truncate_cache(valid_len)
            
            draft_input = correction_token.unsqueeze(0)
        else:
            draft_input = input_ids
        
        next_chunk = draft_model.generate(
            draft_input,
            max_new_tokens=gamma,
            use_cache=use_cache,
            repetition_penalty=1.5
        )
        
        input_length_passed = draft_input.shape[1]
        newly_generated_gamma = next_chunk[:, input_length_passed:]
        # print(f"\nNewly generated gamma: {tokenizer.decode(newly_generated_gamma[0].tolist())}")
        speculated_ids = torch.cat([input_ids, newly_generated_gamma], dim=1)
        # print(f"DEBUG: Input Len: {input_ids.shape[1]} | New Guesses: {newly_generated_gamma.shape[1]}")
        
    
    acceptance_rate = total_accepted_tokens / draft_generated_tokens
    mean_accepted = total_accepted_tokens / total_steps
    
    if return_stats:
        return input_ids, acceptance_rate, mean_accepted
    else:
        return input_ids
            
if __name__ == '__main__':
    print(">> Loading Main Model...")
    main_model = get_model(model_name="main")
    main_model.eval()
    print("\n>> Loading Draft Model...")
    draft_model = get_model(model_name="draft")
    draft_model.eval()
    
    prompt = input("\nPlease enter the prompt: ")
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=DEVICE
    )
    
    output_ids, acceptance_rate, mean_accepted = generate_speculative(main_model, draft_model, input_ids, gamma=5, use_cache=True, return_stats=True)
    text = tokenizer.decode(output_ids[0].tolist())
    
    print(f"\n>> Output: {text}")
    print(f"\nAcceptance rate: {acceptance_rate}")
    print(f"\nMean Accepted tokens: {mean_accepted}")
    