import os
import torch
from model.model_architecture import build_model
from model.config import MAIN_MODEL_CONFIG, DRAFT_MODEL_SMALL_CONFIG, DRAFT_MODEL_MEDIUM_CONFIG, ModelConfig
from data.prepare_data import tokenizer
import sys
import argparse

BASE_DIR = "saved_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_config(model_name, config_path=None):
    '''
    Retrieves config for a specific model name
    
    Args:
        model_name: Name of the model for which the config has to be loaded
        config_path: Path where to look for the config file
    '''
    
    # Load config from json file if it exists
    if config_path and os.path.exists(config_path):
        print(f">> Loading config from artifact: {config_path}")
        return ModelConfig.from_json(config_path)

    # Fallback to hardcoded python object is json does not exists
    print(f">> JSON not found. Using hardcoded python config for {model_name} model...")
    if model_name == 'main':
        return MAIN_MODEL_CONFIG
    elif model_name == 'draft_small':
        return DRAFT_MODEL_SMALL_CONFIG
    elif model_name == 'draft_medium':
        return DRAFT_MODEL_MEDIUM_CONFIG
    else:
        raise ValueError(f">> Unknown model name: {model_name}")
        
def get_model(model_name, checkpoint_dir=BASE_DIR, device=DEVICE):
    '''
    Loads model by name: "main" or "draft"
    
    Args:
        model_name: Name of the model to load
        checkpoint_dir: Base dir from where the checkpoint will be loaded
        device: Device to use
    '''
    
    # Construct path dynamically based on the model requested
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_model.pt")
    config_path = os.path.join(checkpoint_dir, f"{model_name}_config.json")
    
    # Load the checkpoint of the requested model if exist
    if os.path.exists(checkpoint_path):
        # Load config
        config = get_config(model_name, config_path)
        
        # Build the model
        print(f">> Building {model_name} model...")
        model = build_model(device=device, config=config)
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
def generate(prompt = None, model = None, device=DEVICE, max_new_tokens=512, use_cache=True):
    '''
    Generate the output tokens based on the given prompt
    
    Args:
        prompt: Initial decoder tokens
        model: Model to use for generation of the tokens
        device: Device to use for generation
        max_new_tokens: Maximum number of tokens to generate
        use_cache: Boolean variable to determine whether to use cache or not
    '''
    
    if model == None:
        print(f"No model found, either checkpoint doesn't exist or the model is not passed as the parameter.")
        sys.exit()
    
    # If model is using cache, then reset cache before generation
    if use_cache:
        for block in model.blocks:
            block.sa_heads.reset_cache()
            
    # Take prompt as input if not already provided
    if prompt is None:
        prompt = input("Please enter the prompt: ")
    # Tokenize the prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    # Generate output for the given input ids
    model.eval()
    with torch.no_grad():
        output = model.generate(
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.5,
            use_cache=use_cache
        )[0].tolist()
    text = tokenizer.decode(output)
    
    print(f"\n>> Output: {text}")
    
if __name__ == '__main__':
    # CLI Arguments
    parser = argparse.ArgumentParser(description="Generate Function")
    
    parser.add_argument(
        "--model", type=str, default="main",
        choices=["main", "draft_small", "draft_medium"], help="Select model to use for generation"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable KV cache"
    )
    args = parser.parse_args()
    
    model_name = args.model
    use_cache = not args.no_cache
    
    # Load the model and generate the output
    model = get_model(model_name=model_name)
    generate(model=model, use_cache=use_cache)