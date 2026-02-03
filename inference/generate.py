import os
import torch
from model.model_architecture import build_model
from model.config import MAIN_MODEL_CONFIG, DRAFT_MODEL_CONFIG, ModelConfig
from data.prepare_data import tokenizer
import sys

model_name = os.environ.get("MODEL_NAME", "main").lower()
base_dir = "saved_models"

checkpoint_path = os.path.join(base_dir, f"{model_name}_model.pt")
config_path = os.path.join(base_dir, f"{model_name}_config.json")
config = DRAFT_MODEL_CONFIG if model_name.lower() == 'draft' else MAIN_MODEL_CONFIG
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_config():
    # Load config from json file if it exists
    if os.path.exists(config_path):
        print(f">> Loading config from artifact: {config_path}")
        return ModelConfig.from_json(config_path)

    # Fallback to hardcoded python object is json does not exists
    print(f">> JSON not found. Using hardcoded python config for {model_name} model...")
    if model_name == 'main':
        return MAIN_MODEL_CONFIG
    elif model_name == 'draft':
        return DRAFT_MODEL_CONFIG
    else:
        raise ValueError(f">> Unknown model name: {model_name}")
        
def get_model():
    if os.path.exists(checkpoint_path):
        # Load config
        config = get_config()
        
        # Build the model
        print(f">> Building {model_name} model...")
        model = build_model(device=device, config=config)
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    else:
        return None
        
def generate(prompt = None):
    model = get_model()
    
    if model == None:
        print(f"No checkpoint found at {checkpoint_path}. Please train the model and then try again.")
        sys.exit()
    
    if prompt is None:
        prompt = input("Please enter the prompt: ")
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    model.eval()
    with torch.no_grad():
        output = model.generate_with_cache(
            idx=input_ids,
            max_new_tokens=200,
            temperature=1.0,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )[0].tolist()
    text = tokenizer.decode(output)
    
    print(f"\n>> Output: {text}")
    
if __name__ == '__main__':
    generate()