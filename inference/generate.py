import os
import torch
from model.model_architecture import build_model
from model.config import MAIN_MODEL_CONFIG
from data.prepare_data import tokenizer
import sys

model_name = os.environ.get("MODEL_NAME", "main")
checkpoint_path = f"saved_models/{model_name.lower()}_model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    if os.path.exists(checkpoint_path):
        # Later change it for the checkpoint to include model config
        model = build_model(device=device, config=MAIN_MODEL_CONFIG)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    else:
        return None
        
def generate():
    model = get_model()
    
    if model == None:
        print("No trained model exists. Please train the model and then try again.")
        sys.exit()
    
    prompt = input("Please enter the prompt: ")
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    model.eval()
    with torch.no_grad():
        output = model.generate_with_cache(
            idx=input_ids,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_k=50
        )[0].tolist()
    text = tokenizer.decode(output)
    
    print(f">> Output: {text}")
    
if __name__ == '__main__':
    generate()