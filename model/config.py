from dataclasses import dataclass

@dataclass
class ModelConfig:
    '''
    Hyperparameters:
        n_heads: Number of self-attention heads
        n_layers: Number of decoder layers
        n_embd: Embedding Dimension
        context_length: Maximum sequence length allowed
        vocab_size: Number of distinct characters in the database
        dropout: Dropout ratio
    '''
    
    n_heads: int
    n_layers: int
    n_embd: int
    context_length: int = 1024
    vocab_size: int = 50304 # GPT2 vocab size = 50257, we use 50304 because it is a multiple of 64 (GPU Optimization)
    dropout: float = 0.1
    
    # Allows loading from json
    @classmethod
    def from_json(cls, path):
        import json
        with open(path, "r") as f:
            params = json.load(f)
        return cls(**params)
    
    # Allows saving to json
    def save_to_json(self, path):
        import json
        from dataclasses import asdict
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
    
MAIN_MODEL_CONFIG = ModelConfig(
    n_heads=12,
    n_layers=12,
    n_embd=768,
)

DRAFT_MODEL_CONFIG = ModelConfig(
    n_heads=6,
    n_layers=6,
    n_embd=512,
)

if __name__ == "__main__":
    import os
    
    # Ensure the destination exists
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save directly next to the checkpoints
    print(f"Generating artifacts in {save_dir}/ ...")
    
    MAIN_MODEL_CONFIG.save_to_json(os.path.join(save_dir, "main_config.json"))
    DRAFT_MODEL_CONFIG.save_to_json(os.path.join(save_dir, "draft_config.json"))