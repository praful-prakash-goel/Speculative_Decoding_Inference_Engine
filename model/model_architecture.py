import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

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

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    Apply rotation to query and key tensors
    """
    
    def __init__(self, dim, max_seq_len, base=10_000):
        '''
        Args:
            dim: Dimension of the self attention head
            max_seq_len: Maximum sequence length defined to pre-compute the full table
            base: RoPE frequency base (θ). Controls the rate of rotation across positions.
                  larger base -> slower-changing angles, higher base -> faster-changing angles
        '''
        
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # It implements the formula theta_i = base ^ (-2i/d)
        # Earlier dimensions will have higher frequency and later dimensions will have lower frequency
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Pre-compute full table for max_seq_len
        t = torch.arange(max_seq_len).float()
        # time x speed = angle; i -> time, j -> speed
        freqs = torch.einsum("i,j->ij", t, inv_freq) # shape: (max_seq_len, head_dim/2)
        
        # Cache cos and sin tables
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)
    
    def forward(self, seq_len, device, offset=0):
        '''
        Args:
            seq_len: Length of the current sequence/input
            device: Device on which embeddings should be loaded
            offset: Starting index of the current sequence/input
        '''
        
        # Lookup cos and sin tables for given seq_len
        if seq_len + offset > self.max_seq_len:
            # If we somehow exceed max_seq_len, then recalculate dynamically.
            t = torch.arange(seq_len, device=device).float() + offset
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            return freqs.cos(), freqs.sin()
        
        # Fast path: slice the cached cos and sin tables
        return (
            self.cos_cached[offset : offset + seq_len].to(device),
            self.sin_cached[offset : offset + seq_len].to(device)
        )

def rotate_half(x):
    '''
    Args:
        x: Input Sequence of shape: (batch, n_heads, seq_len, head_dim)
    '''
    
    # Rotate the last dimension pair
    x1 = x[..., ::2]    # Take indices 0, 2, 4... (The 'x' coordinate)
    x2 = x[..., 1::2]   # Take indices 1, 3, 5... (The 'y' coordinate)
    
    return torch.stack((-x2, x1), dim=-1).flatten(-2)
    
def apply_rope(q, k, cos, sin):
    '''
    Args:
        q: Query tensor of shape: (..., seq_len, head_dim)
        k: Key tensor of shape: (..., seq_len, head_dim)
        cos: Cosine values for RoPE of shape: (seq_len, head_dim/2)
        sin: Sine values for RoPE of shape: (seq_len, head_dim/2)
    '''
    
    # Expand cos and sin to full head dimension
    cos = torch.repeat_interleave(cos, 2, dim=-1) # [c1, c2] -> [c1, c1, c2, c2]
    sin = torch.repeat_interleave(sin, 2, dim=-1) # [s1, s2] -> [s1, s1, s2, s2]
    
    cos = cos.unsqueeze(0).unsqueeze(0) # shape: (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0) # shape: (1, 1, seq_len, head_dim)
    
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot, k_rot
    
class SelfAttention(nn.Module):
    """Multple Attention Heads in parallel"""
    
    def __init__(self, config, n_kv_heads, head_dim):
        '''
        Args:
            config: Configuration/Hyperparameters for the model
            n_kv_heads: Number of self attention heads for key and value vectors
                        n_kv_heads = n_heads -> MHA; n_kv_heads < n_heads -> GQA; n_kv_heads = 1 -> MQA
            head_dim: Dimension of one self-attention head
        '''
        
        super().__init__()
        self.config = config
        assert config.n_heads % n_kv_heads == 0, "Invalid GQA"
        self.n_heads = config.n_heads
        self.n_kv_heads  = n_kv_heads
        self.head_dim = head_dim
        
        # Query: n_heads * head_dim
        self.query = nn.Linear(config.n_embd, config.n_heads * head_dim, bias=False)
        # Key and Value: n_kv_heads * head_dim each
        self.key = nn.Linear(config.n_embd, n_kv_heads * head_dim, bias=False)
        self.value = nn.Linear(config.n_embd, n_kv_heads * head_dim, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=config.context_length)
        
        self.k_cache = None
        self.v_cache = None
        self.curr_len = 0
        
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.curr_len = 0
    
    def append_cache(self, T, new_k, new_v):
        '''
        Args:
            T: Sequence length (T == 1 -> during incremental decoding AND T >= 1 -> during prefill and verification)
            new_k: New key vector
            new_v: New value vector
        '''
        self.k_cache[:, :, self.curr_len : self.curr_len + T] = new_k
        self.v_cache[:, :, self.curr_len : self.curr_len + T] = new_v
        self.curr_len += T
    
    def truncate_cache(self, new_len):
        '''
        Args:
            new_len: New sequence length after some tokens have been rejected by the main model
        '''
        # Truncate cache if some tokens are rejected by main model
        self.curr_len = new_len
    
    def forward(self, x, use_cache=False):
        '''
        Args:
            x: Input sequence of shape: (B, T, C)
            use_cache: Boolean variable to determine whether to use KV Cache or not
        '''
        
        B, T, C = x.shape
        # Compute the query, key and value vectors
        q = self.query(x)
        k_org = self.key(x)
        v_org = self.value(x)
        # Reshape q, k and v vectors
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_org = k_org.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_org = v_org.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        rope_offset = self.curr_len if use_cache else 0
        cos, sin = self.rope(seq_len=T, device=x.device, offset=rope_offset)
        q, k_org = apply_rope(q, k_org, cos, sin)
        
        if self.n_heads != self.n_kv_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = torch.repeat_interleave(k_org, repeat_factor, dim=1)
            v = torch.repeat_interleave(v_org, repeat_factor, dim=1)
        else:
            k = k_org
            v = v_org
        
        if use_cache:
            # Use Key-Value Caching
            if self.k_cache is None:
                self.k_cache = torch.zeros(
                    (1, self.n_kv_heads, self.config.context_length, self.head_dim),
                    dtype=x.dtype, device=x.device
                )
                self.v_cache = torch.zeros_like(self.k_cache)
                self.curr_len = 0
            
            # If cache is full, then stop generation
            assert self.curr_len + T <= self.config.context_length, \
                "KV Cache overflow: Generation exceeded context length"
            
            # Fetch old key and value vectors upto curr_len
            old_k = self.k_cache[:, :, :self.curr_len]
            old_v = self.v_cache[:, :, :self.curr_len]
            
            # Expand cache
            if self.n_heads != self.n_kv_heads:
                old_k = torch.repeat_interleave(old_k, repeat_factor, dim=1)
                old_v = torch.repeat_interleave(old_v, repeat_factor, dim=1)
                
            # Use the full key and value vectors for attention
            full_k = torch.cat([old_k, k], dim=2)
            full_v = torch.cat([old_v, v], dim=2)
            total_len = self.curr_len + T
            
            mask = torch.ones((T, total_len), device=x.device, dtype=torch.bool)
            mask[:, :self.curr_len] = True # Attend to all past tokens
            mask[:, self.curr_len:] = torch.tril(mask[:, self.curr_len:]) # causal mask within chunk 
                
            # Calculate attention scores
            out = F.scaled_dot_product_attention(
                q, full_k, full_v,
                dropout_p=self.config.dropout if self.training else 0.0,
                attn_mask=mask,
                is_causal=False
            )
            
            self.append_cache(T, k_org, v_org)
        else:  
            # Calculate attention scores through SDPA for better speed
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )
            
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (Batch, Time, Channel)
        return self.proj(out)
    
class FeedForwardNetwork(nn.Module):
    """A simple linear layer followed by a non-linear layer"""
    
    def __init__(self, config):
        '''
        Args:
            config: Configuration/Hyperparameters for the model
        '''
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        '''
        Args:
            x: Input sequence of shape: (B, T, C)
        '''
        
        return self.net(x)

class Block(nn.Module):
    """Decoder Block having all the components"""
    
    def __init__(self, config, n_kv_heads):
        '''
        Args:
            config: Configuration/Hyperparameters for the model
            n_kv_heads: Number of self attention heads for key and value vectors
                        n_kv_heads = n_heads -> MHA; n_kv_heads < n_heads -> GQA; n_kv_heads = 1 -> MQA
        '''
        
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_heads
        # Self Attention heads
        self.sa_heads = SelfAttention(config, n_kv_heads, head_dim)
        # Feed-forward Neural Network
        self.ffwd = FeedForwardNetwork(config)
        # RMSNorm
        self.rms1 = nn.RMSNorm(config.n_embd)
        self.rms2 = nn.RMSNorm(config.n_embd)
    
    def forward(self, x, use_cache=False):
        '''
        Args:
            x: Input sequence of shape: (B, T, C)
            use_cache: Boolean variable to determine whether to use KV Cache or not
        '''
        
        x = x + self.sa_heads(self.rms1(x), use_cache=use_cache) # Pre-normalization, residual connection + output from sa_heads
        x = x + self.ffwd(self.rms1(x)) # Pre-normalization, residual connection + output from ffwd net
        return x

class DecoderModel(nn.Module):
    def __init__(self, config):
        '''
        Args:
            config: Configuration/Hyperparameters for the model
        '''
        
        super().__init__()
        self.config = config
        # Token Embedding Table
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Multiple Decoder Blocks in sequence
        self.blocks = nn.ModuleList([
            Block(config=config, n_kv_heads=config.n_heads//4)
            for _ in range(config.n_layers)    
        ])
        # Final RMS Norm
        self.rms_f = nn.RMSNorm(config.n_embd)
        # Language Model Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, use_cache=False):
        '''
        Args:
            idx: Input sequence of shape: (B, T)
                where B is batch size and T is sequence length.
            targets: Target sequence of shape: (B, T)
                    If provided, cross-entropy loss is computed.
            use_cache: Boolean variable to determine whether to use KV Cache or not
        '''
        
        x = self.token_embedding(idx) # (Batch, Time, Channels), Channels = n_embd
        # RoPE handles positional encoding inside each block
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.rms_f(x)
        logits = self.lm_head(x) # (Batch, Time, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (N, C) -> N no. of samples (each with C scores)
            targets = targets.view(B*T) # (N,) -> N class indices (0, C-1)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None
    ):
        '''
        Args:
            idx: (B, T_start) initial decoder token ids (prompt)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (controls the creativity of the model)
            do_sample: If True sample, else greedy argmax
            top_k: If sampling and top_k provided, restrict to top_k (controls the diversity of sampling)
        '''
        
        self.eval()
        # Generate Tokens autoregressively
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length:]
            # Generate initial logits
            logits, _ = self(idx_cond)
            # Take only the last logit
            last_logits = logits[:, -1, :] # (B, C)
            
            if temperature != 1.0 and temperature > 0.0:
                last_logits = last_logits / temperature
                
            if do_sample:
                probs = F.softmax(last_logits, dim=-1)
                if top_k is not None and top_k > 0:
                    # Take the top_k probs sorted in descending order
                    topk_vals, _ = probs.topk(top_k, dim=-1) # (B, k)
                    # Take the minimum of the top_k probs
                    min_topk = topk_vals[..., -1].unsqueeze(-1) # (B, 1)
                    allowed_mask = probs >= min_topk
                    # Mask all the probs less than min allowed prob
                    probs = probs * allowed_mask.to(probs.dtype)
                    # Normalize the probs
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # If sampling is not allowed then take the argmax of the logits
                idx_next = torch.argmax(last_logits, dim=-1, keepdim=True)
                
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        
        return idx
    
    @torch.no_grad()
    def generate_with_cache(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None
    ):
        '''
        Args:
            idx: Initial decoder token ids (prompt)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (Controls the creativity of the model)
            do_sample: If True sample else greedy argmax
            top_k: If sampling and top_k is provided, restrict to top_k (controls the diversity of sampling)
        '''
        
        self.eval()
        # reset cache
        for block in self.blocks:
            block.sa_heads.reset_cache()
        
        # Prefill the decoder cache once
        logits, _ = self(idx, use_cache=True)
        last_logits = logits[:, -1, :]
        
        if temperature != 1.0 and temperature > 0.0:
            last_logits = last_logits / temperature
            
        if do_sample:
            probs = F.softmax(last_logits, dim=-1)
            if top_k is not None and top_k > 0:
                # Take the top_k probs sorted in descending order
                topk_vals, _ = probs.topk(top_k, dim=-1) # (B, k)
                # Take the minimum of the top_k probs
                min_topk = topk_vals[..., -1].unsqueeze(-1) # (B, 1)
                allowed_mask = probs >= min_topk
                # Mask all the probs which are less than min_topk
                probs = probs * allowed_mask.to(probs.dtype)
                # Normalize the probs
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # If sampling is not allowed then take the argmax of the logits
            idx_next = torch.argmax(last_logits, dim=-1, keepdim=True)
        
        idx = torch.cat([idx, idx_next], dim=1)
        full_seq = idx.clone()
        
        for _ in range(max_new_tokens):
            # Take only the last token. Position is relative to the cache size
            last_token = idx[:, -1:]
            
            logits, _ = self(last_token, use_cache=True)
            last_logits = logits[:, -1, :]
            
            if temperature != 1.0 and temperature > 0.0:
                last_logits = last_logits / temperature
            
            if do_sample:
                probs = F.softmax(last_logits, dim=-1)
                if top_k is not None and top_k > 0:
                    # Take the top_k probs sorted in descending order
                    topk_vals, _ = probs.topk(top_k, dim=-1) # (B, k)
                    # Take the minimum of the topk_vals
                    min_topk = topk_vals[..., -1].unsqueeze(-1) # (B, 1)
                    allowed_mask = topk_vals >= min_topk
                    # Allow only those probs which are greater than min_topk
                    probs = probs * allowed_mask.to(probs.dtype)
                    # Normalize the probs
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # If sampling is not allowed then take the argmax of logits
                idx_next = torch.argmax(last_logits, dim=-1, keepdim=True)
            
            idx = torch.cat([idx, idx_next], dim=1)
            full_seq = torch.cat([full_seq, idx_next], dim=1)
        
        return full_seq
    
def build_model(device, config):
    model = DecoderModel(config=config)
    model.to(device=device)
    
    return model