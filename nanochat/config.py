"""
config for tiny nanochat
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """gpt model config"""
    # model architecture
    block_size: int = 256  # context length (reduced for memory)
    vocab_size: int = 100256  # tiktoken cl100k_base vocab size
    n_layer: int = 3  # number of transformer layers (reduced)
    n_head: int = 4  # number of attention heads
    n_embd: int = 192  # embedding dimension (reduced)
    dropout: float = 0.1
    bias: bool = False  # use bias in layernorm/linear layers
    
    # training
    batch_size: int = 4  # reduced for memory
    gradient_accumulation_steps: int = 4  # effective batch size = 16
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 10  # reduced for memory
    sample_interval: int = 200  # generate samples every N iters
    warmup_iters: int = 100
    min_lr: float = 3e-5
    
    # data
    data_path: str = "training_data/chatml_format.txt"
    max_data_examples: int = 5000  # limit examples loaded into memory (None = all)
    
    # output
    out_dir: str = "out"
    always_save_checkpoint: bool = True
    
    # generation
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 200

