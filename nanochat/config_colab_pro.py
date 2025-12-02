"""
config for colab pro (a100/l4) - larger model, more data
"""

from dataclasses import dataclass
from nanochat.config import GPTConfig


@dataclass
class GPTConfigColabPro(GPTConfig):
    """gpt model config optimized for colab pro gpus"""
    # model architecture - larger for a100/l4
    block_size: int = 512  # longer context
    vocab_size: int = 100256  # tiktoken cl100k_base vocab size
    n_layer: int = 6  # more layers
    n_head: int = 8  # more heads
    n_embd: int = 512  # larger embeddings (~15M params)
    dropout: float = 0.1
    bias: bool = False
    
    # training - larger batches for faster training
    batch_size: int = 16  # larger batch
    gradient_accumulation_steps: int = 2  # effective batch size = 32
    learning_rate: float = 3e-4
    max_iters: int = 10000  # more iterations
    eval_interval: int = 200
    eval_iters: int = 20  # more eval samples
    sample_interval: int = 300
    warmup_iters: int = 200
    min_lr: float = 3e-5
    
    # data - load more examples
    data_path: str = "training_data/chatml_format.txt"
    max_data_examples: int = 20000  # more data
    
    # output
    out_dir: str = "out"
    always_save_checkpoint: bool = True
    
    # generation
    max_new_tokens: int = 512  # longer generations
    temperature: float = 0.8
    top_k: int = 200

