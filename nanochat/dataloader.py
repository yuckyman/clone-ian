"""
data loader for chatml format
"""

import torch
import random
from pathlib import Path
from typing import List, Tuple
try:
    from nanochat.tokenizer import Tokenizer
except ImportError:
    from tokenizer import Tokenizer


def load_chatml_data(data_path: str, tokenizer: Tokenizer, max_examples: int = None) -> List[List[int]]:
    """load chatml format data and tokenize (with optional limit for memory)"""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"data file not found: {data_path}")
    
    examples = []
    current_example = ""
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            # limit examples if specified
            if max_examples and len(examples) >= max_examples:
                break
                
            # empty line separates examples
            if not line.strip():
                if current_example.strip():
                    # tokenize the full conversation
                    tokens = tokenizer.encode(current_example.strip())
                    if len(tokens) > 0:
                        examples.append(tokens)
                    current_example = ""
            else:
                current_example += line
    
        # handle last example if no trailing newline
        if current_example.strip() and (not max_examples or len(examples) < max_examples):
            tokens = tokenizer.encode(current_example.strip())
            if len(tokens) > 0:
                examples.append(tokens)
    
    print(f"loaded {len(examples)} examples")
    return examples


def get_batch(
    data: List[List[int]],
    tokenizer: Tokenizer,
    batch_size: int,
    block_size: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """get a batch of training data"""
    # sample random examples (with replacement if needed)
    if len(data) < batch_size:
        examples = random.choices(data, k=batch_size)
    else:
        examples = random.sample(data, batch_size)
    
    # create batch
    x_batch = []
    y_batch = []
    
    for example in examples:
        # crop to block_size + 1 (for targets)
        if len(example) > block_size + 1:
            start_idx = random.randint(0, len(example) - block_size - 1)
            tokens = example[start_idx:start_idx + block_size + 1]
        else:
            tokens = example[:block_size + 1]
        
        # pad if needed (use pad token or last token)
        if len(tokens) < block_size + 1:
            padding = [0] * (block_size + 1 - len(tokens))  # use 0 as padding
            tokens = tokens + padding
        
        x = tokens[:-1]
        y = tokens[1:]
        
        x_batch.append(x)
        y_batch.append(y)
    
    x = torch.tensor(x_batch, dtype=torch.long, device=device)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)
    
    # mask padding in targets
    y[y == 0] = -1  # ignore padding in loss
    
    return x, y

