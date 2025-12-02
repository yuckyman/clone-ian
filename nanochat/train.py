"""
training script for tiny nanochat
"""

import os
import time
import math
import torch
import torch.nn as nn
from pathlib import Path
from contextlib import nullcontext

import sys
from pathlib import Path

# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from nanochat.gpt import GPT
    from nanochat.config import GPTConfig
    from nanochat.tokenizer import Tokenizer
    from nanochat.dataloader import load_chatml_data, get_batch
except ImportError:
    # fallback for direct execution
    from gpt import GPT
    from config import GPTConfig
    from tokenizer import Tokenizer
    from dataloader import load_chatml_data, get_batch


def get_lr(it, config):
    """learning rate schedule with warmup"""
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.max_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, data, tokenizer, config, device):
    """estimate loss on validation set"""
    out = {}
    model.eval()
    
    # split data
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split_data, tokenizer, config.batch_size, config.block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean()
        out[f'{split_name}_ppl'] = math.exp(losses.mean().item())  # perplexity
    model.train()
    return out


@torch.no_grad()
def generate_sample(model, tokenizer, config, device, prompt="<|im_start|>user\nhey<|im_end|>\n<|im_start|>assistant\n"):
    """generate a sample from the model"""
    model.eval()
    
    # encode prompt
    tokens = tokenizer.encode(prompt)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # generate
    generated = model.generate(
        tokens_tensor,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        stop_token=tokenizer.im_end_id
    )
    
    # decode
    response_tokens = generated[0][len(tokens):].tolist()
    if tokenizer.im_end_id and tokenizer.im_end_id in response_tokens:
        response_tokens = response_tokens[:response_tokens.index(tokenizer.im_end_id)]
    
    response = tokenizer.decode(response_tokens).strip()
    model.train()
    return response


def main():
    # config - auto-detect colab pro or use default
    import os
    in_colab = 'google.colab' in sys.modules or os.path.exists('/content')
    
    # check for colab pro (a100/l4) - use larger config
    if in_colab:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                if 'a100' in gpu_name or 'l4' in gpu_name or 'v100' in gpu_name:
                    print("🚀 detected high-end gpu, using colab pro config")
                    try:
                        from nanochat.config_colab_pro import GPTConfigColabPro
                        config = GPTConfigColabPro()
                    except ImportError:
                        from config_colab_pro import GPTConfigColabPro
                        config = GPTConfigColabPro()
                else:
                    config = GPTConfig()
            else:
                config = GPTConfig()
        except:
            config = GPTConfig()
    else:
        config = GPTConfig()
    
    # device (colab-friendly)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"using device: {device} ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"using device: {device} (apple silicon)")
    else:
        device = 'cpu'
        print(f"using device: {device}")
    
    # tokenizer
    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size
    print(f"vocab size: {config.vocab_size}")
    
    # data
    print(f"loading data from {config.data_path}...")
    # limit examples for memory efficiency
    train_data = load_chatml_data(config.data_path, tokenizer, max_examples=config.max_data_examples)
    
    # split train/val
    split_idx = int(0.9 * len(train_data))
    train_data_split = train_data[:split_idx]
    val_data_split = train_data[split_idx:]
    print(f"train: {len(train_data_split)}, val: {len(val_data_split)}")
    
    # model
    model = GPT(config)
    model = model.to(device)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    
    # create output directory
    out_dir = Path(config.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    # training loop
    iter_num = 0
    best_val_loss = 1e9
    running_loss = 0.0
    running_count = 0
    start_time = time.time()
    
    print("\n" + "="*60)
    print("starting training...")
    print("="*60 + "\n")
    
    while True:
        # lr schedule
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # evaluate
        if iter_num % config.eval_interval == 0:
            eval_start = time.time()
            # combine train and val for eval (we'll split in estimate_loss)
            all_data = train_data_split + val_data_split
            losses = estimate_loss(model, all_data, tokenizer, config, device)
            eval_time = time.time() - eval_start
            
            # calculate average running loss
            avg_loss = running_loss / max(running_count, 1)
            
            # time stats
            elapsed = time.time() - start_time
            iters_per_sec = iter_num / elapsed if elapsed > 0 else 0
            
            print("\n" + "-"*60)
            print(f"iter {iter_num} | {elapsed:.1f}s elapsed | {iters_per_sec:.2f} iters/s")
            print(f"train loss: {losses['train']:.4f} (ppl: {losses['train_ppl']:.2f}) | "
                  f"val loss: {losses['val']:.4f} (ppl: {losses['val_ppl']:.2f})")
            print(f"avg running loss: {avg_loss:.4f} | lr: {lr:.2e}")
            print(f"eval time: {eval_time:.2f}s")
            
            # save checkpoint
            if losses['val'] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'val_loss': losses['val'],
                }
                torch.save(checkpoint, out_dir / 'ckpt.pt')
                print(f"💾 saved checkpoint (best val loss: {best_val_loss:.4f})")
            print("-"*60 + "\n")
            
            # reset running stats
            running_loss = 0.0
            running_count = 0
        
        # forward backward update with gradient accumulation
        loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = get_batch(train_data_split, tokenizer, config.batch_size, config.block_size, device)
            logits, loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            loss_accum += loss.item()
        
        optimizer.step()
        
        # track running loss
        running_loss += loss_accum
        running_count += 1
        
        # print loss occasionally
        if iter_num % 10 == 0 and iter_num > 0:
            avg_loss = running_loss / running_count
            print(f"iter {iter_num} | loss: {loss_accum:.4f} (avg: {avg_loss:.4f}) | lr: {lr:.2e}")
        
        # generate samples periodically
        if iter_num > 0 and iter_num % config.sample_interval == 0:
            print("\n📝 generating samples...")
            sample_prompts = [
                "<|im_start|>user\nhey<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nwhat's up?<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nhow are you?<|im_end|>\n<|im_start|>assistant\n",
            ]
            try:
                for i, prompt in enumerate(sample_prompts[:2]):  # show 2 samples
                    sample = generate_sample(model, tokenizer, config, device, prompt)
                    user_msg = prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
                    print(f"  [{i+1}] user: {user_msg}")
                    print(f"      assistant: {sample[:150]}{'...' if len(sample) > 150 else ''}")
                print()
            except Exception as e:
                print(f"⚠️  sample generation failed: {e}\n")
        
        iter_num += 1
        
        # termination
        if iter_num > config.max_iters:
            break
    
    print("training complete!")


if __name__ == "__main__":
    main()

