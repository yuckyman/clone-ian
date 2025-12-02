"""
minimal gpt model - nanochat style
decoder-only transformer for chat
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """causal self-attention with kv cache support"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention makes training faster
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("⚠️  flash attention not available, using slow attention")
    
    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        
        # calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # kv cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None
        
        # causal self-attention
        if self.flash:
            # efficient attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1), float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv_cache


class MLP(nn.Module):
    """feedforward network"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x, kv_cache=None):
        # attention
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        # feedforward
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache


class GPT(nn.Module):
    """gpt model"""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        
        # init weights
        self.apply(self._init_weights)
        
        # report params
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"sequence length {t} exceeds block size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # forward
        tok_emb = self.transformer.wte(idx)  # token embeddings
        pos_emb = self.transformer.wpe(pos)  # position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # transformer blocks
        for block in self.transformer.h:
            x, _ = block(x, kv_cache=None)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        """generate tokens (simple version without kv cache for correctness)"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # crop to block size
            idx_cond = idx[:, -self.config.block_size:]
            
            # forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, [-1], :]  # only last token
            
            # sample
            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # append
            idx = torch.cat((idx, idx_next), dim=1)
            
            # stop token
            if stop_token is not None and idx_next.item() == stop_token:
                break
        
        return idx

