# Tiny Model + LoRA Training Setup

## Overview

Create two training pipelines:

1. **Tiny GPT from scratch** - Train a small transformer (1-10M params) on your message data to test the pipeline
2. **LoRA fine-tuning** - Set up LoRA training on a larger base model (e.g., Llama 3.2 1B, Mistral 7B) for better personality grafting

## Implementation Plan

### 1. Tiny GPT Training (From Scratch)

- **File**: `train_tiny_gpt.py`
- Implement nanoGPT-style architecture (GPT-2 decoder blocks)
- Configurable model size (1M-10M params)
- Character-level or BPE tokenization
- Training loop with checkpointing
- Evaluation metrics (loss, perplexity)
- Generate samples during training

- **File**: `tiny_gpt_config.yaml`
- Model hyperparameters (layers, heads, dims)
- Training config (batch size, learning rate, epochs)
- Data paths
- Output directories

### 2. LoRA Training Setup

- **File**: `train_lora.py`
- Use Unsloth framework (fast, memory efficient, cloud-friendly)
- Support multiple base models (Llama 3.2 1B/3B, Mistral 7B, Qwen)
- Load ShareGPT format data from `training_data/sharegpt_format.jsonl`
- LoRA configuration (rank, alpha, target modules)
- Training with gradient checkpointing
- Model saving/export

- **File**: `lora_config.yaml`
- Base model selection
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters
- Data format settings

### 3. Supporting Files

- **File**: `requirements.txt`
- PyTorch, transformers, datasets
- Unsloth for LoRA
- Tokenizers, tiktoken
- Training utilities (wandb optional)

- **File**: `tokenizer_setup.py`
- Build tokenizer for tiny GPT (character or BPE)
- Handle your message corpus vocabulary

- **File**: `README_TRAINING.md`
- Setup instructions
- Cloud deployment guide (Colab, RunPod)
- Usage examples
- Troubleshooting

### 4. Data Preparation Updates

- Update `prepare_training_data.py` if needed for tokenizer compatibility
- Ensure proper train/val splits

## Technical Details

**Tiny GPT Architecture:**

- GPT-2 style decoder-only transformer
- Configurable: 2-4 layers, 128-256 hidden dim, 2-4 attention heads
- Character-level or small BPE vocab (256-512 tokens)
- Context length: 256-512 tokens

**LoRA Configuration:**

- Rank: 8-16 (start small)
- Alpha: 16-32
- Target: attention layers (q_proj, v_proj, k_proj, o_proj)
- Use 4-bit quantization for memory efficiency

**Training Strategy:**

- Tiny GPT: Train on full corpus, ~10-50 epochs
- LoRA: Fine-tune on ShareGPT format, 1-3 epochs typically sufficient

## Files to Create/Modify

1. `train_tiny_gpt.py` - Tiny GPT training script
2. `tiny_gpt_config.yaml` - Tiny model configuration
3. `train_lora.py` - LoRA training script  
4. `lora_config.yaml` - LoRA configuration
5. `tokenizer_setup.py` - Tokenizer creation
6. `requirements.txt` - Dependencies
7. `README_TRAINING.md` - Documentation

## Cloud Deployment

- Both scripts will be cloud-ready (Colab, RunPod, etc.)
- Include instructions for GPU setup
- Support for wandb logging (optional)
- Checkpoint saving for resuming training