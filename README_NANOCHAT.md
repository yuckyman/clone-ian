# tiny nanochat

minimal implementation of karpathy's nanochat, trained on your text messages to sound like you.

## setup

```bash
pip install -r requirements.txt
```

## training

train the model on your chatml formatted data:

```bash
# from project root
python -m nanochat.train

# or use the script
./run_train.sh
```

this will:
- load data from `training_data/chatml_format.txt`
- train a tiny gpt model (default: 4 layers, 256 dim, ~1M params)
- save checkpoints to `out/ckpt.pt`

### config

edit `nanochat/config.py` to adjust:
- model size (n_layer, n_head, n_embd)
- training params (batch_size, learning_rate, max_iters)
- data path

## chat

once trained, chat with your model:

```bash
# from project root
python -m nanochat.chat --checkpoint out/ckpt.pt

# or use the script
./run_chat.sh --checkpoint out/ckpt.pt
```

options:
- `--temperature`: sampling temperature (default: 0.8)
- `--top_k`: top-k sampling (default: 200)
- `--max_tokens`: max tokens to generate (default: 256)

## architecture

- **gpt.py**: minimal decoder-only transformer (nanoGPT style)
- **tokenizer.py**: tiktoken wrapper (gpt-4 style tokenization)
- **dataloader.py**: loads chatml format data
- **train.py**: training loop with validation
- **chat.py**: simple cli chat interface

## model size

default config creates a ~1M parameter model:
- 4 layers
- 4 attention heads
- 256 embedding dimension
- 512 context length

you can make it even smaller by reducing these in `config.py`, or larger if you have more compute.

## notes

- uses chatml format (`<|im_start|>role\ncontent<|im_end|>\n`)
- supports kv cache for faster generation
- flash attention if available (pytorch 2.0+)
- simple cosine lr schedule with warmup

## colab setup

see `README_COLAB.md` for instructions on running this on google colab with free gpu!

quick start:
1. open `colab_notebook.ipynb` in google colab
2. clone your repo
3. upload training data
4. train!

## next steps

- add web ui (like nanochat's ui.html)
- experiment with different model sizes
- try different tokenizers
- add more training data formats

## attribution

the `nanochat/` module is based on [karpathy/nanochat](https://github.com/karpathy/nanochat), but has been modified and adapted for this project. original implementation by andrej karpathy.

