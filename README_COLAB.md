# running nanochat on google colab

quick guide to train your model on colab's free gpu

## option 1: use the notebook (easiest)

1. open `colab_notebook.ipynb` in google colab
2. run cells sequentially
3. upload your `training_data/chatml_format.txt` when prompted
4. train!

## option 2: clone and run

```python
# in a colab cell
!git clone https://github.com/YOUR_USERNAME/clone-ian.git
%cd clone-ian
!pip install -q torch tiktoken numpy

# upload training_data/chatml_format.txt via file browser, then:
!python -m nanochat.train
```

## option 3: use train_colab.py

```python
# clone repo first, then:
!python train_colab.py
```

## data setup

you have a few options for getting your training data into colab:

1. **upload via file browser**: use colab's file browser (left sidebar) to upload `training_data/chatml_format.txt`
2. **mount google drive**: 
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # then copy from drive or symlink
   ```
3. **include in repo**: if your data is small enough, commit it to the repo (but be careful with privacy!)

## colab-specific tips

- **free gpu**: colab gives you a free t4 gpu - much faster than cpu!
- **session timeout**: colab sessions timeout after inactivity. save checkpoints frequently.
- **memory limits**: the default config is already memory-optimized for colab's free tier
- **download checkpoints**: use `files.download('out/ckpt.pt')` to save your model

## adjusting for colab

**free tier (t4)**: default config is optimized:
- small batch size (4)
- gradient accumulation (effective batch size 16)
- limited data loading (5000 examples)
- small model (~500k params)

**colab pro (a100/l4)**: automatically uses larger config:
- larger batch size (16)
- more data (20000 examples)
- bigger model (~15M params, 6 layers, 512 dim)
- longer context (512 tokens)

if you want to manually override, edit `nanochat/config.py` or use `config_colab_pro.py`

## troubleshooting

**out of memory**: reduce `batch_size` or `max_data_examples` in config

**slow training**: make sure you're using gpu (check device output)

**data not found**: verify the path to `training_data/chatml_format.txt`

