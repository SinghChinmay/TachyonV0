# TachyonV0

TachyonV0 is a PyTorch language model experiment built around a custom `WaveEngine` block. The repository currently contains:

- a model definition in `tachyon_v0_model.py`
- a training script in `tachyon_v0_train.py`
- a chat/inference script in `tachyon_v0_chat.py`

## Project structure

### `tachyon_v0_model.py`

Defines the model architecture:

- `WaveEngine`: applies `w * (x1 * cos(x2))`
- `TachyonV0Block`: stacks two wave interactions
  - time-axis interaction (`wave_h`)
  - embedding-dimension interaction (`wave_v`)
- `TachyonV0`: token embeddings, positional embeddings, repeated blocks, final layer norm, and language-model head

Default model settings in code:

- vocabulary size: `50257`
- embedding size: `4096`
- layers: `64`
- block size: `1024`

### `tachyon_v0_train.py`

Implements training with:

- `GPT2Tokenizer` from Hugging Face
- an `IterableDataset` that streams text from `dataset.txt`
- `AdamW` optimization
- gradient clipping
- checkpoint resume/save support through `tachyon_v0.pt`
- graceful save on `Ctrl+C`

Important hardcoded training settings:

- batch size: `1`
- block size: `1024`
- learning rate: `5e-5`
- checkpoint path: `tachyon_v0.pt`
- dataset path: `dataset.txt`

The device is chosen automatically in this order:

1. CUDA
2. Apple Metal (`mps`)
3. CPU

### `tachyon_v0_chat.py`

Provides an interactive chat loop that:

- loads the tokenizer and model
- restores weights from `tachyon_v0.pt` when available
- generates up to 100 tokens
- uses top-k sampling with:
  - `k = 40`
  - temperature `0.7`

## Requirements

This repository does not currently include a `requirements.txt` or `pyproject.toml`, but the code requires at least:

- Python 3
- `torch`
- `transformers`

Install dependencies manually, for example:

```bash
pip install torch transformers
```

## Training

1. Create a text dataset file named `dataset.txt` in the repository root.
2. Run the training script from the repository root:

```bash
python tachyon_v0_train.py
```

During training:

- the script will resume from `tachyon_v0.pt` if it exists
- the model saves on `Ctrl+C`
- the model also autosaves every 100,000 steps

## Chat / inference

After training, or after placing a compatible checkpoint at `tachyon_v0.pt`, run:

```bash
python tachyon_v0_chat.py
```

Type a prompt and press Enter. Use `exit` or `quit` to leave the chat loop.

## Notes

- The repository currently uses root-level scripts instead of a package layout.
- The model configuration is very large (`4096` embedding size and `64` layers), so training and inference can be resource intensive.
- If `tachyon_v0.pt` is missing, chat mode still runs, but it uses randomly initialized weights.
