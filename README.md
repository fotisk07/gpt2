# GPT-2 Based Text Generation

This repository contains an implementation of a GPT-2 inspired language model in PyTorch. The model is trained on character-level sequences and can generate text based on an input prompt. It includes a simple bigram model, a full GPT-2 model, and a training script for learning from a dataset.

## Repository Structure

- `models.py`: Defines the model architecture, including:
  - `BigramModel`: A simple bigram model for language modeling.
  - `GPT2`: A GPT-2 based transformer model.
  - `CausalSelfAttention`, `FeedForward`, and `Block` for the transformer layers.
- `gpt2.py`: Training script for the GPT-2 model.
- `addition.py`: Demonstrates model performance on numerical addition tasks.
- `requirements.txt`: Lists dependencies required to run the code.
- `input.txt`: Contains the Tiny Shakespeare dataset used for training.

## Installation

Ensure you have Python 3.8+ and install dependencies using:

```sh
pip install -r requirements.txt
```

## Usage

### Training the GPT-2 Model

To train the GPT-2 model using the Tiny Shakespeare dataset:

```sh
python gpt2.py
```

The script automatically loads `input.txt`, preprocesses it, and trains the model.

### Generating Text

Once trained, the model can generate text by providing an initial context:

```python
import torch
from models import GPT2

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2(vocab_size, context_lenght, n_embds, n_layers, num_heads, device=device)
model.load_state_dict(torch.load("m.pt"))
model.to(device)

context = torch.zeros((1, 1), dtype=torch.long).to(device)
output = model.generate(context, max_new_tokens=500)
print(decode(output[0].tolist()))
```

### Training on Addition Tasks

To train the model on numerical addition:

```sh
python addition.py
```

Results will be logged in `Out.txt`, showcasing the model's predictions on random addition problems.

## Dataset: Tiny Shakespeare

The repository includes `input.txt`, a character-level dataset based on Tiny Shakespeare. If you want to download your own copy:

```sh
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Replace `input.txt` with your own dataset if needed.

## Notes

- The model supports GPU acceleration using PyTorch.
- You can modify hyperparameters such as `context_length`, `num_heads`, and `num_layers` in the scripts.
- Results are saved and periodically logged for evaluation.
- The project has mostly educational value and is not meant to be run in production.
