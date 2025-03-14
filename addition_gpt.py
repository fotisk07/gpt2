"""
Proof of concept code for creating a transformer that learns how to add.
Works well for 2,3,4 digits numbers, haven't tested for more.
A lot of things can be improved
- Simpler (better ?) encoding, no + no = just give the numbers of predifined size : TODO
- Implement train/test split : DONE
- Refactor code to be more project like : DONE
- Check out karpathy config implementation
-
"""

import torch
import numpy
from models import GPT2, train


# hyperparams
num_digits = 2
context_lenght = 2 * num_digits + num_digits
n_embds = 128
num_heads = 8
n_layers = 3

lr = 1e-3
batch_size = 2
max_steps = 1000
eval_interval = 500
eval_iters = 100
output_size = 5
train_val_split = 0.8
data_size = int(numpy.sqrt(10**num_digits))
# ---------------------

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

characters = [str(i) for i in range(0, 10)]
vocab_size = len(characters)
i2c = {i: c for i, c in enumerate(characters)}
c2i = {c: i for i, c in enumerate(characters)}
encode = lambda text: [c2i[letter] for letter in text]
decode = lambda tokens: "".join(i2c[token] for token in tokens)

source = source = torch.randint(0, 10**num_digits, (data_size, 2))
res = source.sum(dim=-1)
data = torch.cat((source, res.unsqueeze(1)), dim=1)

train_data = data[: int(train_val_split * data_size)]
val_data = data[int(train_val_split * data_size) :]


def get_data(split="train"):
    # example for 2 digit numbers 01+01=0002
    # We save in the format 0101002
    # x : [0, 1, 0, 1, 0, 0]
    # y : [-1, -1, -1, -1, 0, 2]
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data), (batch_size, 1))

    x = torch.zeros((batch_size, context_lenght), dtype=torch.long)
    y = torch.zeros((batch_size, context_lenght), dtype=torch.long)
    y[:, : 2 * num_digits] = -1

    for i, id in enumerate(idx):
        x_string = (
            f"{str(data[id, 0].item()):0{num_digits}}"
            f"{str(data[id, 1].item()):0{num_digits}}"
            f"{str(data[id, -1].item())[::-1]:0{num_digits+1}}"[:-1]
        )
        y_string = f"{str(data[id, -1].item())[::-1]:0{num_digits+1}}"[1:]
        x[i, :] = torch.tensor(encode(x_string))
        y[i, 2 * num_digits :] = torch.tensor(encode(y_string))

    return x.to(device), y.to(device)


if __name__ == "__main__":
    m = GPT2(
        vocab_size, context_lenght, n_embds, n_layers, num_heads, device=device
    ).to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    print("Using ", device)
    train(m, optimizer, get_data, max_steps, eval_interval, eval_iters)


context = torch.tensor([0, 1, 0, 1]).unsqueeze(0)
print(decode(m.generate(context, max_new_tokens=4)[0].tolist()))
