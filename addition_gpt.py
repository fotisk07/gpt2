"""
Proof of concept code for creating a transformer that learns how to add.
Works well for 2,3,4 digits numbers, haven't tested for more.
A lot of things can be improved
- Simpler (better ?) encoding, no + no = just give the numbers of predifined size : TODO
- Implement train/test split : TODO
- Refactor code to be more project like : DONE
- Check out karpathy config implementation
-
"""

import torch
from model import GPT2

# hyperparams
num_digits = 4
context_lengt = num_digits * 3 + 3
n_embds = 128
num_heads = 8
n_layers = 3

lr = 1e-3
batch_size = 64
max_steps = 5000
eval_interval = 500
eval_iters = 100
output_size = 5

# ---------------------

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

characters = [str(i) for i in range(0, 10)] + ["+", "="]
vocab_size = len(characters)
i2c = {i: c for i, c in enumerate(characters)}
c2i = {c: i for i, c in enumerate(characters)}
encode = lambda text: [c2i[letter] for letter in text]
decode = lambda tokens: "".join(i2c[token] for token in tokens)


def get_data():
    # example for 1+1
    # x : [1, 10, 1, 11, 1, 0,0 ,..]
    # y : [-1, -1, -1, 1, 0, 0, 0, ...]
    # context leght : 3* n_dig + 2

    source = torch.randint(0, 10**num_digits, (batch_size, 2))
    res = source.sum(dim=1)

    x = torch.zeros(batch_size, context_lengt, dtype=torch.long)
    y = torch.zeros(batch_size, context_lengt, dtype=torch.long)
    for i, (to_add, c) in enumerate(zip(source, res)):
        a, b = to_add[0], to_add[1]
        question = encode(f"{a.item()}+{b.item()}=")
        answer = encode(f"{c.item()}"[::-1])
        equation = question + answer
        padding = (context_lengt + 1 - len(equation)) * [0]
        x[i] = torch.tensor(equation + padding, dtype=torch.long)[:-1]
        y[i, : len(question) - 1] = -1
        y[i, len(question) - 1 :] = torch.tensor(answer + padding, dtype=torch.long)

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    loss_total = 0
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_data()
        _, loss = m(x, y)
        losses[i] = loss
    loss_total = loss.mean()
    return loss_total


@torch.no_grad()
def show_outputs():
    # Generate some data
    source = torch.randint(0, 10**num_digits, (output_size, 2))
    res = source.sum(dim=1)
    print("-----Examples-------")
    for i, (to_add, c) in enumerate(zip(source, res)):
        a, b = to_add[0], to_add[1]
        question = torch.tensor(encode(f"{a.item()}+{b.item()}="))
        predicted_res = decode(
            m.generate(question.unsqueeze(0), max_new_tokens=5)[0].tolist()
        )[len(question) :][::-1]
        print(f"{a.item()}+{b.item()}={predicted_res} ||| Answer :  {c.item()}")
    print("-------------------")


if __name__ == "__main__":
    m = GPT2(vocab_size, context_lengt, n_embds, n_layers, num_heads, device=device).to(
        device
    )
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    print("Using ", device)
    for step in range(max_steps):
        x, y = get_data()
        x, y = x.to(device), y.to(device)
        logits, loss = m(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            total_loss = estimate_loss()
            print(f"Step {step} | Loss : {total_loss:.4f} |")
            show_outputs()
