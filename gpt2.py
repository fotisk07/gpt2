import torch
from models import GPT2, train

# hyperparams
split = 0.9
batch_size = 64
context_lenght = 256
n_embds = 384
n_heads = 6
n_layers = 6
lr = 1e-3
max_steps = 5000
eval_iters = 100
eval_interval = 500
# ---------------------

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("input.txt") as file:
    data = file.read()

characters = sorted(list(set(data)))
vocab_size = len(characters)
i2c = {i: c for i, c in enumerate(characters)}
c2i = {c: i for i, c in enumerate(characters)}

encode = lambda text: [c2i[letter] for letter in text]
decode = lambda tokens: "".join(i2c[token] for token in tokens)

data = encode(data)
n = int(len(data) * split)
train_data, val_data = torch.tensor(data[:n]), torch.tensor(data[n:])


def get_data(split="train"):
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data) - context_lenght - 1, (batch_size, 1))  # because y
    x = torch.stack([data[id : id + context_lenght] for id in idx])
    y = torch.stack([data[id + 1 : id + context_lenght + 1] for id in idx])
    return x, y


@torch.no_grad()
def estimate_loss():
    loss_dict = dict()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_data(split)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[i] = loss
        loss_dict[split] = loss.mean()
    return loss_dict


######### Train Loop ############
model = GPT2(
    vocab_size=vocab_size,
    context_lenght=context_lenght,
    n_embds=n_embds,
    n_layers=n_layers,
    num_heads=n_heads,
    device=device,
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print("Using ", device)
train(model, optimizer, get_data, max_steps, eval_interval, eval_iters)

context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
