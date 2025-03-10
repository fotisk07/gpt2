import torch
import torch.nn as nn

# hyperparams
split = 0.9
batch_size = 32
context_lenght = 8
n_embds = 8
lr = 1e-2
max_steps = 3000
eval_iters = 200
# ---------------------

torch.manual_seed(1337)

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


class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embds = nn.Embedding(vocab_size, n_embds)
        self.pos_embds = nn.Embedding(context_lenght, n_embds)
        self.lm_head = nn.Linear(n_embds, vocab_size)

    def forward(self, inputs, targets=None):
        # inputs [B, T], targets [B, T]
        B, T = inputs.shape
        token_embdedings = self.tok_embds(inputs)  # [B, T, Ne]
        position_embdedings = self.pos_embds(torch.arange(T))  # [B, T, Ne]

        x = token_embdedings + position_embdedings
        logits = self.lm_head(x)

        B, T, C = logits.shape

        if targets is None:
            loss = None
        else:
            loss = nn.functional.cross_entropy(logits.view(-1, C), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is [B, T]
        B, T = idx.shape
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -context_lenght:])  # [B, T, C]
            probabs = nn.functional.softmax(logits, dim=-1)
            prediction = probabs[:, -1]  # only last T
            next_token = torch.multinomial(prediction, B)
            idx = torch.concat((idx, next_token), dim=-1)
        return idx


@torch.no_grad()
def estimate_loss():
    loss_dict = dict()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_data(split)
            _, loss = model(x, y)
            losses[i] = loss
        loss_dict[split] = loss.mean()
    return loss_dict


######### Train Loop ############
model = GPT2()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for step in range(max_steps):
    x, y = get_data()
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        loss_dict = estimate_loss()
        print(
            f"Step {step} | Train Loss : {loss_dict['train']:.4f} |"
            f"Val Loss : {loss_dict['val']:.4f}"
        )

context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
