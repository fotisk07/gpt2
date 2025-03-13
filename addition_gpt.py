import torch
import torch.nn as nn

# hyperparams
num_digits = 10
context_lengt = num_digits * 3 + 2
n_embds = 64
num_heads = 8
n_layers = 2

lr = 1e-3
batch_size = 64
max_steps = 5000
eval_interval = 1000
eval_iters = 100
output_size = 4

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
    # x : [1, 10, 1, 11, 0]
    # y : [-1, -1, -1, 0, 1]
    # context leght : 3* n_dig + 2

    source = torch.randint(0, 10**num_digits, (batch_size, 2))
    res = source.sum(dim=1)

    x = torch.zeros(batch_size, context_lengt, dtype=torch.long)
    y = torch.zeros(batch_size, context_lengt, dtype=torch.long)
    y[:, : 2 * num_digits + 1] = -1  # mask the equation part
    for i, (to_add, c) in enumerate(zip(source, res)):
        a, b = to_add[0], to_add[1]
        equation = encode(f"{a.item():0{num_digits}}+{b.item():0{num_digits}}=")
        result = encode(f"{c.item():0{num_digits+1}}"[::-1])
        x[i] = torch.tensor(equation + result, dtype=torch.long)[:-1]
        y[i, 2 * num_digits + 1 :] = torch.tensor(result, dtype=torch.long)

    return x.to(device), y.to(device)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embds, num_heads, context_lenght):
        super().__init__()

        head_size = n_embds // num_heads
        self.c_attn = nn.Linear(n_embds, head_size * num_heads * 3)
        self.proj = nn.Linear(n_embds, n_embds)
        self.register_buffer(
            "tril", torch.tril(torch.ones((context_lenght, context_lenght)))
        )
        self.head_size = head_size
        self.num_heads = num_heads

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.head_size * self.num_heads, dim=2)

        k, v, q = (
            k.view(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3),
            v.view(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3),
            q.view(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3),
        )
        # (B, NH, T, HS) @ (B,NH, HS, T) -> (B,NH,T,T)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = nn.functional.softmax(wei, dim=-1)

        # (B,NH, T, T) @ (B, NH, T, HS) -> (B, NH, T, HS)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embds):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embds, 4 * n_embds),
            nn.ReLU(),
            nn.Linear(4 * n_embds, n_embds),
        )

    def forward(self, input):
        return self.layers(input)


class Block(nn.Module):
    def __init__(self, num_embds, num_heads):
        super().__init__()
        self.mha = CausalSelfAttention(
            n_embds=n_embds, num_heads=num_heads, context_lenght=context_lengt
        )
        self.ffw = FeedForward(n_embds)
        self.ln1 = nn.LayerNorm(n_embds)
        self.ln2 = nn.LayerNorm(n_embds)

    def forward(self, x):
        # x [B, T, Ne]
        x = x + self.mha(self.ln1(x))  # x [B, T, Ne]
        x = x + self.ffw(self.ln2(x))  # x [B, T, Ne]
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, context_lenght, n_embds, n_layers, num_heads):
        super().__init__()
        self.tok_embds = nn.Embedding(vocab_size, n_embds)
        self.pos_embds = nn.Embedding(context_lenght, n_embds)
        self.blocks = nn.Sequential(
            *[Block(num_embds=n_embds, num_heads=num_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embds)
        self.lm_head = nn.Linear(n_embds, vocab_size)

        self.context_lenght = context_lenght

    def forward(self, inputs, targets=None):
        # inputs [B, T], targets [B, T]
        B, T = inputs.shape
        token_embdedings = self.tok_embds(inputs)  # [B, T, Ne]
        position_embdedings = self.pos_embds(
            torch.arange(T, device=device)
        )  # [B, T, Ne]

        x = token_embdedings + position_embdedings  # [B, T, Ne]
        x = self.blocks(x)  # [B, T, Nembs]
        x = self.ln_f(x)  # [B, T, Nembs]
        logits = self.lm_head(x)  # [B, T, VS]

        B, T, C = logits.shape

        if targets is None:
            loss = None
        else:
            loss = nn.functional.cross_entropy(
                logits.view(-1, C), targets.reshape(-1), ignore_index=-1
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is [B, T]
        B, T = idx.shape
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -context_lengt:])  # [B, T, C]
            probabs = nn.functional.softmax(logits, dim=-1)  # [B, T, C]
            prediction = probabs[
                :, -1, :
            ]  # Get the last token prediction for each batch
            next_token = torch.multinomial(prediction, 1)  # Sample 1 token per batch
            next_token = next_token.squeeze(1)  # Remove the extra dimension, shape [B]
            idx = torch.cat(
                (idx, next_token.unsqueeze(1)), dim=-1
            )  # Add the new token to idx
        return idx


@torch.no_grad()
def estimate_loss():
    loss_total = 0
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_data()
        _, loss = model(x, y)
        losses[i] = loss
    loss_total = loss.mean()
    return loss_total


@torch.no_grad()
def show_outputs(output_size):
    x, _ = get_data()
    results = model.generate(x[:output_size, :-2], max_new_tokens=3)

    print("---- Examples --------")
    for res in results:
        output = (
            decode(res[: -num_digits - 1].tolist())
            + decode(res[-num_digits - 1 :].tolist())[::-1]
        )
        print(output)
    print("----------------------")


if __name__ == "__main__":
    model = GPT2(vocab_size, context_lengt, n_embds, n_layers, num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Using ", device)
    for step in range(max_steps):
        x, y = get_data()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            total_loss = estimate_loss()
            print(f"Step {step} | Loss : {total_loss:.4f} |")
            show_outputs(output_size)

    torch.save(model.state_dict(), "model.pt")
# print(
#     decode(
#         model.generate(torch.tensor(encode("10+10=")).unsqueeze(0), max_new_tokens=3)[
#             0
#         ].tolist()
#     )
# )
