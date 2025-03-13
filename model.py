import torch
import torch.nn as nn


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embds = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        # inputs [B, T], targets [B, T]
        logits = self.embds(inputs)  # [B, T, VS]

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
            logits, _ = self(idx)  # [B, T, C]
            probabs = nn.functional.softmax(logits, dim=-1)
            prediction = probabs[:, -1]  # only last T
            next_token = torch.multinomial(prediction, B)
            idx = torch.concat((idx, next_token), dim=-1)
        return idx


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
    def __init__(self, n_embds, num_heads, context_lenght):
        super().__init__()
        self.mha = CausalSelfAttention(
            n_embds=n_embds, num_heads=num_heads, context_lenght=context_lenght
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
    def __init__(
        self, vocab_size, context_lenght, n_embds, n_layers, num_heads, device
    ):
        super().__init__()
        self.tok_embds = nn.Embedding(vocab_size, n_embds)
        self.pos_embds = nn.Embedding(context_lenght, n_embds)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embds=n_embds, num_heads=num_heads, context_lenght=context_lenght
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embds)
        self.lm_head = nn.Linear(n_embds, vocab_size)

        self.context_lenght = context_lenght
        self.device = device

    def forward(self, inputs, targets=None):
        # inputs [B, T], targets [B, T]
        B, T = inputs.shape
        token_embdedings = self.tok_embds(inputs)  # [B, T, Ne]
        position_embdedings = self.pos_embds(
            torch.arange(T, device=self.device)
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
            logits, _ = self(idx[:, -self.context_lenght :])  # [B, T, C]
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
