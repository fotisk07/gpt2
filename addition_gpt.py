import torch
from models import GPT2
import random

# hyperparams
num_digits = 3
context_lenght = 2 * num_digits + num_digits
n_embds = 128
num_heads = 8
n_layers = 3

lr = 1e-3
batch_size = 64
max_steps = 5000
eval_interval = 1000
eval_iters = 20
output_size = 5
train_val_split = 0.8
data_size = 100 * int(10**num_digits)
file_path = "Out.txt"
# ---------------------

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

characters = [str(i) for i in range(0, 10)]
vocab_size = len(characters)
i2c = {i: c for i, c in enumerate(characters)}
c2i = {c: i for i, c in enumerate(characters)}
encode = lambda text: [c2i[letter] for letter in text]
decode = lambda tokens: "".join(i2c[token] for token in tokens)

source = torch.randint(0, 10**num_digits, (data_size, 2))
res = source.sum(dim=-1)
data = torch.cat((source, res.unsqueeze(1)), dim=1)

train_data = data[: int(train_val_split * data_size)]
val_data = data[int(train_val_split * data_size) :]

with open(file_path, "w") as f:  # Empty file at the beginning
    f.write("Performance showcase:\n")


def get_data(split="train"):
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data), (batch_size, 1))

    x = torch.zeros((batch_size, context_lenght), dtype=torch.long)
    y = torch.zeros((batch_size, context_lenght), dtype=torch.long)
    y[:, : 2 * num_digits - 1] = -1

    for i, id in enumerate(idx):
        x_string = (
            f"{str(data[id, 0].item()):0{num_digits}}"
            f"{str(data[id, 1].item()):0{num_digits}}"
            f"{str(data[id, -1].item())[::-1]:0{num_digits+1}}"[:-1]
        )
        y_string = f"{str(data[id, -1].item())[::-1]:0{num_digits+1}}"
        x[i, :] = torch.tensor(encode(x_string))
        y[i, 2 * num_digits - 1 :] = torch.tensor(encode(y_string))

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    loss_dict = dict()
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_data(split)
            _, loss = model(x, y)
            losses[i] = loss
        loss_dict[split] = loss.mean()
    model.train()
    return loss_dict


def showcase_performance(model, file_path="performance_log.txt", num_samples=5):

    with open(file_path, "a") as f:
        f.write("\nPerformance showcase:\n")
        for _ in range(num_samples):
            a, b = random.randint(0, 10**num_digits - 1), random.randint(
                0, 10**num_digits - 1
            )
            result = a + b
            input_str = f"{a:0{num_digits}}{b:0{num_digits}}"
            x = torch.tensor(encode(input_str)).unsqueeze(0).to(device)
            prediction = decode(
                model.generate(x, max_new_tokens=num_digits + 1)[
                    0, x.shape[1] :
                ].tolist()
            )[::-1]
            f.write(
                f"{a} + {b} = {result}, Model predicted: {a} + {b} = {prediction}\n"
            )


if __name__ == "__main__":
    m = GPT2(
        vocab_size, context_lenght, n_embds, n_layers, num_heads, device=device
    ).to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    print("Using ", device)

    for step in range(max_steps):
        x, y = get_data()
        logits, loss = m(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            loss_dict = estimate_loss(m)
            print(
                f"Step {step} | Train Loss : {loss_dict['train']:.4f} | Val Loss : {loss_dict['val']:.4f}"
            )
            showcase_performance(m, file_path, num_samples=output_size)

    torch.save(m.state_dict(), "m.pt")
