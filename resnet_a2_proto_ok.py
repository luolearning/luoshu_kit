import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from luoshu_kit.luoshu_kit_a2_proto import inject

def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = pick_device()
print("Using device:", device)

ds = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
loader = DataLoader(ds, batch_size=128, shuffle=True)
x, y = next(iter(loader))
x, y = x.to(device), y.to(device)

model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

bridge = inject(
    model,
    layer_name="layer2",
    input_shape=(1, 1, 28, 28),
    grid_policy=3,
    align_mode="nearest",
    device=device,
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(20):
    optimizer.zero_grad()

    logits = model(x)
    ce_loss = F.cross_entropy(logits, y)
    reg_loss = bridge.regularize(lambda_a=5.0, lambda_p=1.0)
    loss = ce_loss + reg_loss

    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        pred = logits.argmax(1)
        acc = (pred == y).float().mean().item()
        print(
            f"step={step} | "
            f"ce_loss={float(ce_loss.detach().cpu()):.4f} | "
            f"reg_loss={float(reg_loss.detach().cpu()):.4f} | "
            f"total={float(loss.detach().cpu()):.4f} | "
            f"acc={acc:.4f}"
        )

print("final diagnostics =", bridge.diagnostics())
