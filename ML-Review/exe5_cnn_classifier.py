"""
5. Small CNN on synthetic shapes, ~25 min
No dataset download needed — three procedurally generated 16x16 patterns (plus / diagonal
cross / square outline) stand in for MNIST-style images. Same architecture idea though:
Conv -> ReLU -> Pool, twice, then flatten into a linear classifier head.

Fill in the TODOs: the flatten dimension (use exe4's shape formula: 16 -> pool -> 8 -> pool -> 4),
the layers, the forward pass, and the training loop. Target: >90% val accuracy.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # this machine has a libiomp/libomp DLL conflict; without this, `import torch` crashes

import torch
import torch.nn as nn

torch.manual_seed(0)

SIZE = 16


def make_plus(size=SIZE):
    img = torch.zeros(size, size)
    c = size // 2
    img[c - 1:c + 1, :] = 1.0
    img[:, c - 1:c + 1] = 1.0
    return img


def make_cross(size=SIZE):
    img = torch.eye(size) + torch.eye(size).flip(0)
    return img.clamp(0, 1)


def make_square(size=SIZE, margin=3):
    img = torch.zeros(size, size)
    img[margin, margin:size - margin] = 1.0
    img[size - 1 - margin, margin:size - margin] = 1.0
    img[margin:size - margin, margin] = 1.0
    img[margin:size - margin, size - 1 - margin] = 1.0
    return img


def make_dataset(n_per_class: int = 150, noise: float = 0.15):
    generators = [make_plus, make_cross, make_square]
    images, labels = [], []
    for label, gen in enumerate(generators):
        base = gen()
        for _ in range(n_per_class):
            images.append(base + torch.randn(SIZE, SIZE) * noise)
            labels.append(label)
    x = torch.stack(images).unsqueeze(1)  # shape: (N, 1, 16, 16) — channel dim for Conv2d
    y = torch.tensor(labels)
    perm = torch.randperm(len(x))
    return x[perm], y[perm]


x, y = make_dataset()
n_train = int(0.8 * len(x))
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:], y[n_train:]

"""

Why it's needed:

images is a list of N tensors, each shaped (16, 16) — just height × width, no channel dimension, because make_plus/make_cross/make_square return plain 2D grids.
torch.stack(images) stacks them along a new dim 0 → shape (N, 16, 16).
But nn.Conv2d always expects 4D input: (batch, channels, height, width). A 3D tensor (N, 16, 16) doesn't fit that — channels is missing entirely.
.unsqueeze(1) inserts a new dimension of size 1 at index 1 → (N, 16, 16) becomes (N, 1, 16, 16). That 1 is the channel count (grayscale = 1 channel, as opposed to 3 for RGB).
"""

"""
Step 1 — in_channels is fixed by what feeds into the layer, not chosen.

The dataset tensor x has shape (N, 1, 16, 16) — see line 52: x = torch.stack(images).unsqueeze(1).
unsqueeze(1) inserts a channel dim of size 1 because these are grayscale images (one intensity value per pixel, no RGB).
So conv1's in_channels must be 1 — it's just matching the data, not a guess.

Step 2 — out_channels is a free hyperparameter you pick, not derived.

out_channels=8 means "learn 8 independent 3×3 filters." Each filter slides over the input and produces one output feature map (one channel).
Each filter has shape (in_channels, kernel, kernel) = (1, 3, 3) here, so conv1 has 8 filters × 1×3×3 = 72 learnable weights (+8 biases).
Result: conv1 maps (N, 1, 16, 16) → (N, 8, 16, 16) — spatial size unchanged (per the formula from before), channel count jumps to whatever you chose (8).
There's no "correct" number — it's a capacity/design choice. Convention: start small, double channels as spatial size shrinks (8 → 16 → 32...), which is exactly the pattern here (8 then 16).

Step 3 — conv2's in_channels is then fixed by conv1's output.

conv1 output has 8 channels → conv2 must take in_channels=8 to match. This isn't a choice either; it's a shape-compatibility constraint (mismatched channels → runtime error).
conv2's out_channels=16 is again a free choice — following the "double channels" convention as spatial size halves (16×16 → 8×8 via pooling).

So the rule of thumb: in_channels = whatever the previous tensor already has; out_channels = however many filters you decide to learn. The only hard constraint is in_channels(layer N) == out_channels(layer N-1) (or the raw data's channel count for the first layer).
"""
"""
The formula only applies to spatial dimensions (H, W) at each layer that has a kernel/stride — walking through forward() (lines 80-88):

Step	Layer	Formula applies?	Calculation	Result
input	—	—	—	(N, 1, 16, 16)
conv1(x)	Conv2d(1→8, k=3, p=1, s=1)	✅ yes	(16-3+2·1)/1+1	(N, 8, 16, 16)
relu(...)	ReLU	❌ no (elementwise, no shape change)	—	(N, 8, 16, 16)
pool(...)	MaxPool2d(2) (k=2, p=0, s=2)	✅ yes	(16-2)/2+1	(N, 8, 8, 8)
conv2(x)	Conv2d(8→16, k=3, p=1, s=1)	✅ yes	(8-3+2·1)/1+1	(N, 16, 8, 8)
relu(...)	ReLU	❌ no	—	(N, 16, 8, 8)
pool(...)	MaxPool2d(2)	✅ yes	(8-2)/2+1	(N, 16, 4, 4)
flatten(x, start_dim=1)	—	❌ no (just reshapes, 16·4·4=256)	—	(N, 256)
fc(x)	Linear(256, 3)	❌ no (fixed by flatten_dim, not the conv formula)	—	(N, 3)
So concretely: every Conv2d and every MaxPool2d/AvgPool2d call — anything with a kernel sliding over the spatial dims. ReLU, flatten, and Linear never touch H/W via that formula — ReLU preserves shape entirely, flatten just collapses existing dims, and Linear's output size is whatever you define (n_classes=3 here), unrelated to the conv/pool math.

The number you ultimately need it for is flatten_dim on line 76 — you apply the formula at each conv/pool step in sequence (16→16→8→8→4) to know the final 4×4 spatial size before flattening.

"""

class CNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        # TODO 1: conv1 = Conv2d(1 -> 8 channels, kernel_size=3, padding=1)   # keeps size 16x16
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # TODO 2: conv2 = Conv2d(8 -> 16 channels, kernel_size=3, padding=1)  # keeps size 8x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # halves spatial size each call
        self.relu = nn.ReLU()

        # TODO 3: work out flatten_dim after conv1 -> relu -> pool -> conv2 -> relu -> pool
        # 16x16 --pool--> 8x8 --pool--> 4x4, with 16 channels from conv2
        flatten_dim = 16 * 4 * 4
        # TODO 4: fc = Linear(flatten_dim, n_classes)
        self.fc = nn.Linear(flatten_dim, n_classes)

    def forward(self, x):
        # TODO 5: x = pool(relu(conv1(x)))
        x = self.pool(self.relu(self.conv1(x)))
        # TODO 6: x = pool(relu(conv2(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # TODO 7: flatten x to (batch, flatten_dim) — see torch.flatten(x, start_dim=1)
        x = torch.flatten(x, start_dim=1)
        # TODO 8: return fc(x)  (raw logits, no softmax — CrossEntropyLoss applies it internally)
        return self.fc(x)


model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(30):
    # TODO 9: standard training step — zero_grad, forward, loss, backward, optimizer.step()
    logits = model(x_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  loss {loss.item():.4f}")

with torch.no_grad():
    val_logits = model(x_val)
    val_preds = val_logits.argmax(dim=1) # argmax(dim=1) -> [0, 1, 2, 1]

    """
    argmax(dim=1) — collapses dim 1 (classes), keeps dim 0 (samples) → output shape (N,). For each row (sample), it finds which of the 3 class-scores is highest. This is what you want: one predicted class per sample.

    argmax(dim=0) — collapses dim 0 (samples), keeps dim 1 (classes) → output shape (3,). For each column (class), it finds which sample scored highest on that class. Wrong axis entirely — you'd get 3 numbers (one per class), not N predictions.
    """

    accuracy = (val_preds == y_val).float().mean().item()

print(f"\nvalidation accuracy: {accuracy:.3f}  (target: >0.90)")
