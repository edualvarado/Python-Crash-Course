"""
3. MLP on XOR — a linear model can't do this, ~25 min
XOR-style data (4 clusters, diagonal ones share a label) is the classic example of a
problem that's not linearly separable — a single linear layer cannot solve it, but a
2-layer MLP with a nonlinearity between the layers can. That nonlinearity is the whole
point of "multi-layer".

Fill in the TODOs: the model class and the training loop. Data generation is done for you.
Target: >95% accuracy on the held-out set.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # this machine has a libiomp/libomp DLL conflict; without this, `import torch` crashes

import torch
import torch.nn as nn

torch.manual_seed(0)

def make_xor_data(n_per_cluster: int = 100, noise: float = 0.15):
    centers = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    labels_per_center = torch.tensor([0.0, 0.0, 1.0, 1.0])  # diagonal pairs share a label

    x_parts, y_parts = [], []
    for center, label in zip(centers, labels_per_center):
        x_parts.append(center + torch.randn(n_per_cluster, 2) * noise)
        y_parts.append(torch.full((n_per_cluster,), label))

    x = torch.cat(x_parts)
    y = torch.cat(y_parts)
    perm = torch.randperm(len(x))
    return x[perm], y[perm]


x, y = make_xor_data()
n_train = int(0.8 * len(x))
x_train, y_train = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:], y[n_train:]


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 16):
        super().__init__()
        # TODO 1: define the layers — Linear(in_dim, hidden) -> ReLU -> Linear(hidden, hidden)
        # -> ReLU -> Linear(hidden, 1). Output is a single logit (no sigmoid — BCEWithLogitsLoss
        # applies it internally).
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )


    def forward(self, x):
        # TODO 2: run x through the layers you defined and return the logits (shape [batch])
        return self.net(x).squeeze(-1)


"""
This doesn't crash — it silently broadcasts, which is worse. Toy example, batch of 3:
logits (unsqueezed) = [[2.0], [-1.0], [0.5]]   shape (3, 1)
y_train              = [1.0, 0.0, 1.0]          shape (3,)
PyTorch broadcasting aligns from the right: (3,1) vs (3,) → treats the second as (1,3) → broadcasts both to (3,3). The loss function ends up pairing every prediction against every label — entry [0,1] computes loss between logits[0]=2.0 and y_train[1]=0.0, a completely mismatched pair that should never be compared. You get a 9-term mean instead of the correct 3-term mean, and gradients backprop from the wrong labels into the wrong predictions. .squeeze(-1) collapses (batch,1) → (batch,) so shapes match exactly and each prediction only ever sees its own label.
"""

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCEWithLogitsLoss()

"""
2. SGD vs. Adam:

┌──────────────────────────┬───────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
│                          │                    SGD                    │                                       Adam                                        │
├──────────────────────────┼───────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Update                   │ param -= lr * grad                        │ uses running averages of the gradient and its square                              │
├──────────────────────────┼───────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Learning rate            │ one fixed value, same for every parameter │ adapts per parameter, automatically                                               │
├──────────────────────────┼───────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Memory of past gradients │ none (plain SGD, no momentum)             │ yes — exponentially-decaying averages of both mean and variance of past gradients │
└──────────────────────────┴───────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

Adam = momentum + adaptive per-parameter step size, informally:
m_t = β1·m_{t-1} + (1-β1)·grad          # running average of the gradient (momentum)
v_t = β2·v_{t-1} + (1-β2)·grad²         # running average of the squared gradient
param -= lr * m_t / (sqrt(v_t) + eps)   # bigger past gradients → smaller effective step
- Momentum (m_t) smooths out noisy gradients and keeps moving in a consistent direction instead of oscillating.
- Adaptive scaling (v_t) shrinks the step for parameters with large/noisy gradients and lets parameters with small/quiet gradients take relatively bigger steps — so you don't need one perfectly-tuned global lr for every parameter.
"""

for epoch in range(200):
    # TODO 3: standard training step — zero_grad, forward, loss, backward, optimizer.step()
    logits = model(x_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

with torch.no_grad():
    val_logits = model(x_val)
    val_preds = (val_logits >= 0).float()
    accuracy = (val_preds == y_val).float().mean().item()

print(f"\nvalidation accuracy: {accuracy:.3f}  (target: >0.95)")
