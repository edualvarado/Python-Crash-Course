"""
1b. Same fit as exe1, using torch.optim instead of manual updates
Same problem as exe1_gradient_descent.py (fit y = 3x + 2), but the manual
`w -= lr * w.grad` / `.zero_()` steps are replaced by an optimizer. Compare
this to exe1 line by line: optimizer.zero_grad() replaces TODO 5, optimizer.step()
replaces TODO 4 — the optimizer just does the same in-place updates for you,
looping over whatever parameters you handed it.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # this machine has a libiomp/libomp DLL conflict; without this, `import torch` crashes

import torch

torch.manual_seed(0)

N = 200
x = torch.linspace(-5, 5, N)
y = 3 * x + 2 + torch.randn(N) * 0.5

def train (x: torch.Tensor, y: torch.Tensor, epochs: int = 200, lr: float = 0.01):
    w = torch.zeros(1, requires_grad = True)
    b = torch.zeros(1, requires_grad = True)
    optimizer = torch.optim.SGD([w,b], lr=lr)

    for epoch in range(epochs):
        y_hat = w * x + b
        loss = (y - y_hat).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return w.detach(), b.detach()

w, b = train(x, y)


# ---

N = 200
x = torch.linspace(-5, 5, N)
y = 3 * x + 2 + torch.randn(N) * 0.5


def train(x: torch.Tensor, y: torch.Tensor, epochs: int = 200, lr: float = 0.01):
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=lr)

    for epoch in range(epochs):
        y_hat = w * x + b
        loss = (y_hat - y).pow(2).mean()

        optimizer.zero_grad()  # same job as w.grad.zero_() / b.grad.zero_() in exe1
        loss.backward()
        optimizer.step()       # same job as w -= lr * w.grad / b -= lr * b.grad in exe1

        if epoch % 40 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")

    return w.detach(), b.detach()


w, b = train(x, y)
print(f"\nfinal: w={w.item():.3f} (expect ~3.0), b={b.item():.3f} (expect ~2.0)")
