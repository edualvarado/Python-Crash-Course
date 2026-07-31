"""
1. Gradient descent with autograd — Foundations, ~15 min
Fit y = 3x + 2 (+ noise) with a single linear unit, using torch's autograd instead of
a closed-form solution. The point is the mechanics: forward -> loss -> backward -> update -> zero_grad.

Fill in the TODOs in `train`. Run the file — it should print a final (w, b) close to (3.0, 2.0)
and a loss that decreases every epoch.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # this machine has a libiomp/libomp DLL conflict; without this, `import torch` crashes

import torch

torch.manual_seed(0)

N = 200
x = torch.linspace(-5, 5, N)
y = 3 * x + 2 + torch.randn(N) * 0.5


def train(x: torch.Tensor, y: torch.Tensor, epochs: int = 200, lr: float = 0.01):
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for epoch in range(epochs):
        # TODO 1: forward pass — predict y_hat from x using w, b
        y_hat = w * x + b

        # TODO 2: compute mean squared error loss between y_hat and y
        loss = (y_hat - y).pow(2).mean()

        # TODO 3: backpropagate the loss
        loss.backward()

        with torch.no_grad():
            # TODO 4: update w and b in-place using their .grad and lr
            w -= lr * w.grad
            b -= lr * b.grad

            # TODO 5: zero out w.grad and b.grad so they don't accumulate next epoch
            w.grad.zero_()
            b.grad.zero_()

        if epoch % 40 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")

    return w.detach(), b.detach()


w, b = train(x, y)
print(f"\nfinal: w={w.item():.3f} (expect ~3.0), b={b.item():.3f} (expect ~2.0)")
