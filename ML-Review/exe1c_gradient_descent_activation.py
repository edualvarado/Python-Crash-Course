"""
1c. Gradient descent through an activation function — Foundations, ~15 min
Same target as exe1 (y = 3x + 2 + noise), but the single unit now applies a sigmoid on
top of the linear part: y_hat = sigmoid(w*x + b). Sigmoid squashes any input to (0, 1),
so it can never match y's actual range (~-13 to 17 over x in [-5, 5]). The point is to
watch gradient descent get stuck: as |w*x + b| grows trying to compensate, sigmoid
saturates and its gradient collapses toward zero, so learning stalls well before the fit
is any good. Compare the final loss and y_hat range here to exe1's clean convergence.
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
        z = w * x + b            # pre-activation: same linear part as exe1
        y_hat = torch.sigmoid(z)  # activation: squashes z to (0, 1)

        loss = (y_hat - y).pow(2).mean()

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

            w.grad.zero_()
            b.grad.zero_()

        if epoch % 40 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")

    return w.detach(), b.detach()


w, b = train(x, y)

with torch.no_grad():
    y_hat = torch.sigmoid(w * x + b)

print(f"\nfinal: w={w.item():.3f}, b={b.item():.3f}")
print(f"target y range:   [{y.min().item():7.2f}, {y.max().item():7.2f}]")
print(f"model y_hat range: [{y_hat.min().item():6.2f}, {y_hat.max().item():6.2f}]  (sigmoid caps this to (0,1) -- can't cover the target range)")
