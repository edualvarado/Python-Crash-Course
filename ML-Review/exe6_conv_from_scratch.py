"""
6. Naive 2D convolution from scratch (numpy), ~30 min

Day 6 of the Google interview study plan. Implement the forward pass of a 2D
convolution using nothing but numpy -- no autograd, no nn.Conv2d. Two parts:

  1. Single-channel, single-filter: x is (H, W), w is (kh, kw).
  2. Multi-channel, multi-filter:   x is (C_in, H, W), w is (C_out, C_in, kh, kw).

Output size formula (same one from exe4, but now you're building the operation that
produces that shape instead of just predicting it):
  out = floor((in + 2*pad - dilation*(kernel-1) - 1) / stride) + 1
With dilation=1 this simplifies to floor((in + 2*pad - kernel) / stride) + 1.

Part 1 is checked against a hand-computed example -- work out `expected1` yourself on
paper before running this file, then fill it in.
Part 2 is checked against torch's nn.Conv2d as ground truth (its weights are copied in
from yours, so it's a fair comparison, not a different computation).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must be set + torch imported before numpy,
# otherwise torch's fbgemm.dll resolves against numpy's already-loaded OpenMP DLL and fails (WinError 127)
import torch
import torch.nn as nn

import numpy as np


def conv2d_single_channel(x: np.ndarray, w: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    x: (H, W) input
    w: (kh, kw) single kernel
    Returns: (out_h, out_w) feature map.

    Plain nested-loop convolution -- no im2col yet, that's the IF TIME follow-up.
    """
    # TODO 1: pad x if padding > 0 (np.pad)
    H, W = x.shape
    kH, kW = w.shape
    x = np.pad(x, (padding, padding))

    # TODO 2: compute out_h, out_w from the output size formula
    out_h = (H - kH + 2*padding) // stride + 1
    out_w = (W - kW + 2*padding) // stride + 1

    # TODO 3: nested loop over output positions (i, j); for each, slice the
    #         corresponding (kh, kw) input region, elementwise multiply by w, sum
    #         -> that's out[i, j]

    """
    Rule of thumb: as many slices as you need, all inside one bracket, comma-separated — x[a:b, c:d] for 2D, x[a:b, c:d, e:f] for 3D, etc.
    """

    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i][j] = (x[i*stride : i*stride + kH, j*stride : j*stride + kW] * w).sum()
            
    return out


def conv2d_multi(x: np.ndarray, w: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    x: (C_in, H, W) input
    w: (C_out, C_in, kh, kw) filters
    Returns: (C_out, out_h, out_w) feature map.

    Hint: for a fixed output filter c_out, sum conv2d_single_channel(x[c_in], w[c_out, c_in])
    over all c_in -- that gives you one (out_h, out_w) map. Stack across c_out.
    """
    # TODO 1: figure out out_h, out_w (reuse the same formula / call conv2d_single_channel
    #         once to get the shape, or compute it directly)
    # TODO 2: loop over output filters c_out, loop over input channels c_in, accumulate

    C_in, H, W = x.shape # (3,6,6)
    C_out, C_in, kH, kW = w.shape # (4,3,3,3)

    result = []
    for c_out in range(C_out):
        out = 0
        for c_in in range(C_in):
            out += conv2d_single_channel(x[c_in], w[c_out, c_in], stride, padding)
        result.append(out)

    return np.stack(result)


# --- Part 1: hand-computed check ---
x1 = np.array([
    [1, 2, 0, 1],
    [0, 1, 2, 1],
    [1, 0, 1, 2],
    [2, 1, 0, 1],
], dtype=float)

w1 = np.array([
    [1, 0],
    [0, 1],
], dtype=float)

# TODO: hand-compute the output of conv2d_single_channel(x1, w1, stride=1, padding=0)
# on paper first (what shape should it be? use the formula above), then fill it in here.
expected1 = None

result1 = conv2d_single_channel(x1, w1, stride=1, padding=0)
print("Part 1 result:\n", result1)
if expected1 is not None:
    print("Matches hand computation:", np.allclose(result1, expected1))
else:
    print("Fill in `expected1` with your hand-computed answer to self-check.")

# stride/padding edge case -- just check the shape matches the formula
result1b = conv2d_single_channel(x1, w1, stride=2, padding=1)
expected_shape = ((4 + 2 * 1 - 2) // 2 + 1,) * 2
print(f"\nPart 1b (stride=2, padding=1) shape: {None if result1b is None else result1b.shape}"
      f"  expected via formula: {expected_shape}")

# --- Part 2: multi-channel check against torch nn.Conv2d ---
rng = np.random.default_rng(0)
C_in, C_out, H, W, kh, kw = 3, 4, 6, 6, 3, 3
x2 = rng.standard_normal((C_in, H, W))
w2 = rng.standard_normal((C_out, C_in, kh, kw))

mine2 = conv2d_multi(x2, w2, stride=1, padding=1)

conv = nn.Conv2d(C_in, C_out, kernel_size=kh, stride=1, padding=1, bias=False)
with torch.no_grad():
    conv.weight.copy_(torch.tensor(w2, dtype=torch.float32))
    torch_out = conv(torch.tensor(x2, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

if mine2 is not None:
    print("\nPart 2 matches torch nn.Conv2d:", np.allclose(mine2, torch_out, atol=1e-4))
else:
    print("\nImplement conv2d_multi, then rerun to compare against torch.")
