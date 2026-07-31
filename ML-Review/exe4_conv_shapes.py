"""
4. Conv/pool output shapes and receptive field — CNN foundations, ~20 min
The two formulas that come up in every CNN interview question:

  output size:     out = floor((in + 2*pad - kernel) / stride) + 1
  receptive field:  RF' = RF + (kernel - 1) * jump      (jump' = jump * stride)

Fill in both functions, then the script checks your `conv_output_size` against real
nn.Conv2d layers, and your `receptive_field_after` against the classic fact that three
stacked 3x3 stride-1 convs see the same input area as a single 7x7 conv.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # this machine has a libiomp/libomp DLL conflict; without this, `import torch` crashes

import torch
import torch.nn as nn


def conv_output_size(in_size: int, kernel: int, stride: int, padding: int) -> int:
    """Spatial size of one dimension after a conv/pool layer."""
    # TODO 1: implement out = floor((in_size + 2*padding - kernel) / stride) + 1
    # (integer division // already floors for non-negative operands)
    return (in_size + 2 * padding - kernel) // stride + 1

"""
What "receptive field" means, in plain terms: for one single output value deep in the network, it's how many original input pixels can possibly affect that one number. Early layers: a small window. Later layers: much larger, because each later-layer unit is built from a window of already-aggregated earlier units.

The tricky variable is jump, not rf. jump = "how many input pixels do you move by, when you move one step in the current layer's output grid." This is the thing that's easy to lose track of, and it's exactly why the formula needs to track it separately.

Concrete trace, two conv layers, actual pixel indices:

Layer 1: kernel=3, stride=1, on a 1D input [0,1,2,3,4,5,...].
- output1[0] = built from input [0,1,2]
- output1[1] = built from input [1,2,3]

So output1's receptive field = 3 pixels (rf=3). Moving one step in output1 (index 0→1) shifts the input window by exactly 1 pixel — that's jump=1. Formula check: rf = 1 + (3-1)·1 = 3 ✓, jump = 1·1 = 1 ✓.

Layer 2: kernel=3, stride=2, applied to output1.
- output2[0] = built from output1[0], output1[1], output1[2]
→ those cover input [0,1,2] ∪ [1,2,3] ∪ [2,3,4] = input [0..4] (5 pixels)
- output2[1] = built from output1[2], output1[3], output1[4] (stride=2 means we jump 2 positions in output1 for the next output2 unit)
→ those cover input [2,3,4] ∪ [3,4,5] ∪ [4,5,6] = input [2..6] (also 5 pixels — same size, just shifted)

Two things to read off this:
1. rf=5 — both output2[0] and output2[1] see a 5-pixel window. Formula: rf = 3 + (3-1)·1 = 5 ✓
2. The window shifted by 2 input pixels (started at input 0, now starts at input 2) for just one step in output2. That shift is jump. Formula: jump = 1 · 2 = 2 ✓ — because layer 2's stride (2) multiplies onto the previous jump (1), since each step in output2 now corresponds to 2 steps in output1, and each output1 step was already worth 1 input pixel.

So the two update rules, in words:
- rf += (kernel-1) * jump — adding kernel-1 more units at the current layer's resolution, but each of those units is worth jump input pixels (not 1), because jump already accounts for everything upstream compressing/downsampling the input.
- jump *= stride — this layer's stride multiplies onto however much ground each previous step already covered, since strides compound through the network.

Why this matters for your interview: it's the mechanism behind "three 3×3 convs = one 7×7's receptive field, with far fewer parameters" (27 weights vs 49) — and it's also why stride/pooling layers grow receptive field fast (multiplicatively, via jump), while stride-1 convs grow it slowly (additively) — a real architectural tradeoff, not just a formula to memorize.
"""

"""
How output2[0] is actually built — it's a weighted sum (dot product), not concatenation
Layer 2 slides its own kernel over output1 (which is just "the input" from Layer 2's point of view) and computes a weighted sum at each position:
output2[0] = w0·output1[0] + w1·output1[1] + w2·output1[2] + bias
"""

# --- worked example: what this function is actually computing ---
# Concrete 2-layer case, kernel=3/stride=1 for both, on a 1D input for clarity (same
# logic applies per-dimension in 2D). Receptive field (RF) is always defined for ONE
# single output unit -- "which input pixels can possibly affect this one number."
#
# The full input, explicitly (positions 0-9, just plain numbers/pixels, nothing fancier):
#   input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Everything below refers to *indices into this array*, not the values stored in it --
# "input {0,1,2}" means "positions 0, 1, 2 of `input`", i.e. input[0:3].
#
# Layer 1 config:  kernel = 3, stride = 1
# Layer 2 config:  kernel = 3, stride = 1
#
# Layer 1 (kernel=3, stride=1): each output1 unit is a weighted sum (dot product) of a
# 3-pixel input window (3 = this layer's kernel), starting `stride` positions apart,
# NOT a concatenation -- e.g. output1[0] = w0*input[0] + w1*input[1] + w2*input[2] + bias.
#   output1[0] <- input {0, 1, 2}      (window starts at 0*stride = 0)
#   output1[1] <- input {1, 2, 3}      (window starts at 1*stride = 1)
# -> RF of a single output1 unit = 3 (just the kernel size -- nothing upstream yet)
# -> jump = 1 (moving one step in output1 shifts the input window by 1 pixel = stride)
#
# Layer 2 (kernel=3, stride=1) applied to output1: output2[0] is a weighted sum of THREE
# output1 units -- output1[0], output1[1], output1[2] -- so output2[0]'s receptive field
# is the UNION of those three units' own receptive fields (overlaps counted once, not summed):
#   output1[0] -> input {0,1,2}
#   output1[1] -> input {1,2,3}
#   output1[2] -> input {2,3,4}
#   union      -> input {0,1,2,3,4}  = 5 distinct pixels
# -> RF after layer 2 = 5, matching: rf = 3 + (kernel-1)*jump = 3 + 2*1 = 5
# -> jump after layer 2 = 1*stride = 1 (would be 2 if layer 2's stride were 2 instead --
#    each step in output2 would then skip 2 output1 units, doubling the pixel shift)
#
# What the loop below does per layer, in words:
#   rf   = rf + (kernel-1)*jump   <- (kernel-1) EXTRA units get unioned in beyond the
#                                     first, each one's window offset by `jump` pixels
#   jump = jump * stride          <- this layer's stride compounds onto every layer before it
def receptive_field_after(layers: list[tuple[int, int]]) -> int:
    """
    layers: list of (kernel, stride) pairs, applied in order.
    Returns the receptive field (in input pixels) of one output unit after all layers.
    Start from rf=1, jump=1 and update per layer:
        rf   = rf + (kernel - 1) * jump
        jump = jump * stride
    """
    rf, jump = 1, 1
    for kernel, stride in layers:
        # TODO 2: update rf and jump using the formulas above
        rf += (kernel - 1) * jump
        jump *= stride
    return rf


# --- checks against real conv layers ---
cases = [
    (16, 3, 1, 0),
    (16, 3, 1, 1),
    (16, 3, 2, 1),
    (28, 5, 1, 0),
    (32, 4, 2, 1),
]
for in_size, kernel, stride, padding in cases:
    conv = nn.Conv2d(1, 1, kernel_size=kernel, stride=stride, padding=padding)
    actual = conv(torch.randn(1, 1, in_size, in_size)).shape[-1]
    mine = conv_output_size(in_size, kernel, stride, padding)
    status = "OK" if mine == actual else "MISMATCH"
    print(f"in={in_size:3d} k={kernel} s={stride} p={padding}  mine={mine}  actual={actual}  [{status}]")

# --- receptive field of three stacked 3x3 stride-1 convs ---
rf = receptive_field_after([(3, 1), (3, 1), (3, 1)])
print(f"\nreceptive field after three 3x3 stride-1 convs: {rf}  (expect 7)")
