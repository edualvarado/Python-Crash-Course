"""
2. Loss functions vs. evaluation metrics — Foundations, ~15 min
The loss you optimize (differentiable, e.g. BCE) is not the metric you report (often
non-differentiable, e.g. accuracy/precision/recall). This exercise fixes a toy set of
logits + labels so you can compute both by hand and sanity-check the numbers yourself.

Fill in the TODOs. Expected values are given in comments — if your numbers don't match,
your formula is wrong, not the data.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # this machine has a libiomp/libomp DLL conflict; without this, `import torch` crashes

import torch
import torch.nn as nn

# 10 fixed examples: raw model outputs (logits, pre-sigmoid) and true binary labels.
logits = torch.tensor([2.0, -1.0, 0.5, -3.0, 1.5, -0.2, 3.0, -2.5, 0.1, -0.8])
labels = torch.tensor([1.0,  0.0, 1.0,  0.0, 1.0,  0.0, 1.0,  0.0, 0.0,  1.0])

# TODO 1: compute the BCEWithLogitsLoss between `logits` and `labels` using nn.BCEWithLogitsLoss
loss_fn = nn.BCEWithLogitsLoss()
bce_loss = loss_fn(logits, labels)
print(f"BCE loss: {bce_loss.item():.4f}")

# bce_loss_manual: same computation, but by hand from the raw formula -- sigmoid the
# logits into probabilities first, then plug into -[y*log(p) + (1-y)*log(1-p)], mean
# over the batch (nn.BCEWithLogitsLoss defaults to reduction='mean'). Once you fix TODO 1
# above, this should match it almost exactly (tiny float differences only).
#
# Why sigmoid converts a logit to a probability (not just "a squashing function"):
#   odds(p)  = p / (1-p)                      range (0, inf) -- lopsided around 1
#   logit(p) = log(p / (1-p))                 range (-inf, inf), 0 at p=0.5 -- this is
#              what a raw linear layer (w.x+b, unconstrained) is trained to output
#   Solve logit(p) = x for p (invert it):
#     p/(1-p) = e^x  ->  p = e^x(1-p)  ->  p(1+e^x) = e^x  ->  p = 1/(1+e^-x)
#   That last expression IS sigmoid. So sigmoid = logit^-1, algebraically -- not an
#   arbitrary choice. Round-trip check: x=2.0 -> sigmoid(2.0)=0.881 -> logit(0.881)
#   = log(0.881/0.119) = log(7.4) = 2.0 again.
p = torch.sigmoid(logits)
bce_loss_manual = -(labels * torch.log(p) + (1 - labels) * torch.log(1 - p)).mean()
print(f"BCE loss (manual): {bce_loss_manual.item():.4f}")

# TODO 2: convert logits -> predicted class (0 or 1) by thresholding sigmoid(logits) at 0.5
# hint: sigmoid(x) >= 0.5  <=>  x >= 0
preds = (logits >= 0).float()
# preds = tensor([1., 0., 1., 0., 1., 0., 1., 0., 1., 0.])
print(f"preds:  {preds.tolist()}")
print(f"labels: {labels.tolist()}")

# TODO 3: count true positives, false positives, false negatives, true negatives
# (positive class = label 1)
tp = ((preds == 1) & (labels == 1)).sum().item()
fp = ((preds == 1) & (labels == 0)).sum().item()
fn = ((preds == 0) & (labels == 1)).sum().item()
tn = ((preds == 0) & (labels == 0)).sum().item()
print(f"tp={tp} fp={fp} fn={fn} tn={tn}  (expect tp=4 fp=1 fn=1 tn=4)")

# TODO 4: compute accuracy, precision, recall, f1 from tp/fp/fn/tn (plain formulas, no torchmetrics)
accuracy = (tp + tn) / (tp + tn + fp + fn) # Accuracy — correct predictions over everything
precision = tp / (tp + fp) # Precision — of everything you predicted positive, how much was right.
recall = tp / (tp + fn)  # of everything actually positive, how much did you catch
f1 = 2 * precision * recall / (precision + recall) # harmonic mean of precision and recall (not the plain average):

print(f"accuracy:  {accuracy:.3f}  (expect 0.800)")
print(f"precision: {precision:.3f}  (expect 0.800)")
print(f"recall:    {recall:.3f}  (expect 0.800)")
print(f"f1:        {f1:.3f}  (expect 0.800)")
