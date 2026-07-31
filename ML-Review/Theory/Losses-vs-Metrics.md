Related: [[Google - ML SWE Interview - Study Plan]], [[Entropy-CrossEntropy-KLDivergence]]

# Losses vs. Metrics — BCE, Logits/Labels, and Classification Metrics

Underlies exe2. Goal: explain why the loss you train on and the metric you report are different things, what a logit actually represents, how to turn it into a hard prediction, and the confusion-matrix metrics that follow.

---

## 1. Logits and labels — what the data actually represents

**Definition:** in a dataset like `logits = [2.0, -1.0, ...]`, `labels = [1.0, 0.0, ...]`, **each row is one independent sample**, not a feature. `logits[i]` is a model's raw output (pre-sigmoid) for sample `i` — this already assumes some upstream model ran; the exercise skips the actual input (image/audio/whatever) and starts from the model's output, to isolate just the loss/metric computation. `labels[i]` is the ground-truth class for sample `i`: `0` or `1`.

**One-hot, correctly scoped:** one-hot describes the classes *within one example*, not across different examples. For binary classification, one example's full one-hot is length 2: `label=1` → `[0,1]`, `label=0` → `[1,0]`. The two entries are always complementary (second = 1 − first), so storing both is redundant — a single bit (`0` or `1`) already encodes it. So `labels = [1,0,1,0,...]` is **10 separate compressed one-hot labels** (one per sample), not one one-hot vector spread across 10 classes.

**Likely follow-up:** *"Why not store both class probabilities per sample?"* → Redundant — they must sum to 1, so the second number carries zero extra information.

---

## 2. From logit to probability — why sigmoid

**Definition:** a raw linear layer output (`w·x+b`) is unconstrained — any real number. A probability must be in `(0,1)`. Sigmoid, `σ(x) = 1/(1+e^-x)`, bridges the two: monotonic (order-preserving), smooth (differentiable), maps all of ℝ into `(0,1)`.

**Why sigmoid specifically — it's the algebraic inverse of log-odds, not an arbitrary choice:**
```
odds(p)  = p / (1-p)                  range (0, inf), lopsided around 1
logit(p) = log(p / (1-p))             range (-inf, inf), 0 at p=0.5
```
A raw linear layer's output range matches log-odds' range exactly — so it's natural to train the network to output log-odds directly (hence the name "logit" for the raw output). To go back from log-odds to a probability, solve `logit(p) = x` for `p`:
```
p/(1-p) = e^x  ->  p(1+e^x) = e^x  ->  p = 1/(1+e^-x)
```
That's sigmoid. **`sigmoid = logit⁻¹`, algebraically** — not a convenient squashing function picked by convention.

**Numeric round-trip:** `x=2.0 → σ(2.0) ≈ 0.881 → logit(0.881) = log(0.881/0.119) = log(7.4) ≈ 2.0` ✓

---

## 3. From probability to a hard prediction — thresholding

**Rule:** predict class 1 iff `p ≥ 0.5`. **Why 0.5, not some other cutoff:** there are only 2 classes, so `P(y=1) ≥ 0.5` automatically means `P(y=1) ≥ P(y=0)` — class 1 is at least as likely as class 0. The error-minimizing decision, given only a probability estimate, is to guess whichever class you currently believe is more likely (the Bayes-optimal rule under symmetric 0/1 cost) — 0.5 is exactly the tie-breaking point between the two.

**Shortcut — skip computing sigmoid entirely:** since `sigmoid(x) ≥ 0.5 ⟺ x ≥ 0` (sigmoid crosses 0.5 exactly at input 0, and is monotonic), thresholding the raw logit at 0 gives the identical decision:
```python
preds = (logits >= 0).float()
```
- `logits >= 0` is a **vectorized** elementwise comparison (no Python loop) → a boolean tensor (`True`/`False`)
- `.float()` casts `True→1.0`, `False→0.0`, matching `labels`' dtype for later comparison

**Worked trace (10-sample toy set):**

| logit | ≥0? | pred | label | outcome |
|---|---|---|---|---|
| 2.0 | T | 1 | 1 | correct (TP) |
| -1.0 | F | 0 | 0 | correct (TN) |
| 0.5 | T | 1 | 1 | correct (TP) |
| -3.0 | F | 0 | 0 | correct (TN) |
| 1.5 | T | 1 | 1 | correct (TP) |
| -0.2 | F | 0 | 0 | correct (TN) |
| 3.0 | T | 1 | 1 | correct (TP) |
| -2.5 | F | 0 | 0 | correct (TN) |
| 0.1 | T | 1 | **0** | false positive |
| -0.8 | F | 0 | **1** | false negative |

8 correct (4 TP, 4 TN), 1 FP, 1 FN.

**Threshold isn't fixed by law:** 0.5 assumes false positives and false negatives cost the same. If they don't (e.g. a missed real event is worse than a false alarm), you move the threshold — which trades precision against recall (section 6).

---

## 4. BCE (Binary Cross-Entropy) — the loss

**What it is:** `nn.BCEWithLogitsLoss` — the 2-class special case of cross-entropy (full derivation from entropy → cross-entropy → BCE: [[Entropy-CrossEntropy-KLDivergence]]). Per-sample: `-[y·log(p) + (1-y)·log(1-p)]`, mean-reduced over the batch by default.

**Usage pattern** (it's a class — instantiate, then call):
```python
loss_fn = nn.BCEWithLogitsLoss()
bce_loss = loss_fn(logits, labels)   # NOT nn.BCEWithLogitsLoss(logits, labels) -- that
                                       # passes your tensors as constructor args (weight,
                                       # size_average), not as data to score.
```

**Why BCE and not accuracy as the training loss:** accuracy is a hard 0/1 decision — flat almost everywhere, zero/undefined gradient, can't drive gradient descent. BCE is smooth everywhere, so backprop can use it.

**Why BCE rewards confidence, not just direction.** For one already-labeled sample, `y` is certain (not some abstract "70% probability") — but the *model* doesn't see `y`, only the input, so its output `q=sigmoid(logit)` is a stated confidence. If `y=1`, the loss collapses to `-log(q)` (the `(1-y)` term vanishes): guessing `q` close to 1 is rewarded, `q` close to 0.5 (weak, hedging) is penalized more, `q` close to 0 (confidently wrong) is penalized hardest. Concretely, for sample 6 (`logit=3.0`, `label=1.0`, model's actual guess `q=0.953`):

| guess `q` | `-log(q)` |
|---|---|
| 0.5 | 0.693 |
| 0.8 | 0.223 |
| 0.953 (model's real guess) | 0.049 |
| 1.0 (perfect) | 0.000 |

This is why a model that's *right but unconfident* still gets penalized more than one that's *right and confident* — accuracy alone can't see that difference, BCE can.

**Why minimizing BCE actually pushes `q` toward the true probability** (not just toward the right class): see [[Entropy-CrossEntropy-KLDivergence]] section 2 — BCE is a proper scoring rule, minimized in expectation exactly when the guess equals the true probability, provable by taking `d(E[loss])/dq = 0`.

---

## 5. Confusion matrix — TP / FP / FN / TN

Positive class = label 1.

| | actual = 1 | actual = 0 |
|---|---|---|
| **predicted = 1** | TP | FP |
| **predicted = 0** | FN | TN |

**Vectorized counting** (preds, labels already 0.0/1.0 float tensors):
```python
tp = ((preds == 1) & (labels == 1)).sum().item()
fp = ((preds == 1) & (labels == 0)).sum().item()
fn = ((preds == 0) & (labels == 1)).sum().item()
tn = ((preds == 0) & (labels == 0)).sum().item()
```
`&` is elementwise boolean AND; `.sum()` counts `True`s; `.item()` converts the resulting 0-dim tensor to a plain Python number. On the 10-sample trace above: `tp=4, fp=1, fn=1, tn=4`.

---

## 6. Accuracy, Precision, Recall, F1

```python
accuracy  = (tp + tn) / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)   # harmonic mean
```

**Precision vs. recall — two different questions about the same confusion matrix:**
- **Precision** = "of everything I *predicted* positive, how much was actually right?" — denominator is `tp+fp`, **your output**.
- **Recall** = "of everything that's *actually* positive, how much did I catch?" — denominator is `tp+fn`, **the ground truth**.

Same `tp` on top, different universe on the bottom — that's the whole difference. They can be gamed independently: flag only one thing you're certain about → precision→1 but recall craters (missed everything else); flag everything as positive → recall→1 but precision craters (drowning in false alarms). Neither alone tells the full story, which is why F1 exists.

**Grounded example — blink detection:**
- High precision, low recall: only flags "blink" when very confident → misses real blinks → character's eyes barely close.
- High recall, low precision: flags "blink" liberally → catches every real blink but also false alarms → character blinks too much, looks twitchy.

**Why F1 uses the harmonic mean, not the arithmetic mean:** harmonic mean punishes imbalance much harder. Precision=1.0, recall=0.0 (never catches anything, but always right when it does guess positive): arithmetic mean = 0.5 (looks fine!); harmonic mean = `2·1·0/(1+0) = 0` (correctly flags the model as useless). F1 can't be propped up by excelling at one while ignoring the other.

**Note on this exercise's specific numbers:** precision and recall both come out to 0.800 here — a coincidence of this dataset (`fp=1` happens to equal `fn=1`), not a general rule. A rare-event confusion matrix like `tp=3, fp=1, fn=9` gives `precision=0.75` but `recall=0.25` — clearly different, a cautious-but-incomplete detector.

---

## Why this matters for the XR Audio Face Tracking role

- Every per-frame binary classifier in this stack (VAD, blink detection, expression-onset detection) trains on BCE and reports accuracy/precision/recall/F1 — being able to derive *why* they're different, unprompted, is exactly what this exercise is testing.
- The precision/recall tradeoff is a real design decision for this role, not textbook trivia: which error type is worse (missed blink vs. false blink; missed speech vs. false speech-triggered animation) determines which way you'd move the 0.5 threshold.

---

## 90-second recap (say this out loud, unprompted)

> "The model's raw output is a logit — an unconstrained real number meant to represent log-odds. Sigmoid converts it to a probability because sigmoid is literally the algebraic inverse of the log-odds function. To get a hard prediction, threshold at 0.5 probability, which is the same as thresholding the raw logit at 0, since sigmoid crosses 0.5 exactly at 0 — that's the Bayes-optimal 'guess the more likely class' rule. Training uses BCE, not accuracy, because BCE is differentiable and rewards calibrated confidence, not just correctness — accuracy has no usable gradient. Once you have hard predictions, you build a confusion matrix — TP, FP, FN, TN — and accuracy, precision, and recall all come from different slices of it. Precision asks how trustworthy your positive calls are; recall asks how complete they are; they can be gamed independently, which is why F1, the harmonic mean of the two, exists — it collapses if either one is bad, unlike a plain average."

---

## Sources / cross-links

- [[Google - ML SWE Interview - Study Plan]] — Day 6 (exe2)
- [[Entropy-CrossEntropy-KLDivergence]] — full cross-entropy/BCE derivation this file builds on
