Related: [[Google - ML SWE Interview - Study Plan]], [[RNN-LSTM]], [[Audio-Driven-Facial-Animation]], [[Multimodal-Fusion]]

# Real-Time / Edge Deployment Constraints

Day 4 topic 4 — the last of the four. Goal: explain quantization, pruning, and distillation unprompted, <2 min each, and be ready to argue *why* they matter specifically for a headset, not just recite the definitions.

---

## 1. Quantization

**Definition:** Reducing the numerical precision used to store a model's weights and/or activations — most commonly FP32 (32-bit floating point) → INT8 (8-bit integer). Each weight, normally a 32-bit float, gets mapped to one of 256 integer levels via a scale factor (and sometimes a zero-point offset) per tensor or channel. This shrinks model size roughly 4× and can run dramatically faster on hardware with dedicated low-precision integer arithmetic units, at the cost of a typically small accuracy drop — neural networks are generally fairly robust to this, since the precision loss behaves a bit like mild, structured noise. Two ways to do it: **post-training quantization (PTQ)** — quantize an already-trained FP32 model directly, fast and cheap but sometimes a bigger accuracy hit — vs. **quantization-aware training (QAT)** — simulate the precision loss *during* training so the model's weights adapt to be robust to it, better accuracy but requires retraining.

**Example 1 (XR):** Quantizing a trained FaceFormer-style audio encoder + mesh decoder from FP32 to INT8 before deploying on-headset — same model, ~4× smaller, much faster per-frame inference, small accuracy tradeoff that's acceptable given a hard frame budget.

**Example 2 (own domain tie):** Quantizing a SoleSense-style TCN (Temporal Convolutional Network) model to run on a low-power embedded chip inside a smart insole, rather than a full PC — a real hardware constraint, not a hypothetical one.

**Likely follow-up:** *"Why INT8 specifically, why not go lower (INT4, binary)?"* → INT8 is the sweet spot most current mobile/edge hardware has dedicated fast integer arithmetic units for — a real hardware-support reason, not just a theoretical choice. Going lower (INT4, or 1-bit binary weights) saves more memory/compute but typically costs noticeably more accuracy unless the model is specifically designed and trained for extreme quantization, and less hardware has efficient native support for it yet.

---

## 2. Pruning

**Definition:** Removing redundant or low-importance weights (or entire structural units — channels, attention heads, whole layers) from a trained network, based on some importance criterion (weight magnitude — small weights near zero contribute little; gradient-based sensitivity; or explicit sparsity-inducing regularization during training), to shrink the model and speed up inference. Two flavors: **unstructured pruning** — remove individual weights anywhere in a weight matrix, producing a sparse matrix; higher possible compression ratio, but you need specialized sparse-matrix hardware/software to actually realize a speedup, since a sparse matrix stored densely saves no compute. **Structured pruning** — remove entire channels/filters/attention heads/layers; typically a lower compression ratio for the same accuracy hit, but the result is just a smaller *dense* model that runs faster on completely standard hardware, no special support needed.

**Example 1 (XR):** Pruning less-important attention heads or channels out of a trained cross-attention fusion model ([[Multimodal-Fusion]]) to hit a hard per-frame compute budget on headset hardware — structured pruning specifically, since headset chips generally lack specialized sparse-matrix acceleration.

**Example 2 (own domain tie):** Pruning redundant channels from a SoleSense-style TCN after training but before deploying it to a resource-constrained embedded board — same "shrink a trained model down" motivation, no audio/visual involved.

**Likely follow-up:** *"Why not just train a smaller model from scratch instead of pruning a big one down?"* → Empirically, training a large model and then pruning it down tends to reach better accuracy at a given final size than training a small architecture of that same size from scratch (the intuition behind the "lottery ticket hypothesis" — a big model is easier to optimize and tends to contain a smaller subnetwork that alone would train just as well, but that subnetwork is very hard to find or train correctly from a random initialization directly). Big-then-shrink generally beats small-from-the-start at matched final size.

---

## 3. Knowledge distillation

**Definition:** Training a smaller "student" model to mimic a larger, already-trained "teacher" model's outputs, rather than training the student only on the original ground-truth hard labels. The student is trained (often in addition to the hard labels) to match the teacher's full output distribution — its soft predictions across all options, not just the single correct answer. That extra signal, sometimes called "dark knowledge," carries information a hard label doesn't (e.g. how similar the teacher considers two different wrong answers to be), and lets a small student reach better accuracy than training that same small architecture on hard labels alone would.

**Example 1 (XR):** Distilling a large, high-accuracy but slow model (trained offline with a large compute/data budget) down into a small student model that's what actually ships on-headset. The teacher never runs at inference time — only during the student's training.

**Example 2 (own domain tie):** A large offline motion-synthesis model used as a teacher for a small real-time student deployed in an interactive application — a common pattern in real-time graphics/animation research, and directly aligned with what a real-time behavior engine (Pneuma Labs) needs to do.

**Likely follow-up:** *"What's actually being 'distilled' — what does the student learn that hard labels wouldn't teach it?"* → The teacher's soft output distribution encodes relative similarity/uncertainty beyond the single correct answer — e.g. for a viseme prediction, the teacher might say "70% confident this is viseme A, but 25% it could plausibly be the very similar viseme B." That relative-similarity structure helps the student learn a smoother, better-generalizing decision boundary than one-hot hard labels alone would.

---

## 4. The latency-accuracy tradeoff, and why it matters on-device

**Definition:** All three techniques above trade some accuracy for reduced compute, memory, and latency, and they're routinely combined (quantized *and* pruned *and* distilled). The real practical question when deploying to constrained hardware isn't "how do I make this as accurate as possible" — it's "where do I sit on this tradeoff curve," and that's dictated by the actual hard constraint of the target device, not chosen freely.

The XR-specific hard constraint: a **per-frame compute budget tied directly to the display refresh rate**. A 90Hz headset display gives roughly 11ms per frame total — and the face-tracking/animation model's inference is only one consumer of that budget, shared with rendering, head/hand/eye tracking, and everything else running that frame. Miss the budget and frames drop or perceived latency rises — for a live avatar, that's immediately and viscerally noticeable (a laggy or jittery digital face reads as deeply uncanny) in a way a few percentage points of offline accuracy simply doesn't.

**Example 1:** A full-precision (FP32), unpruned, non-distilled large model might have the best offline accuracy metric — but if it can't run inside the 11ms frame budget on headset-class hardware, it's not a valid choice at all, regardless of accuracy. An INT8-quantized, structurally-pruned, distilled small model that loses a few points of accuracy but comfortably fits the budget is the only real option between the two.

**Example 2 (own domain tie):** SoleSense's causal, dilated TCN (small channel count, few residual blocks) was itself a real-time/edge-driven *architecture* choice from the start, rather than a big model shrunk after the fact — same underlying goal (fit a hard per-frame budget on constrained hardware), different technique (architectural efficiency vs. post-training compression).

**Likely follow-up:** *"How would you decide how much to compress — where do you stop?"* → Work backward from the hard latency budget (how many ms of the frame the tracking model is actually allotted, after rendering and other systems take their share), then apply quantization/pruning/distillation — usually all three together — until inference comfortably fits inside that budget, validating that accuracy stays above whatever the product's minimum acceptable quality bar is. The budget is the hard constraint; accuracy is optimized subject to it, not the other way around.

---

## 5. On-device runtimes (quick reference — worth being able to name)

The tools that actually take a quantized/pruned/distilled model and run it efficiently on real hardware:
- **TensorRT** — NVIDIA's inference optimization library/runtime, GPU-focused.
- **TFLite (TensorFlow Lite)** — Google's runtime for mobile/edge devices.
- **CoreML** — Apple's on-device inference framework (iOS, and relevant here, Apple's XR hardware).
- **ONNX (Open Neural Network Exchange) Runtime** — a cross-platform inference engine for models exported into the hardware-agnostic ONNX interchange format.

Not something to go deep on unprompted, but naming one or two correctly if asked "how would you actually ship this" signals you've thought past the algorithm level to the deployment level.

---

## Why this matters for the XR Audio Face Tracking role

- This is the constraint that makes every other Day 4 topic *real* rather than purely academic: the RNN-vs-Transformer latency tradeoff ([[RNN-LSTM]]), the frame-wise-vs-seq2seq choice ([[Audio-Driven-Facial-Animation]]), and cross-attention's O(n·m) cost ([[Multimodal-Fusion]]) all cash out here — into a literal millisecond budget tied to the display refresh rate.
- The honest, senior-sounding answer to "how would you make X fast enough" is almost always some combination of these three techniques plus an architecture-level choice (bounded context, causal-only, smaller model) made *up front* — not squeezing an already-large offline-quality model after the fact.
- Expect this to come up as a direct follow-up to topics 1–3: after any "how would this work" answer, be ready for "and how would you make that run in real time on the headset."

---

## 90-second recap (say this out loud, unprompted)

> "Deploying a model to real-time/edge hardware means trading some accuracy for speed and size, using three main techniques, usually combined. Quantization reduces numerical precision — typically FP32 down to INT8 — shrinking the model about 4x and speeding it up on hardware with dedicated low-precision arithmetic, for a small accuracy cost. Pruning removes redundant weights or whole structural units like channels or attention heads from an already-trained model; structured pruning — removing whole units rather than scattered individual weights — is what actually speeds things up on standard hardware without needing specialized sparse-matrix support. Knowledge distillation trains a small student model to mimic a larger teacher's full output distribution, not just hard labels, which lets the student reach better accuracy than training that same small architecture from scratch. All three trade accuracy for latency, and the real question is where you sit on that curve — which for XR is dictated by a hard per-frame compute budget tied to the display refresh rate, something like 11 milliseconds at 90Hz, shared with rendering and other tracking systems. A model that's more accurate offline but doesn't fit that budget simply isn't a valid choice — you work backward from the latency budget and compress until you fit it, then check accuracy is still acceptable, not the other way around."

---

## Sources / cross-links

- [[Google - ML SWE Interview - Study Plan]] — Day 4, topic 4 (final topic)
- [[RNN-LSTM]] — the RNN-vs-Transformer latency tradeoff this budget cashes out
- [[Audio-Driven-Facial-Animation]] — frame-wise-vs-seq2seq, the other latency-shaped decision
- [[Multimodal-Fusion]] — cross-attention's O(n·m) cost, the other compute concern
- [[Google - Technical Interview]] — role/JD context
