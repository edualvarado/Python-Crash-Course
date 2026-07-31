Related: [[Google - ML SWE Interview - Study Plan]]

# RNN / LSTM Basics

Day 4 carryover topic. Goal: explain each concept unprompted, <2 min, then land on "why this matters for audio-driven facial animation" — that's the actual interview payoff.

---

## 1. Recurrent Neural Networks (RNN)

**Definition:** A neural network for sequential data that processes one timestep at a time, maintaining a hidden state `h_t` that's updated at each step: `h_t = f(W_x·x_t + W_h·h_{t-1} + b)`. The hidden state acts as a compressed summary of everything seen so far, letting the network's output at time `t` depend on the full history, not just the current input. Same weights (`W_x`, `W_h`) are reused at every timestep — parameter sharing across time, analogous to how a CNN (Convolutional Neural Network — a network that slides shared filter weights across an image or spatial grid) shares weights across space.

**Example 1 (audio):** Feeding a sequence of MFCC/mel-spectrogram frames into an RNN to predict per-frame viseme/blendshape coefficients — the hidden state carries phonetic context from preceding frames, which matters because coarticulation means a phoneme's mouth shape depends on its neighbors.

**Example 2 (IMU/motion):** An IMU (Inertial Measurement Unit — the accelerometer/gyroscope sensor package in mocap suits and wearables) stream (accel/gyro at each timestep) fed through an RNN to regress joint angles or ground-reaction-force — hidden state accumulates motion context (e.g., stance vs swing phase) that a single frame alone can't disambiguate.

**Likely follow-up:** *"Why not just use a sliding window of frames with a plain feedforward net?"* → With window size W, a feedforward/conv net predicting the output at `t` runs the full network over all W frames `[t-W+1, ..., t]`, every single step — a fixed O(W) cost per output, paid identically whether the relevant frame is 1 step back or W steps back. Two knock-on costs: a **hard ceiling** (anything `>W` steps back is simply invisible, no matter how much data you have), and **redundant recomputation** (frame `t` gets reprocessed in every window that includes it — up to W times — so total compute over a length-N sequence is O(N·W), not O(N)). An RNN is O(1) per step: each frame is folded into the hidden state exactly once and reused going forward, with no hyperparameter window size baked into the architecture — context can in principle reach arbitrarily far back (in practice bounded by vanishing gradients, see below).

---

## 2. Vanishing / Exploding Gradients

**Definition:** Training an RNN via backpropagation-through-time (BPTT) multiplies gradients by the same recurrent weight matrix (and derivative of the activation) once per timestep. Over a long sequence this is a repeated multiplication — if the dominant factor is <1, gradients shrink toward zero (vanishing: early timesteps get ~no learning signal); if >1, they blow up (exploding: unstable updates, NaNs). Vanishing is the more common practical failure mode and is what motivated gating architectures.

**Example 1 (audio):** A sentence-length audio clip (~seconds, hundreds of frames) — a vanilla RNN trained on this will struggle to let a phoneme at frame 5 influence the loss computed at frame 300, because the gradient connecting them has decayed through ~295 multiplications.

**Example 2 (motion):** A long mocap sequence where a stylistic cue at the start of a gait cycle should influence foot placement several steps later — vanilla RNN loses that signal over the intervening timesteps.

**Likely follow-up:** *"How do you fix exploding gradients specifically?"* → Gradient clipping (clip the norm of the gradient to a max threshold before the update) — cheap, standard, orthogonal to the gating fix for vanishing gradients.

**Likely follow-up:** *"Isn't the typical fix skip/residual connections?"* → Yes, for vanishing gradients that's the general deep-learning fix, and LSTM's gating is the RNN-specific instance of the same idea:
- **ResNet** (feedforward/CNN): `y = x + F(x)` → `dy/dx = 1 + dF/dx`. The "+1" guarantees an identity gradient path through arbitrarily many layers regardless of `F`.
- **Highway Networks** — explicitly built as "LSTM gating applied to feedforward nets": `y = T(x)·F(x) + (1-T(x))·x`, a *gated* skip connection (learned per-unit). Historically LSTM (1997) came first and inspired this, not the other way around.
- **LSTM cell state** `c_t = f_t⊙c_{t-1} + i_t⊙c̃_t`: the `f_t⊙c_{t-1}` term *is* a gated identity/skip path through time. If `f_t ≈ 1`, `∂c_t/∂c_{t-1} ≈ f_t`, bounded near 1 instead of crushed by repeated `tanh`/weight-matrix multiplications the way vanilla RNN's `h_t = tanh(W·h_{t-1} + ...)` is. Same principle as ResNet — an additive path that lets gradient bypass the nonlinear transform — except *how much* to skip is learned per-timestep via the gate rather than fixed at 1.
- Transformers use residual connections around every attention/FFN sublayer for the identical reason; Residual LSTMs apply the same trick depth-wise across stacked recurrent layers.
- Caveat: skip connections mainly fix **vanishing**, not **exploding** — the identity path doesn't stop the gated/`F(x)` branch from blowing up, so exploding still needs gradient clipping (+ orthogonal init / LayerNorm inside the cell) as a separate fix.

One-liner if this comes up: *"LSTM's cell state is basically a gated residual connection through time — same fix as ResNet, just learned instead of fixed."*

---

## 3. LSTM (and GRU)

**Definition:** LSTM (Long Short-Term Memory) adds a separate cell state `c_t` (a slower-changing "memory highway") alongside the hidden state, regulated by three learned gates: **forget gate** (what to discard from `c_t-1`), **input gate** (what new info to write in), **output gate** (what to expose to `h_t`). The cell state update is largely additive (`c_t = f_t⊙c_{t-1} + i_t⊙c̃_t`) rather than repeated matrix multiplication, which gives gradients a near-linear path backward through time — the core fix for vanishing gradients. GRU (Gated Recurrent Unit) is a simplified variant: merges cell/hidden state into one, uses 2 gates (reset, update) instead of 3 — fewer parameters, often comparable performance, faster to train.

**Example 1 (audio):** FaceFormer-style / VOCA-style audio-to-face pipelines historically used LSTMs to map audio feature sequences to blendshape/vertex trajectories over time, before Transformer-based cross-attention became the dominant approach — LSTM's gating let them retain relevant phonetic context across a full utterance without the vanishing-gradient collapse a vanilla RNN would hit.

**Example 2 (motion):** Motion synthesis models predicting the next pose conditioned on a long motion history (e.g. locomotion continuing a style over many steps) use LSTM/GRU cells so that a stylistic or gait-phase signal from many frames back can still gate the current prediction, instead of decaying away.

**Likely follow-up:** *"Why has attention/Transformers largely replaced LSTM for this?"* → Three reasons: (1) LSTM is inherently sequential — timestep `t` needs `h_{t-1}`, so no parallelization across time during training, which is slow on modern hardware; (2) even with gating, very long-range dependencies still degrade gradually — attention gives an O(1) path (not O(n)) between any two positions regardless of distance; (3) self-/cross-attention directly models *which* audio frames matter most for *which* output frame (explicit alignment), which is exactly the coarticulation/lip-sync alignment problem — this is why FaceFormer moved to Transformer cross-attention. Trade-off: attention is O(n²) in sequence length vs RNN's O(n), so for very long streams or tight real-time/edge budgets, RNN/GRU variants (or windowed/causal attention) can still win on latency.

---

## Why this matters for the XR Audio Face Tracking role

Both audio and facial motion are **time series** — the whole role sits on modeling temporal dependencies correctly:

- Audio → viseme/blendshape mapping is fundamentally sequence-to-sequence with alignment (coarticulation = neighboring phonemes affect the current mouth shape) — exactly the problem RNN/LSTM (and now attention) are built for.
- Facial motion output must be temporally *smooth*, not per-frame independent — recurrent state (or a causal/windowed attention mechanism) is what prevents jitter and lets the model condition each frame on recent history.
- Real-time/edge constraint (Day 4 topic #4) ties back here: LSTM/GRU's O(n) sequential compute and small state can be a deliberate architecture choice over full self-attention when the latency budget doesn't afford O(n²) attention — a concrete point to make if asked "would you use a Transformer or an RNN here" instead of just calling Transformers strictly better.

---

## 90-second recap (say this out loud, unprompted)

> "RNNs process sequences step by step, keeping a hidden state that summarizes history — same weights reused every timestep. The problem is vanishing gradients: backprop-through-time repeatedly multiplies through the recurrent weights, so long-range dependencies decay and early timesteps stop getting a learning signal. LSTM fixes this with a separate cell state updated additively via forget/input/output gates, giving gradients a near-linear path back through time; GRU is a lighter 2-gate simplification. This matters directly for audio-driven facial animation — audio and facial motion are both time series with real temporal dependencies: coarticulation means neighboring phonemes affect the current mouth shape, and output motion needs to stay temporally smooth. LSTM/GRU-based models (like early VOCA-style pipelines) handled this before Transformer cross-attention took over for its O(1) path length between any two positions and its explicit alignment modeling — though RNN/GRU's cheaper O(n) sequential compute can still be the right call under a tight real-time/edge latency budget."

---

## Sources / cross-links

- [[Google - ML SWE Interview - Study Plan]] — Day 3 (origin)/Day 4 (this makeup slot)
- [[Google - Technical Interview]] — role/JD context
