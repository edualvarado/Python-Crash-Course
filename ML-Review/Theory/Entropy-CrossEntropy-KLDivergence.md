Related: [[Google - ML SWE Interview - Study Plan]]

# Information Entropy, Cross-Entropy & KL Divergence

Underlies exe2 (losses vs. metrics). Goal: derive BCE (Binary Cross-Entropy) from first principles instead of reciting the formula — entropy → cross-entropy → one-hot collapse → the two-class special case.

---

## 1. Information Entropy H(p)

**Definition:** `H(p) = -Σ p(c)·log p(c)`, summed over classes `c`. The expected number of bits (log base 2) needed to encode outcomes drawn from distribution `p` — a measure of how much inherent uncertainty/"surprise" the distribution has. Peaked distribution → low entropy (outcome is predictable). Uniform distribution → maximum entropy for that number of outcomes (every outcome equally surprising).

**Example 1:** Fair coin, p=[0.5, 0.5] → H = 1 bit. Biased coin, p=[0.99, 0.01] → H ≈ 0.08 bits — you're almost never surprised, so little information is conveyed per flip.

**Example 2 (audio):** A VAD (Voice Activity Detection) label distribution in a mostly-quiet recording, e.g. 95% silence / 5% speech, has low entropy — there's little inherent uncertainty in the *labels themselves*, independent of any model. Entropy is a property of the data, not a property of a predictor.

**Likely follow-up:** *"Why the log?"* → Information from independent events should add (two independent coin flips = 2 bits, not 1), and probabilities of independent events multiply — log converts multiplication into addition. Base 2 gives units of bits; base e gives "nats" (what most ML frameworks use internally).

---

## 2. Cross-Entropy H(q, p)

**Definition:** `H(q, p) = -Σ q(c)·log p(c)`. `q` = true distribution, `p` = model's predicted distribution. The expected number of bits to encode data actually distributed as `q`, using a code built assuming distribution `p`. Key inequality: `H(q,p) ≥ H(q)`, equality only when `p = q` exactly — using the wrong distribution always costs extra bits on top of the unavoidable floor set by `q`'s own entropy.

**One-hot collapse:** for one labeled example, `q` is one-hot (1 at the true class `y`, 0 elsewhere), so every term with `q(c)=0` vanishes:
```
H(q,p) = -log p(y)
```
Cross-entropy on one example is just the negative log-probability the model assigned to the correct class. That's the loss you actually compute in code.

**Numeric example (C=3, cat/dog/bird):** true class = cat, q=[1,0,0]. Model predicts p=[0.7, 0.2, 0.1].
```
H = -log(0.7) ≈ 0.357
```
Only the true class's predicted probability matters — the other two terms are multiplied by 0.

**BCE reduction (C=2):** plug the one-hot formula into both possible binary labels and combine with `y` as a switch:
```
y=1: H = -log(p)
y=0: H = -log(1-p)
H = -[y·log(p) + (1-y)·log(1-p)]   <- this is BCE
```
Numerically: y=1, p=0.8 → BCE ≈ 0.223 (confident + correct, low loss). y=1, p=0.2 → BCE ≈ 1.609 (confident + wrong, high loss).

**Likely follow-up:** *"Is BCE a different loss from cross-entropy?"* → No — BCE is `nn.CrossEntropyLoss`'s 2-class special case, same derivation, `nn.CrossEntropyLoss` is the >2-class generalization.

---

## 3. KL (Kullback-Leibler) Divergence

**Definition:** `KL(q‖p) = H(q,p) - H(q) = Σ q(c)·log(q(c)/p(c))`. The *extra* bits wasted specifically because you used `p` instead of the true `q` — cross-entropy minus the unavoidable entropy floor. Non-negative (Gibbs' inequality), equals 0 iff `p = q` everywhere. **Not symmetric** — `KL(q‖p) ≠ KL(p‖q)` in general, so it's a divergence, not a true distance metric.

**The relationship to say out loud:**
```
Cross-Entropy = Entropy + KL divergence
H(q,p) = H(q) + KL(q‖p)
```
`H(q)` is a fixed property of the true data/labels — it doesn't depend on the model's parameters at all. So minimizing cross-entropy loss during training and minimizing `KL(q‖p)` between the true and predicted distributions are the *same* optimization problem. That's the actual justification for cross-entropy as a training objective — not convenience, but that it's mathematically equivalent to pulling the model's predicted distribution toward the true one.

**Example 1 (forward to Day 7/8 — GNM/diffusion prep):** a VAE (Variational Autoencoder) loss — the ELBO (Evidence Lower BOund) — includes an explicit `KL(q(z|x) ‖ p(z))` term regularizing the learned latent distribution toward a prior (usually standard normal). This is what keeps the latent space smooth and sample-able for generation, directly relevant to generative face/shape models like GNM.

**Example 2 (back to Day 4 — edge deployment):** knowledge distillation trains a small student to match a softened teacher output distribution, using cross-entropy/KL between teacher and student logits — this transfers "dark knowledge" (relative confidence across wrong classes) beyond what a one-hot hard label carries.

**Likely follow-up:** *"Why does asymmetry matter?"* → `KL(q‖p)` heavily penalizes `p` assigning near-zero probability to something `q` says is likely (that term blows up toward ∞). Which direction you pick changes what kind of mismatch gets punished hardest — mode-covering vs. mode-seeking behavior, relevant if VAE (mode-covering) vs. GAN (Generative Adversarial Network, mode-seeking-ish in practice) training dynamics come up.

### KL vs. cross-entropy — what does each actually measure, and is minimizing KL ever better?

- **What each measures:** cross-entropy `H(q,p)` mixes two things — the entropy `q` has on its own (`H(q)`, unavoidable, doesn't depend on the model) plus the extra cost from `p` being wrong (`KL(q‖p)`). KL isolates only the second part: pure "how wrong is my model," with the target's own inherent randomness subtracted out. `KL(q‖p) = 0` exactly when `p=q`, regardless of how random `q` itself is; `H(q,p)` never hits exactly 0 unless `q` is also deterministic.

- **Standard classification (one-hot hard labels): no difference at all — not even a constant one.** A one-hot `q` has zero entropy (`H(q)=0`: you know the outcome with certainty; `0·log0=0` by convention, `1·log1=0`). So `H(q,p) = H(q) + KL(q‖p) = 0 + KL(q‖p)` — cross-entropy and KL are *exactly* equal, term for term. Frameworks implement CE rather than KL purely because it's simpler to compute — skipping a term that's already zero.

- **Soft targets (label smoothing, knowledge-distillation with a softened teacher):** now `H(q) > 0`, so CE and KL differ by that constant and the *reported loss values* diverge. But `H(q)` still doesn't depend on the model's parameters (the teacher/target is fixed, not backpropagated into), so `∂H(q,p)/∂θ = ∂KL(q‖p)/∂θ` — **gradients and parameter updates are still identical.** Minimizing CE vs. minimizing KL trains the same model; only the absolute number you'd log differs.

- **So is there ever a real (not cosmetic) advantage to KL over CE?** Yes — exactly when the "true" distribution `q` is *not fixed data* but is itself being learned, so `H(q)` is no longer constant w.r.t. the parameters being optimized. Clearest case: a VAE's ELBO optimizes `KL(q(z|x)‖p(z))`, where `q(z|x)` is the encoder's own output — `H(q(z|x))` moves as the encoder trains, so silently dropping it (treating the term as CE-shaped instead) would change what's actually being minimized. That's why VAE losses are written as explicit KL terms, never as "cross-entropy" — the equivalence that makes CE and KL interchangeable in classification breaks the moment the target distribution has learnable parameters of its own.

**One-liner:** *"For fixed hard labels, cross-entropy and KL divergence aren't just similar, they're identical — the entropy term separating them is exactly zero. They only diverge, and the choice starts to actually matter, once the 'true' distribution is itself something the model is learning to shape, like a VAE's latent posterior."*

---

---

## 4. Why KL matters for VAEs (Day 7/8 pay-off)

**The problem it solves:** a plain autoencoder only minimizes reconstruction error — nothing stops the encoder from scattering different inputs' latent codes into disconnected clumps with gaps between them. Reconstruction works, but generation doesn't: pick a random point in latent space and decode it, you likely land in a gap the decoder never saw during training → garbage output. Good autoencoder, useless generator.

**What the KL term does:** VAE (Variational Autoencoder) training adds `KL(q(z|x) ‖ p(z))` to the loss, where `q(z|x)` is the encoder's output distribution for a given input (usually a Gaussian with predicted mean/variance), and `p(z)` is a fixed simple prior, almost always `N(0, I)`. This pulls *every* input's `q(z|x)` toward overlapping that same centered blob instead of letting the encoder scatter them freely:

1. **Continuity/coverage** — latent space becomes one dense, overlapping region instead of isolated islands; nearby `z` decode to similar, meaningful outputs, no dead zones.
2. **Sampling becomes possible** — because every training input was pulled toward `N(0,I)`, at generation time you sample `z ~ N(0,I)` directly (no encoder needed) and decode — the decoder has actually seen z's from that region during training.

**Bonus effect:** it also stops the encoder from collapsing to near-zero variance (cheating by memorizing a point per input, like a plain autoencoder). For a Gaussian, `KL(N(μ,σ²)‖N(0,1)) = 0.5·(σ² + μ² − 1 − log σ²)` — as `σ²→0`, `−log σ² → ∞`, so collapsing variance is penalized heavily. The KL term forces genuine stochasticity — the "variational" part of the name.

**Full objective (ELBO — Evidence Lower BOund):**
```
ELBO = E_z~q(z|x)[log p(x|z)]  -  KL(q(z|x) ‖ p(z))
        \_____reconstruction_____/   \___regularization___/
```
Balance matters: no KL term → chaotic, ungeneratable latent space (plain autoencoder). Too much KL weight → "posterior collapse," encoder ignores the input and always outputs ≈ the prior, reconstructions degrade. (β-VAE literally exposes this weight as a tunable hyperparameter.)

**Ties back to section 3's "when does KL vs. CE actually matter":** here `q(z|x)` is a distribution output by a network being actively trained — not a fixed one-hot label — so its entropy is *not* constant and can't be dropped. This is exactly the case where the CE/KL equivalence breaks, and why VAE losses are written as an explicit KL term.

**Grounded example (GNM-adjacent):** a generative face/shape model's latent `z` might encode identity + expression. Without KL, you couldn't reliably sample "a new plausible face" — most of latent space was never visited during training. With KL regularizing toward `N(0,I)`, you can sample directly, and interpolating between two faces' `z` gives smooth morphs instead of garbage in between.

**Likely follow-up:** *"What breaks if you just skip the KL term entirely?"* → You get a deterministic-ish autoencoder with good reconstructions but an unstructured, ungeneratable latent space — exactly the failure mode KL exists to prevent. Skipping it is a legitimate ablation to name if asked "what does the KL term buy you," since it isolates the regularization's effect.

---

## Why this matters for the XR Audio Face Tracking role

- Every classification head in this stack — VAD, blink detection, expression/viseme classification — trains on cross-entropy or BCE. Deriving it from entropy rather than reciting the formula is what separates "knows the API" from "understands why."
- KL divergence resurfaces directly in Day 7/8 material: VAE-style latent regularization for generative face/shape models (GNM), and in Day 4's knowledge-distillation edge-deployment content.

---

## 90-second recap (say this out loud, unprompted)

> "Entropy measures the inherent uncertainty in a distribution — bits needed to encode its outcomes, low for a peaked distribution, high for a uniform one. Cross-entropy is the cost of encoding data from the true distribution using a code built for the model's predicted distribution — always at least the true entropy, with equality only when the predicted distribution exactly matches. For one labeled example the true distribution is one-hot, so cross-entropy collapses to the negative log-probability the model assigned to the correct class — that's the loss you actually compute, and BCE is just its two-class special case. KL divergence is the gap between the two: cross-entropy minus entropy, the extra bits wasted specifically because the prediction is wrong. Since the true distribution's entropy doesn't depend on the model at all, minimizing cross-entropy during training is mathematically identical to minimizing KL divergence between predicted and true distributions — that's the real justification for cross-entropy as a loss, not just convenience."

---

## Sources / cross-links

- [[Google - ML SWE Interview - Study Plan]] — exe2 (losses vs. metrics)
- Day 7/8 — GNM paper / diffusion-from-scratch (KL term in VAE-style objectives)
- Day 4 — real-time/edge deployment (knowledge distillation)
