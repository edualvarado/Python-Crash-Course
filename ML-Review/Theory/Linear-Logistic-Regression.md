Related: [[Google - ML SWE Interview - Study Plan]], [[Losses-vs-Metrics]], [[Entropy-CrossEntropy-KLDivergence]]

# Linear & Logistic Regression — Closed-Form, Gradient Derivations, and Why They're Related

Day 2 IF-TIME item, still outstanding: "refresh the math, be ready to derive on a whiteboard." Underlies exe1 (linear regression via autograd). Goal: derive both models' gradients by hand, and show they're the same equation in disguise.

---

## 1. Linear regression — model and loss

**Model:** `y_hat = w·x + b` (or `Xw` with `x` prepended by a 1s column to fold `b` in, for multi-feature `X`).

**Loss — MSE:** `L = (1/n) Σ (y_hat_i - y_i)²`. Squared, not absolute, for two reasons: differentiable everywhere (`|·|` has a kink at 0), and convex in `w,b` — a single global minimum, no local-minima search needed.

**Assumptions (for the *statistical* guarantees, not for point prediction to work at all):** linear relationship between `X` and `y`; errors independent; homoscedastic (constant error variance across the input range); no severe multicollinearity among features. Point prediction from a fitted `w,b` doesn't require any of this — but confidence intervals / p-values on the coefficients do.

---

## 2. Closed-form solution — normal equations

Minimize `L(w) = ||Xw - y||²` directly by calculus instead of iterating:

```
L(w) = (Xw - y)ᵀ(Xw - y)
dL/dw = 2Xᵀ(Xw - y) = 0
Xᵀ X w = Xᵀ y
w = (Xᵀ X)⁻¹ Xᵀ y
```

**Why gradient descent is used in practice anyway:** `(XᵀX)⁻¹` is `O(d³)` in the number of features `d` — fine for a handful of features, infeasible for a deep net's millions of parameters. GD's per-step cost is `O(d)`, and it generalizes to loss surfaces that have no closed form at all (anything with a nonlinearity in it). Normal equations are the "why does GD even converge to the right answer" sanity check, not the production method.

---

## 3. Gradient descent derivation — linear regression (whiteboard-ready)

Per-sample loss (drop the batch mean for the derivation, it just distributes):

```
L = (y_hat - y)²,  y_hat = w·x + b

dL/dy_hat = 2(y_hat - y)
dy_hat/dw = x            ->  dL/dw = 2(y_hat - y)·x
dy_hat/db = 1             ->  dL/db = 2(y_hat - y)
```

Update: `w -= lr · dL/dw`, `b -= lr · dL/db`. This is exactly `exe1_gradient_descent.py`'s TODO 2–4, just with autograd computing the chain rule instead of doing it by hand. The **residual** `(y_hat - y)` scaled by the input `x` is the whole update — worth internalizing, because section 5 shows logistic regression collapses to the identical form.

---

## 4. Logistic regression — from linear to probability

Same linear core `z = w·x + b`, but passed through sigmoid to constrain the output to `(0,1)`: `p = σ(z)`. Full derivation of *why* sigmoid specifically (it's the algebraic inverse of log-odds) is in [[Losses-vs-Metrics]] section 2 — not repeated here.

**"Linear" refers to the decision boundary, not the probability surface.** `p` as a function of `x` is an S-curve (nonlinear), but the boundary `p = 0.5` is exactly `z = 0`, i.e. `w·x + b = 0` — a hyperplane, linear in `x`. That's the sense in which logistic regression is a *linear* classifier: everything on one side of a straight line/plane gets one class.

**Why not train logistic regression with MSE:** substituting `p = σ(z)` into MSE gives a loss that's **non-convex** in `z` — multiple local minima, GD can get stuck. Worse, MSE's gradient through sigmoid is `2(p-y)·σ'(z)`, and `σ'(z) = p(1-p)` → 0 as `p` approaches 0 or 1. So a *confidently wrong* prediction (`p≈0` when `y=1`) produces a near-zero gradient — the model can't learn from its worst mistakes. BCE (section 5) is specifically the loss that fixes both problems.

---

## 5. Gradient derivation — logistic regression (the elegant result)

BCE with `p = σ(z)`:

```
L = -[y·log(p) + (1-y)·log(1-p)]
```

Chain rule through the sigmoid, using `dp/dz = p(1-p)` (standard sigmoid-derivative identity):

```
dL/dp = -y/p + (1-y)/(1-p)

dL/dz = dL/dp · dp/dz
      = [-y/p + (1-y)/(1-p)] · p(1-p)
      = -y(1-p) + (1-y)p
      = -y + yp + p - yp
      = p - y
```

**`dL/dz = p - y`** — the gradient w.r.t. the *raw logit* is just the residual, exactly the same shape as linear regression's `(y_hat - y)`. Then `dL/dw = (p-y)·x` by the same chain rule as section 3. This is not a coincidence: BCE was constructed (as the negative log-likelihood of a Bernoulli, section 2 of [[Entropy-CrossEntropy-KLDivergence]]) precisely so that the messy `σ'(z)` term cancels against the `1/p, 1/(1-p)` terms from the log — the vanishing-gradient problem from section 4 is gone by design. **This cancellation is the single most likely "derive it" whiteboard ask** — know it cold, both directions (why MSE fails, why BCE's gradient is clean).

---

## 6. Multi-class extension — softmax regression

Generalizes directly: `K` logits `z_1..z_K` → softmax `p_i = e^{z_i} / Σ_j e^{z_j}` → categorical cross-entropy `L = -Σ_i y_i log(p_i)` (`y` one-hot). Same cancellation happens per-class: `dL/dz_i = p_i - y_i`. Binary logistic regression is the `K=2` special case (and is exactly what section 5 derives, with `p_1=p, p_0=1-p`).

---

## 7. Regularization — closed form still exists for Ridge

Adding L2 (`Ridge`) penalty `λ||w||²` to linear regression keeps a closed form and fixes a real numerical problem:

```
w = (XᵀX + λI)⁻¹ Xᵀy
```

`λI` makes the matrix invertible even when `XᵀX` is singular/ill-conditioned (collinear features, or more features than samples) — Ridge is as much a numerical-stability fix as a variance-reduction one. L1 (`Lasso`) has no closed form (the penalty isn't differentiable at 0) but drives coefficients exactly to zero — implicit feature selection. Both are the direct fix for the "high variance" failure mode in [[ML_fundamentals]]'s bias-variance section: shrinking `w` trades a little bias for a lot less variance.

---

## Why this matters for the XR Audio Face Tracking role

- Logistic regression is the baseline you'd reach for before an RNN/CNN on any per-frame binary signal in this stack (VAD, blink onset) — cheap, interpretable, and the floor a fancier model has to beat to justify its cost.
- Linear regression is the same story for a continuous per-frame target (a single blendshape coefficient) — establishes whether the nonlinearity a bigger model adds is actually earning its complexity.
- The `p - y` cancellation is a fluency signal: being able to produce it unprompted shows the loss/activation pairing (BCE+sigmoid, CE+softmax) is understood as a designed unit, not memorized separately.

---

## 90-second recap (say this out loud, unprompted)

> "Linear regression fits `y = wx+b` by minimizing MSE — convex, so there's a closed form, the normal equations `w=(XᵀX)⁻¹Xᵀy`, but in practice we use gradient descent because the matrix inversion doesn't scale to high-dimensional features. The per-sample gradient is just the residual times the input: `2(y_hat-y)·x`. Logistic regression reuses the same linear core `z=wx+b` but squashes it through sigmoid to get a probability — the decision boundary `z=0` is still linear, only the probability surface is curved. You can't train it with MSE: substituting sigmoid in makes the loss non-convex, and the gradient vanishes exactly when the model is confidently wrong, because sigmoid saturates. Binary cross-entropy fixes both, and if you differentiate BCE-of-sigmoid with respect to the raw logit, everything cancels down to the exact same form as linear regression: `p - y`. That cancellation isn't an accident — BCE is constructed as the Bernoulli negative log-likelihood specifically so it happens. Softmax regression is the same result generalized to K classes, `p_i - y_i` per class."

---

## Sources / cross-links

- [[Google - ML SWE Interview - Study Plan]] — Day 2 IF-TIME, cheat sheet one-liners
- [[Losses-vs-Metrics]] — full sigmoid derivation (why it's `logit⁻¹`), BCE usage pattern, confusion-matrix metrics
- [[Entropy-CrossEntropy-KLDivergence]] — BCE as Bernoulli negative log-likelihood, why it's a proper scoring rule
- [[ML_fundamentals]] — bias-variance tradeoff that Ridge/Lasso regularization trades against
