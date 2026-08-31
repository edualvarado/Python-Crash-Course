Related: [[Google - ML SWE Interview - Study Plan]], [[Audio-Driven-Facial-Animation]], [[Multimodal-Fusion]], [[RNN-LSTM]]

# GNM Head — A Generative aNthropometric Model of the human head

**Ploumpis, Bednarik, Zoss, ... Beeler, Zafeiriou — Google, arXiv 2607.23687, 26 Jul 2026.**
Code + model public: `github.com/google/GNM`, licensed for academic **and** commercial use.
Name is a deliberate homophone of "genome".

Day 7/8 item. This is the target department's own paper — Thabo Beeler, Sergio Orts-Escolano, Erroll Wood, Timo Bolkart, Prashanth Chandran are all Google XR/face people. Fluency here reads as genuine engagement with their work.

---

## 0. TL;DR — the 60-second version

Existing 3D Morphable Models (3DMMs) of the head — FLAME, Basel Face Model (BFM), Large Scale Face Model (LSFM) — treat the head as a **hollow shell**: outer skin only, no teeth, no tongue, crude spherical eyeballs. GNM is Google's replacement that puts **face + neck + eyeballs + teeth + tongue into one unified statistical space**, built on ~5,000 scanned subjects and ~150,000 expression samples from a 22-camera rig.

Three things make it new:
1. **Teeth shape lives in the identity basis** — your dental arch is part of who you are, not a generic prop.
2. **The expression basis is split by region** (left eye / right eye / lower face / tongue / pupil) instead of one global basis, so opening the mouth can't accidentally trigger an eye wink.
3. **Eyeballs are anatomically parameterised** — a two-sphere sclera/cornea model with per-identity corneal curvature, plus a dedicated pupil-dilation control.

Headline number: **0.748 mm** mean scan-to-mesh error on 15,000 held-out scans vs FLAME's **0.971 mm** — ~23% better, and better across every demographic subgroup and every facial region.

It is still a **linear PCA model with linear blend skinning**. That is the deliberate trade: it keeps the interpretability and graphics-pipeline compatibility that neural implicit head models throw away.

---

## 1. The problem it attacks

**Definition:** A 3DMM (3D Morphable Model) compresses the geometry of a human head into a low-dimensional, controllable latent space — originally Blanz & Vetter (1999), Principal Component Analysis (PCA) over registered exemplar meshes. Modern uses go well beyond graphics: they are geometric conditioning signals for diffusion models, generators of privacy-safe synthetic training data, and the structural scaffold that anchors neural rendering (Neural Radiance Fields / NeRF, 3D Gaussian Splatting / 3DGS) so it doesn't deform non-physically.

**The gap:** every widely-used public head model omits internal anatomy. No teeth, no tongue, no real eyeball structure. Consequences the paper names explicitly:
- Generative models have no geometric constraint on the mouth interior → visibly bad open-mouth frames.
- You lose semantic control over **lip and tongue coarticulation** — exactly the non-verbal cue that matters for speech.
- No corneal geometry → you cannot simulate the LED glints that gaze-tracking algorithms key off.

**Why it's an XR paper, not just a graphics paper:** the motivation given is the "uncanny valley" in Augmented Reality (AR) telepresence. An avatar whose mouth interior is a black hole reads as fake the moment it speaks.

---

## 2. Data — the part that's actually the moat

| | |
|---|---|
| Capture rig | 22 × ZCam E2 S6G cameras at 6144×4096, 14 controllable lights, uniform diffused illumination |
| Coverage | ~150° horizontal, 60° vertical in front of the subject |
| Subjects | **~5,000 individuals**, demographically diverse (gender / age / ethnicity all reported) |
| Samples | **~150,000** expression samples |
| Throughput | custom parallel pipeline, ~10,000 samples/day |
| Expression protocol | 10 categories: flexing, 10 standard **visemes**, lip motion (roll/pucker/funnel), global emotions, tongue motion, jaw motion, wink/squint, gaze, eyebrows, cheek suck/blow |

Plus artist-made synthetic assets where real data is impossible to capture:
- **5,000** procedurally generated dental shapes (upper + lower + gums) from a rigged template
- a tongue rig fitted to expressive scans and to sparse 3D keypoints (~2.5K samples)
- **200** artist-sculpted head meshes used to reconstruct the cranium (hair occludes it, so it can never be measured directly)

**Likely follow-up:** *"Why synthetic teeth rather than scanned?"* → You cannot see most of the dentition in a multi-view face capture; the lips and cheeks occlude it. An artist-rigged parametric tooth generator gives you full, clean coverage of the shape space, and the model only has to learn how the visible portion correlates with the rest.

---

## 3. Model formulation — know this cold

GNM is a function producing a mesh of **N_V = 17,821 vertices** (skin + teeth + tongue + eyeballs):

```
M(Θ; Ψ) : R^|Θ| → R^(N_V × 3)
```

**Parameters Θ = (β, φ, θ, τ):**
- `β` — identity coefficients
- `φ` — expression coefficients
- `θ` — angle-axis rotations for **K = 4 joints**
- `τ` — global translation

**Fixed model data Ψ = (T, J, I, E, Q, W, p):** template mesh, template joint locations, identity basis, expression basis, joint-location identity basis, Linear Blend Skinning (LBS) weights, kinematic chain.

**The core equation (Eq. 1) — this is the whole 3DMM idea in one line:**

```
T(β, φ) = T̄ + Σ βᵢ Iᵢ + Σ φᵢ Eᵢ
```

Template mesh, plus a weighted sum of identity displacement fields, plus a weighted sum of expression displacement fields. Then LBS poses it: `M = L(V_B, X; W)`.

**The 4 joints are: left eye, right eye, neck, head.** Note what is *missing* — **there is no jaw joint**. FLAME's standard variant articulates the jaw with a skinning joint; GNM puts jaw motion in the **linear expression basis** instead. If asked why: a linear basis learned from real jaw motion captures the soft-tissue deformation that accompanies the rotation, which a rigid joint rotation plus fixed blendweights cannot.

Joint locations are **identity-dependent**: `J(β) = J̄ + Σ βᵢ Qᵢ`, so the skeleton scales with each subject's head shape rather than sitting at fixed coordinates.

---

## 4. Composite regional bases — the key architectural idea

Instead of one global PCA for identity and one for expression, both bases are **concatenations of per-region PCAs**:

```
I = [ I_head | I_eye | I_teeth ]
E = [ E_left-eye | E_right-eye | E_lower-face | E_tongue | E_pupil ]
```

**Exact component counts (Table 1) — worth memorising, it's a concrete detail:**

| Identity | | | | Expression | | | | |
|---|---|---|---|---|---|---|---|---|
| head | eye | teeth | **total** | left eye | right eye | lower face | tongue | pupil |
| 170 | 3 | 80 | **253** | 100 | 100 | 150 | 31 | 1 |

Expression total: **382**.

**Why regional:**
1. **No semantic leakage.** Global PCA couples everything — nudge one coefficient and the skull deforms too. Regional masks mean a jaw movement cannot trigger an eye wink.
2. **Perfect left/right mirroring.** Only `E_left-eye` is computed; the right eye is its mirror. A global basis learns the asymmetries present in the finite dataset and bakes them in.
3. **Specialised datasets per region.** Teeth come from 5,000 synthetic dental shapes; tongue from a fitted rig. You could not merge those into one PCA.

Regions use continuous vertex masks `S_r ∈ [0,1]` with small overlaps, linearly blended — so components are orthogonal *within* a region but not across regions.

**Two PCA details an interviewer could probe:**
- **Eigenvectors are scaled by their eigenvalues**, so basis vectors are orthogonal but *not* unit length. Effect: coefficient ranges are unified across components — β₁ and β₂₀₀ have comparable natural scale, which makes L2 regularisation during fitting behave sensibly.
- **Expression PCA is uncentered.** If you centered it and absorbed the mean into the template, the "neutral" face would end up with slightly closed eyes and a slightly open mouth (the dataset mean of all expressions). Uncentered means **β = φ = 0 gives a genuinely neutral face**. Exception: the tongue mean *is* absorbed, as the first component of `E_tongue`, so φ = 0 gives a retracted tongue tucked inside the mouth rather than one protruding through the lips.

---

## 5. The three sub-models

### Teeth — in the *identity* basis
5,000 artist-generated dental shapes → PCA → 80 components, ~99% variance. Critically this sits in **I** (identity), not **E** (expression): dental arch shape is who you are. Lower-teeth *motion* rides along with the jaw inside **E**.

### Tongue — in the *expression* basis
31 components from ~2.5K samples. The tongue is notoriously hard to register from raw scans, so they fit an artist tongue rig to (a) real expressive scans performing tongue motions, constrained not to penetrate the lips, and (b) sparse 3D keypoint data. This is the component that makes **phonetic articulation** representable at all.

### Eyeballs — a two-sphere anatomical model
- Large **scleral** sphere, radius held **constant at ≈14.6 mm** (so it stays compatible with the template's eyelids)
- Smaller **corneal** sphere forming the anterior segment; the line joining the two centres is the **optical axis**
- Sampled from physiological Gaussians: limbus radius `r_l` (μ=6.0, σ=0.44 mm), cornea radius `r_c` (μ=8.5, σ=0.73 mm)
- Method: sample 10,000 **2D cross-section polylines**, PCA those, then interpolate the 2D basis onto the 3D mesh in polar coordinates — legitimate because an eyeball is a **surface of revolution**. Result: only **3 identity components** for the whole eye.
- **`E_pupil` is 1 hand-crafted component**, coefficient in [−3, 3]: −3 = pupil contracts to a point, 0 = half the iris radius, +3 = full iris. Covers ~1–4 mm radius.

**Why the cornea matters (say this if gaze comes up):** corneal curvature dictates where **LED glints** form on the eye, and glints are the primary feature in gaze-target prediction. Baking corneal variation into the identity space means synthetic training data for a gaze tracker has biologically plausible glint geometry per identity.

---

## 6. Registration pipeline (how the data became a model)

**Iterative coregistration** — alternate between (a) registering scans with the current model as a prior and (b) rebuilding the model from the new registrations. Same loop FLAME used. Bootstrapped with artist-sculpted internal parts injected into the template.

Fit is done by **differentiable inverse rendering** in Mitsuba, using edge sampling for visibility gradients. Losses:
- L1 on RGB, surface normals, and per-pixel semantic segmentation (renderer emits normal + semantic Arbitrary Output Variables)
- Structural Similarity Index Measure (SSIM) on RGB for camera-space alignment
- **dense landmarks** carry the occluded regions (teeth, tongue) where image evidence is absent
- regularisers: gradient preconditioner, L2 on per-vertex offsets, graph-Laplacian L2, edge-deviation penalty, and a differentiable **self-intersection** loss for ears/tongue/lips

**Stabilisation** — a detail worth knowing because it's a subtle problem: to build a clean expression basis you must remove rigid head motion, i.e. align the *unknown underlying skull* across expressions of the same person. The paper calls this "notoriously difficult". Two stages: automatic confidence-map stabilisation, then a **semi-automatic PCA pass where a human operator visually inspects per-region components and discards ones representing spurious rigid motion**. Human in the loop, in a Google paper, in 2026 — that's an honest signal about how hard it is.

**Cranium** — never measurable (hair). Solved by cross-domain latent regression: project registered faces onto an auxiliary model built from 200 artist-sculpted heads to get a plausible cranium, then non-rigidly map back, preserving it.

---

## 7. Fitting GNM to images (the downstream use)

Detect ~**600 dense 2D landmarks**, then optimise model parameters + camera intrinsics to minimise:

```
E_total = w_lan·E_lan + w_prior·E_prior + w_anat·E_anat + w_temp·E_temp
```

- `E_lan` — dense landmark re-projection error (the data term)
- `E_prior` — L2 on β and φ (keeps you on the plausible manifold)
- `E_anat` — **anatomical penalty preventing self-intersection** between skin and eyeballs / mouth cavity. This term only exists because the model *has* internal anatomy.
- `E_temp` — temporal smoothness across video frames; set to 0 for single images

Multi-view video: solve **one shared identity β across all frames**, per-frame expression / rotation / translation. That factorisation is standard and worth stating unprompted.

---

## 8. Semantic Sampler — dual CVAE

**The problem:** raw PCA coefficients are statistically meaningful but not *semantically* meaningful. You cannot ask a PCA basis for "a 40-year-old East Asian man, 70% happy and 30% surprised".

**The fix:** two separate Conditional Variational Autoencoders (CVAEs) — one for identity, one for expression — mapping semantic conditions to GNM coefficients:

```
β = f_id(z_id, c_id)        φ = f_exp(z_exp, c_exp)
```

- `c_id` = one-hot gender (2) ‖ one-hot ethnicity (4) = **6-D**
- `c_exp` = one-hot over **20 action-driven expression classes** (happiness, disgust, surprise, snarl, cheek suck, pucker, cheek blow, funneler, lips-roll-in, tongue centering, platysma, single-eye wink, ...)
- `z` = 64-D latent ~ N(0, I) capturing within-class variation
- Multi-Layer Perceptrons (MLPs) with ReLU; identity encoder 256/128/64, expression encoder 512/128/256/64; decoders mirror. Trained on 12K samples.

**Three training tricks worth naming — this is the most "ML interview" part of the paper:**

1. **L1 reconstruction loss, not L2.** L2 smooths high-frequency shape detail; L1 preserves sharp anatomical boundaries and extreme expressions.
2. **Cyclical KL annealing** — `w_KL` warmed 0 → 0.05 over the first 4,000 steps, to prevent **posterior collapse**: the classic CVAE failure where the decoder learns to ignore `z` entirely and read only the conditioning vector `c`, so all samples of a class come out identical. (Direct continuation of the KL material in `Entropy-CrossEntropy-KLDivergence.md` and your `exe7_vae_from_scratch.py`.)
3. **Mixup on the conditional inputs** — feed the decoder convex combinations `λc_A + (1−λ)c_B` with `λ ~ Beta(0.2, 0.2)` during training, so interpolating *between* discrete classes at inference (40% ethnicity A / 60% B, or 70% happy / 30% surprise) yields anatomically viable geometry instead of garbage between the modes.

---

## 9. Results — the numbers to quote

**Generalisation** (scan-to-mesh distance, 15,000 held-out scans, two-stage fit with Iterative Closest Point refinement, no regularisation so both models are maximally expressive):

| Method | Mean (mm) | Median (mm) | Std (mm) |
|---|---|---|---|
| FLAME (w/o jaw joint) | 0.968 | 0.779 | 0.729 |
| FLAME | 0.971 | 0.780 | 0.734 |
| **GNM** | **0.748** | **0.623** | **0.529** |

**Single-view reconstruction** (2,000 synthetic images, 7,000 ground-truth dense landmarks, identical pipeline for both):

| Method | Mean (mm) | Median (mm) |
|---|---|---|
| FLAME | 2.172 | 2.086 |
| **GNM** | **1.683** | **1.589** |

Also reported: lower error in **every** facial region (lower lip, upper lip, cheek, nose, forehead), and in **every** demographic subgroup (gender, 3 age bands, 8 ethnicity groups) — that subgroup breakdown is itself a deliberate fairness-reporting choice worth noticing.

**The two standard 3DMM metrics — know both, they're a matched pair:**
- **Generalisation** — can the model represent *novel* shapes? Measured by fitting to held-out scans. Too little capacity → high error.
- **Specificity** — do *random samples* from the model stay on the manifold of plausible human heads? Measured by sampling β ~ N(0, Σ) and finding the distance to the nearest real scan. Too much capacity → you can represent anything, including monsters.

They trade off. A model that wins on both, as GNM claims, is the actual result. GNM also reaches a given error with **fewer components** than FLAME — i.e. a more compact basis.

---

## 10. Where GNM sits vs what you already know

| Model | Representation | Scope | Internal anatomy | Control |
|---|---|---|---|---|
| **3DMM / blendshapes** (classic) | linear PCA / artist blendshapes | face | none | interpretable, low fidelity |
| **BFM / LSFM** | linear PCA | face / full cranium | none | global, coupled deformations |
| **FLAME** | linear PCA + LBS, jaw+neck+eye joints | head | none (hollow shell) | global expression basis |
| **SMPL** | linear PCA + LBS, pose blendshapes | **full body** | none | same mathematical family — GNM is to heads what SMPL is to bodies |
| **TailorMe** | anatomically constrained volumetric | body | bone/fat layers | anatomical, not facial |
| **NPHM / imHead / AIM** | neural implicit | head | handles topology change | **black-box latent, no semantic control** |
| **Gaussian Head Avatar / StyleMorpheus** | 3DGS / NeRF | head | teeth+eyes "baked" into the volume | photorealistic but identity-specific, heavy |
| **GNM** | linear PCA + LBS, **regional bases** | head+neck+eyes+teeth+tongue | **yes, explicit and separable** | regional + CVAE semantic sampler |

**The one-sentence positioning:** GNM deliberately stays *linear* to keep graphics-pipeline compatibility and semantic control, and buys back the fidelity that neural methods promise by (a) far better data and (b) splitting the bases regionally instead of globally.

---

## 11. Honest weaknesses (have these ready — they're also your questions)

- **Still linear.** Cannot change topology. A neural implicit model handles mouth opening as a genuine topological event; GNM approximates it within a fixed mesh.
- **Fitting is optimisation-based** — Adam, 5,000 steps, per scan. That is not a real-time face tracker; it's an offline fitting pipeline. Real-time use needs a feed-forward regressor onto GNM coefficients, which the paper doesn't provide.
- **Geometry only** — no appearance/texture/albedo model in this report, despite texture being used during registration.
- **Demographic categories are coarse** — binary gender, 4 ethnicity classes in the sampler. The paper flags this itself, at length, as a fairness limitation driven by data availability.
- **Human in the loop** for stabilisation — a manual component in an otherwise automated pipeline, and a scaling bottleneck.
- **Cranium is inferred, not measured** — from 200 artist meshes.

---

## Why this matters for the XR Audio Face Tracking role

- **It is the output representation your problem needs.** Audio-driven facial animation (`Audio-Driven-Facial-Animation.md`) has to predict *something*. FaceFormer regresses raw mesh vertices. Predicting GNM's **382 regional expression coefficients** instead of 17,821 × 3 vertices is smaller, smoother, anatomically constrained by construction, and cannot produce a self-intersecting face. That is a strong, concrete answer to "how would you architect an audio-to-face model here".
- **The tongue and teeth are the coarticulation story.** Your notes already argue that coarticulation is why frame-independent mappings fail. GNM is the first public model that can actually *represent* the tongue positions that phonetic articulation requires. Connect these two out loud.
- **Regional expression bases = regional losses.** Because the lower face is a separate block of coefficients, you can weight the mouth region higher in an audio-driven loss — audio tells you a lot about the mouth and nothing about eyebrow raises. That falls straight out of the architecture.
- **The eye model is the gaze/eye-tracking story.** Corneal curvature → LED glints → gaze estimation, all in a headset.
- **Privacy-safe synthetic data.** A named GNM use case, and a real answer to "how would you get training data" when face data is legally fraught. Ties to your `exe8` work: generate paired synthetic audio/face data with GNM as the geometry source.
- **Real-time is the open gap.** Everything in `Real-Time-Edge-Deployment.md` applies: the paper's fitting is offline optimisation, and the obvious productisation is distilling it into a feed-forward network under a per-frame budget. Saying this shows you read it critically rather than admiringly.

**Two questions to ask the interviewer:**
1. *"The fitting pipeline is optimisation-based — 5,000 Adam steps with landmark, prior, anatomical and temporal terms. Is the real-time path a feed-forward regressor onto GNM coefficients, or something else? And does the anatomical self-intersection penalty survive that distillation, or do you have to re-impose it?"*
2. *"The expression basis splits into left eye / right eye / lower face — which maps suspiciously well onto an XR headset's sensor layout, where cameras see the eyes and the microphone informs the lower face. Was the regional split motivated by that fusion structure, or is the alignment a happy accident?"*

---

## 90-second recap (say this out loud, unprompted)

> "GNM is Google's new parametric head model, released publicly this July. The problem it solves is that every existing 3D morphable model — FLAME, Basel, LSFM — treats the head as a hollow shell: outer skin only, no teeth, no tongue, a sphere for an eyeball. That breaks down exactly where it matters for telepresence, because you can't synthesise a plausible open mouth or model lip and tongue coarticulation without the internal anatomy, and you can't simulate corneal glints for gaze tracking without a real eye model.
>
> GNM unifies face, neck, eyeballs, teeth and tongue in one statistical space — about 17,800 vertices, built from roughly 5,000 scanned subjects and 150,000 expression samples on a 22-camera rig, plus artist-made assets for the parts you physically can't capture, like teeth and the cranium under the hair.
>
> Architecturally it's still a linear model — PCA identity and expression bases plus linear blend skinning over four joints — which is a deliberate choice to stay controllable and pipeline-compatible, unlike neural implicit head models whose latent spaces are black boxes. The novel part is that the bases are *composite and regional* rather than global: identity splits into head, eyes and teeth; expression splits into left eye, right eye, lower face, tongue and pupil. That gives localised control — opening the mouth can't trigger an eye wink — and lets each region be trained on its own specialised dataset. Notably there's no jaw joint; jaw motion lives in the linear expression basis, which captures the accompanying soft-tissue deformation better than a rigid rotation would.
>
> On top of the PCA space they train a dual conditional VAE as a 'semantic sampler', so you can drive it with demographic and expression categories instead of raw eigen-coefficients, with cyclical KL annealing to avoid posterior collapse and mixup on the conditioning so interpolating between classes stays anatomically valid.
>
> Results: 0.75 millimetres mean scan-to-mesh error on 15,000 held-out scans versus FLAME's 0.97, and better across every facial region and every demographic subgroup. The honest limitations are that it's still linear so it can't change topology, the fitting is offline optimisation rather than real-time regression, and there's no appearance model yet — which is where I'd expect the interesting engineering work to be."

---

## Sources / cross-links

- Paper: `99 - Others/Interviews/gnm.pdf` (arXiv 2607.23687v1) · code: `github.com/google/GNM`
- [[Audio-Driven-Facial-Animation]] — coarticulation, VOCA/FaceFormer, what a model should output
- [[Multimodal-Fusion]] — cross-attention; relevant to combining camera + audio into GNM coefficients
- [[Real-Time-Edge-Deployment]] — the distillation/latency gap GNM's offline fitting leaves open
- [[Entropy-CrossEntropy-KLDivergence]] — the KL term in the semantic sampler's CVAE loss
- `exe7_vae_from_scratch.py` — same VAE machinery; the sampler is a conditional version of it
- `exe8_transformer_from_scratch.py` — an audio→coefficient model is exactly what would drive GNM
- Vault notes for comparison: `[[SMPL - a skinned multi-person linear model]]`, `[[TailorMe - Self-Supervised Learning of an Anatomically Constrained Volumetric Human Shape Model]]`
