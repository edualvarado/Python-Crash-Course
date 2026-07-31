Related: [[Google - ML SWE Interview - Study Plan]]

# Audio-Driven Facial Animation

Day 4 topic 2. Goal: explain each concept unprompted, <2 min, then be ready to name-drop and briefly contrast Wav2Lip / FaceFormer / VOCA / SadTalker / MakeItTalk — this is the most role-specific material in the whole plan.

---

## Prerequisites: FFT, STFT, and mel-spectrogram

Skip this section if these are already second nature — otherwise read it before topic 1 below, since MFCC, mel-spectrogram, and even wav2vec2's input all build directly on it. Presented in dependency order: FFT is the core math tool; STFT applies it in short windows over time; the mel-spectrogram is STFT's output remapped onto a perceptual frequency scale.

### FFT (Fast Fourier Transform)

**Definition:** An efficient O(N log N) algorithm for computing the Discrete Fourier Transform (DFT) — the operation that decomposes a discrete signal of N samples into its frequency components: `X[k] = Σₙ x[n]·e^(−j2πkn/N)` for k=0,...,N−1. Each `X[k]` is a complex number giving the amplitude and phase of the sinusoidal component at frequency `k/N` cycles/sample. Computed naively, the DFT is O(N²) — every output bin sums over all N input samples. The FFT (classically the Cooley-Tukey algorithm) exploits the DFT's recursive symmetry to divide-and-conquer the computation (e.g. splitting even/odd-indexed samples), getting the exact same result in O(N log N) — for N=1024 that's ~10K operations instead of ~1M.

**Example 1:** Inside STFT (below), each ~25ms window (e.g. 400 samples at 16kHz) gets FFT'd into ~200 frequency bins. FFT is what makes running this at ~100 windows/sec over hours of speech computationally practical — naive O(N²) DFT would make real-time spectral feature extraction infeasible at this rate.

**Example 2 (own domain tie):** Same algorithm, same speedup, used to pull gait cadence out of windowed IMU (Inertial Measurement Unit — the accelerometer/gyroscope sensor package in motion-capture and phone/wearable hardware) accelerometer data — finding the dominant frequency (steps/sec) in a signal, identical math applied to a different sensor.

**Likely follow-up:** *"Why does the speedup matter specifically for this role?"* → Real-time/edge audio processing has a tight per-frame compute budget — O(N log N) vs O(N²) is the difference between feasible real-time feature extraction and not, especially at the sample rates/window sizes involved on constrained XR (Extended Reality — the umbrella term for VR/AR/mixed-reality headset hardware) hardware.

### STFT (Short-Time Fourier Transform)

**Definition:** Decomposes a signal's frequency content *over time* by computing an FFT on short, overlapping windows (frames) of the waveform instead of the whole signal at once: `STFT(x)[m,k] = Σₙ x[n]·w[n − mH]·e^(−j2πkn/N)`, where `w` is a window function (e.g. Hann, to taper each frame's edges and avoid spectral leakage), `H` is the hop size (stride between successive windows), `N` is the FFT size, `m` indexes time frames, `k` indexes frequency bins. The result is a complex-valued 2D array (time × frequency); its magnitude is the **spectrogram**. A single whole-clip FFT would tell you *what* frequencies are present overall but destroy *when* they occurred — STFT assumes the signal is roughly stationary within each short window and Fourier-transforms each independently, recovering a time-frequency representation.

**Example 1:** This is literally step 1 of the MFCC/mel-spectrogram pipeline below: raw waveform → STFT → magnitude → mel filterbank → log → (DCT for MFCC). The window length and hop size chosen here set the frame rate of the extracted audio features, which then needs to line up with (or be resampled to) the output video/blendshape frame rate.

**Example 2 (own domain tie):** Same tool used in gait/IMU analysis — computing a spectrogram of an accelerometer signal to track how frequency content (e.g. step cadence) changes over a walking bout, rather than just getting one number for the whole recording.

**Likely follow-up:** *"What's the tradeoff in choosing window length?"* → Time-frequency uncertainty: a longer window gives better frequency resolution (narrower, more distinguishable bins) but blurs time resolution (smears together events close in time); a shorter window gives better time resolution but coarser frequency resolution. Speech's phonemes change roughly every 50–200ms, so the standard practical default is a ~25ms window with a ~10ms hop (60% overlap) — short enough to track phoneme-rate changes, long enough for usable frequency resolution.

### Mel-spectrogram

**Definition:** A spectrogram (STFT magnitude) whose frequency axis has been remapped from linear Hz onto the **mel scale** — a perceptually-motivated nonlinear scale where equal distances correspond to roughly equal perceived pitch differences (`mel(f) = 2595·log₁₀(1 + f/700)`). In practice: STFT → magnitude → pass through a bank of triangular "mel filters" that pool the linear frequency bins into fewer (typically 40–128) mel bands → often take the log (perceived loudness is roughly logarithmic, and logging compresses dynamic range for stabler model training). Human pitch perception is roughly linear below ~1kHz and logarithmic above it, so the mel scale allocates more resolution to the low frequencies where speech's most informative content lives (fundamental frequency, early formants) and compresses the higher frequencies that matter less for phoneme identity.

**Example 1:** raw waveform → STFT → magnitude → mel filterbank → log → **mel-spectrogram** → (optionally) DCT (Discrete Cosine Transform — decomposes a signal into a sum of cosine waves; similar in spirit to FFT but real-valued and effective at compacting/decorrelating energy into few coefficients) → **MFCC**. Many modern audio encoders (e.g. Tacotron-style TTS, some speech front-ends) stop at the log-mel-spectrogram rather than going all the way to MFCC, since the DCT step discards information a neural net could otherwise exploit.

**Example 2 (own domain tie):** Same "reweight the raw frequency axis toward what's actually informative" idea as emphasizing biomechanically-relevant frequency bands (e.g. typical gait frequency ~1–3Hz) over irrelevant high-frequency sensor noise when building IMU features.

**Likely follow-up:** *"Why does perceptual motivation matter for a model, not just human ears?"* → It concentrates a fixed bin budget (e.g. 80 mel bins) where the signal that's actually discriminative for phoneme/speech content lives, rather than spreading it evenly across linear Hz — generally more discriminative per bin than 80 evenly-spaced linear bins would be, especially with limited training data.

---

## 1. Audio feature extraction: MFCC, mel-spectrogram, learned embeddings

**Definition:** Raw waveform audio (16–44.1kHz) is far too high-resolution and noisy to feed directly into most audio-to-face models frame by frame, so it's converted into a lower-dimensional per-frame representation first, aligned to a fixed frame rate.
- **Mel-spectrogram:** see Prerequisites above — a 2D time-frequency representation, the standard input either fed directly to a model or compressed further into MFCC.
- **MFCC (Mel-Frequency Cepstral Coefficients):** apply a DCT to the log mel-spectrogram (see Prerequisites above) to decorrelate and compress it into ~13–40 coefficients per frame — the classic hand-engineered speech-recognition feature, still used but largely superseded for representation-heavy tasks.
- **Learned embeddings (wav2vec2, HuBERT):** self-supervised Transformer encoders pretrained on large unlabeled speech corpora (contrastive prediction for wav2vec2, masked cluster prediction for HuBERT) — output contextualized frame-level embeddings that capture phonetic/prosodic structure beyond what a hand-crafted spectral summary can, and are the dominant audio front-end for state-of-the-art (SOTA) models like FaceFormer. Full breakdown of how this actually learns is below.

**Example 1:** FaceFormer uses a pretrained wav2vec2 encoder, fine-tuned jointly with its face-mesh decoder — richer than MFCC because it carries higher-level phonetic/prosodic structure learned from massive unlabeled speech data, not just a fixed spectral transform.

**Example 2 (own domain tie):** Same problem as feature-engineering raw IMU signals before regression (SoleSense) — you wouldn't feed raw high-frequency accel/gyro straight into a small regression head without some feature extraction or learned front-end first; audio's raw waveform is the identical problem in a different sensor modality.

**Likely follow-up:** *"Why not feed the raw waveform directly, end-to-end?"* → You can, but raw waveform is ~100–1000× the temporal resolution of the ~25–30fps output you need, so you'd need a learned downsampling front-end anyway — which is essentially what wav2vec2 already is. Training an audio encoder from scratch on limited face-paired data is also far less sample-efficient than fine-tuning one already pretrained via self-supervision on huge unlabeled speech corpora.

### wav2vec2 — what it is, and how it actually learns (from scratch)

**wav2vec2 is not built on top of STFT/mel-spectrogram/MFCC — it replaces that whole pipeline.** Instead of a fixed, hand-designed Fourier-based transform, it *learns* its own way of turning raw waveform into useful per-frame vectors, trained on huge amounts of unlabeled speech before ever seeing a face-animation dataset.

**First, what is an "embedding"?**
An embedding is just a list of numbers (a vector) representing something — a word, an image patch, a chunk of audio — placed at a point in a high-dimensional space so that *similar* things end up with *nearby* vectors and *different* things end up far apart. No single number in the vector means anything on its own; only distances and directions relative to other embeddings matter. Two audio chunks that both sound like the vowel "ah" should get nearby vectors; an "ah" chunk and an "sss" chunk should be far apart. Critically, nobody hand-designs where each sound goes — a model *learns* the placement, driven entirely by a training objective. This is the opposite of MFCC, where the "coordinates" (the DCT coefficients) are computed by a fixed formula, not learned.

**Second, what does "self-supervised" mean?**
Ordinary supervised learning needs a human-provided answer for every example ("this audio = the word 'cat'") — that doesn't scale to the enormous amount of raw unlabeled audio available (podcasts, audiobooks, etc.); nobody has time to transcribe all of it. Self-supervised learning invents its own training signal directly from unlabeled data: hide part of the input, and train the model to guess the hidden part from what's still visible. No human labels anything — you already know the right answer because you're the one who hid it. Same idea as a fill-in-the-blank test: "The cat sat on the ___" — the blank is obviously "mat," and no external grader is needed to know that.

**Now, wav2vec2 step by step:**

1. **Raw waveform in.** A few seconds of raw audio samples (16,000 numbers per second at 16kHz) — no STFT, no mel-spectrogram, just the raw signal.
2. **CNN feature encoder.** A stack of 1D convolution layers slides over the raw waveform and compresses it into a shorter sequence of vectors, roughly one every 20ms. This plays the same *role* as the Prerequisites' STFT step (raw samples → one summary vector per short window) — except this "summary" isn't a fixed Fourier transform, it's whatever the CNN's weights currently compute, and those weights are exactly what training is going to adjust.
3. **Build an answer key: quantization.** Each of those same CNN output vectors is also snapped to the nearest entry in a learned "codebook" — a lookup table of a few hundred/thousand representative prototype sound-chunk vectors. This snapped, discrete version becomes the "ground truth answer" for that position later on. Discretizing matters because "predict the exact right real-valued vector" is a mushy, ill-defined target (there's no single correct real number), whereas "pick the right entry out of a finite list" is a clean, well-posed problem.
4. **Mask part of the sequence.** A run of consecutive positions in the CNN's output sequence gets randomly blanked out — the audio equivalent of covering up a few consecutive words in a sentence.
5. **Transformer fills in the blanks.** A Transformer encoder processes the whole sequence, masked spots included, using self-attention so every position can look at every *other* visible position — both before and after a blank — to build a guess about what belongs there. The vector this produces at each position is what people actually mean by "the wav2vec2 embedding."

   > **Aside — why *encoder*, and which "masking" is this?** Encoder vs. decoder is older than Transformers: an **encoder** takes an input and maps it into a representation of its content (an "understand/compress" step); a **decoder** takes a representation and maps it *out* into something, usually generated step by step (a "produce/generate" step) — e.g. a classic autoencoder's encoder compresses an image to a vector, its decoder reconstructs the image back. wav2vec2's job is to represent an already fully-available input (a fixed audio clip), not to generate a new sequence, so there's no reason to stop any position from attending to any other position — full bidirectional context is exactly what an encoder is for.
   >
   > This is a **different masking** from decoder/causal masking, despite the shared word: here the input is fully known, and a few spots are deliberately blanked out as a *training trick*, with the model free to use everything else — both directions — to guess what's missing (a fill-in-the-blank on a complete sequence). Causal masking, used in a *decoder* generating output step by step (e.g. FaceFormer's own decoder producing mesh frames one at a time, cross-attending back into this wav2vec2 encoding), is a hard architectural constraint, not a training trick — a generation step genuinely cannot see positions after itself, because they don't exist yet.

6. **The quiz: contrastive loss.** At each masked position, take the Transformer's guess vector and compare it against (a) the true quantized answer from step 3, and (b) a handful of "distractor" quantized vectors sampled from other random positions. The model is scored on making its guess land close to the true answer and far from the distractors — a multiple-choice question with one right option and several wrong ones.
7. **Backpropagation.** Every wrong guess produces an error signal that flows backward through the Transformer, the CNN encoder, and the codebook, nudging all their weights slightly toward a better next guess. Repeated over millions of masked positions across a huge unlabeled speech corpus, the CNN gradually learns to extract genuinely useful acoustic structure, and the Transformer learns to use surrounding context to guess well — purely as a side effect of getting better at the quiz. Nobody ever tells the model "this is a phoneme" or "this is the /k/ sound"; that structure emerges on its own.
8. **Throw away the quiz, keep the machine.** After pretraining, the codebook and the contrastive-loss scaffolding are discarded — they were only training wheels. What's kept is the CNN + Transformer stack. Feed it any new raw waveform (no masking this time) and its output sequence of vectors — the embeddings — becomes the input features for a downstream task, e.g. FaceFormer's decoder mapping them to face-mesh motion.

**Back to "what is an embedding":** after training, wav2vec2's output vector at a given 20ms frame sits in its vector space such that acoustically/phonetically similar sounds (two different recordings of someone saying "ba") land near each other, and different sounds land far apart — exactly the property an embedding is supposed to have. Unlike MFCC's fixed mel-scale-plus-DCT formula, this placement was never hand-specified — it emerged entirely from millions of rounds of the masked-prediction quiz above.

**Example (own domain tie):** Same self-supervised recipe used to pretrain vision/motion backbones — e.g. masked autoencoders for images (mask patches, train a model to reconstruct them) or masked-motion pretraining for mocap sequences (hide a chunk of frames, predict the missing pose) — identical "hide part of the input, predict it from context" trick, just applied to a different modality than audio.

**Likely follow-up:** *"Why quantize at all — why not just directly regress the true continuous vector?"* → Regressing a raw continuous target directly is underspecified and prone to collapse — the model could cheat by outputting the average vector for everything and still get a deceptively low error, since there's no natural notion of "close enough" for a continuous target. Turning it into a discrete, multiple-choice-style contrastive task (true code vs. a handful of distractor codes) gives a well-defined, non-degenerate learning signal instead.

---

## 2. Viseme/phoneme mapping and coarticulation

**Definition:** A **phoneme** is the smallest distinguishing unit of speech sound (e.g. /p/, /b/, /m/); a **viseme** is its visual counterpart — the mouth-shape category. The mapping isn't 1:1: multiple phonemes can share a viseme (/p/, /b/, /m/ look near-identical on the mouth despite being different sounds), so visemes form a coarser set (~10–15) than phonemes (~40+ in English). **Coarticulation** — neighboring phonemes influence a given phoneme's articulation — means the mouth shape for a phoneme depends on what comes before *and after* it, not the phoneme in isolation; a per-frame-independent phoneme→shape mapping ignoring this produces jerky, unnatural motion.

**Example 1:** "stew" vs "stee" — the /s/ and /t/ show anticipatory lip-rounding ahead of the upcoming /u/ in "stew" but not in "stee": identical consonants, visibly different mouth shape, purely from what follows.

**Example 2 (own domain tie):** Directly analogous to human motion synthesis, where a given frame's pose is shaped by both past and future context (e.g. footstep placement anticipating an upcoming turn) — the same "context beyond the current instant matters" principle, and exactly why frame-independent mappings fail and RNN/LSTM/attention (topic 1) are needed at all.

**Likely follow-up:** *"How do models actually handle coarticulation architecturally?"* → By never mapping frame-by-frame independently — each output frame is conditioned on a temporal window or the full audio sequence via RNN hidden state or Transformer self-/cross-attention over neighboring frames, so anticipatory/carry-over effects are learned implicitly from data rather than hand-coded as rules.

---

## 3. Mapping architecture: frame-wise regression vs seq2seq

**Definition:** Two general framings for the audio→face task.
- **Frame-wise regression:** predict blendshape/viseme coefficients per audio frame, with local temporal context via a recurrent hidden state or a windowed input — simpler, low-latency, streaming-friendly.
- **Seq2seq (sequence-to-sequence):** encode the full audio sequence (or a chunk), then decode the full output motion sequence with (potentially bidirectional) access to the whole context — captures longer-range dependencies, smoother/more coherent output, but higher latency since it needs to see more/all of the input first.

Real-time/edge-constrained systems (like XR face tracking) favor frame-wise, causal, small-lookahead formulations, trading some smoothness for latency.

**Example 1:** A live XR avatar in a call needs frame-wise, causal (or bounded-lookahead) prediction — you can't wait for a full sentence to finish before animating the mouth without introducing unacceptable lag.

**Example 2:** Offline dubbing/re-animation pipelines (generating a lip-synced video from a full prerecorded audio track) can afford full bidirectional seq2seq context since there's no real-time constraint — this is the regime Wav2Lip typically operates in.

**Likely follow-up:** *"Which would you use for this role, given the latency constraint?"* → Frame-wise/causal with a small bounded lookahead (a few tens of ms of future audio context is standard in real-time lip-sync systems) — enough to capture some anticipatory coarticulation without blowing a real-time XR latency budget. Full bidirectional seq2seq is off the table for a live display-refresh-rate budget.

---

## 4. Named models — know what each one actually outputs

| Model | Year | Output representation | Core mechanism |
|---|---|---|---|
| **VOCA** | 2019 | 3D mesh vertex offsets | Temporal-conv (convolution applied along the time axis, like the SoleSense TCN from Day 2/RNN-LSTM note), speech + one-hot subject-identity label (disentangles speaking style from content); introduced the VOCASET 4D (3D mesh geometry + time) audio-face dataset |
| **Wav2Lip** | 2020 | 2D video (re-dubbing) | GAN (Generative Adversarial Network — a generator and discriminator trained against each other) based; existing video + target audio in, lip-synced video out; trained against a pretrained "lip-sync expert" discriminator (SyncNet) for strong audio-visual sync accuracy |
| **FaceFormer** | 2022 | 3D mesh vertex sequence | Autoregressive Transformer decoder, cross-attends over a wav2vec2 audio encoding; explicitly modeled for long-term context and coarticulation via self-attention; trained on VOCASET |
| **SadTalker / MakeItTalk** | 2022/2020 | 2D video from a single photo | Predict 3D head pose + expression/landmark coefficients from audio, render back into 2D — animate a static portrait rather than drive a rigged mesh |

**Definition (the grouping that matters):** these split cleanly by **output representation** — 2D video re-dubbing (Wav2Lip) vs 3D mesh/vertex-driven (VOCA, FaceFormer) vs single-photo portrait animation (SadTalker, MakeItTalk). An XR face-tracking role drives a rigged 3D avatar, so it maps most directly onto the **VOCA/FaceFormer family**; the others solve an adjacent but different output problem, even though the audio-feature-extraction and coarticulation-modeling core is shared across all five.

**Likely follow-up:** *"Which of these is most relevant to what this role would actually build?"* → FaceFormer's family — 3D mesh/blendshape output, audio-conditioned, real-time-adaptable. Wav2Lip's 2D re-dubbing and SadTalker/MakeItTalk's single-photo animation are closer to consumer video/dubbing applications than to a live XR avatar pipeline.

---

## Why this matters for the XR Audio Face Tracking role

This is the single most role-specific topic in the plan — direct application, not generic ML fluency:

- The whole pipeline is: **raw audio → learned features → temporally-aware model (RNN/LSTM/Transformer, topic 1) → viseme/blendshape coefficients**, with coarticulation and latency as the two competing constraints to manage throughout.
- Naming VOCA/FaceFormer specifically (not just Wav2Lip, which is the one most people know from media coverage) signals you understand the **output-representation distinction** — mesh/blendshape-driven vs 2D video — which is exactly what an XR avatar pipeline needs versus what a re-dubbing tool needs.
- The frame-wise-vs-seq2seq tradeoff (topic 3) connects straight into Day 4 topic 4 (real-time/edge constraints) — expect the interviewer to probe "how would you make this run in real time," and the honest answer is causal/bounded-lookahead frame-wise prediction, same logic as the RNN-vs-attention latency tradeoff from topic 1.

---

## 90-second recap (say this out loud, unprompted)

> "Audio-driven facial animation maps a speech signal to a moving face over time. First you extract features from raw audio — MFCC or mel-spectrogram classically, or a pretrained self-supervised encoder like wav2vec2/HuBERT now, since that carries richer phonetic and prosodic structure. Those features get mapped to viseme or blendshape coefficients per frame — but not independently per frame, because of coarticulation: a phoneme's mouth shape depends on its neighbors, so the model needs temporal context via a recurrent hidden state or attention, not a frame-by-frame lookup. Architecturally that's either frame-wise causal regression, which is low-latency and streaming-friendly, or full seq2seq, which is smoother but needs to see more of the input first — real-time XR needs the former, or something close to it with bounded lookahead. On named models: VOCA was an early temporal-conv model predicting 3D mesh vertex offsets from speech plus a subject-identity label; FaceFormer is the Transformer successor, cross-attending over a wav2vec2 encoding for long-range coarticulation modeling — both output 3D mesh, which is what an XR avatar pipeline actually needs. Wav2Lip and SadTalker/MakeItTalk solve an adjacent problem — 2D video re-dubbing or single-photo animation — sharing the same audio-processing core but a different output representation."

---

## Sources / cross-links

- [[Google - ML SWE Interview - Study Plan]] — Day 4, topic 2
- [[RNN-LSTM]] — the temporal-modeling mechanism underlying the audio→viseme mapping
- [[Google - Technical Interview]] — role/JD context
