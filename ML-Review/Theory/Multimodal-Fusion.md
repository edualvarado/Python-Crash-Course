Related: [[Google - ML SWE Interview - Study Plan]], [[RNN-LSTM]], [[Audio-Driven-Facial-Animation]]

# Multimodal Fusion (Audio + Visual)

Day 4 topic 3. Goal: explain early vs. late vs. cross-attention fusion unprompted, <2 min, and be ready to argue for cross-attention specifically in an XR context.

---

## 1. First, a distinction worth being precise about: fusion vs. cross-modal generation

**Definition:** Topic 2 (VOCA/Wav2Lip/FaceFormer) is **cross-modal generation** — one live input modality (audio) mapped to a *different* output modality (a face mesh). There's only one input stream at inference time. **Multimodal fusion** is a different problem: *two or more* live input streams, each providing partial/complementary evidence about the *same* target output, combined to produce one better answer than either stream alone could.

**Example 1 (why this matters for the role specifically):** An XR (Extended Reality) headset physically occludes part of the face — the headset itself covers the eyes/upper face, and an inward-facing camera typically only sees a partial, steeply-angled view of the mouth/lower face. Audio is often the *only* reliable full signal for certain expressions or phonemes when the camera view is poor or occluded. Fusing camera + audio — not relying on camera alone — is a first-order product need for this role, not a nice-to-have.

**Example 2 (own domain tie):** SoleSense fuses two live sensor streams — IMU (accelerometer/gyroscope) and insole pressure — to predict gait metrics; neither stream alone is fully sufficient. Structurally the same problem as camera+audio fusion, just a different pair of sensors.

**Likely follow-up:** *"So which topic actually needs fusion — audio-to-face generation, or facial tracking?"* → Generation (topic 2) is single-input, no fusion needed. *Tracking* — reconstructing what the wearer's actual face is doing right now, for their live avatar — is the fusion problem: combining whatever partial camera view exists with the always-available audio.

---

## 2. Early fusion

**Definition:** Concatenate raw or low-level features from each modality right at the input, before either has been through much modality-specific processing, then run a single shared model on the combined representation.

**Example 1 (XR case):** Stack a per-frame audio feature (e.g. a wav2vec2 embedding) and a per-frame visual feature (e.g. camera landmark coordinates) into one combined vector, and feed that straight into a single downstream network predicting facial expression.

**Example 2 (own domain tie):** Concatenating raw IMU and insole-pressure readings into one feature vector before a single shared regression head — same paradigm, no audio/visual involved at all.

**Likely follow-up:** *"What's the main weakness?"* → It forces very different modalities — different statistics, different natural sample rates, different noise characteristics — into a shared representation before either has been transformed into a comparable space, and it doesn't explicitly handle temporal misalignment between the streams (a microphone and a camera aren't naturally in lockstep). A single early layer has to somehow absorb both problems at once.

---

## 3. Late fusion

**Definition:** Each modality is processed independently through its own separate model, producing independent predictions, and only the *final outputs* are combined — by averaging, a weighted sum, or a small combiner network on top of the concatenated final outputs. The modalities never see each other until the very last step.

**Example 1 (XR case):** A camera-only model predicts facial expression from visible landmarks; a fully independent audio-only model predicts expression from speech (audio → viseme, topic 2's problem in miniature); the final output is some combination of the two — e.g. weighted more toward camera when the mouth is visible, more toward audio when it's occluded.

**Example 2 (own domain tie):** Two separate SoleSense-style models — one trained on IMU alone, one on insole pressure alone — combined only by a final blend, rather than sharing information internally.

**Likely follow-up:** *"What's lost compared to early or cross-attention fusion?"* → By the time you combine, each branch has already discarded whatever information wasn't relevant to *its own* unimodal prediction. Genuinely cross-modal cues that only make sense jointly — e.g. a specific ambiguous partial mouth shape combined with a specific phoneme implying one unambiguous expression — can't be exploited, because neither branch ever saw the other modality's signal while it was still deciding what to keep.

---

## 4. Cross-attention fusion (current state-of-the-art / SOTA)

**Definition:** Each modality gets its own encoder producing a sequence of modality-specific features (e.g. wav2vec2 for audio, a CNN/vision-transformer for camera frames — see [[Audio-Driven-Facial-Animation]] for the encoder/decoder distinction). A cross-attention mechanism then lets one modality's features "look up" relevant information in the other modality's features at each layer, often in both directions — exchanging information repeatedly *during* processing, not just once at the start (early fusion) or once at the end (late fusion). Mechanically it's the same scaled dot-product attention math as a Transformer's self-attention ([[RNN-LSTM]] has the reference formula), except the Query comes from one modality while the Key/Value come from the *other* modality — that swap is exactly what makes it "cross" rather than "self" attention.

**Example 1 (XR case):** At each timestep, the audio stream's query vector attends over the camera stream's key/value vectors (and vice versa), letting the model learn, per-timestep, how much to trust each modality — leaning on audio when the camera's mouth view is occluded by the headset, leaning on camera when the audio is noisy or ambiguous.

**Example 2 (temporal alignment):** This is also what handles the audio/camera sync problem directly — audio sampled at 16kHz and a camera running at 30–90fps aren't naturally frame-aligned, and there's often a small hardware latency offset between microphone and camera capture. Cross-attention lets the model learn a *soft, weighted* temporal alignment between the two streams instead of assuming a fixed 1:1 frame correspondence.

**Likely follow-up:** *"Why is this the SOTA choice despite being more expensive?"* → It directly solves what early and late fusion each get wrong: modalities inform each other *during* feature-building (unlike late fusion), without forcing raw, incompatible signals together before either is in a comparable space (unlike early fusion), and it explicitly, adaptively models temporal alignment instead of assuming it. The cost is genuine — attention between two sequences of length n and m is O(n·m) — and it's harder to train, but it generally wins on quality, especially with real-world capture where mic/camera sync is imperfect.

---

## Why this matters for the XR Audio Face Tracking role

- This is the fusion problem the role's product actually has: a headset that occludes part of the face, an inward camera with a partial/steep-angle view, and a microphone with a full but ambiguous (viseme-level, not expression-level) signal. Naming the right paradigm — cross-attention — and explaining *why* (adaptive per-timestep trust + temporal alignment) is a much stronger answer than just listing all three.
- Connects directly to topic 1 (temporal modeling) and topic 2 (coarticulation): cross-attention fusion is solving a structurally identical "which context matters, and how much" problem — just across *modalities* here instead of across *time*.
- Connects directly to topic 4 (real-time/edge): full cross-attention's O(n·m) cost is a genuine latency concern on XR hardware — same tradeoff logic as the RNN-vs-Transformer discussion in topic 1. If asked how to make fusion cheap enough for a live budget, the honest answer is to bound the attention window temporally (attend only over a small recent window of the other modality, not the full sequence) rather than run full unbounded cross-attention — the same "bounded lookahead" idea from topic 2/3's frame-wise-vs-seq2seq tradeoff.

---

## 90-second recap (say this out loud, unprompted)

> "Multimodal fusion is a different problem from audio-to-face generation — it's combining two or more live input streams that each carry partial evidence about the same target, rather than mapping one modality into another. There are three standard paradigms. Early fusion concatenates raw or low-level features from each modality right at the input and runs one shared model on top — simple, but it forces mismatched modalities together before either is in a comparable representation, and it doesn't handle timing misalignment between streams. Late fusion runs each modality through a fully independent model and only combines the final outputs — lets you reuse strong pretrained unimodal models, but by the time you combine, each branch has already thrown away information that wasn't relevant to its own prediction, so genuinely cross-modal cues get lost. Cross-attention fusion, the current state of the art, gives each modality its own encoder and then lets the modalities attend into each other's features at multiple points during processing — the Query comes from one modality, Key and Value from the other. This is what lets the model learn, per-timestep, how much to trust each modality, and it explicitly handles the fact that audio and camera aren't naturally frame-aligned. For this role specifically — a headset that occludes part of the face, a partial camera view of the mouth, and a full but ambiguous audio signal — cross-attention fusion is the right answer, at the cost of being more computationally expensive, which is where bounding the attention window becomes the real-time-friendly compromise."

---

## Sources / cross-links

- [[Google - ML SWE Interview - Study Plan]] — Day 4, topic 3
- [[RNN-LSTM]] — scaled dot-product attention mechanics, self- vs cross-attention
- [[Audio-Driven-Facial-Animation]] — encoder/decoder distinction, the audio-side pipeline being fused
- [[Google - Technical Interview]] — role/JD context
