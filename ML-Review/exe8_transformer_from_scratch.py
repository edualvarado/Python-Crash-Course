"""
8. Transformer from scratch, ~45 min

Builds a Transformer encoder from the ground up -- scaled dot-product attention, multi-head
attention (MHA), sinusoidal positional encoding (PE), the position-wise feed-forward network
(FFN), and a pre-norm residual block -- then wires it into a small audio-to-face model.

The task is deliberately the role's task in miniature: map a sequence of synthetic "audio
features" to per-frame blendshape coefficients, where each target frame depends on its
NEIGHBORING phonemes, not just the current one. That is coarticulation, and it is the reason
frame-independent mappings fail (see `Theory/Audio-Driven-Facial-Animation.md` topic 2).

Theory this cashes out:
  - `Theory/RNN-LSTM.md`            -- scaled dot-product attention formula, why attention
                                       replaced LSTM (O(1) path between any two positions),
                                       and the O(n^2)-vs-O(n) latency tradeoff
  - `Theory/Multimodal-Fusion.md`   -- self- vs cross-attention is ONE line of code apart:
                                       where the Key/Value come from
  - `Theory/Audio-Driven-Facial-Animation.md` -- coarticulation, and the frame-wise (causal,
                                       low-latency) vs seq2seq (bidirectional, smoother)
                                       architectural choice

TODO list (each one is marked `# TODO n:` below, with the solution filled in underneath):
  1. scaled dot-product attention          -- the core formula, with masking
  2. causal mask                           -- what makes attention streaming-safe
  3. multi-head attention                  -- split into heads, attend, recombine
  4. sinusoidal positional encoding        -- attention is permutation-invariant without it
  5. position-wise feed-forward network    -- the other half of a Transformer block
  6. pre-norm encoder block                -- residual + LayerNorm wiring
  7. the audio-to-face model               -- stack the blocks, project to blendshapes
  8. training loop                         -- standard supervised step
  9. cross-attention                       -- same MHA module, Key/Value from a second stream

What the script proves when it runs:
  - your attention matches torch's `F.scaled_dot_product_attention` to ~1e-6
  - your MHA matches `nn.MultiheadAttention` to ~1e-6 (weights copied across)
  - the bidirectional model BEATS the causal model on this task, because the target at
    frame t depends on the phoneme at t+1 -- a causal model literally cannot see it.
    That gap is the accuracy price of low latency, measured rather than asserted.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # this machine has a libiomp/libomp DLL conflict; without this, `import torch` crashes

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
# A toy stand-in for an audio-driven facial animation pipeline:
#   input  : T frames of D_AUDIO-dimensional audio features (think MFCC / wav2vec2 output)
#   output : T frames of N_BLENDSHAPES coefficients in [0, 1] (think jawOpen, mouthPucker, ...)

N_PHONEMES = 6        # size of the toy phoneme inventory
D_AUDIO = 16          # audio feature dimension per frame
N_BLENDSHAPES = 4     # facial blendshape coefficients per frame
SEQ_LEN = 24          # frames per utterance
D_MODEL = 32          # Transformer working width (must be divisible by N_HEADS)
N_HEADS = 4
D_FF = 64             # feed-forward inner width; conventionally ~4x D_MODEL
N_LAYERS = 2


def make_dataset(n_seq: int = 512, noise: float = 0.15):
    """
    Synthetic audio -> blendshape data WITH coarticulation baked in.

    Two random-but-fixed lookup tables define the ground truth:
      phoneme_to_audio      : which audio features a phoneme produces
      phoneme_to_blendshape : which mouth shape a phoneme wants

    The key line is the target: the blendshape at frame t is a weighted blend of the
    PREVIOUS, CURRENT and NEXT phoneme. That forward-looking term is what makes this a
    sequence problem rather than a per-frame lookup -- and it is exactly why a causal
    model is handicapped here.
    """
    g = torch.Generator().manual_seed(42)
    phoneme_to_audio = torch.randn(N_PHONEMES, D_AUDIO, generator=g)
    phoneme_to_blendshape = torch.randn(N_PHONEMES, N_BLENDSHAPES, generator=g)

    # a random phoneme sequence per utterance
    phonemes = torch.randint(0, N_PHONEMES, (n_seq, SEQ_LEN), generator=g)

    # audio features: the phoneme's signature plus sensor noise
    x = phoneme_to_audio[phonemes]
    x = x + noise * torch.randn(x.shape, generator=g)

    # coarticulation: pad by one frame on each side so t-1 and t+1 always exist
    padded = F.pad(phonemes.unsqueeze(1).float(), (1, 1), mode="replicate").squeeze(1).long()
    prev_bs = phoneme_to_blendshape[padded[:, :-2]]    # phoneme at t-1
    curr_bs = phoneme_to_blendshape[padded[:, 1:-1]]   # phoneme at t
    next_bs = phoneme_to_blendshape[padded[:, 2:]]     # phoneme at t+1

    y = torch.sigmoid(1.0 * curr_bs + 0.6 * prev_bs + 0.6 * next_bs)
    return x, y


# ---------------------------------------------------------------------------
# TODO 1: scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    The one formula the whole architecture is built on:

        Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_head) ) @ V

    Shapes (B = batch, H = heads, T_q/T_k = query/key sequence length, d = head dim):
        q    : (B, H, T_q, d)
        k, v : (B, H, T_k, d)
        mask : (T_q, T_k) or broadcastable, BOOLEAN, True = BLOCKED
        ->   : out (B, H, T_q, d),  attn (B, H, T_q, T_k)

    Reading it in words: every query position scores itself against every key position,
    those scores are turned into a probability distribution, and the output is a weighted
    average of the value vectors. That is the "O(1) path between any two positions" from
    `Theory/RNN-LSTM.md` -- position 0 and position 100 are one matmul apart, whereas an
    LSTM would need 100 sequential steps to connect them.

    Why divide by sqrt(d)? Q @ K^T is a sum of d products. If the entries are roughly unit
    variance, the dot product has variance ~d, so it grows with head width. Feed large
    values into softmax and it saturates -- one weight goes to ~1, the rest to ~0, and the
    gradient through softmax vanishes. Dividing by sqrt(d) keeps the scores at unit scale,
    independent of head width. This is the "scaled" in scaled dot-product attention.
    """
    d_head = q.size(-1)

    # TODO 1a: scores = q @ k^T / sqrt(d_head)      -> (B, H, T_q, T_k)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)

    # TODO 1b: apply the mask by setting blocked positions to -inf BEFORE the softmax.
    # -inf (not 0, not a large negative) because exp(-inf) == 0 exactly, so a blocked
    # position contributes literally nothing and the remaining weights still sum to 1.
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # TODO 1c: softmax over the KEY axis (dim=-1), so each query's weights sum to 1
    attn = torch.softmax(scores, dim=-1)

    # TODO 1d: weighted sum of the values
    out = attn @ v

    return out, attn


# ---------------------------------------------------------------------------
# TODO 2: causal mask
# ---------------------------------------------------------------------------

def causal_mask(seq_len: int, device=None):
    """
    Upper-triangular boolean mask, True strictly above the diagonal.

    True = blocked, matching the convention in scaled_dot_product_attention above.

        seq_len = 4          key ->
                          0   1   2   3
                    0 [   F   T   T   T ]   frame 0 sees only frame 0
        query   |   1 [   F   F   T   T ]   frame 1 sees frames 0-1
          |     v   2 [   F   F   F   T ]
          v         3 [   F   F   F   F ]   frame 3 sees everything

    This is the difference between a model you can run live on a headset and one that has
    to wait for the whole utterance. With the mask, frame t's output never depends on
    t+1, so you can emit it the moment frame t arrives. Without it, the model is
    bidirectional: better quality here (the target genuinely depends on t+1), but you
    cannot stream it. That is the frame-wise-vs-seq2seq tradeoff from
    `Theory/Audio-Driven-Facial-Animation.md`, made concrete.
    """
    # TODO 2: build the upper-triangular boolean mask (diagonal=1 keeps the diagonal visible,
    # so a frame can always attend to itself)
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


# ---------------------------------------------------------------------------
# TODO 3: multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Run several attention "heads" in parallel over slices of the feature dimension,
    then concatenate and mix.

    Why heads at all? A single softmax produces ONE weighted average per query -- it is
    forced to commit to one notion of relevance. Splitting D_MODEL into H independent
    subspaces lets different heads specialize: one can track the immediately preceding
    frame, another a longer coarticulation window, another something spectral. Cost is
    unchanged -- H heads of width D_MODEL/H is the same total compute as one head of
    width D_MODEL. You are re-partitioning the width, not adding any.

    Note `forward` takes x_q and x_kv SEPARATELY. That single detail is the whole
    self-vs-cross attention distinction (`Theory/Multimodal-Fusion.md`):
        self-attention  : x_q is x_kv                  (audio attends to audio)
        cross-attention : x_q audio, x_kv video        (audio queries the video stream)
    Same module, same math -- only where the Key/Value come from changes.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # One projection each for Query, Key, Value, plus the output mix.
        # bias=False keeps the comparison against nn.MultiheadAttention below exact.
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x):
        """(B, T, D) -> (B, H, T, d_head). Heads move to their own axis so the attention
        matmuls below batch over them for free."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        """(B, H, T, d_head) -> (B, T, D). The inverse of _split_heads.
        `.contiguous()` is required because `.transpose` only changes the stride metadata;
        `.view` needs a contiguous buffer to reinterpret."""
        B, H, T, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d)

    def forward(self, x_q, x_kv, mask=None):
        # TODO 3a: project inputs to Query, Key, Value and split into heads
        q = self._split_heads(self.w_q(x_q))
        k = self._split_heads(self.w_k(x_kv))
        v = self._split_heads(self.w_v(x_kv))

        # TODO 3b: run attention (the mask broadcasts over batch and head axes)
        out, attn = scaled_dot_product_attention(q, k, v, mask)

        # TODO 3c: merge heads back together and apply the output projection.
        # Without w_o the heads would never talk to each other -- concatenation alone
        # leaves each head's output confined to its own slice of the feature vector.
        out = self.w_o(self._merge_heads(out))

        return out, attn


# ---------------------------------------------------------------------------
# TODO 4: sinusoidal positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Attention is permutation-invariant: shuffle the input frames and the set of outputs is
    just shuffled the same way. Nothing in Q @ K^T knows that frame 3 comes after frame 2.
    For audio that is fatal -- coarticulation is a statement about ORDER.

    So we add a fixed, position-dependent vector to every frame:

        PE[t, 2i]   = sin( t / 10000^(2i/d) )
        PE[t, 2i+1] = cos( t / 10000^(2i/d) )

    Each dimension pair is a sinusoid of a different wavelength, from ~2 frames up to
    ~10000*2*pi frames. Low dimensions oscillate fast (fine "which frame exactly"), high
    dimensions oscillate slowly (coarse "which region of the utterance"). Together they
    give every position a unique fingerprint, and -- because sin/cos of a shifted angle
    are linear combinations of the unshifted ones -- relative offsets stay linearly
    decodable, which is what the model actually needs.

    Fixed rather than learned means it extrapolates to sequence lengths never seen in
    training, which matters for streaming audio of unbounded length.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        # TODO 4: build the (max_len, d_model) table of sinusoids
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()             # (max_len, 1)
        # computed in log space for numerical stability: 10000^(-2i/d) == exp(-2i * ln(10000)/d)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions

        # register_buffer: part of the module's state (moves with .to(device), saved in
        # state_dict) but NOT a parameter -- no gradients, the optimizer ignores it.
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D) -- add the first T rows of the table, broadcasting over the batch
        return x + self.pe[: x.size(1)].unsqueeze(0)


# ---------------------------------------------------------------------------
# TODO 5: position-wise feed-forward network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    A two-layer MLP (multi-layer perceptron) applied to each position INDEPENDENTLY --
    the same weights for every frame, no mixing across time.

    Division of labour inside a Transformer block:
      - attention  MIXES information ACROSS positions, but is linear in the values
      - FFN        TRANSFORMS each position on its own, and supplies the nonlinearity

    Without the FFN, stacking attention layers would collapse into not much more than one
    big weighted average. Conventionally the inner width is ~4x d_model: expand, apply the
    nonlinearity in the wider space, project back.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # TODO 5: Linear -> GELU -> Linear
        # GELU (Gaussian Error Linear Unit) rather than ReLU: it is smooth near zero, which
        # is the modern default in Transformers (BERT, GPT). ReLU works fine too.
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# TODO 6: pre-norm encoder block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    One Transformer encoder layer: self-attention sublayer, then FFN sublayer, each wrapped
    in a residual connection with LayerNorm.

    PRE-norm (normalize going INTO the sublayer):
        x = x + Sublayer(LayerNorm(x))
    POST-norm (the original 2017 paper):
        x = LayerNorm(x + Sublayer(x))

    Pre-norm is used here because it trains stably without a learning-rate warmup schedule
    -- the residual path stays an untouched identity from input to output, so gradients
    reach early layers cleanly. Post-norm puts a LayerNorm on that path and needs warmup to
    avoid diverging. This is the same reasoning as residual connections in ResNets, and the
    same point made about Transformers in `Theory/RNN-LSTM.md`.

    LayerNorm (not BatchNorm) because normalizing across the FEATURE axis of a single
    frame is independent of batch composition and of sequence length -- essential when
    inference runs on one live stream with batch size 1.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # TODO 6a: self-attention sublayer, pre-norm + residual.
        # x_q and x_kv are the same tensor -- that is what makes it SELF-attention.
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, mask)
        x = x + attn_out

        # TODO 6b: feed-forward sublayer, pre-norm + residual
        x = x + self.ff(self.norm2(x))

        return x, attn_weights


# ---------------------------------------------------------------------------
# TODO 7: the audio-to-face model
# ---------------------------------------------------------------------------

class AudioToFaceTransformer(nn.Module):
    """
    audio features -> blendshape coefficients, per frame.

        (B, T, D_AUDIO) -> input projection -> + positional encoding
                        -> N_LAYERS encoder blocks
                        -> output head -> (B, T, N_BLENDSHAPES) in [0, 1]

    `causal=True` masks the future and makes this streaming-capable; `causal=False` lets
    every frame see the whole utterance. Same weights, same parameter count -- the ONLY
    difference is the mask. The training run below fits both and reports the gap.
    """

    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.input_proj = nn.Linear(D_AUDIO, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL)
        self.blocks = nn.ModuleList(
            EncoderBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)
        )
        self.norm_out = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_BLENDSHAPES)

    def forward(self, x, return_attn: bool = False):
        # TODO 7a: project audio features into the model width, then add positional info
        h = self.pos_enc(self.input_proj(x))

        # TODO 7b: build the mask once and reuse it across layers
        mask = causal_mask(x.size(1), x.device) if self.causal else None

        # TODO 7c: run the stack, keeping the last layer's attention for inspection
        attn_weights = None
        for block in self.blocks:
            h, attn_weights = block(h, mask)

        # TODO 7d: final norm, then project to blendshapes.
        # sigmoid because blendshape coefficients live in [0, 1].
        out = torch.sigmoid(self.head(self.norm_out(h)))

        return (out, attn_weights) if return_attn else out


# ---------------------------------------------------------------------------
# TODO 8: training loop
# ---------------------------------------------------------------------------

def train(model, x_train, y_train, x_val, y_val, epochs: int = 300, lr: float = 3e-3):
    """
    Standard supervised loop. MSE (mean squared error) because blendshape coefficients are
    continuous regression targets, not classes.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # TODO 8: the five-line training step -- forward, loss, zero_grad, backward, step
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                val_loss = F.mse_loss(model(x_val), y_val)
            print(f"    epoch {epoch:4d}  train {loss.item():.5f}  val {val_loss.item():.5f}")

    with torch.no_grad():
        return F.mse_loss(model(x_val), y_val).item()


# ===========================================================================
# Checks
# ===========================================================================

print("=" * 72)
print("Check 1: does scaled_dot_product_attention match torch?")
print("=" * 72)

B, H, T, Dh = 2, 4, 6, 8
q = torch.randn(B, H, T, Dh)
k = torch.randn(B, H, T, Dh)
v = torch.randn(B, H, T, Dh)

mine, _ = scaled_dot_product_attention(q, k, v)
theirs = F.scaled_dot_product_attention(q, k, v)
print(f"  unmasked  max abs diff: {(mine - theirs).abs().max().item():.2e}")

# MASK CONVENTION GOTCHA, worth knowing cold for an interview:
#   F.scaled_dot_product_attention : bool attn_mask True = ALLOWED to attend
#   nn.MultiheadAttention          : bool attn_mask True = BLOCKED from attending
# They are opposites in the same library. Ours follows the nn.MultiheadAttention
# convention (True = blocked), so we invert when calling the functional version.
cm = causal_mask(T)
mine_causal, _ = scaled_dot_product_attention(q, k, v, cm)
theirs_causal = F.scaled_dot_product_attention(q, k, v, attn_mask=~cm)
print(f"  causal    max abs diff: {(mine_causal - theirs_causal).abs().max().item():.2e}")
print(f"  (also matches is_causal=True: "
      f"{(mine_causal - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max().item():.2e})")

print()
print("=" * 72)
print("Check 2: does MultiHeadAttention match nn.MultiheadAttention?")
print("=" * 72)

mha = MultiHeadAttention(D_MODEL, N_HEADS)
ref = nn.MultiheadAttention(D_MODEL, N_HEADS, bias=False, batch_first=True)

# copy our weights into torch's module so the comparison is exact.
# torch packs Q, K, V into a single (3*d_model, d_model) matrix, in that order.
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([mha.w_q.weight, mha.w_k.weight, mha.w_v.weight], dim=0))
    ref.out_proj.weight.copy_(mha.w_o.weight)

x = torch.randn(2, SEQ_LEN, D_MODEL)
mine_out, mine_attn = mha(x, x)
ref_out, ref_attn = ref(x, x, x, need_weights=True, average_attn_weights=True)
print(f"  self-attention output   max abs diff: {(mine_out - ref_out).abs().max().item():.2e}")
print(f"  attention weights       max abs diff: "
      f"{(mine_attn.mean(dim=1) - ref_attn).abs().max().item():.2e}  (ours averaged over heads)")

cm_full = causal_mask(SEQ_LEN)
mine_c, _ = mha(x, x, cm_full)
ref_c, _ = ref(x, x, x, attn_mask=cm_full, need_weights=False)   # same convention: True = blocked
print(f"  causal-masked output    max abs diff: {(mine_c - ref_c).abs().max().item():.2e}")

print()
print("=" * 72)
print("Check 3: attention weights behave as expected")
print("=" * 72)

# Every row of the attention matrix is a probability distribution over key positions.
_, attn = mha(x, x)
print(f"  rows sum to 1:            {torch.allclose(attn.sum(dim=-1), torch.ones(2, N_HEADS, SEQ_LEN), atol=1e-5)}")

# Under the causal mask, everything strictly above the diagonal must be exactly zero --
# not merely small. That is what -inf before the softmax buys you.
_, attn_c = mha(x, x, cm_full)
upper = attn_c.masked_select(cm_full.expand_as(attn_c))
print(f"  future weights all zero:  {bool((upper == 0).all())}  (max = {upper.max().item():.1e})")

print()
print("=" * 72)
print("Check 4: causal vs bidirectional on the coarticulation task")
print("=" * 72)

x_all, y_all = make_dataset()
x_train, y_train = x_all[:384], y_all[:384]
x_val, y_val = x_all[384:], y_all[384:]
print(f"  audio {tuple(x_train.shape)} -> blendshapes {tuple(y_train.shape)}")

# Baseline: predict the training mean for every frame. Any real model must beat this.
baseline = F.mse_loss(y_train.mean(dim=(0, 1)).expand_as(y_val), y_val).item()
print(f"  constant-prediction baseline val MSE: {baseline:.5f}\n")

print("  [bidirectional]  every frame sees the whole utterance (seq2seq, not streamable)")
torch.manual_seed(1)
bi_val = train(AudioToFaceTransformer(causal=False), x_train, y_train, x_val, y_val)

print("\n  [causal]         every frame sees only the past (streaming-capable)")
torch.manual_seed(1)
causal_val = train(AudioToFaceTransformer(causal=True), x_train, y_train, x_val, y_val)

print()
print("-" * 72)
print(f"  bidirectional val MSE : {bi_val:.5f}")
print(f"  causal        val MSE : {causal_val:.5f}")
print(f"  causal is {causal_val / bi_val:.2f}x worse")
print("-" * 72)
print("""
  The target at frame t depends on the phoneme at t+1 (coarticulation), and the causal
  model is structurally forbidden from seeing it. The gap is not a tuning failure -- it is
  the information the mask removed.

  That is the real-time tradeoff from Theory/Audio-Driven-Facial-Animation.md, measured:
  bidirectional is more accurate but cannot emit frame t until the utterance ends; causal
  emits immediately. The usual production compromise is neither extreme -- allow a small
  bounded lookahead (attend to t+1..t+k only), buying back most of the accuracy for a
  fixed, budgetable k-frame latency.
""")

print("=" * 72)
print("Check 5: cross-attention -- TODO 9")
print("=" * 72)
print("""  Self- and cross-attention are the SAME module. Only the Key/Value source changes:
      self-attention  ->  mha(audio, audio)
      cross-attention ->  mha(audio, video)     Query from audio, Key/Value from video
  This is the fusion mechanism from Theory/Multimodal-Fusion.md, and the reason it handles
  audio/camera sync: the attention matrix IS a learned soft temporal alignment between two
  streams that were never frame-aligned to begin with.
""")

# TODO 9: cross-attention -- Query from audio, Key/Value from a second (video) stream.
# Different sequence lengths on purpose: 24 audio frames vs 10 video frames, which is the
# realistic case (16kHz microphone vs 30-90fps camera). Attention handles the mismatch for
# free -- the output length always follows the QUERY, and the attention matrix is
# (T_audio x T_video), i.e. exactly the alignment between the two rates.
audio_feats = torch.randn(2, 24, D_MODEL)
video_feats = torch.randn(2, 10, D_MODEL)

cross_out, cross_attn = mha(audio_feats, video_feats)
print(f"  audio (query) : {tuple(audio_feats.shape)}")
print(f"  video (key/val): {tuple(video_feats.shape)}")
print(f"  output        : {tuple(cross_out.shape)}   <- follows the QUERY length (24)")
print(f"  attention map : {tuple(cross_attn.shape)}   <- (B, H, T_audio, T_video) alignment")
print(f"  each audio frame's weights over video frames sum to 1: "
      f"{torch.allclose(cross_attn.sum(dim=-1), torch.ones(2, N_HEADS, 24), atol=1e-5)}")

print()
print("=" * 72)
print("""Recap for the interview:
  - attention = softmax(QK^T / sqrt(d)) V ; the sqrt(d) stops softmax saturating
  - heads re-partition d_model, they do not add compute; w_o is what mixes them
  - positional encoding exists because attention is permutation-invariant
  - the FFN is where the nonlinearity and per-position capacity live
  - pre-norm (x + Sublayer(LN(x))) trains without warmup; LayerNorm not BatchNorm because
    inference is batch size 1 on a live stream
  - causal mask = streaming; the accuracy cost is real and measured above
  - cross-attention is self-attention with Key/Value from the other modality
  - cost is O(T^2) in sequence length -- the reason bounded/windowed attention exists""")
print("=" * 72)
