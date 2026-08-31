"""
9. Attention from blank -- Day 9, ~60 min

`exe8_transformer_from_scratch.py` was written FOR you, not BY you. This file is the
re-do of its core, from nothing. Only the two pieces that are actually examinable:

    1. scaled_dot_product_attention
    2. causal_mask
    3. MultiHeadAttention

The other ~300 lines of exe8 (FFN, encoder block, the audio-to-face model, the training
loop) are reference material, NOT a memorisation target -- the coding segment tomorrow is
LeetCode-style. Do not re-type them.

RULES
  - Do not open exe8 until every check below passes. The ground truth here is torch's own
    `F.scaled_dot_product_attention` / `nn.MultiheadAttention`, so you can verify yourself.
  - Narrate before you type. That is the habit the mock said was missing.
  - 60 minutes. If you stall for more than ~10 min on one piece, open exe8, read that piece
    only, close it, and re-type from memory.

SAY THESE OUT LOUD BEFORE YOU START CODING -- no notes, no peeking:
  a) Why is there a 1/sqrt(d_head) in the formula? What specifically goes wrong without it,
     and why does the failure get worse as the head gets wider?
        A: A dot product of two d-dim vectors (each ~unit variance) has variance ≈ d. More dims → bigger raw scores, just from dimension count, not from actual relevance. Big scores saturate softmax (one weight →1, rest →0) → gradient vanishes, can't learn. Dividing by √d cancels that growth.
  b) Why does splitting d_model into H heads cost the same as one head of width d_model?
     What do you get in exchange for free?
        A: Total attention compute = (seq_len)² × d_model. Doesn't matter if that width is 1 head of 32 or 4 heads of 8 — same total FLOPs. What you get for "free": 4 independent softmax patterns instead of 1, so each head can specialize (recent-past, spectral shape, etc.) instead of being forced to average everything into one pattern.
  c) What is w_o for? What breaks if you concatenate the heads and stop there?
        A: After attention, you concatenate the H heads back into one vector — that's just stacking, each head's output sits untouched in its own slice. w_o is a linear layer that mixes across slices so heads can combine. Without it, heads never interact after attention.
  d) Why is positional encoding needed at all? State the property of attention that forces
     it, and why that property is fatal for audio specifically.
        A: Softmax(QK^T)V treats input as an unordered set: permute tokens, outputs permute the same way — nothing marks "this came right after that." Audio order = time, and coarticulation is specifically about the neighboring frame. No position → model can't tell "A then B" from "B then A."

Then answer them again at the end, and see whether the code changed your answers.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # libiomp/libomp conflict on this machine

import math

import torch                      # NOTE: torch BEFORE numpy on this machine, always
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

D_MODEL = 32 #  width of each token's feature vector flowing through the transformer (analogous to embedding size).
N_HEADS = 4 # how many attention heads split that 32-wide vector (each head gets 32/4 = 8 dims, d_head=8).
SEQ_LEN = 24 # number of frames per sequence (24 audio frames per utterance in this toy setup).


# ---------------------------------------------------------------------------
# TODO 1: scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q, K, V are three different linear projections (Linear(d_model, d_model)) of the same input — each token gets transformed 3 ways:
        - Q (query) — "what am I looking for?" (per token)
        - K (key) — "what do I contain?" (per token, used to be matched against)
        - V (value) — "what do I actually give you if you pick me?"

    Mechanics:
        1. score[i,j] = Q[i] · K[j] — how relevant token j is to token i
        2. scale by 1/√d, softmax over j → weights summing to 1 (per query row)
        3. output[i] = Σ_j weight[i,j] * V[j] — weighted blend of values

    Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_head) ) @ V

    Shapes (B = batch, H = heads, T_q / T_k = query / key length, d = head dim):
        q          : (B, H, T_q, d)
        k, v       : (B, H, T_k, d)
        mask       : (T_q, T_k) or broadcastable, BOOLEAN, True = BLOCKED
        returns    : out  (B, H, T_q, d)
                     attn (B, H, T_q, T_k), each row summing to 1 over the KEY axis

    Contract the checks below rely on:
      - blocked positions must come out of the softmax as EXACTLY 0.0, not merely small.
      - T_q and T_k may differ (that is cross-attention); the output length follows T_q.
    """

    #1. scores = q @ k^T / √d_head → shape (B,H,T_q,T_k)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    # k.transpose(-2, -1): (B,H,T_k,d) → (B,H,d,T_k)
    # matmul(q, k^T): (B,H,T_q,d) @ (B,H,d,T_k) → (B,H,T_q,T_k) ✓ (matches the spec)

    #2. mask — where mask is True, set that score to -inf (before softmax)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    #3. softmax the scores over the last axis (the key axis, T_k) → attn
    # You want: for each query (fixed i), the weights across all keys j sum to 1 — that's "how do I distribute my attention across all the candidates I could look at." Softmax normalizes along the axis you tell it to; picking -1 (T_k) means each row attn[b,h,i,:] sums to 1.
    attn = F.softmax(scores, dim=-1) # (B,H,T_q,T_k)

    #4. weighted sum = attn @ v → out        
    out = torch.matmul(attn, v) # (B,H,T_q,T_k) @ (B,H,T_k,d) → (B,H,T_q,d)

    return out, attn

# ---------------------------------------------------------------------------
# TODO 2: causal mask
# ---------------------------------------------------------------------------

def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    Boolean (seq_len, seq_len) mask, True = BLOCKED, matching TODO 1's convention.

    True strictly ABOVE the diagonal. The diagonal itself is False -- a frame must always
    be allowed to attend to itself. So query row t may see key columns 0..t inclusive.
    """
    # - torch.ones(seq_len, seq_len, dtype=torch.bool) → all True
    # - triu(..., diagonal=1) keeps only entries where j - i ≥ 1 (strictly above diagonal), zeros out everything else (diagonal and below → False)

    # Result: True exactly where j > i, False on and below diagonal — matches the spec. Run the file and check Check 2 & 3.

    # Two things change, not just the sign:

    # - Function: torch.tril (lower triangular), not triu.
    # - Diagonal: -1 to exclude the main diagonal itself (strictly below only).

    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)



# ---------------------------------------------------------------------------
# TODO 3: multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    H attention heads over disjoint slices of the feature dimension, concatenated and mixed.

    INTERFACE REQUIREMENTS (the checks copy weights across, so these names matter):
      - four `nn.Linear(d_model, d_model, bias=False)` projections named
        `w_q`, `w_k`, `w_v`, `w_o`
      - `forward(x_q, x_kv, mask=None) -> (out, attn)`

    Note that forward takes the query stream and the key/value stream SEPARATELY. That one
    detail is the entire self- vs cross-attention distinction. Be ready to say which is
    which, and what x_kv is in each case.

    Shapes:
        x_q  : (B, T_q, d_model)
        x_kv : (B, T_kv, d_model)
        out  : (B, T_q, d_model)
        attn : (B, H, T_q, T_kv)
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO 3a: assert divisibility, store d_model / n_heads / d_head,
        #          create the four projections named as documented above.
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias= False)
        self.w_k = nn.Linear(d_model, d_model, bias= False)
        self.w_v = nn.Linear(d_model, d_model, bias= False)
        self.w_o = nn.Linear(d_model, d_model, bias= False)

        # - w_q: Linear(d_model, d_model) — takes the input x and projects it into "queries". q = x @ w_q
        # - w_k: same shape, projects x into "keys". k = x @ w_k
        # - w_v: same shape, projects x into "values". v = x @ w_v
        # - w_o: applied after attention — takes the recombined multi-head output and mixes it back into one d_model-wide vector (from your earlier question: without it, heads never talk to each other).

        # Concretely: x is (B, T, d_model). self.w_q(x) runs it through the linear layer → still (B, T, d_model), but now it's "queries" instead of raw input — a learned transformation, not the same numbers. Then _split_heads reshapes that into (B, H, T, d_head) so each head gets its own slice.

        # w_q, w_k, w_v are used before attention (to produce Q/K/V); w_o is used after (to remix heads). All four are just nn.Linear layers with learnable weights, updated via backprop during training.

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, H, T, d_head). Heads get their own axis so the attention
        matmuls batch over them for free."""

        """
        Setup: B=1, T=3 (3 tokens), d_model=4, H=2 → d_head=2

        x, shape (1, 3, 4) — one row per token:
        token0 = [ 1,   2,   3,   4]
        token1 = [10,  20,  30,  40]
        token2 = [100,200, 300, 400]

        Step 1 — x.view(B, T, H, d_head) → (1, 3, 2, 2). Splits each token's 4 values into 2 chunks of 2, in order:
        token0: head0=[1,2]     head1=[3,4]
        token1: head0=[10,20]   head1=[30,40]
        token2: head0=[100,200] head1=[300,400]

        Step 2 — .transpose(1, 2) → (1, 2, 3, 2) = (B, H, T, d_head). Regroups by head instead of by token:
        head0: token0=[1,2]    token1=[10,20]   token2=[100,200]
        head1: token0=[3,4]    token1=[30,40]   token2=[300,400]

        Now head0 is its own clean (T=3, d_head=2) matrix — 3 tokens, each contributing only its first 2 dims. Same for head1. That's what gets fed into scaled_dot_product_attention independently per head.
        """

        # TODO 3b

        B, T, d_model = x.shape
        H = self.n_heads
        d_head = self.d_head

        # (B, T, d_model) -> (B, T, H, d_head) -> (B, H, T, d_head)
        return x.view(B, T, H, d_head).transpose(1,2)


    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_head) -> (B, T, d_model). Inverse of _split_heads.
        Watch out: one of the two ops here leaves the tensor non-contiguous, and the next
        one refuses to work on a non-contiguous buffer. Know which, and why."""
        # TODO 3c

        """
        .contiguous() physically copies the tensor's data into a new memory buffer, laid out in standard row-major order matching its current logical shape.

        Why it's needed: a tensor is a flat memory buffer + metadata (shape + strides) telling PyTorch how to read that buffer as multi-dimensional. .transpose() doesn't touch the buffer at all — it just swaps two stride values, so the same memory gets reinterpreted with axes swapped. Fast (no copy), but afterward the strides no longer match the simple "row-major" pattern .view() requires.
        """

        B, H, T, d_head = x.shape
        d_model = H * d_head

        # (B, H, T, d_head) -> (B, T, H, d_head) -> (B, T, d_model)
        return x.transpose(1,2).contiguous().view(B, T, d_model)
    

    def forward(self, x_q, x_kv, mask=None):
        # TODO 3d: project -> split -> attend -> merge -> output projection
        """
        self.w_q is an nn.Linear object, and in PyTorch, calling a module like a function (self.w_q(x_q)) is how you run it — it's shorthand for self.w_q.__call__(x_q), which internally does x_q @ W.T (+ bias, but we set bias=False) and returns the result.
        """
        # 1. Project first, whole width: q = x_q @ W_q, k = x_kv @ W_k, v = x_kv @ W_v — each still (B, T, d_model), no heads yet.
        # 2. Then split each into heads: _split_heads(q) → (B, H, T, d_head), same for k, v.
        q = self._split_heads(self.w_q(x_q))
        k = self._split_heads(self.w_k(x_kv))
        v = self._split_heads(self.w_v(x_kv))

        # 3. Attend per head (your TODO 1 function, batched over B, H).
        out, attn = scaled_dot_product_attention(q, k, v, mask)

        # 4. Merge heads back → (B, T, d_model).
        # 5. Project once more through w_o to let heads mix.
        out = self.w_o(self._merge_heads(out))
        
        return out, attn

# ===========================================================================
# Checks -- run the file, work top to bottom, stop when one fails
# ===========================================================================

def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def report(name: str, diff: float, tol: float = 1e-5) -> None:
    print(f"  {'PASS' if diff < tol else 'FAIL'}  {name:<38} max abs diff: {diff:.2e}")


section("Check 1: scaled_dot_product_attention vs torch")

B, H, T, Dh = 2, 4, 6, 8
q, k, v = (torch.randn(B, H, T, Dh) for _ in range(3))

try:
    mine, attn = scaled_dot_product_attention(q, k, v)
    report("unmasked", (mine - F.scaled_dot_product_attention(q, k, v)).abs().max().item())
    print(f"        attn rows sum to 1: "
          f"{torch.allclose(attn.sum(-1), torch.ones(B, H, T), atol=1e-5)}")
except NotImplementedError as e:
    print(f"  SKIP  {e}")

section("Check 2: causal_mask")

try:
    cm = causal_mask(4)
    print(f"  shape {tuple(cm.shape)}, dtype {cm.dtype}")
    print(f"  {'PASS' if cm.dtype is torch.bool else 'FAIL'}  dtype is bool")
    print(f"  {'PASS' if not cm.diagonal().any() else 'FAIL'}  diagonal is False (self-attention allowed)")
    print(f"  {'PASS' if cm.sum().item() == 6 else 'FAIL'}  6 blocked entries for seq_len=4 "
          f"(got {cm.sum().item()})")
    print(f"  {'PASS' if bool(cm[0, 1]) and not bool(cm[1, 0]) else 'FAIL'}  "
          f"blocks the future, not the past")
except NotImplementedError as e:
    print(f"  SKIP  {e}")

section("Check 3: masked attention vs torch")

# MASK CONVENTION GOTCHA -- worth knowing cold, torch is inconsistent with ITSELF here:
#   F.scaled_dot_product_attention : bool attn_mask  True = ALLOWED
#   nn.MultiheadAttention          : bool attn_mask  True = BLOCKED
# Ours follows nn.MultiheadAttention (True = blocked), hence the ~ when calling F.
try:
    cm_t = causal_mask(T)
    mine_c, attn_c = scaled_dot_product_attention(q, k, v, cm_t)
    report("vs is_causal=True",
           (mine_c - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max().item())
    report("vs attn_mask=~mask",
           (mine_c - F.scaled_dot_product_attention(q, k, v, attn_mask=~cm_t)).abs().max().item())
    blocked = attn_c.masked_select(cm_t.expand_as(attn_c))
    print(f"  {'PASS' if bool((blocked == 0).all()) else 'FAIL'}  future weights are EXACTLY zero "
          f"(max {blocked.max().item():.1e})")
except NotImplementedError as e:
    print(f"  SKIP  {e}")

section("Check 4: MultiHeadAttention vs nn.MultiheadAttention")

try:
    mha = MultiHeadAttention(D_MODEL, N_HEADS)
    ref = nn.MultiheadAttention(D_MODEL, N_HEADS, bias=False, batch_first=True)

    # torch packs Q, K, V into one (3*d_model, d_model) matrix, in that order
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.w_q.weight, mha.w_k.weight, mha.w_v.weight], 0))
        ref.out_proj.weight.copy_(mha.w_o.weight)

    x = torch.randn(2, SEQ_LEN, D_MODEL)

    mine_out, mine_attn = mha(x, x)
    ref_out, ref_attn = ref(x, x, x, need_weights=True, average_attn_weights=True)
    report("self-attention output", (mine_out - ref_out).abs().max().item())
    report("attention weights (head-averaged)", (mine_attn.mean(1) - ref_attn).abs().max().item())

    cm_full = causal_mask(SEQ_LEN)
    mine_cm, _ = mha(x, x, cm_full)
    ref_cm, _ = ref(x, x, x, attn_mask=cm_full, need_weights=False)  # same convention: True = blocked
    report("causal-masked output", (mine_cm - ref_cm).abs().max().item())
except NotImplementedError as e:
    print(f"  SKIP  {e}")

section("Check 5: cross-attention -- shapes only, no reference needed")

# 24 audio frames vs 10 video frames: the realistic case (16 kHz mic vs 30-90 fps camera).
# Attention handles the rate mismatch for free. Before running this, predict both shapes.
try:
    audio = torch.randn(2, 24, D_MODEL)
    video = torch.randn(2, 10, D_MODEL)
    out, attn_x = mha(audio, video)
    print(f"  query  (audio)   {tuple(audio.shape)}")
    print(f"  key/val (video)  {tuple(video.shape)}")
    print(f"  {'PASS' if out.shape == (2, 24, D_MODEL) else 'FAIL'}  output {tuple(out.shape)} "
          f"-- follows the QUERY length")
    print(f"  {'PASS' if attn_x.shape == (2, N_HEADS, 24, 10) else 'FAIL'}  attn "
          f"{tuple(attn_x.shape)} -- (B, H, T_audio, T_video), a learned soft alignment")
except (NotImplementedError, NameError) as e:
    print(f"  SKIP  {e}")

print("""
Once everything passes: diff against exe8 to catch anything you got right by accident,
then answer questions (a)-(d) out loud again. Then stop -- do NOT re-type the rest of exe8.
""")
