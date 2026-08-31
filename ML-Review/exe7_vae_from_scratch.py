"""
7. VAE (Variational Autoencoder) from scratch, ~30 min

Encoder -> reparameterization trick -> decoder -> ELBO loss (reconstruction + KL).
Reuses exe5's synthetic 16x16 shape dataset (plus / diagonal cross / square outline) so
you build directly on the CNN encoder/decoder pattern from Day 6, instead of a new dataset.

This is the hands-on follow-through of the KL-divergence theory in
`Theory/Entropy-CrossEntropy-KLDivergence.md` (section 4) -- the closed-form Gaussian KL
term you derived there is exactly what regularizes this model's latent space so that
sampling z ~ N(0,I) at generation time actually produces something shape-like.

Fill in the TODOs: encoder, reparameterization trick, decoder, ELBO loss, training loop.
Target: reconstructions that visibly resemble the input shape; samples drawn from
z ~ N(0,I) and decoded should look shape-like, not pure noise -- that's the real test,
reconstruction quality alone doesn't prove the latent space is well-regularized.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # this machine has a libiomp/libomp DLL conflict; without this, `import torch` crashes

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

SIZE = 16
LATENT_DIM = 8


# --- dataset (same generators as exe5) ---
def make_plus(size=SIZE):
    img = torch.zeros(size, size)
    c = size // 2
    img[c - 1:c + 1, :] = 1.0
    img[:, c - 1:c + 1] = 1.0
    return img


def make_cross(size=SIZE):
    img = torch.eye(size) + torch.eye(size).flip(0)
    return img.clamp(0, 1)


def make_square(size=SIZE, margin=3):
    img = torch.zeros(size, size)
    img[margin, margin:size - margin] = 1.0
    img[size - 1 - margin, margin:size - margin] = 1.0
    img[margin:size - margin, margin] = 1.0
    img[margin:size - margin, size - 1 - margin] = 1.0
    return img


def make_dataset(n_per_class: int = 150, noise: float = 0.1):
    generators = [make_plus, make_cross, make_square]
    images = []
    for gen in generators:
        base = gen()
        for _ in range(n_per_class):
            images.append((base + torch.randn(SIZE, SIZE) * noise).clamp(0, 1))
    x = torch.stack(images).unsqueeze(1)  # (N, 1, 16, 16)
    return x[torch.randperm(len(x))]


x = make_dataset()
n_train = int(0.9 * len(x))
x_train, x_val = x[:n_train], x[n_train:]


class VAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        # --- Encoder: image -> (mu, logvar) ---
        # TODO 1: a small conv stack, e.g.
        #   Conv2d(1, 8, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)   # 16x16 -> 8x8
        #   Conv2d(8, 16, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)  # 8x8 -> 4x4
        # then flatten (16*4*4) into two separate Linear heads: fc_mu, fc_logvar,
        # each mapping flattened features -> latent_dim. (Reuse the shape-tracking logic
        # from exe4/exe5 if useful.)
        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        """
        work out flatten_dim after conv1 -> relu -> pool -> conv2 -> relu -> pool -> flatten
        (N,1,16,16) -> (N, 8, 16,16) -> (N,8, 8, 8) -> (N, 16, 8, 8) -> (N, 16, 4, 4) -> (N, 16*4*4)

        """

        flatten_dim = 16 * 4 * 4
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, latent_dim)

        # --- Decoder: z -> reconstructed image ---
        # TODO 2: mirror the encoder: Linear(latent_dim -> 16*4*4), reshape to (16,4,4),
        # then a small transposed-conv (or upsample+conv) stack back up to (1, 16, 16),
        # with a final Sigmoid so output pixels are in [0, 1] (matches BCE target range).
        
        self.fc = nn.Linear(latent_dim, flatten_dim)

        # Option 1
        # self.conv3 = nn.ConvTranspose2d(16, 8, kernel_size = 2, stride = 2)
        # self.conv4 = nn.ConvTranspose2d(8, 1, kernel_size = 2, stride = 2)
        # Option 2
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv3 = nn.Conv2d(16, 8, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(8, 1, kernel_size = 3, padding = 1)

        """
        work out flatten_dim after fc -> reshape -> upsample -> conv3 -> relu -> upsample -> conv4 -> sigmoid
        (N, 16*4*4) -> (N, 16, 4, 4) -> (N, 16, 8, 8) -> (N, 8, 8, 8) -> (N, 8, 16, 16) -> (N, 1, 16, 16)
        """

        self.sigmoid = nn.Sigmoid()


    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder conv stack + flatten. Returns (mu, logvar), each (batch, latent_dim)."""
        # TODO 3
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim = 1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        The reparameterization trick: sampling z ~ N(mu, sigma^2) directly isn't
        differentiable. Instead sample eps ~ N(0, I) and compute z = mu + sigma * eps --
        the randomness lives in `eps` (constant w.r.t. the network's parameters), so
        gradients can flow through mu and sigma during backward().
        """
        # TODO 4: std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + std * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Run the decoder. Returns reconstructed image, same shape as the input."""
        # TODO 5
        z = self.fc(z) # Takes (N, 8) -> (N, 256) = (N, 16*4*4)
        z = z.reshape(-1, 16, 4, 4) # (N, 16*4*4) -> (N, 16, 4, 4)
        z = self.relu(self.conv3(self.upsample(z))) # (N, 16, 4, 4) -> (N, 8, 8, 8) 
        z = self.conv4(self.upsample(z)) # (N, 8, 8, 8) -> (N, 1, 16, 16)
        return self.sigmoid(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_recon, mu, logvar)."""
        # TODO 6: encode -> reparameterize -> decode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return (x_recon, mu, logvar)

def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Negative ELBO = reconstruction loss + KL term.
      recon_loss: BCE between x_recon and x, summed over pixels (matches the Sigmoid decoder output)
      kl_loss:    closed-form KL(N(mu, sigma^2) || N(0, I)) for a diagonal Gaussian
                  = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                  (derivation: Theory/Entropy-CrossEntropy-KLDivergence.md, section 4)
    Return recon_loss + kl_loss (summed over the batch -- divide by batch size outside
    if you want a per-example average for logging).
    """

    """
    - F.binary_cross_entropy (what TODO 7 calls for): expects x_recon to already be a probability in [0, 1] — i.e. sigmoid has already been applied. It computes -[y·log(p) + (1-y)·log(1-p)] directly on those probabilities.
    - nn.BCEWithLogitsLoss (and its functional twin F.binary_cross_entropy_with_logits): expects raw, unbounded logits — the network's output before any sigmoid. It applies sigmoid internally, fused into the loss computation.

    Why the "with logits" version exists — numerical stability. If you compute sigmoid(x) then log(...) as two separate steps, and x is a very negative logit, sigmoid(x) can round to exactly 0.0 in floating point — then log(0) = -inf, and your loss/gradients become NaN. BCEWithLogitsLoss sidesteps this by never explicitly computing sigmoid(x); it uses the algebraically equivalent but numerically stable identity:

    BCE_with_logits(x, y) = max(x, 0) - x·y + log(1 + exp(-|x|))

    This is stable for any real x, however extreme, because the exp term only ever gets a non-positive argument.

    Why this exercise correctly uses plain binary_cross_entropy, not the logits version: your decode() already ends with self.sigmoid(z) — it has to, because decode() is also called directly on prior samples at the bottom of the file (generated = model.decode(z_sample)) where you want actual pixel probabilities out, not raw logits, to eyeball the generated shape. So x_recon arriving at vae_loss is already post-sigmoid, in [0,1]. Feeding that into BCEWithLogitsLoss would be a bug — it would apply sigmoid a second time to an already-squashed value, dragging everything toward 0.5 and corrupting the loss.

    The alternative design (common in bigger VAEs) is: drop the Sigmoid from the decoder entirely, have it output raw logits, use BCEWithLogitsLoss for training, and apply sigmoid manually only at the point you need actual pixel values (visualization/generation). Legitimate tradeoff, but unnecessary complexity for this toy 16×16 case — what you have is correct as-is.
    """

    # TODO 7: recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")

    # TODO 8: kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = vae_loss

for epoch in range(50):
    # TODO 9: standard training step over x_train (single batch is fine for this toy case):
    # zero_grad, forward, loss = vae_loss(...), backward, step
    
    x_recon, mu, logvar = model(x_train)
    loss = loss_fn(x_recon, x_train, mu, logvar)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.1f}" if loss is not None else f"epoch {epoch:3d}  (TODO 9 not implemented yet)")

# --- check 1: reconstruction quality on held-out val data ---
with torch.no_grad():
    out = model(x_val)
if out is not None:
    x_recon, mu, logvar = out
    recon_error = F.mse_loss(x_recon, x_val).item()
    print(f"\nval reconstruction MSE: {recon_error:.4f}")

# --- check 2: the actual generative test -- sample directly from the prior N(0, I) and
# decode. If the KL term did its job, this should look shape-like, not noise.
with torch.no_grad():
    z_sample = torch.randn(4, LATENT_DIM)
    generated = model.decode(z_sample)
if generated is not None:
    print(f"generated samples shape: {generated.shape}  (expect (4, 1, {SIZE}, {SIZE}))")
    print("Eyeball `generated` (e.g. plt.imshow(generated[i, 0])) -- should look shape-like, not pure noise.")
