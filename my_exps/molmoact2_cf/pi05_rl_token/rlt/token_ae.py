"""RL token: an encoder-decoder bottleneck over frozen-VLA embeddings.

Paper Sec. III-A. The encoder appends a learned <rl> embedding to the VLA's
final-layer token sequence and reads out that position. The decoder must
reconstruct the whole sequence from that one vector, which is what forces the
readout to stay informative:

    z_rl = g([z_1..z_M, e_rl])_{M+1}                                  (Eq. 1)
    L_ro = E_D[ sum_i || h(d([z_rl, z_1..z_{i-1}]))_i - z_i ||^2 ]    (Eq. 2)

VLA embeddings enter as stop-gradient targets (z-bar in the paper). We never use
the optional alpha * L_vla term: the VLA is frozen throughout.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import VLA_TOKEN_DIM


def sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    """Fixed positional encoding, (length, dim)."""
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(dim, device=device, dtype=dtype).unsqueeze(0)
    angles = pos / (10000 ** (2 * (i // 2) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class TokenEncoder(nn.Module):
    """VLA token sequence -> z_rl."""

    def __init__(
        self,
        token_dim: int = VLA_TOKEN_DIM,
        z_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(token_dim, d_model)
        self.rl_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_dim))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens (B,S,token_dim), mask (B,S) 1=valid -> z_rl (B,z_dim)."""
        b, s, _ = tokens.shape
        x = self.in_proj(tokens)
        rl = self.rl_embed.view(1, 1, -1).expand(b, 1, -1)
        x = torch.cat([x, rl], dim=1)  # <rl> last, so we read position M+1
        x = x + sinusoidal_positions(s + 1, self.d_model, x.device, x.dtype)

        pad = None
        if mask is not None:
            # <rl> is always valid. PyTorch wants True = ignore.
            pad = ~torch.cat([mask, mask.new_ones(b, 1)], dim=1).bool()

        return self.out_proj(self.encoder(x, src_key_padding_mask=pad)[:, -1])


class TokenDecoder(nn.Module):
    """z_rl -> reconstructed token sequence, teacher-forced and causal."""

    def __init__(
        self,
        token_dim: int = VLA_TOKEN_DIM,
        z_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.z_proj = nn.Linear(z_dim, d_model)
        self.tok_proj = nn.Linear(token_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, token_dim)

    def forward(self, z_rl: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """z_rl (B,z_dim) + targets (B,S,token_dim) -> predictions (B,S,token_dim).

        Position i sees z_rl (cross-attention memory) and targets[:i] only; z_rl
        doubles as the start token. So everything the decoder knows about the
        sequence has to pass through the bottleneck.
        """
        b, s, _ = targets.shape
        memory = self.z_proj(z_rl).unsqueeze(1)  # (B,1,d)

        tgt = memory if s == 1 else torch.cat([memory, self.tok_proj(targets[:, :-1])], dim=1)
        tgt = tgt + sinusoidal_positions(s, self.d_model, tgt.device, tgt.dtype)

        causal = nn.Transformer.generate_square_subsequent_mask(s, device=tgt.device)
        return self.out_proj(self.decoder(tgt, memory, tgt_mask=causal))


class RLTokenAE(nn.Module):
    """Encoder + decoder trained jointly by L_ro. Only the encoder is used later."""

    def __init__(
        self,
        token_dim: int = VLA_TOKEN_DIM,
        z_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.config = dict(
            token_dim=token_dim, z_dim=z_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers
        )
        self.z_dim = z_dim
        self.encoder = TokenEncoder(**self.config)
        self.decoder = TokenDecoder(**self.config)

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(tokens, mask)

    def reconstruction_loss(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """L_ro (Eq. 2). `tokens` are VLA outputs, so they are targets only."""
        targets = tokens.detach()  # z-bar: the VLA never learns here
        z = self.encode(targets, mask)
        pred = self.decoder(z, targets)

        err = (pred - targets).pow(2).mean(dim=-1)  # (B,S)
        if mask is None:
            loss = err.mean()
        else:
            w = mask.to(err.dtype)
            loss = (err * w).sum() / w.sum().clamp_min(1.0)

        return loss, {"recon_loss": loss.item(), "z_norm": z.norm(dim=-1).mean().item()}

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "RLTokenAE":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        # load_state_dict copies into the freshly constructed CPU module even when
        # the file was mapped onto a GPU. Put the weights where the caller asked.
        return model.to(map_location)


def write_random_ae(path: str, **kwargs) -> "RLTokenAE":
    """A randomly initialized AE so a run can train the encoder from scratch."""
    model = RLTokenAE(**kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return model
