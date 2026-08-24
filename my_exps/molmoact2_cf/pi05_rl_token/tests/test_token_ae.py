"""The RL token bottleneck. The two tests that matter are padding isolation and
decoder causality -- both fail silently and would quietly corrupt z."""

import numpy as np
import torch

from rlt.token_ae import RLTokenAE, write_random_ae
from rlt.vla import TokenEncoder

B, S, D, Z = 3, 17, 2560, 32


def build() -> RLTokenAE:
    torch.manual_seed(0)
    return RLTokenAE(token_dim=D, z_dim=Z, d_model=64, n_heads=4, n_layers=2)


def masked_batch():
    tokens = torch.randn(B, S, D)
    mask = torch.ones(B, S)
    mask[1, 12:] = 0
    mask[2, 5:] = 0
    return tokens, mask


def test_encode_shape_and_gradients():
    ae = build()
    tokens, mask = masked_batch()
    assert ae.encode(tokens, mask).shape == (B, Z)

    loss, stats = ae.reconstruction_loss(tokens, mask)
    loss.backward()
    assert not [n for n, p in ae.named_parameters() if p.grad is None]
    assert stats["recon_loss"] > 0


def test_padded_positions_do_not_reach_z():
    """Junk beyond the mask must not move the readout."""
    ae = build()
    tokens, mask = masked_batch()
    with torch.no_grad():
        base = ae.encode(tokens, mask)
        noisy = tokens.clone()
        noisy[1, 12:] += 100.0
        noisy[2, 5:] -= 250.0
        after = ae.encode(noisy, mask)
    assert (base - after).abs().max().item() < 1e-4


def test_decoder_is_causal():
    """Prediction i must not see target i, or reconstruction is trivial."""
    ae = build()
    tokens, mask = masked_batch()
    with torch.no_grad():
        z = ae.encode(tokens, mask)
        base = ae.decoder(z, tokens)
        bumped = tokens.clone()
        bumped[:, -1] += 50.0
        after = ae.decoder(z, bumped)
    assert (base[:, -1] - after[:, -1]).abs().max().item() < 1e-4


def test_vla_embeddings_stay_frozen():
    """L_ro must not push gradients back into the VLA outputs (z-bar in Eq. 2)."""
    ae = build()
    tokens = torch.randn(B, S, D, requires_grad=True)
    loss, _ = ae.reconstruction_loss(tokens, torch.ones(B, S))
    loss.backward()
    assert tokens.grad is None


def test_bottleneck_actually_compresses():
    ae = build()
    assert ae.z_dim < D  # 32 << 2560; the point of the readout


def test_save_load_roundtrip(tmp_path):
    ae = build()
    tokens, mask = masked_batch()
    with torch.no_grad():
        before = ae.encode(tokens, mask)

    path = tmp_path / "ae.pt"
    ae.save(str(path))
    with torch.no_grad():
        after = RLTokenAE.load(str(path)).encode(tokens, mask)
    assert torch.allclose(before, after, atol=1e-6)


def test_load_puts_the_module_on_the_requested_device(tmp_path):
    path = tmp_path / "ae.pt"
    build().save(str(path))
    loaded = RLTokenAE.load(str(path), map_location="cpu")
    assert next(loaded.parameters()).device.type == "cpu"


def test_load_map_location_cuda_puts_weights_on_cuda(tmp_path):
    if not torch.cuda.is_available():
        return
    ae = build()
    path = tmp_path / "ae.pt"
    ae.save(str(path))
    loaded = RLTokenAE.load(str(path), map_location="cuda:0")
    assert next(loaded.parameters()).device.type == "cuda"


def test_token_encoder_moves_cpu_module_onto_its_device(tmp_path):
    """The mug AC crash: shared encoder left on CPU, tokens on CUDA."""
    if not torch.cuda.is_available():
        return
    path = tmp_path / "ae.pt"
    write_random_ae(str(path), token_dim=32, z_dim=8, d_model=32, n_heads=4, n_layers=1)
    enc = TokenEncoder(str(path), "cuda:0")
    enc.encoder.cpu()
    z = enc.encode(np.zeros((4, 32), np.float32), np.ones(4, np.float32))
    assert z.shape == (8,)
    assert next(enc.encoder.parameters()).device.type == "cuda"
