"""Numerical checks of the theoretical properties of Random Wavelet Features."""

import math

import numpy as np
import pytest
import torch

from srwf.features import RandomWaveletFeatures
from srwf.wavelets import WAVELETS

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("name", ["mexican_hat", "morlet"])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_mother_wavelet_has_zero_mean(name, d):
    w = WAVELETS[name]
    g = torch.Generator().manual_seed(0)
    u = torch.randn(400000, d, generator=g) * 3.0
    dens = torch.exp(-0.5 * u.pow(2).sum(1) / 9.0) / (2 * math.pi * 9.0) ** (d / 2)
    v = torch.zeros(d)
    v[0] = 1.0
    vals = w.fn(u.pow(2).sum(1), u @ v, d) if name == "mexican_hat" else w.fn(u.pow(2).sum(1), u @ v, d, 5.0)
    integral = (vals / dens).mean()
    scale = (vals.abs() / dens).mean()
    assert abs(float(integral)) < 0.02 * float(scale)


@pytest.mark.parametrize("wavelet", ["mexican_hat", "morlet"])
@pytest.mark.parametrize("anchors", [0, 6])
def test_gram_matrix_is_positive_semidefinite(wavelet, anchors):
    torch.manual_seed(0)
    x = torch.randn(300, 3)
    f = RandomWaveletFeatures(3, 256, levels=4, wavelet=wavelet, anchors=anchors).initialize(x)
    with torch.no_grad():
        if anchors:
            f.anchor_log_weight.normal_(0.0, 2.0)
        f.level_log_weight.normal_(0.0, 1.0)
        k = f.kernel(x, x)
    assert torch.allclose(k, k.t(), atol=1e-10)
    assert float(torch.linalg.eigvalsh(k).min()) > -1e-8 * float(k.diagonal().max())


def _kernel_estimate(seed, x1, x2, d=2, n=64):
    f = RandomWaveletFeatures(d, n, levels=3, seed=seed)
    f.initialize(torch.cat([x1, x2]), centres="box", base_scale=1.0)
    f.block_rms.fill_(1.0)
    with torch.no_grad():
        return (f(x1) * f(x2)).sum(1)


def test_unbiased_and_variance_decays_as_one_over_d():
    """E[k_hat] = k and Var[k_hat] = O(1/D)."""
    torch.manual_seed(1)
    x1, x2 = torch.rand(5, 2), torch.rand(5, 2)
    small = torch.stack([_kernel_estimate(s, x1, x2, n=32) for s in range(400)])
    big = torch.stack([_kernel_estimate(s, x1, x2, n=128) for s in range(400)])
    ratio = (small.var(0) / big.var(0)).mean()
    assert 2.5 < float(ratio) < 6.0
    se = (small.std(0) / math.sqrt(400)).max()
    assert torch.allclose(small.mean(0), big.mean(0), atol=float(5 * se))


def test_uniform_error_decreases_with_feature_count():
    """The sup-norm approximation error shrinks as D grows."""
    torch.manual_seed(2)
    x = torch.rand(120, 2)
    ref = _feature_kernel(x, 16384, seed=999)
    errs = []
    for n in (128, 512, 2048):
        e = np.mean([float((_feature_kernel(x, n, seed=s) - ref).abs().max()) for s in range(5)])
        errs.append(e)
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] / errs[2] > 2.5


def _feature_kernel(x, n, seed):
    f = RandomWaveletFeatures(2, n, levels=3, growth=1.0, seed=seed)
    f.initialize(x, centres="box", base_scale=1.0)
    f.block_rms.fill_(1.0)
    with torch.no_grad():
        return f.kernel(x, x)


def test_kernel_is_non_stationary():
    torch.manual_seed(3)
    x = torch.rand(2000, 1)
    f = RandomWaveletFeatures(1, 512, levels=4, anchors=4).initialize(x)
    with torch.no_grad():
        f.anchor_log_weight[-1].copy_(torch.tensor([4.0, 0.0, 0.0, -4.0]))
        a = torch.tensor([[0.1], [0.11]])
        b = torch.tensor([[0.9], [0.91]])
        ka, kb = f.kernel(a, a), f.kernel(b, b)
    assert not torch.allclose(ka, kb, rtol=0.1)
