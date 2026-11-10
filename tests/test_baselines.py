import numpy as np
import pytest
import torch

baselines = pytest.importorskip("srwf.baselines")


def _kernels(d):
    torch.manual_seed(0)
    gibbs = baselines.GibbsKernel(torch.randn(16, d))
    with torch.no_grad():
        gibbs.weights.normal_()
    return [gibbs, baselines.DeepKernel(d, hidden=(16,), features=2)]


@pytest.mark.parametrize("d", [1, 3])
def test_nonstationary_gram_is_psd(d):
    x = torch.randn(60, d)
    for k in _kernels(d):
        k = k.double()
        gram = k(x.double()).to_dense()
        assert gram.shape == (60, 60)
        torch.testing.assert_close(gram, gram.T)
        assert torch.linalg.eigvalsh(gram).min() > -1e-8
        torch.testing.assert_close(k(x.double(), diag=True), gram.diagonal())
        assert k(x[:7].double(), x[:5].double()).to_dense().shape == (7, 5)


def test_gibbs_lengthscale_is_positive_and_input_dependent():
    k = _kernels(1)[0]
    ell = k.lengthscale_at(torch.linspace(-3, 3, 50)[:, None])
    assert ell.shape == (50,) and bool((ell > 0).all()) and float(ell.std()) > 0


@pytest.mark.parametrize("kernel", ["gibbs", "dkl"])
def test_nonstationary_baselines_fit(kernel):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (80, 1))
    y = np.sin(6 * x[:, 0]) + 0.05 * rng.standard_normal(80)
    for model in (
        baselines.ExactGP(kernel=kernel, steps=30, device="cpu"),
        baselines.SVGP(num_inducing=16, kernel=kernel, epochs=10, batch_size=40, device="cpu"),
    ):
        p = model.fit(x, y).predict(x[:10])
        assert p.mean.shape == p.std.shape == (10,)
        assert np.all(np.isfinite(p.mean)) and np.all(p.std > 0)
