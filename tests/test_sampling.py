import numpy as np
import pytest
import torch

from srwf import RFFGP, RWFGP
from srwf.sampling import PosteriorFunction


def _data(n: int = 40, d: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 3, size=(n, d))
    y = np.sin(3 * x[:, 0]) * np.exp(-0.3 * x[:, 0] ** 2) + 0.05 * rng.standard_normal(n) + 4.0
    return x, y


@pytest.mark.parametrize("cls", [RWFGP, RFFGP])
def test_posterior_moments_match_predict(cls):
    x, y = _data()
    model = cls(num_features=64, steps=30, device="cpu", seed=0).fit(x, y)
    xs = np.random.default_rng(1).uniform(-2, 3, size=(25, 2))
    f = PosteriorFunction(model, n_samples=20000, seed=3)
    s = f.numpy(xs)
    pred = model.predict(xs, return_f_std=True)
    se = pred.f_std / np.sqrt(s.shape[0])
    assert np.all(np.abs(s.mean(0) - pred.mean) < 5 * se + 1e-6)
    np.testing.assert_allclose(s.std(0), pred.f_std, rtol=0.05)

    xt = torch.as_tensor(xs, dtype=torch.float32)
    with torch.no_grad():
        z = model.blr.features(model._x(xt)).double()
    cov = (z @ torch.cholesky_inverse(model.blr.chol) @ z.t()).numpy() * model.y_sd**2
    emp = np.cov(s, rowvar=False)
    assert np.abs(emp - cov).max() < 0.05 * np.abs(cov).max()


def test_functions_are_consistent_and_paired():
    x, y = _data()
    model = RWFGP(num_features=64, steps=10, device="cpu").fit(x, y)
    f = PosteriorFunction(model, n_samples=4, seed=0)
    xs = torch.rand(10, 2)
    full = f(xs)
    paired = f(xs.unsqueeze(0).expand(4, -1, -1).contiguous())
    torch.testing.assert_close(full, paired)
    torch.testing.assert_close(full[:, :3], f(xs[:3]))
    again = PosteriorFunction(model, n_samples=4, seed=0)(xs)
    torch.testing.assert_close(full, again)


def test_gradient_matches_finite_differences():
    x, y = _data()
    model = RWFGP(num_features=64, steps=10, device="cpu").fit(x, y)
    f = PosteriorFunction(model, n_samples=3, seed=0)
    xs = torch.rand(3, 5, 2, dtype=torch.float32)
    g = f.gradient(xs).double()
    h = 1e-2
    for j in range(2):
        e = torch.zeros(2)
        e[j] = h
        with torch.no_grad():
            fd = (f(xs + e) - f(xs - e)) / (2 * h)
        torch.testing.assert_close(g[..., j], fd, rtol=2e-2, atol=2e-2)


def test_prior_samples_have_prior_variance():
    x, y = _data()
    model = RWFGP(num_features=64, steps=10, device="cpu").fit(x, y)
    xs = np.random.default_rng(2).uniform(-2, 3, size=(30, 2))
    s = PosteriorFunction(model, n_samples=20000, prior=True, seed=1).numpy(xs)
    xt = model._x(torch.as_tensor(xs, dtype=torch.float32))
    with torch.no_grad():
        z = model.blr.features(xt).double()
    var = (z.pow(2).sum(1) * model.blr.prior_var).detach().numpy() * model.y_sd**2
    np.testing.assert_allclose(s.var(0), var, rtol=0.06)
    np.testing.assert_allclose(s.mean(0), model.y_mu, atol=5 * np.sqrt(var.max() / 20000))
