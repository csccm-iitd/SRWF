import numpy as np
import torch

from srwf import RWFGP
from srwf.bo import FeatureSampler, maximise, sobol, thompson_sampling


def test_maximise_finds_known_maxima():
    centres = torch.tensor([[0.2, -0.5], [0.7, 0.3], [-0.4, 0.8]])

    def f(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return -(x.unsqueeze(0) - centres[:, None]).pow(2).sum(-1)
        return -(x - centres[:, None]).pow(2).sum(-1)

    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    x, v = maximise(f, bounds, n_candidates=256, n_local=0, steps=100, lr=0.02)
    np.testing.assert_allclose(x, centres.numpy(), atol=1e-2)
    assert np.all(v <= 0) and np.all(v > -1e-3)


def test_maximise_respects_bounds():
    bounds = np.array([[0.0, 1.0]])
    f = lambda x: x[..., 0].reshape(1, -1) if x.ndim == 2 else x[..., 0]  # noqa: E731
    x, _ = maximise(f, bounds, n_candidates=64, n_local=0, steps=20, lr=0.1)
    assert 0.99 <= x[0, 0] <= 1.0


def test_sobol_within_bounds():
    b = np.array([[-5.0, 10.0], [0.0, 15.0]])
    x = sobol(100, b, seed=0)
    assert x.shape == (100, 2) and np.all(x >= b[:, 0]) and np.all(x <= b[:, 1])


def test_thompson_sampling_improves_on_quadratic():
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    objective = lambda x: -((x - np.array([0.6, -0.8])) ** 2).sum(1)  # noqa: E731
    sampler = FeatureSampler(lambda x: RWFGP(128, 3, steps=30, device="cpu"))
    trace = thompson_sampling(objective, bounds, sampler, budget=20, n_init=5, batch_size=3, seed=0, n_candidates=256)
    assert len(trace.y) == 20 and len(trace.fit_seconds) == 5
    assert trace.best[-1] > max(trace.best[4], -0.1)


def test_log_expected_improvement_matches_closed_form():
    from scipy.stats import norm

    from srwf.bo import log_expected_improvement

    u = torch.linspace(-30.0, 5.0, 71, dtype=torch.float64)
    std = torch.full_like(u, 0.3)
    got = log_expected_improvement(u * std + 1.0, std, best=1.0).numpy()
    want = np.log(0.3 * (u.numpy() * norm.cdf(u.numpy()) + norm.pdf(u.numpy())))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)
    far = log_expected_improvement(
        torch.tensor([-1e4], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64), 0.0
    )
    assert torch.isfinite(far).all()


def test_posterior_moments_match_predict_and_differentiate():
    from srwf.bo import PosteriorMoments

    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(30, 2))
    y = np.sin(x[:, 0]) + 0.1 * x[:, 1] + 3.0
    model = RWFGP(num_features=64, steps=20, device="cpu").fit(x, y)
    xs = torch.as_tensor(rng.uniform(-2, 2, size=(7, 2)), dtype=torch.float32).requires_grad_(True)
    mean, std = PosteriorMoments(model)(xs)
    pred = model.predict(xs.detach().numpy(), return_f_std=True)
    np.testing.assert_allclose(mean.detach().numpy(), pred.mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(std.detach().numpy(), pred.f_std, rtol=1e-4, atol=1e-6)
    (g,) = torch.autograd.grad((mean + std).sum(), xs)
    assert g.shape == xs.shape and torch.isfinite(g).all()
