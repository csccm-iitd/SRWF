import math

import numpy as np
import pytest
import torch

from srwf import RFFGP, RWFGP, BayesianLinearRegression, MixtureGP, RandomWaveletFeatures
from srwf.blr import cholesky
from srwf.data.synthetic import multi_step_dataset
from srwf.metrics import evaluate, gaussian_crps


def _exact_gp(z, y, noise, prior_var, zs):
    k = prior_var * z @ z.T + noise * np.eye(len(y))
    ks = prior_var * zs @ z.T
    mean = ks @ np.linalg.solve(k, y)
    var = prior_var * (zs * zs).sum(1) - np.einsum("ij,ji->i", ks, np.linalg.solve(k, ks.T)) + noise
    lml = -0.5 * y @ np.linalg.solve(k, y) - 0.5 * np.linalg.slogdet(k)[1] - 0.5 * len(y) * math.log(2 * math.pi)
    return mean, var, lml


@pytest.mark.parametrize("n", [50, 400])
def test_blr_matches_function_space_gp(n):
    torch.manual_seed(0)
    x = torch.randn(n, 2, dtype=torch.float64)
    y = torch.sin(3 * x[:, 0]) + 0.1 * torch.randn(n, dtype=torch.float64)
    xs = torch.randn(30, 2, dtype=torch.float64)
    f = RandomWaveletFeatures(2, 128, levels=3).double().initialize(x)
    blr = BayesianLinearRegression(f, noise=0.05, prior_var=0.7).double()
    blr.condition(x, y)
    mean, var = blr.predict(xs)
    with torch.no_grad():
        z, zs = f(x).numpy(), f(xs).numpy()
    m_ref, v_ref, lml_ref = _exact_gp(z, y.numpy(), 0.05, 0.7, zs)
    np.testing.assert_allclose(mean.numpy(), m_ref, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(var.numpy(), v_ref, rtol=1e-6, atol=1e-8)
    with torch.no_grad():
        lml = float(blr.log_marginal_likelihood(x, y))
        lml_primal = float(blr.log_marginal_likelihood_from_stats(blr.stats(x, y)))
    assert lml == pytest.approx(lml_ref, rel=1e-8)
    assert lml_primal == pytest.approx(lml_ref, rel=1e-8)


def test_loo_matches_refitting():
    torch.manual_seed(1)
    x = torch.randn(60, 1, dtype=torch.float64)
    y = torch.cos(2 * x[:, 0]) + 0.1 * torch.randn(60, dtype=torch.float64)
    f = RandomWaveletFeatures(1, 32, levels=3).double().initialize(x)
    blr = BayesianLinearRegression(f, noise=0.02).double()
    with torch.no_grad():
        mu, v = blr.loo(x, y)
    for i in (0, 17, 59):
        keep = torch.arange(60) != i
        blr.condition(x[keep], y[keep])
        m_i, v_i = blr.predict(x[i : i + 1])
        assert float(mu[i]) == pytest.approx(float(m_i), rel=1e-7)
        assert float(v[i]) == pytest.approx(float(v_i), rel=1e-7)


def test_chunked_statistics_are_exact():
    torch.manual_seed(2)
    x = torch.randn(1000, 3, dtype=torch.float64)
    y = torch.randn(1000, dtype=torch.float64)
    blr = BayesianLinearRegression(RandomWaveletFeatures(3, 64).double().initialize(x)).double()
    with torch.no_grad():
        a, b = blr.stats(x, y, chunk=1000), blr.stats(x, y, chunk=37)
    assert torch.allclose(a.ztz, b.ztz) and torch.allclose(a.ztr, b.ztr) and a.n == b.n


def test_crps_matches_monte_carlo():
    rng = np.random.default_rng(0)
    y, m, s = 0.3, -0.2, 0.7
    draws = m + s * rng.standard_normal(400000)
    mc = np.mean(np.abs(draws - y)) - 0.5 * np.mean(np.abs(draws[::2] - draws[1::2]))
    assert gaussian_crps([y], [m], [s]) == pytest.approx(mc, abs=3e-3)


def test_rwf_beats_rff_on_multi_step():
    xtr, ytr, xte, yte = multi_step_dataset(n_train=1500, n_test=500, seed=0)
    rwf = RWFGP(num_features=512, levels=8, steps=150, device="cpu").fit(xtr, ytr)
    rff = RFFGP(num_features=512, steps=150, device="cpu").fit(xtr, ytr)
    e_rwf = evaluate(yte, *_ms(rwf.predict(xte)))
    e_rff = evaluate(yte, *_ms(rff.predict(xte)))
    assert e_rwf["rmse"] < 0.6 * e_rff["rmse"]
    assert e_rwf["nll"] < e_rff["nll"]
    assert 0.85 < e_rwf["cov95"] <= 1.0


def test_predict_shapes_and_samples():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(200, 2))
    y = np.sin(4 * x[:, 0]) * x[:, 1]
    m = RWFGP(num_features=128, levels=3, steps=20, device="cpu", anchors=4).fit(x, y)
    p = m.predict(x[:7], return_f_std=True)
    assert p.mean.shape == p.std.shape == p.f_std.shape == (7,)
    assert np.all(p.std >= p.f_std)
    assert m.sample_posterior(x[:7], 5).shape == (5, 7)
    assert m.sample_prior(x[:7], 3).shape == (3, 7)


def _ms(p):
    return p.mean, p.std


def test_warm_start_reuses_hyperparameters():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(150, 2))
    y = np.sin(5 * x[:, 0])
    fitted = RWFGP(num_features=128, levels=3, steps=50, device="cpu").fit(x, y)
    warm = RWFGP(num_features=128, levels=3, steps=0, device="cpu").fit(x, y, warm_start=fitted)
    np.testing.assert_allclose(fitted.predict(x[:5]).mean, warm.predict(x[:5]).mean, atol=1e-5)


def test_cholesky_jitter_leaves_input_unchanged():
    z = torch.randn(5, 40, dtype=torch.float64)
    a = z.t() @ z
    before = a.clone()
    chol = cholesky(a)
    assert torch.equal(a, before)
    assert torch.allclose(chol @ chol.t(), a, atol=1e-4 * float(a.diagonal().mean()))


def test_mixture_is_moment_matched_gaussian_mixture():
    rng = np.random.default_rng(2)
    x = rng.uniform(size=(120, 2))
    y = np.sin(5 * x[:, 0]) + 0.05 * rng.normal(size=120)
    kw = dict(num_features=96, levels=3, steps=10, device="cpu")
    mix = MixtureGP.of(RWFGP, members=3, seed=4, **kw).fit(x, y)
    p = mix.predict(x[:9], return_f_std=True)
    single = [RWFGP(seed=4 + k, **kw).fit(x, y).predict(x[:9]) for k in range(3)]
    np.testing.assert_allclose(single[0].mean, mix.models[0].predict(x[:9]).mean)
    mean = np.mean([q.mean for q in single], 0)
    second = np.mean([q.std**2 + q.mean**2 for q in single], 0)
    np.testing.assert_allclose(p.mean, mean, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(p.std**2, second - mean**2, rtol=1e-6, atol=1e-8)
    assert np.all(p.std >= p.f_std)
    assert mix.sample_posterior(x[:9], 5).shape == (5, 9)
    one = MixtureGP.of(RWFGP, members=1, seed=4, **kw).fit(x, y).predict(x[:9])
    np.testing.assert_allclose(one.std, single[0].std, rtol=1e-6)


def test_dual_solver_matches_primal_posterior():
    torch.manual_seed(1)
    x = torch.randn(60, 2, dtype=torch.float64)
    y = torch.cos(2 * x[:, 1]) + 0.1 * torch.randn(60, dtype=torch.float64)
    xs = torch.randn(25, 2, dtype=torch.float64)
    f = RandomWaveletFeatures(2, 200, levels=3).double().initialize(x)
    primal = BayesianLinearRegression(f, noise=0.05, prior_var=0.7).double().condition(x, y)
    dual = BayesianLinearRegression(f, noise=0.05, prior_var=0.7).double().condition(x, y, solver="dual")
    assert dual.dual and not primal.dual
    for include_noise in (True, False):
        expected, got = primal.predict(xs, include_noise=include_noise), dual.predict(xs, include_noise=include_noise)
        for a, b in zip(expected, got, strict=True):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(primal.mean_w.numpy(), dual.mean_w.numpy(), rtol=1e-6, atol=1e-9)
    w = dual.posterior_weights(20000, generator=torch.Generator().manual_seed(0))
    zs = f(xs).detach()
    draws = zs @ w
    _, var = primal.predict(xs, include_noise=False)
    np.testing.assert_allclose(draws.mean(1).numpy(), (zs @ primal.mean_w).numpy(), atol=0.05)
    np.testing.assert_allclose(draws.var(1).numpy(), var.numpy(), rtol=0.08, atol=1e-3)
    m = RWFGP(num_features=128, levels=3, steps=5, device="cpu", solver="dual").fit(x.numpy(), y.numpy())
    assert m.blr.dual and m.sample_posterior(xs.numpy(), 3).shape == (3, 25)


def test_restarts_keep_the_best_feature_draw():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(100, 2))
    y = np.sin(6 * x[:, 0]) + 0.1 * rng.normal(size=100)
    kw = dict(num_features=64, levels=3, steps=15, device="cpu")
    single = [RWFGP(seed=10 + k, **kw).fit(x, y) for k in range(3)]
    best = int(np.argmin([min(m.info.losses) for m in single]))
    multi = RWFGP(seed=10, restarts=3, **kw).fit(x, y)
    assert min(multi.info.losses) == pytest.approx(min(single[best].info.losses))
    np.testing.assert_allclose(multi.predict(x[:6]).mean, single[best].predict(x[:6]).mean, rtol=1e-6, atol=1e-8)
