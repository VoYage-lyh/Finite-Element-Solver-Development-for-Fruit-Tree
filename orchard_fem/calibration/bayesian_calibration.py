"""Bayesian finite-element model updating for fruit-tree vibration models.

Implements Section 2.2 of the parameter-uncertainty harvesting paper:

  θ = (E, ρ, ζ_1, ζ_2, k_c, c_c, k_f, c_f) ∈ ℝ^8

with truncated-normal priors on ``(E, ρ)`` from material tests, log-uniform
priors on the remaining parameters, Gaussian likelihoods on the first three
modal frequencies and the log-FRF magnitudes, and ensemble-MCMC posterior
sampling. We use the affine-invariant ``emcee`` sampler rather than the
gradient-based NUTS named in the paper because the forward operator
(FEniCSx/PETSc beam solver) is not differentiable through autodiff. emcee
handles black-box log-posteriors directly and reaches comparable per-evaluation
mixing rates on this dimensionality.

The forward operator is supplied by the caller as a ``ForwardOperator``
callable that takes a parameter dict and returns a :class:`ForwardResult`
(modal frequencies + complex FRF on a user-specified frequency grid).
This decoupling makes the calibration testable with cheap stub solvers and
keeps the heavy FEniCSx import out of the calibration module's top level.

Example::

    from orchard_fem.calibration.bayesian_calibration import (
        BayesianPrior, BayesianLikelihood, run_emcee_calibration,
    )
    from orchard_fem.io.material_test import priors_from_material_test, ...

    priors = priors_from_material_test(material_summary)
    likelihood = BayesianLikelihood(
        modal_frequencies_hz=[5.84, 10.62, 17.95],
        frf_frequencies_hz=frf_freqs,
        frf_log_magnitudes=np.log(frf_mag),
    )
    posterior = run_emcee_calibration(forward_operator, priors, likelihood,
                                      n_walkers=32, n_steps=4000, n_burn=1500)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
#  Prior / likelihood / posterior types
# ────────────────────────────────────────────────────────────────────────────
PriorKind = Literal["truncnorm", "loguniform", "uniform"]


@dataclass(frozen=True)
class BayesianPrior:
    """One parameter's prior.

    Parameters
    ----------
    name:
        Parameter name. Used as the dict key passed to the forward operator
        (e.g. ``"E"``, ``"zeta1"``, ``"k_c"``).
    kind:
        ``"truncnorm"`` — truncated normal with ``mean``, ``sd``, ``bounds``;
        ``"loguniform"`` — uniform on ``log(bound[0])..log(bound[1])``;
        ``"uniform"``    — uniform on ``[bound[0], bound[1]]``.
    mean, sd:
        Required for ``truncnorm``; ignored otherwise.
    bounds:
        ``(lower, upper)`` inclusive support. Required for all kinds.
    """

    name: str
    kind: PriorKind
    bounds: tuple[float, float]
    mean: float | None = None
    sd: float | None = None

    def __post_init__(self) -> None:
        lo, hi = self.bounds
        if hi <= lo:
            raise ValueError(f"Prior {self.name}: bounds must be increasing.")
        if self.kind == "truncnorm" and (self.mean is None or self.sd is None):
            raise ValueError(f"Prior {self.name}: truncnorm requires mean and sd.")
        if self.kind == "loguniform" and lo <= 0.0:
            raise ValueError(f"Prior {self.name}: loguniform requires lower > 0.")

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one sample from the prior (for walker initialisation)."""
        lo, hi = self.bounds
        if self.kind == "truncnorm":
            assert self.mean is not None and self.sd is not None
            # Reject-sample to enforce bounds — simple and fast for usual ranges.
            for _ in range(100):
                x = float(rng.normal(self.mean, self.sd))
                if lo <= x <= hi:
                    return x
            return float(np.clip(self.mean, lo, hi))
        if self.kind == "loguniform":
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(rng.uniform(lo, hi))

    def log_pdf(self, x: float) -> float:
        """Return log-prior density (unnormalised for log-uniform)."""
        lo, hi = self.bounds
        if x < lo or x > hi:
            return -np.inf
        if self.kind == "truncnorm":
            assert self.mean is not None and self.sd is not None
            z = (x - self.mean) / self.sd
            return -0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - math.log(self.sd)
        if self.kind == "loguniform":
            return -math.log(x) - math.log(math.log(hi) - math.log(lo))
        # uniform
        return -math.log(hi - lo)


@dataclass(frozen=True)
class BayesianLikelihood:
    """Observation data and noise model for the calibration.

    The log-likelihood is the sum of two independent Gaussian components,
    matching Eq. (6)–(7) of the paper:

      log p(f̂_r | θ) = N(f_r_sim(θ),  (σ_f · f̂_r)²)        — relative noise
      log p(log|Ĥ(ω_i)| | θ) = N(log|H_sim(ω_i; θ)|, σ_H²)   — log-magnitude

    Parameters
    ----------
    modal_frequencies_hz:
        Measured modal frequencies ``f̂_r`` (typically 3 values: f_1, f_2, f_3).
    modal_sigma_rel:
        Relative noise level for frequency observations; the paper uses ``0.03``.
    frf_frequencies_hz:
        Frequency grid for the FRF observations.
    frf_log_magnitudes:
        ``ln |Ĥ(ω_i)|`` on the same grid (natural log to match Eq. 7).
    frf_sigma_log:
        Standard deviation of FRF log-magnitudes; paper uses ``0.15``.
    """

    modal_frequencies_hz: np.ndarray
    frf_frequencies_hz: np.ndarray
    frf_log_magnitudes: np.ndarray
    modal_sigma_rel: float = 0.03
    frf_sigma_log: float = 0.15

    def __post_init__(self) -> None:
        if self.frf_frequencies_hz.shape != self.frf_log_magnitudes.shape:
            raise ValueError("frf_frequencies_hz and frf_log_magnitudes must align.")
        if self.modal_sigma_rel <= 0.0 or self.frf_sigma_log <= 0.0:
            raise ValueError("Noise standard deviations must be positive.")


@dataclass(frozen=True)
class ForwardResult:
    """What a forward operator must return.

    Parameters
    ----------
    modal_frequencies_hz:
        Simulated natural frequencies (first ``n_modes`` modes); compared
        elementwise to :attr:`BayesianLikelihood.modal_frequencies_hz`.
    frf_complex:
        Simulated complex FRF on the same frequency grid as
        :attr:`BayesianLikelihood.frf_frequencies_hz`.
    """

    modal_frequencies_hz: np.ndarray
    frf_complex: np.ndarray


ForwardOperator = Callable[[dict[str, float]], ForwardResult]


@dataclass
class PosteriorResult:
    """Container for posterior samples and convergence diagnostics."""

    parameter_names: list[str]
    samples: np.ndarray  # (n_chains, n_steps, n_params) or flattened (n_samples, n_params)
    log_posterior: np.ndarray  # (n_chains, n_steps) or (n_samples,)
    acceptance_fraction: float
    n_walkers: int
    n_steps: int
    n_burn: int

    def flatten(self) -> np.ndarray:
        """Return (n_walkers·(n_steps − n_burn), n_params) post-burn samples."""
        if self.samples.ndim == 2:
            return self.samples
        return self.samples[:, self.n_burn:, :].reshape(-1, self.samples.shape[-1])

    def parameter_array(self, name: str) -> np.ndarray:
        """Return all post-burn samples for one parameter as a 1-D array."""
        idx = self.parameter_names.index(name)
        return self.flatten()[:, idx]

    def quantiles(
        self,
        name: str,
        q: tuple[float, ...] = (0.05, 0.50, 0.95),
    ) -> dict[float, float]:
        """Return requested quantiles for one parameter."""
        flat = self.parameter_array(name)
        return {qi: float(np.quantile(flat, qi)) for qi in q}

    def credible_interval(self, name: str, alpha: float = 0.90) -> tuple[float, float, float]:
        """Return ``(q_lo, q_med, q_hi)`` for a centred *alpha* credible interval."""
        lo = (1.0 - alpha) / 2.0
        hi = 1.0 - lo
        q = self.quantiles(name, q=(lo, 0.5, hi))
        return q[lo], q[0.5], q[hi]


# ────────────────────────────────────────────────────────────────────────────
#  Log-posterior assembly
# ────────────────────────────────────────────────────────────────────────────
def _log_prior(theta: np.ndarray, priors: list[BayesianPrior]) -> float:
    total = 0.0
    for value, prior in zip(theta, priors):
        lp = prior.log_pdf(float(value))
        if not math.isfinite(lp):
            return -np.inf
        total += lp
    return total


def _log_likelihood(
    fwd: ForwardResult,
    likelihood: BayesianLikelihood,
) -> float:
    # Modal-frequency term (relative noise → variance ∝ f̂²)
    n_modes = min(len(fwd.modal_frequencies_hz), len(likelihood.modal_frequencies_hz))
    if n_modes == 0:
        return -np.inf
    f_meas = np.asarray(likelihood.modal_frequencies_hz[:n_modes], dtype=float)
    f_sim = np.asarray(fwd.modal_frequencies_hz[:n_modes], dtype=float)
    sigma_f = likelihood.modal_sigma_rel * f_meas
    if (sigma_f <= 0.0).any() or not np.isfinite(f_sim).all():
        return -np.inf
    z_f = (f_sim - f_meas) / sigma_f
    log_modal = -0.5 * float(np.sum(z_f * z_f))
    log_modal += -float(np.sum(np.log(sigma_f))) - 0.5 * n_modes * math.log(2.0 * math.pi)

    # Log-FRF magnitude term
    mag = np.abs(fwd.frf_complex)
    if (mag <= 0.0).any() or not np.isfinite(mag).all():
        return -np.inf
    log_mag = np.log(mag)
    z_h = (log_mag - likelihood.frf_log_magnitudes) / likelihood.frf_sigma_log
    n_h = z_h.size
    log_frf = -0.5 * float(np.sum(z_h * z_h))
    log_frf += -n_h * math.log(likelihood.frf_sigma_log) - 0.5 * n_h * math.log(
        2.0 * math.pi
    )
    return log_modal + log_frf


def make_log_posterior(
    priors: list[BayesianPrior],
    likelihood: BayesianLikelihood,
    forward_operator: ForwardOperator,
) -> Callable[[np.ndarray], float]:
    """Return a callable ``log_posterior(theta)`` suitable for emcee."""
    names = [p.name for p in priors]

    def log_posterior(theta: np.ndarray) -> float:
        lp = _log_prior(theta, priors)
        if not math.isfinite(lp):
            return -np.inf
        params = dict(zip(names, [float(v) for v in theta]))
        try:
            fwd = forward_operator(params)
        except Exception:
            return -np.inf
        ll = _log_likelihood(fwd, likelihood)
        if not math.isfinite(ll):
            return -np.inf
        return lp + ll

    return log_posterior


# ────────────────────────────────────────────────────────────────────────────
#  emcee runner
# ────────────────────────────────────────────────────────────────────────────
def _initialise_walkers(
    priors: list[BayesianPrior],
    n_walkers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_walkers`` initial positions independently from the priors."""
    return np.array([[p.sample(rng) for p in priors] for _ in range(n_walkers)])


def run_emcee_calibration(
    forward_operator: ForwardOperator,
    priors: list[BayesianPrior],
    likelihood: BayesianLikelihood,
    *,
    n_walkers: int = 32,
    n_steps: int = 4000,
    n_burn: int = 1500,
    seed: int = 2026,
    progress: bool = False,
    n_threads: int = 1,
) -> PosteriorResult:
    """Run affine-invariant ensemble MCMC (emcee) to sample the posterior.

    Parameters
    ----------
    forward_operator:
        Callable ``params_dict -> ForwardResult``. Should be deterministic and
        reasonably fast — every walker step evaluates it once.
    priors:
        Parameter priors. The order of this list defines the parameter ordering
        in :class:`PosteriorResult.samples`.
    likelihood:
        Measured-data container.
    n_walkers:
        Number of emcee walkers. Recommended ≥ ``2·n_params``; default 32.
    n_steps:
        Total MCMC steps per walker. Half of paper's 5000 NUTS steps is usually
        plenty for emcee thanks to ensemble mixing.
    n_burn:
        Number of warm-up steps to discard at the start of every walker chain.
    seed:
        RNG seed for walker initialisation and emcee's sampler.
    progress:
        Forwarded to ``emcee.EnsembleSampler.run_mcmc`` for tqdm output.
    n_threads:
        Number of parallel forward-operator evaluations per step. Requires the
        forward operator to be picklable when ``n_threads > 1``.

    Returns
    -------
    PosteriorResult
        Samples shape ``(n_walkers, n_steps, n_params)``.
    """
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "emcee is required for Bayesian calibration. "
            "Install with: pip install emcee"
        ) from exc

    log_post = make_log_posterior(priors, likelihood, forward_operator)
    rng = np.random.default_rng(seed)
    p0 = _initialise_walkers(priors, n_walkers, rng)
    n_params = len(priors)

    if n_threads > 1:
        # multiprocessing pool — caller's forward_operator must be picklable.
        from multiprocessing import Pool

        with Pool(n_threads) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, n_params, log_post, pool=pool,
            )
            sampler.run_mcmc(p0, n_steps, progress=progress)
            chain = sampler.get_chain()
            log_prob = sampler.get_log_prob()
            acc = float(np.mean(sampler.acceptance_fraction))
    else:
        sampler = emcee.EnsembleSampler(n_walkers, n_params, log_post)
        sampler.run_mcmc(p0, n_steps, progress=progress)
        chain = sampler.get_chain()           # (n_steps, n_walkers, n_params)
        log_prob = sampler.get_log_prob()     # (n_steps, n_walkers)
        acc = float(np.mean(sampler.acceptance_fraction))

    # Re-shape from emcee's (steps, walkers, params) to (walkers, steps, params)
    chain_w = np.transpose(chain, (1, 0, 2))
    log_prob_w = np.transpose(log_prob, (1, 0))

    return PosteriorResult(
        parameter_names=[p.name for p in priors],
        samples=chain_w,
        log_posterior=log_prob_w,
        acceptance_fraction=acc,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
    )


# ────────────────────────────────────────────────────────────────────────────
#  Convergence diagnostics
# ────────────────────────────────────────────────────────────────────────────
def gelman_rubin(samples: np.ndarray) -> np.ndarray:
    """Return the split-R̂ statistic per parameter.

    Implements the split-chain Gelman–Rubin convergence diagnostic
    (Vehtari et al. 2021, Eq. 4). ``samples`` has shape
    ``(n_walkers, n_steps, n_params)``.
    """
    n_walkers, n_steps, n_params = samples.shape
    if n_steps < 4:
        return np.full(n_params, np.nan)
    half = n_steps // 2
    split = np.concatenate(
        [samples[:, :half, :], samples[:, half : 2 * half, :]], axis=0
    )
    m, n, _ = split.shape
    chain_mean = split.mean(axis=1)         # (m, n_params)
    chain_var = split.var(axis=1, ddof=1)   # (m, n_params)
    grand = chain_mean.mean(axis=0)         # (n_params,)
    B = n * np.sum((chain_mean - grand) ** 2, axis=0) / (m - 1)
    W = chain_var.mean(axis=0)
    var_hat = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_hat / np.maximum(W, 1e-30))
    return rhat


def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """Return the effective sample size per parameter via autocorrelation.

    Returns ``np.nan`` for any parameter whose autocorrelation does not decay
    enough to be estimable; otherwise a positive integer-valued float.
    """
    n_walkers, n_steps, n_params = samples.shape
    ess = np.zeros(n_params)
    for p in range(n_params):
        # Compute autocorrelation by FFT for each walker, then average
        ac_list = []
        for w in range(n_walkers):
            x = samples[w, :, p] - samples[w, :, p].mean()
            if np.allclose(x, 0.0):
                continue
            ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
            ac = ac / ac[0]
            ac_list.append(ac)
        if not ac_list:
            ess[p] = np.nan
            continue
        ac_mean = np.mean(ac_list, axis=0)
        # Geyer's initial monotone sequence: cumulative sum until autocorrelation
        # becomes negative.
        tau = 1.0
        for k in range(1, len(ac_mean)):
            if ac_mean[k] < 0.0:
                break
            tau += 2.0 * ac_mean[k]
        ess[p] = max(1.0, n_walkers * n_steps / tau)
    return ess


__all__ = [
    "BayesianPrior",
    "BayesianLikelihood",
    "ForwardOperator",
    "ForwardResult",
    "PosteriorResult",
    "PriorKind",
    "make_log_posterior",
    "run_emcee_calibration",
    "gelman_rubin",
    "effective_sample_size",
]
