"""Sobol global sensitivity analysis on the recommended work frequency.

Implements Section 2.4 of the parameter-uncertainty harvesting paper:

  Y = f*(θ) — recommended frequency from the Pareto knee
  X = (E, ρ, ζ₁, ζ₂, k_c, c_c, k_f, c_f, d_tr,root, L_tr, d̄_br, L̄_br)

We use SALib's Saltelli sampling (``N(2k+2)`` total evaluations) and the
first-order / total-effect Sobol estimators with bootstrap confidence bounds.
The forward model is supplied by the caller as a function
``params_dict -> recommended_frequency_hz``; this keeps the sensitivity
module decoupled from both the FEM solver and the Pareto wrapper.

Paper baseline: ``N_base = 512`` → ``13312`` evaluations per tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class SobolInputDef:
    """One sensitivity-analysis input variable.

    Parameters
    ----------
    name:
        Parameter name passed through to the forward operator.
    bounds:
        ``(lower, upper)`` sampling range.
    log_scale:
        If ``True``, the Saltelli sample is drawn uniformly in log space
        between the bounds (matches loguniform priors in the Bayesian module).
    """

    name: str
    bounds: tuple[float, float]
    log_scale: bool = False

    def __post_init__(self) -> None:
        lo, hi = self.bounds
        if hi <= lo:
            raise ValueError(f"{self.name}: bounds must be increasing.")
        if self.log_scale and lo <= 0.0:
            raise ValueError(f"{self.name}: log_scale requires lower > 0.")


@dataclass(frozen=True)
class SobolIndex:
    """First-order or total-effect index with bootstrap CI."""

    value: float
    confidence_low: float
    confidence_high: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.value, self.confidence_low, self.confidence_high


@dataclass(frozen=True)
class SobolResult:
    """Result of Sobol analysis on one scalar output.

    Parameters
    ----------
    parameter_names:
        Names of the input variables (in evaluation order).
    first_order:
        Per-parameter ``S_i`` index.
    total_effect:
        Per-parameter ``S_T,i`` index (≥ ``S_i``).
    n_evaluations:
        Total number of forward evaluations performed.
    output_variance:
        ``Var[Y]`` — useful as a check that the model output actually varies.
    output_mean:
        ``E[Y]``.
    """

    parameter_names: list[str]
    first_order: dict[str, SobolIndex]
    total_effect: dict[str, SobolIndex]
    n_evaluations: int
    output_variance: float
    output_mean: float

    def ranked_by_total_effect(self) -> list[tuple[str, SobolIndex]]:
        """Return ``[(name, S_T,i)]`` sorted by total-effect descending."""
        return sorted(
            ((n, self.total_effect[n]) for n in self.parameter_names),
            key=lambda x: x[1].value,
            reverse=True,
        )

    def summary_table(self, top_k: int | None = None) -> str:
        """Human-readable table in the paper's Table-8 style."""
        rows = self.ranked_by_total_effect()
        if top_k is not None:
            rows = rows[:top_k]
        lines = [
            "parameter             S_i  [CI95]              S_T,i [CI95]",
            "-" * 70,
        ]
        for name, st in rows:
            si = self.first_order[name]
            lines.append(
                f"{name:18s}  {si.value:.3f} [{si.confidence_low:.3f},"
                f"{si.confidence_high:.3f}]   "
                f"{st.value:.3f} [{st.confidence_low:.3f},{st.confidence_high:.3f}]"
            )
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
#  Saltelli sampling + forward evaluation
# ────────────────────────────────────────────────────────────────────────────
def _build_salib_problem(inputs: Sequence[SobolInputDef]) -> dict:
    bounds = []
    for inp in inputs:
        if inp.log_scale:
            bounds.append([np.log(inp.bounds[0]), np.log(inp.bounds[1])])
        else:
            bounds.append([inp.bounds[0], inp.bounds[1]])
    return {
        "num_vars": len(inputs),
        "names": [inp.name for inp in inputs],
        "bounds": bounds,
    }


def _untransform_sample(
    sample_row: np.ndarray,
    inputs: Sequence[SobolInputDef],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for value, inp in zip(sample_row, inputs):
        if inp.log_scale:
            out[inp.name] = float(np.exp(value))
        else:
            out[inp.name] = float(value)
    return out


def run_sobol_analysis(
    forward: Callable[[dict[str, float]], float],
    inputs: Sequence[SobolInputDef],
    *,
    n_base: int = 512,
    bootstrap_resamples: int = 200,
    confidence_level: float = 0.95,
    second_order: bool = False,
    seed: int = 2026,
    progress: bool = False,
    n_threads: int = 1,
) -> SobolResult:
    """Run Saltelli sampling and compute first-order / total-effect Sobol indices.

    Parameters
    ----------
    forward:
        Callable ``params_dict -> scalar``. Must accept the names in *inputs*
        and return a finite scalar (e.g. the recommended frequency in Hz).
        If it raises, that sample is replaced by the running mean (fail-soft).
    inputs:
        Sampling-space definition.
    n_base:
        Saltelli base sample size. Total evaluations are ``N_base · (2k+2)``
        where ``k = len(inputs)`` (paper baseline ``N_base = 512``).
    bootstrap_resamples:
        Number of bootstrap resamples used by SALib for the index CI.
    confidence_level:
        Probability level of the CI (default 0.95 — paper format).
    second_order:
        If ``True``, compute second-order indices as well (caller pays the
        higher evaluation cost: ``N_base · (2k+2)`` becomes ``N_base · (k+2)``
        — Saltelli's reduced design — vs. the second-order design ``N_base · (2k+2)``).
    seed:
        RNG seed for Saltelli sampling.
    progress:
        Show a tqdm progress bar over the forward-operator loop if available.
    n_threads:
        If > 1, evaluate the forward operator in a ``multiprocessing.Pool``.
        The forward operator must then be picklable.
    """
    try:
        from SALib.analyze.sobol import analyze
        from SALib.sample.sobol import sample as saltelli_sample
    except ImportError as exc:
        raise ImportError(
            "SALib is required for Sobol analysis. "
            "Install with: pip install SALib"
        ) from exc

    problem = _build_salib_problem(inputs)
    X = saltelli_sample(problem, n_base, calc_second_order=second_order,
                        seed=seed)

    # Evaluate forward operator on every Saltelli row
    n_eval = X.shape[0]
    Y = np.empty(n_eval, dtype=float)

    if n_threads > 1:
        from multiprocessing import Pool

        def _worker(row):
            params = _untransform_sample(row, inputs)
            try:
                return float(forward(params))
            except Exception:
                return np.nan

        with Pool(n_threads) as pool:
            it = pool.imap(_worker, X, chunksize=max(1, n_eval // (n_threads * 8)))
            if progress:
                try:
                    from tqdm import tqdm
                    it = tqdm(it, total=n_eval, desc="Sobol forward")
                except ImportError:
                    pass
            for i, y in enumerate(it):
                Y[i] = y
    else:
        iterator = range(n_eval)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=n_eval, desc="Sobol forward")
            except ImportError:
                pass
        for i in iterator:
            params = _untransform_sample(X[i], inputs)
            try:
                Y[i] = float(forward(params))
            except Exception:
                Y[i] = np.nan

    # Replace NaN with the sample mean to avoid breaking SALib's analyse step
    finite = np.isfinite(Y)
    if not finite.all():
        Y[~finite] = float(np.mean(Y[finite])) if finite.any() else 0.0

    indices = analyze(
        problem, Y,
        calc_second_order=second_order,
        num_resamples=bootstrap_resamples,
        conf_level=confidence_level,
    )

    names = problem["names"]
    first_order: dict[str, SobolIndex] = {}
    total_effect: dict[str, SobolIndex] = {}
    for i, name in enumerate(names):
        s_val = float(indices["S1"][i])
        s_conf = float(indices["S1_conf"][i])
        first_order[name] = SobolIndex(
            value=s_val,
            confidence_low=s_val - s_conf,
            confidence_high=s_val + s_conf,
        )
        st_val = float(indices["ST"][i])
        st_conf = float(indices["ST_conf"][i])
        total_effect[name] = SobolIndex(
            value=st_val,
            confidence_low=st_val - st_conf,
            confidence_high=st_val + st_conf,
        )

    return SobolResult(
        parameter_names=names,
        first_order=first_order,
        total_effect=total_effect,
        n_evaluations=n_eval,
        output_variance=float(np.var(Y, ddof=1)),
        output_mean=float(np.mean(Y)),
    )


__all__ = [
    "SobolIndex",
    "SobolInputDef",
    "SobolResult",
    "run_sobol_analysis",
]
