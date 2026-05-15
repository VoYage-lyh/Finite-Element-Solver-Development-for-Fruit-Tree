"""Branch material-test data and Bayesian prior generation.

Captures the kind of material-test summary reported in Table 2 of the
parameter-uncertainty harvesting paper (fresh density, moisture content,
three-point bending E, three-point compression E, etc.) and converts it into
the truncated-normal priors used by :mod:`orchard_fem.calibration.bayesian_calibration`.

Example::

    from orchard_fem.io.material_test import (
        BranchMaterialTestSummary,
        priors_from_material_test,
    )

    summary = BranchMaterialTestSummary(
        n_samples=30,
        fresh_density_kgm3_mean=765.0,
        fresh_density_kgm3_sd=58.0,
        bending_E_GPa_mean=7.85,
        bending_E_GPa_sd=1.12,
    )
    priors = priors_from_material_test(summary)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchard_fem.calibration.bayesian_calibration import BayesianPrior


@dataclass(frozen=True)
class BranchMaterialTestSummary:
    """Summary statistics from independent branch material tests.

    Fields correspond to Table 2 of the harvesting-parameter paper. Any field
    whose ``..._sd`` is ``None`` is omitted from the auto-generated priors
    (the corresponding parameter then falls back to a configured default prior).

    Parameters
    ----------
    n_samples:
        Number of physical specimens. Used only as metadata for reporting.
    fresh_density_kgm3_mean / _sd:
        Mean / standard deviation of fresh-basis density ``ρ`` [kg/m³].
    moisture_content_pct_mean / _sd:
        Wet-basis moisture content [%]; reported but not used in default priors.
    bending_E_GPa_mean / _sd:
        Three-point bending Young's modulus [GPa]; primary source for the
        equivalent bending modulus prior used in FRF calibration.
    bending_strength_MPa_mean / _sd:
        Bending strength [MPa]; informational only.
    compression_E_GPa_mean / _sd:
        Three-point compression modulus [GPa]; informational only.
    compression_strength_MPa_mean / _sd:
        Compression strength [MPa]; informational only.
    poisson_assumed:
        Poisson ratio assumed for the FE model (typically 0.3 for woody tissue).
    notes:
        Optional free-text provenance description.
    """

    n_samples: int
    fresh_density_kgm3_mean: float
    fresh_density_kgm3_sd: float
    bending_E_GPa_mean: float
    bending_E_GPa_sd: float
    moisture_content_pct_mean: float | None = None
    moisture_content_pct_sd: float | None = None
    bending_strength_MPa_mean: float | None = None
    bending_strength_MPa_sd: float | None = None
    compression_E_GPa_mean: float | None = None
    compression_E_GPa_sd: float | None = None
    compression_strength_MPa_mean: float | None = None
    compression_strength_MPa_sd: float | None = None
    poisson_assumed: float = 0.30
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self.fresh_density_kgm3_sd <= 0.0 or self.bending_E_GPa_sd <= 0.0:
            raise ValueError(
                "Density and bending modulus standard deviations must be positive "
                "to define a non-degenerate prior."
            )

    def bending_E_Pa(self) -> tuple[float, float]:
        """Return ``(mean, sd)`` of bending modulus in Pa instead of GPa."""
        return self.bending_E_GPa_mean * 1.0e9, self.bending_E_GPa_sd * 1.0e9

    # ── I/O ──
    @classmethod
    def from_json(cls, path: str | Path) -> "BranchMaterialTestSummary":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class DefaultPriorSpec:
    """Default prior used when the material-test summary lacks data for a parameter.

    All defaults follow Section 2.2 of the harvesting-parameter paper.
    """

    # log-uniform bounds for modal damping ratios (dimensionless)
    zeta_min: float = 0.01
    zeta_max: float = 0.10
    # log-uniform bounds for clamp equivalent stiffness [N/m]
    k_clamp_min: float = 1.0e4
    k_clamp_max: float = 1.0e7
    # log-uniform bounds for clamp equivalent damping [N·s/m]
    c_clamp_min: float = 1.0e0
    c_clamp_max: float = 1.0e3
    # log-uniform bounds for fruit attachment stiffness [N/m]
    k_fruit_min: float = 1.0e2
    k_fruit_max: float = 1.0e4
    # log-uniform bounds for fruit attachment damping [N·s/m]
    c_fruit_min: float = 1.0e-2
    c_fruit_max: float = 1.0e1


def priors_from_material_test(
    summary: BranchMaterialTestSummary,
    *,
    truncation_sigma: float = 2.0,
    defaults: DefaultPriorSpec = DefaultPriorSpec(),
    include_damping: bool = True,
    include_clamp: bool = True,
    include_fruit: bool = True,
) -> "list[BayesianPrior]":
    """Generate the canonical 8-parameter Bayesian prior list.

    The returned list covers ``(E, ρ, ζ_1, ζ_2, k_c, c_c, k_f, c_f)`` per the
    paper's parameter vector ``θ ∈ ℝ⁸``. Truncated-normal priors for ``(E, ρ)``
    are bounded at ``mean ± truncation_sigma · sd`` (paper uses ``±2σ``).

    Parameters
    ----------
    summary:
        Material-test summary with bending modulus and density stats.
    truncation_sigma:
        Number of standard deviations defining the truncation interval for
        ``E`` and ``ρ``. Default 2.0 matches the paper.
    defaults:
        Default log-uniform bounds for parameters not covered by material tests.
    include_damping, include_clamp, include_fruit:
        If ``False``, omit the corresponding pairs from the prior list (useful
        when the user wants to fix some parameters at point values rather than
        calibrate them).

    Returns
    -------
    list[BayesianPrior]
        Priors ready to pass to :func:`run_nuts_calibration`.
    """
    # Lazy import to avoid circular dep at module load time.
    from orchard_fem.calibration.bayesian_calibration import BayesianPrior

    e_mean, e_sd = summary.bending_E_Pa()
    rho_mean, rho_sd = summary.fresh_density_kgm3_mean, summary.fresh_density_kgm3_sd

    priors: list[BayesianPrior] = [
        BayesianPrior(
            name="E",
            kind="truncnorm",
            mean=e_mean,
            sd=e_sd,
            bounds=(e_mean - truncation_sigma * e_sd, e_mean + truncation_sigma * e_sd),
        ),
        BayesianPrior(
            name="rho",
            kind="truncnorm",
            mean=rho_mean,
            sd=rho_sd,
            bounds=(
                rho_mean - truncation_sigma * rho_sd,
                rho_mean + truncation_sigma * rho_sd,
            ),
        ),
    ]

    if include_damping:
        priors.extend(
            BayesianPrior(name=name, kind="loguniform",
                          bounds=(defaults.zeta_min, defaults.zeta_max))
            for name in ("zeta1", "zeta2")
        )
    if include_clamp:
        priors.append(
            BayesianPrior(name="k_c", kind="loguniform",
                          bounds=(defaults.k_clamp_min, defaults.k_clamp_max))
        )
        priors.append(
            BayesianPrior(name="c_c", kind="loguniform",
                          bounds=(defaults.c_clamp_min, defaults.c_clamp_max))
        )
    if include_fruit:
        priors.append(
            BayesianPrior(name="k_f", kind="loguniform",
                          bounds=(defaults.k_fruit_min, defaults.k_fruit_max))
        )
        priors.append(
            BayesianPrior(name="c_f", kind="loguniform",
                          bounds=(defaults.c_fruit_min, defaults.c_fruit_max))
        )

    return priors


__all__ = [
    "BranchMaterialTestSummary",
    "DefaultPriorSpec",
    "priors_from_material_test",
]
