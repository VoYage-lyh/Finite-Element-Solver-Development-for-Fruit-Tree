"""Modal frequency calibration using scipy.optimize.

Tunes material Young's moduli so that the simulated natural frequencies of the
FEM model match experimentally measured resonance peaks extracted from an FRF.

The calibration minimises the weighted relative frequency error::

    f(x) = Σ_i  [(ω_sim_i − ω_meas_i) / ω_meas_i]²

using L-BFGS-B so that box constraints on each Young's modulus are respected.

Example::

    from orchard_fem.calibration.modal_calibration import (
        CalibrationParameter,
        ModalCalibrationConfig,
        calibrate_from_modal_frequencies,
    )

    config = ModalCalibrationConfig(
        parameters=[
            CalibrationParameter("xylem_default", 8e9, 5e9, 2e10),
            CalibrationParameter("pith_default",  5e8, 3e8, 1.5e9),
        ],
        target_frequencies_hz=[3.2, 7.8, 14.1],
        n_modes=6,
        max_iter=80,
    )
    result = calibrate_from_modal_frequencies(base_model, config)
    print(result.calibrated_values)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchard_fem.domain import OrchardModel


@dataclass(frozen=True)
class CalibrationParameter:
    """One tunable material parameter.

    Parameters
    ----------
    material_id:
        ID of the material to perturb (matches ``MaterialProperties.material_id``).
    initial_value:
        Starting Young's modulus [Pa].
    lower_bound:
        Minimum allowed Young's modulus [Pa].
    upper_bound:
        Maximum allowed Young's modulus [Pa].
    """

    material_id: str
    initial_value: float
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if not (self.lower_bound <= self.initial_value <= self.upper_bound):
            raise ValueError(
                f"CalibrationParameter '{self.material_id}': "
                f"initial_value ({self.initial_value:.3g}) must be within "
                f"[{self.lower_bound:.3g}, {self.upper_bound:.3g}]."
            )


@dataclass(frozen=True)
class ModalCalibrationConfig:
    """Configuration for modal frequency calibration.

    Parameters
    ----------
    parameters:
        List of tunable material parameters with bounds.
    target_frequencies_hz:
        Measured natural frequencies [Hz] to match.  The first
        ``len(target_frequencies_hz)`` simulated modes are compared.
    n_modes:
        Number of modes to request from the modal solver.  Must be ≥
        ``len(target_frequencies_hz)``.
    max_iter:
        Maximum number of L-BFGS-B iterations.
    verbose:
        If True, print objective value at each iteration.
    """

    parameters: list[CalibrationParameter]
    target_frequencies_hz: list[float]
    n_modes: int = 10
    max_iter: int = 100
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("ModalCalibrationConfig requires at least one parameter.")
        if not self.target_frequencies_hz:
            raise ValueError("ModalCalibrationConfig requires at least one target frequency.")
        if self.n_modes < len(self.target_frequencies_hz):
            raise ValueError(
                "n_modes must be >= len(target_frequencies_hz) "
                f"(got n_modes={self.n_modes}, targets={len(self.target_frequencies_hz)})."
            )


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of a calibration run.

    Parameters
    ----------
    calibrated_values:
        Best-found material Young's moduli [Pa], keyed by ``material_id``.
    final_frequencies_hz:
        Simulated natural frequencies at the calibrated parameter values.
    residual_norm:
        Final objective value (dimensionless relative frequency error norm).
    converged:
        ``True`` if the optimiser reported successful convergence.
    n_evaluations:
        Number of FEM modal evaluations performed.
    """

    calibrated_values: dict[str, float]
    final_frequencies_hz: list[float]
    residual_norm: float
    converged: bool
    n_evaluations: int


def _apply_material_values(
    model: "OrchardModel",
    parameters: list[CalibrationParameter],
    values: list[float],
) -> "OrchardModel":
    """Return a clone of *model* with Young's moduli set from *values*."""
    id_to_value = {p.material_id: v for p, v in zip(parameters, values)}
    new_materials = []
    for mat in model.materials:
        if mat.material_id in id_to_value:
            new_materials.append(replace(mat, youngs_modulus=id_to_value[mat.material_id]))
        else:
            new_materials.append(mat)
    return replace(model, materials=new_materials)


def calibrate_from_modal_frequencies(
    base_model: "OrchardModel",
    config: ModalCalibrationConfig,
) -> CalibrationResult:
    """Calibrate material Young's moduli to match measured natural frequencies.

    Parameters
    ----------
    base_model:
        Baseline :class:`~orchard_fem.domain.OrchardModel`.
    config:
        Calibration configuration with parameter bounds and target frequencies.

    Returns
    -------
    CalibrationResult
        Calibrated parameter values and convergence information.

    Raises
    ------
    ImportError
        If ``scipy`` or ``dolfinx`` / ``slepc4py`` are not installed.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "scipy is required for modal calibration. "
            "Install it with: pip install scipy"
        ) from exc

    from orchard_fem.fenicsx.modal import solve_embedded_beam_modal_experiment

    targets = config.target_frequencies_hz
    n_targets = len(targets)
    n_eval = [0]

    def objective(x: list[float]) -> float:
        n_eval[0] += 1
        model = _apply_material_values(base_model, config.parameters, list(x))
        try:
            result = solve_embedded_beam_modal_experiment(model, num_modes=config.n_modes)
        except Exception:
            return 1.0e6

        sim_freqs = [m.frequency_hz for m in result.modes[:n_targets]]
        if len(sim_freqs) < n_targets:
            return 1.0e6

        error = sum(
            ((sim - meas) / meas) ** 2
            for sim, meas in zip(sim_freqs, targets)
            if meas > 0.0
        )
        if config.verbose:
            print(
                f"[calibration] eval {n_eval[0]:4d}  obj={error:.6e}  "
                f"E={[f'{v:.3g}' for v in x]}"
            )
        return float(error)

    x0 = [p.initial_value for p in config.parameters]
    bounds = [(p.lower_bound, p.upper_bound) for p in config.parameters]

    opt_result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": config.max_iter, "ftol": 1.0e-10, "gtol": 1.0e-8},
    )

    best_x = list(opt_result.x)
    final_model = _apply_material_values(base_model, config.parameters, best_x)
    try:
        final_modal = solve_embedded_beam_modal_experiment(
            final_model, num_modes=config.n_modes
        )
        final_freqs = [m.frequency_hz for m in final_modal.modes[:n_targets]]
    except Exception:
        final_freqs = []

    calibrated = {
        p.material_id: float(v)
        for p, v in zip(config.parameters, best_x)
    }

    return CalibrationResult(
        calibrated_values=calibrated,
        final_frequencies_hz=final_freqs,
        residual_norm=float(opt_result.fun),
        converged=bool(opt_result.success),
        n_evaluations=n_eval[0],
    )
