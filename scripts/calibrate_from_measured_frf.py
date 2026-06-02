"""Inverse-fit Rayleigh damping + local nonlinear (k_3, c_2) from a measured FRF.

Reads the single-channel measured H1 FRF for a hammer test (e.g. tree1_p1
left_leader tip-Z, averaged over the first N strikes) and tunes three
physically interpretable parameters on the single-branch single-point sim
model so the simulated displacement compliance matches the measurement in
the band of interest.

Tunable parameters (all in log10 space, so Nelder-Mead stays in valid range)
    rayleigh_beta — stiffness-proportional damping coefficient β.
                    Linear modal damping at angular frequency ω is ζ = β·ω/2,
                    so β ≈ 2 ζ / ω_res. Baseline tree_1.json: β = 1e-4
                    → ζ_1 ≈ 0.4 % at 12.5 Hz (too low; manuscript ~5 %).
    k3_scale       — multiplies the auto-injected cubic stiffness range
                    (joints._K3_DOWNSCALE × k3_scale). Baseline = 1.0
                    keeps the current downscaled range
                    k_3 ∈ ±[0.9, 2.1]×10⁷ N·m⁻³.
    c2_scale       — multiplies the auto-injected quadratic damping range
                    (joints._C2_DOWNSCALE × c2_scale). Baseline = 1.0
                    keeps c_2 ∈ [1.2, 4.3]×10³ N·s²·m⁻².

Cost: sum_i γ²(f_i) · (log|H_sim(f_i)| − log|H_meas(f_i)|)² in [fmin, fmax],
weighted by measured coherence so noisy freqs contribute less.

Optimiser: scipy.optimize.minimize Nelder-Mead (gradient-free, robust to
the unsmooth Pareto/peak-detector cost surface).

Forward evaluations: ~30–60 FE FRF sweeps × ~2 s each ≈ 1–2 min total.

Outputs:
  cache/calibration/tree_<n>_<input>_<output>_fit.npz   (optimal params)
  results/calibration/calibrated_frf_tree_<n>.{png,pdf} (before/after plot)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orchard_fem.io.loaders.orchard import load_orchard_model

CALIBRATION_CACHE = REPO / "cache" / "calibration"


# ─────────────────────────────────────────────────────────────────────────────
#  Forward model
# ─────────────────────────────────────────────────────────────────────────────
def _override_excitation(model, *, branch_id, node, comp, amplitude):
    new_excitation = replace(
        model.excitation,
        target_branch_id=branch_id, target_node=node,
        target_component=comp, amplitude=amplitude, target_s=None,
    )
    return replace(model, excitation=new_excitation)


def _frf_single_point(model, freqs_hz, output_obs_name):
    """Evaluate FRF at the supplied frequencies (one FE solve per call,
    discrete sweep — does not interpolate)."""
    from orchard_fem.fenicsx.frequency_response import (
        solve_embedded_beam_frequency_response_experiment,
    )
    swept = replace(
        model,
        analysis=replace(
            model.analysis,
            frequency_start_hz=float(freqs_hz[0]),
            frequency_end_hz=float(freqs_hz[-1]),
            frequency_steps=int(freqs_hz.size),
        ),
    )
    exp = solve_embedded_beam_frequency_response_experiment(swept, polynomial_degree=1)
    res = exp.result
    if output_obs_name not in res.observation_names:
        raise RuntimeError(f"Observation {output_obs_name!r} not found.")
    idx = res.observation_names.index(output_obs_name)
    f_out = np.array([p.frequency_hz for p in res.points])
    m_out = np.array([p.observation_magnitudes[idx] for p in res.points])
    return f_out, m_out


class ForwardModel:
    """Configure & evaluate the single-branch sim FRF with patchable
    (beta, k3_scale, c2_scale)."""

    def __init__(self, *, tree_n, input_branch, input_node, input_comp,
                 output_branch, output_station, output_comp, amplitude=10.0):
        path = REPO / "trees" / f"tree_{tree_n}.json"
        self.base_model = _override_excitation(
            load_orchard_model(str(path)),
            branch_id=input_branch, node=input_node,
            comp=input_comp, amplitude=amplitude,
        )
        self.output_obs = f"obs_{output_branch}_{output_station}_{output_comp}"
        self.amplitude = amplitude
        # Keep originals so we always restore them after a call.
        import orchard_fem.fenicsx.joints as j
        self._joints = j
        self._orig_k3 = j._K3_DOWNSCALE
        self._orig_c2 = j._C2_DOWNSCALE

    def evaluate(self, freqs_hz, *, beta, k3_scale, c2_scale):
        """Return |H_x| = |U_tip / F| in m·N⁻¹ at the given frequencies."""
        model = replace(
            self.base_model,
            analysis=replace(self.base_model.analysis, rayleigh_beta=float(beta)),
        )
        self._joints._K3_DOWNSCALE = self._orig_k3 * float(k3_scale)
        self._joints._C2_DOWNSCALE = self._orig_c2 * float(c2_scale)
        try:
            f_out, m_out = _frf_single_point(model, freqs_hz, self.output_obs)
        finally:
            self._joints._K3_DOWNSCALE = self._orig_k3
            self._joints._C2_DOWNSCALE = self._orig_c2
        return f_out, m_out / self.amplitude  # m·N⁻¹


# ─────────────────────────────────────────────────────────────────────────────
#  Measured FRF loading
# ─────────────────────────────────────────────────────────────────────────────
def load_measured_disp_frf(test_dir: Path, *, component="Z"):
    """Read the measured H1 estimate and convert |H_a| → |H_x| via /ω²."""
    path = test_dir / "frf_tip.csv"
    if not path.exists():
        raise FileNotFoundError(f"Measured FRF CSV missing: {path}")
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    cols = {name: np.array([float(r[i]) if r[i] else np.nan for r in rows])
            for i, name in enumerate(header)}
    f = cols["frequency_hz"]
    H_a = cols[f"H_{component}_mag_ms2_per_N"]
    coh = cols[f"coherence_{component}"]
    omega = 2.0 * math.pi * f
    H_x = np.full_like(H_a, np.nan)
    mask = omega > 0
    H_x[mask] = H_a[mask] / (omega[mask] ** 2)
    return f, H_x, coh


# ─────────────────────────────────────────────────────────────────────────────
#  Cost function and optimisation
# ─────────────────────────────────────────────────────────────────────────────
def make_cost(forward: ForwardModel, freqs_eval, H_meas_target, coh_weights,
              *, f_res_guess_hz, zeta_max=0.30):
    """Return cost(x) where x = [log10(beta), log10(k3_scale), log10(c2_scale)].

    A soft penalty caps the equivalent first-mode damping at ``zeta_max``
    (default 30 %) so the optimiser cannot crank up Rayleigh β past the
    over-critical regime — that would match the FRF shape arithmetically but
    violate physics (no real branch has ζ > 100 %)."""
    valid = np.isfinite(H_meas_target) & np.isfinite(coh_weights) & (H_meas_target > 0)
    omega_res = 2.0 * math.pi * f_res_guess_hz

    def cost(x):
        beta = 10.0 ** x[0]
        k3_scale = 10.0 ** x[1]
        c2_scale = 10.0 ** x[2]
        try:
            _, H_sim = forward.evaluate(
                freqs_eval, beta=beta, k3_scale=k3_scale, c2_scale=c2_scale,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] forward eval failed: {exc}")
            return 1e6
        H_sim_safe = np.where(H_sim > 0, H_sim, 1e-30)
        resid = np.log(H_sim_safe) - np.log(np.where(H_meas_target > 0, H_meas_target, 1e-30))
        w = coh_weights ** 2
        data_cost = float(
            np.sum(w[valid] * resid[valid] ** 2) / (np.sum(w[valid]) + 1e-30)
        )
        # Physical penalty on Rayleigh β so equivalent ζ at the resonance
        # stays subcritical. Smooth quadratic on the excess.
        zeta_eq = 0.5 * beta * omega_res
        penalty = 50.0 * max(0.0, zeta_eq - zeta_max) ** 2

        c = data_cost + penalty
        cost.calls = getattr(cost, "calls", 0) + 1
        if cost.calls % 5 == 0 or cost.calls < 3:
            print(f"  eval#{cost.calls:3d}  β={beta:.2e}  k3×{k3_scale:.2f}  "
                  f"c2×{c2_scale:.2f}  ζ_eq={zeta_eq*100:.0f}%  →  "
                  f"data={data_cost:.3f}  pen={penalty:.3f}")
        return c

    return cost


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────
def _configure_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def plot_calibration_result(
    freqs_eval, H_sim_before, H_sim_after, H_meas, coh,
    *, params_before, params_after, fmax, out_stem,
):
    _configure_paper_style()
    import matplotlib.pyplot as plt
    PRIMARY = "#2166AC"
    ACCENT = "#B2182B"
    GRID_MAJOR = "#d0d0d0"
    GRID_MINOR = "#ececec"

    fig, (ax_h, ax_c) = plt.subplots(
        2, 1, figsize=(6.8, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    mask = (freqs_eval > 0.5) & (freqs_eval <= fmax)
    ax_h.semilogy(
        freqs_eval[mask], H_sim_before[mask],
        color=PRIMARY, linewidth=1.5, linestyle="--", alpha=0.7,
        label=rf"Sim before  ($\beta={params_before[0]:.1e},\,k_3{{\times}}{params_before[1]:.1f},\,c_2{{\times}}{params_before[2]:.1f}$)",
    )
    ax_h.semilogy(
        freqs_eval[mask], H_sim_after[mask],
        color=ACCENT, linewidth=1.8,
        label=rf"Sim after   ($\beta={params_after[0]:.1e},\,k_3{{\times}}{params_after[1]:.1f},\,c_2{{\times}}{params_after[2]:.1f}$)",
    )
    ax_h.semilogy(
        freqs_eval[mask], H_meas[mask],
        color="black", linewidth=1.3, linestyle=":",
        marker="o", markersize=3.4,
        markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.5,
        label=r"Measured  ($H_1$, 5 impacts)",
    )
    ax_h.set_ylabel(r"$|H_x|=|U_{\rm tip}/F|$ [m$\cdot$N$^{-1}$]")
    ax_h.set_xlim(0.0, fmax)
    ax_h.grid(True, which="major", linewidth=0.6, color=GRID_MAJOR)
    ax_h.grid(True, which="minor", linewidth=0.4, color=GRID_MINOR)
    ax_h.legend(loc="upper right", fontsize=8.5)
    ax_h.set_title(
        rf"Inverse calibration of $(\beta, k_3, c_2)$ from measured FRF"
    )

    ax_c.plot(freqs_eval[mask], coh[mask], color=ACCENT, linewidth=1.0)
    ax_c.set_ylim(0.0, 1.05)
    ax_c.set_ylabel(r"$\gamma^2$")
    ax_c.set_xlabel(r"Frequency $f$ [Hz]")
    ax_c.grid(True, which="major", linewidth=0.6, color=GRID_MAJOR)
    ax_c.grid(True, which="minor", linewidth=0.4, color=GRID_MINOR)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_stem.with_suffix(".png"), dpi=150)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=int, default=1)
    parser.add_argument("--measured-dir", default="results/hammer_test/tree1_p1")
    parser.add_argument("--measured-comp", default="Z",
                        help="Column suffix in frf_tip.csv (H_<comp>_mag_ms2_per_N).")
    parser.add_argument("--input-branch", default="left_leader")
    parser.add_argument("--input-node", default="root",
                        choices=("root", "mid", "tip"))
    parser.add_argument("--input-comp", default="ux",
                        choices=("ux", "uy", "uz"),
                        help="Sim excitation direction (matches transverse 'Z' "
                        "of a vertical leader's accelerometer).")
    parser.add_argument("--output-branch", default=None)
    parser.add_argument("--output-station", default="tip",
                        choices=("root", "mid", "tip"))
    parser.add_argument("--output-comp", default="ux",
                        choices=("ux", "uy", "uz"))
    parser.add_argument("--fmin", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=25.0)
    parser.add_argument("--steps", type=int, default=40,
                        help="Frequency grid size for the inversion (default 40).")
    parser.add_argument("--maxiter", type=int, default=80,
                        help="Nelder-Mead max iterations (default 80).")
    parser.add_argument("--init-beta", type=float, default=2.5e-3,
                        help="Initial Rayleigh β (default 2.5e-3 ≈ ζ_1 = 10 % at 12.5 Hz).")
    parser.add_argument("--init-k3-scale", type=float, default=1.0)
    parser.add_argument("--init-c2-scale", type=float, default=10.0,
                        help="Initial c2 scale; default 10× undoes the 0.1× downscale.")
    args = parser.parse_args()

    output_branch = args.output_branch or args.input_branch

    print("Loading measured FRF …")
    f_meas, H_meas, coh = load_measured_disp_frf(
        REPO / args.measured_dir, component=args.measured_comp,
    )

    band = (f_meas >= args.fmin) & (f_meas <= args.fmax) & np.isfinite(H_meas)
    if not band.any():
        raise RuntimeError("No measured FRF samples in selected band.")
    # Use the same frequency grid for sim evaluations (no interpolation).
    freqs_eval = f_meas[band]
    H_meas_target = H_meas[band]
    coh_band = np.where(np.isfinite(coh[band]), coh[band], 0.0)

    print(f"Measured band: {freqs_eval.min():.1f}–{freqs_eval.max():.1f} Hz, "
          f"{freqs_eval.size} samples, mean γ²={coh_band.mean():.2f}")

    # Build forward model
    forward = ForwardModel(
        tree_n=args.tree,
        input_branch=args.input_branch, input_node=args.input_node,
        input_comp=args.input_comp,
        output_branch=output_branch, output_station=args.output_station,
        output_comp=args.output_comp,
    )

    print("Initial forward eval (sim BEFORE) …")
    t0 = time.time()
    _, H_before = forward.evaluate(
        freqs_eval, beta=1.0e-4, k3_scale=1.0, c2_scale=1.0,
    )
    print(f"  done in {time.time() - t0:.1f}s, peak={H_before.max():.3e} @ "
          f"{freqs_eval[H_before.argmax()]:.2f} Hz")

    print(f"\nNelder-Mead inversion ({args.maxiter} iter max) …")
    from scipy.optimize import minimize
    f_res_guess = float(freqs_eval[H_before.argmax()])
    cost = make_cost(
        forward, freqs_eval, H_meas_target, coh_band,
        f_res_guess_hz=f_res_guess, zeta_max=0.30,
    )
    x0 = np.array([
        math.log10(args.init_beta),
        math.log10(args.init_k3_scale),
        math.log10(args.init_c2_scale),
    ])
    t0 = time.time()
    res = minimize(cost, x0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-4,
                            "maxiter": args.maxiter, "disp": True})
    print(f"\nOptimisation finished in {time.time() - t0:.1f}s, "
          f"final cost={res.fun:.4f}, {res.nfev} evals")

    beta_opt = 10.0 ** res.x[0]
    k3_scale_opt = 10.0 ** res.x[1]
    c2_scale_opt = 10.0 ** res.x[2]

    # Convert to physical interpretations
    omega_res = 2.0 * math.pi * freqs_eval[H_before.argmax()]
    zeta_1_eq = 0.5 * beta_opt * omega_res
    k3_phys_range = (
        -2.1e9 * 0.01 * k3_scale_opt,
        -0.9e9 * 0.01 * k3_scale_opt,
    )  # softening branch shown
    c2_phys_range = (1.2e3 * c2_scale_opt, 4.3e3 * c2_scale_opt)

    print()
    print("=" * 64)
    print("Calibrated parameters:")
    print(f"  rayleigh_beta  = {beta_opt:.3e}      "
          f"(ζ₁ ≈ {zeta_1_eq*100:.1f}% at {freqs_eval[H_before.argmax()]:.1f} Hz)")
    print(f"  k3_scale       = {k3_scale_opt:.3f}     "
          f"(softening range now {k3_phys_range[0]:.2e} … {k3_phys_range[1]:.2e} N·m⁻³)")
    print(f"  c2_scale       = {c2_scale_opt:.3f}     "
          f"(range now {c2_phys_range[0]:.2e} … {c2_phys_range[1]:.2e} N·s²·m⁻²)")
    print("=" * 64)

    print("\nFinal forward eval with optimal params …")
    _, H_after = forward.evaluate(
        freqs_eval,
        beta=beta_opt, k3_scale=k3_scale_opt, c2_scale=c2_scale_opt,
    )
    print(f"  peak={H_after.max():.3e} @ {freqs_eval[H_after.argmax()]:.2f} Hz")
    print(f"  measured peak={H_meas_target.max():.3e} @ "
          f"{freqs_eval[H_meas_target.argmax()]:.2f} Hz")

    CALIBRATION_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CALIBRATION_CACHE / (
        f"tree_{args.tree}_{args.input_branch}_{args.input_node}_{args.input_comp}_fit.npz"
    )
    np.savez(
        cache_path,
        freqs=freqs_eval, H_sim_before=H_before, H_sim_after=H_after,
        H_meas=H_meas_target, coh=coh_band,
        beta_opt=beta_opt, k3_scale_opt=k3_scale_opt, c2_scale_opt=c2_scale_opt,
        zeta_1_eq=zeta_1_eq, cost_final=float(res.fun),
        n_evals=int(res.nfev),
    )
    print(f"\nCache: {cache_path.relative_to(REPO)}")

    out_dir = REPO / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / f"calibrated_frf_tree_{args.tree}"
    plot_calibration_result(
        freqs_eval, H_before, H_after, H_meas_target, coh_band,
        params_before=(1.0e-4, 1.0, 1.0),
        params_after=(beta_opt, k3_scale_opt, c2_scale_opt),
        fmax=args.fmax, out_stem=out_stem,
    )
    print(f"Plot:  {out_stem}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
