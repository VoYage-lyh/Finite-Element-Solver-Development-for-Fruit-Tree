"""Stand-alone SDOF Duffing verification of the c2|v|v describing function.

The FE pipeline pins the auto-injected link's relative DOFs to ~0 via joint
penalty and clamp Dirichlet BCs, so the c2|v|v contribution at those links
is structurally suppressed at the harmonic-balance stage.  To verify the
describing-function math itself is correct, this script runs a stand-alone
1-DOF Duffing oscillator with the same constitutive law

    m u'' + c u' + k u + k3 u^3 + c2 |u'| u' = F0 cos(omega t)

via two independent methods:

* **scipy RK45 time integration** — settle to steady state, extract the
  first-harmonic amplitude by FFT (the "ground truth").
* **First-harmonic balance** using :func:`orchard_fem.dynamics.harmonic_balance.first_harmonic_link_response`
  — the same describing function the FE pipeline uses.

We sweep frequency near resonance for three parameter sets:

* linear baseline (k3 = c2 = 0)
* cubic only (k3 != 0, c2 = 0)
* cubic + quadratic damping (k3 != 0, c2 != 0)

Output (under ``results_nonlinear/verification/``):

* ``duffing_sdof_frf.{png,pdf}`` — three FRFs overlaid, with RK45 markers
  showing the HB prediction matches the time-history ground truth.
* ``duffing_sdof_results.csv``  — frequency, amplitude per method.

If HB matches RK45 within ~5 %, the describing-function math is verified —
even if the FE pipeline can't currently route c2 through penalty-pinned
links.
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orchard_fem.discretization.types import NonlinearLinkDefinition, NonlinearLinkKind
from orchard_fem.dynamics.harmonic_balance import first_harmonic_link_response


# ---------------------------------------------------------------------------
# A representative single-element model.
#   m, c, k chosen so the linear resonance sits at ~10 Hz (omega_n = 2pi*10),
#   damping ratio ~5 %, easy to drive with order-unity F0 force.
# ---------------------------------------------------------------------------
M = 0.2                       # kg
K_LIN = M * (2 * math.pi * 10.0) ** 2   # → ~790 N/m  (linear resonance ~10 Hz)
DAMPING_RATIO = 0.05
C_LIN = 2.0 * DAMPING_RATIO * math.sqrt(K_LIN * M)   # → ~1.26 N·s/m
F0 = 0.5                      # N    — keeps peak response ~mm scale
K3 = -3.0e4                   # N/m^3 — mild softening (u_crit ≈ 16 cm, stable)
C2_LARGE = 8.0                # N·s^2/m^2 — c2*v^2 comparable to F0 at resonance

FREQ_MIN_HZ = 8.0
FREQ_MAX_HZ = 12.0
FREQ_STEPS = 33


# ---------------------------------------------------------------------------
# Reference: scipy time integration to steady state.
# ---------------------------------------------------------------------------
def _rk45_steady_state_amplitude(
    *, k3: float, c2: float, omega: float,
    settle_periods: int = 80, sample_periods: int = 20,
) -> float:
    """Integrate until steady state, then return the first-harmonic amplitude."""
    def rhs(t, y):
        u, v = y
        force = F0 * math.cos(omega * t)
        return [v, (force - C_LIN * v - K_LIN * u - k3 * u**3 - c2 * abs(v) * v) / M]

    period = 2.0 * math.pi / omega
    t_settle = settle_periods * period
    t_sample = sample_periods * period
    sample_pts = max(2048, sample_periods * 64)
    t_eval = np.linspace(t_settle, t_settle + t_sample, sample_pts)
    sol = solve_ivp(
        rhs, [0.0, t_settle + t_sample], [0.0, 0.0],
        t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        # Softening Duffing can leave the basin at extreme amplitudes; the
        # SDOF verification is most informative near resonance, so mark this
        # point as missing and continue.
        return float("nan")
    u = sol.y[0]
    # First-harmonic amplitude via projection onto cos/sin(omega t)
    n = u.size
    t = sol.t
    a_cos = (2.0 / n) * np.sum(u * np.cos(omega * t))
    a_sin = (2.0 / n) * np.sum(u * np.sin(omega * t))
    return float(np.sqrt(a_cos**2 + a_sin**2))


# ---------------------------------------------------------------------------
# First-harmonic balance using the codebase's describing function.
# ---------------------------------------------------------------------------
def _hb_amplitude(*, k3: float, c2: float, omega: float) -> float:
    """Solve the first-harmonic-balance equation for the SDOF amplitude.

    Linearized at amplitude A, the steady-state response satisfies
        [ (k_eq(A) - m omega^2) + j (omega c + omega c_eq(A)) ] * U_complex = F0
    with k_eq = K_LIN + 0.75 k3 A^2 and the c2 contribution captured exactly
    by the codebase's :func:`first_harmonic_link_response`.  We solve via
    fixed-point iteration on A.
    """
    link = NonlinearLinkDefinition(
        label="sdof", first_dof=0, second_dof=-1,
        kind=NonlinearLinkKind.CUBIC_SPRING,
        cubic_stiffness=k3, quadratic_damping=c2,
    )
    # Initial guess: linear amplitude.
    z = (K_LIN - M * omega ** 2) + 1j * (omega * C_LIN)
    u_complex = F0 / z
    for _ in range(60):
        u_r, u_i = u_complex.real, u_complex.imag
        link_response = first_harmonic_link_response(link, u_r, u_i, omega)
        # Block equations (real/imag of  (K-mω²)U + jωC U + F_nl = F_ext):
        # (K-mω²) U_r - ωC U_i + F_nl_r = F0
        # ωC U_r + (K-mω²) U_i + F_nl_i = 0
        a = K_LIN - M * omega ** 2
        b = omega * C_LIN
        # Newton-style update: linearise around current guess and solve.
        det = a * a + b * b
        rhs_r = F0 - link_response.force_real
        rhs_i = -link_response.force_imag
        new_u_r = (a * rhs_r + b * rhs_i) / det
        new_u_i = (a * rhs_i - b * rhs_r) / det
        new_u_complex = complex(new_u_r, new_u_i)
        change = abs(new_u_complex - u_complex) / max(abs(new_u_complex), 1e-14)
        u_complex = 0.5 * (u_complex + new_u_complex)
        if change < 1e-7:
            break
    return float(abs(u_complex))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.fontsize": 11,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


def main() -> int:
    _apply_pub_style()
    out_dir = REPO / "results_nonlinear" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    freqs_hz = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, FREQ_STEPS)
    cases = [
        ("linear", 0.0, 0.0, "#1f77b4", "-",  "Linear ($k_3=0,c_2=0$)"),
        ("cubic_only", K3, 0.0, "#9467bd", "--", f"Cubic only ($k_3={K3:.0e}$, $c_2=0$)"),
        ("full", K3, C2_LARGE, "#d62728", "-",
         f"Cubic + $c_2$ ($k_3={K3:.0e}$, $c_2={C2_LARGE:.0f}$)"),
    ]

    results = {}
    for tag, k3, c2, color, ls, label in cases:
        hb_amps = np.empty(freqs_hz.size)
        rk_amps = np.empty(freqs_hz.size)
        print(f"[{tag}] sweeping {len(freqs_hz)} freqs ...")
        for i, f in enumerate(freqs_hz):
            omega = 2 * math.pi * f
            hb_amps[i] = _hb_amplitude(k3=k3, c2=c2, omega=omega)
            rk_amps[i] = _rk45_steady_state_amplitude(k3=k3, c2=c2, omega=omega)
        results[tag] = dict(
            hb=hb_amps, rk=rk_amps, color=color, ls=ls, label=label, k3=k3, c2=c2,
        )

    # Plot overlay
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for tag, r in results.items():
        ax.semilogy(freqs_hz, r["hb"], color=r["color"], ls=r["ls"], lw=1.8,
                    label=r["label"] + " (HB)")
        ax.semilogy(freqs_hz, r["rk"], color=r["color"], marker="o",
                    ms=3.5, ls="None", alpha=0.55, label=r["label"] + " (RK45)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Steady-state amplitude $A$ (m)")
    ax.set_title(r"SDOF Duffing FRF: HB describing function vs RK45 time integration")
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "duffing_sdof_frf.png")
    fig.savefig(out_dir / "duffing_sdof_frf.pdf")
    plt.close(fig)

    # CSV dump
    with open(out_dir / "duffing_sdof_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["frequency_hz"] + [
            f"{tag}_{m}" for tag, _, _, _, _, _ in cases for m in ("hb", "rk45")
        ]
        w.writerow(header)
        for i, f in enumerate(freqs_hz):
            row = [f]
            for tag, _, _, _, _, _ in cases:
                row.append(results[tag]["hb"][i])
                row.append(results[tag]["rk"][i])
            w.writerow(row)

    # Quantify peak suppression
    print()
    print("=" * 72)
    print(f"{'case':12s}  peak_freq(HB)  peak_amp(HB)    peak_amp(RK45)   HB-vs-RK45 error")
    refs = {}
    for tag, r in results.items():
        idx_hb = int(np.argmax(r["hb"]))
        idx_rk = int(np.argmax(r["rk"]))
        # Use the lower envelope freq for fairness
        amp_hb = r["hb"][idx_hb]
        amp_rk = r["rk"][idx_rk]
        ratio = (amp_hb - amp_rk) / amp_rk * 100.0 if amp_rk > 0 else float("nan")
        print(f"{tag:12s}  {freqs_hz[idx_hb]:>10.2f} Hz   {amp_hb:>10.3e}    "
              f"{amp_rk:>10.3e}        {ratio:+6.2f} %")
        refs[tag] = (freqs_hz[idx_hb], amp_hb, freqs_hz[idx_rk], amp_rk)

    cubic_amp = refs["cubic_only"][1]
    full_amp = refs["full"][1]
    suppression = (cubic_amp - full_amp) / cubic_amp * 100.0 if cubic_amp > 0 else 0.0
    print()
    print(f"c2 peak suppression (cubic_only -> full, HB): {suppression:+.2f} %")
    print(f"Output: {out_dir.relative_to(REPO)}/duffing_sdof_frf.{{png,pdf}}, "
          f"duffing_sdof_results.csv")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
