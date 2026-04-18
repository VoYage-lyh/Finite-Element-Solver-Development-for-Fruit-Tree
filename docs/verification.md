# Verification Suite

This document records the current solver verification cases and the level of confidence each one is meant to provide.

The intent is not to prove that the orchard solver is a full general-purpose FEM package. The intent is to keep a compact, orchard-focused regression guardrail around the parts that matter most to the dedicated fast solver roadmap:

- beam-element kinematics and mass/stiffness assembly,
- boundary-condition handling,
- modal accuracy under mesh refinement,
- nonlinear time integration for localized hardening behavior,
- reduced joint flexibility benchmarks.

## How to run

When CMake/CTest is available:

```bash
ctest -L verification
```

In the current local environment, `cmake` is not available on `PATH`, so these verification cases are also compiled and run directly with `g++` during local development.

## Case Inventory

### 1. Cantilever First Mode

- File: `tests/verification/case_cantilever_first_mode.cpp`
- Model: single branch, fixed-free Euler-Bernoulli cantilever assembled through the full orchard model pipeline.
- Purpose: regression check that the orchard-specific beam assembly, clamp handling, excitation mapping, and observation extraction recover the first bending mode at the application level.
- Reference:

  \[
  \omega_1 = \beta_1^2 \sqrt{\frac{EI}{\rho A L^4}}, \quad \beta_1 L = 1.8751040687
  \]

- Acceptance criterion: peak frequency from the frequency-response scan must stay within 5% of the analytical reference.

### 2. Cantilever Multi-Mode Convergence

- File: `tests/verification/case_cantilever_modes.cpp`
- Model: uniform planar beam assembled from the same Euler-Bernoulli element matrices used by the orchard solver.
- Purpose: verify that modal frequencies converge at the expected rate as the beam mesh is refined.
- Reference:

  \[
  \omega_n = \beta_n^2 \sqrt{\frac{EI}{\rho A L^4}}
  \]

  with

  - \(\beta_1 L = 1.8751040687\)
  - \(\beta_2 L = 4.6940911330\)
  - \(\beta_3 L = 7.8547574382\)

- Acceptance criteria:
  - at `num_elements = 32`, the first three modal frequencies must each be below 0.5% relative error;
  - when refining `2 -> 4 -> 8 -> 16 -> 32` elements, the modal-error `L2` norm must drop by at least a factor of 3 each time.

### 3. Simply Supported Beam Static Deflection

- File: `tests/verification/case_simple_beam_static.cpp`
- Model: uniform beam with simply supported end translations and a concentrated midspan force.
- Purpose: verify the static stiffness path independently of dynamic terms.
- Reference:

  \[
  \delta_{mid} = \frac{F L^3}{48 E I}
  \]

- Acceptance criterion: midspan deflection error must stay below 0.1%.

### 4. Duffing Frequency Response

- File: `tests/verification/case_duffing_frequency_response.cpp`
- Model: single-degree-of-freedom oscillator with cubic hardening spring, integrated in time with the solver's Newmark nonlinear path.
- Purpose: verify that the nonlinear time-history path reproduces the expected hardening trend before full harmonic-balance frequency-domain support is introduced.
- Reference backbone estimate:

  \[
  \omega_{peak} \approx \sqrt{\frac{k + \frac{3}{4} k_3 a^2}{m}}
  \]

  where \(a\) is the measured steady-state amplitude from the latter half of the simulated response.

- Acceptance criteria:
  - the measured peak frequency must agree with the backbone estimate within 5%;
  - the nonlinear peak must lie at least 3% above the linear natural frequency, confirming hardening behavior.

### 5. Hinged Two-Bar Benchmark

- File: `tests/verification/case_hinged_two_bar.cpp`
- Model: two planar beam segments connected by a rotational spring, with a concentrated tip mass.
- Purpose: regression guardrail for future joint-flexibility work.
- Reference: rigid-link spring-mass asymptote

  \[
  \omega_1 \approx \sqrt{\frac{K_\theta}{m L^2}}
  \]

- Acceptance criterion: first frequency must stay within 3% of the asymptotic estimate.

## Scope Notes

- Cases 1-3 are classical analytical beam benchmarks and should remain the primary regression shield for element assembly changes.
- Case 4 is a nonlinear engineering benchmark rather than a closed-form exact solution. It is intentionally used as a practical guardrail for hardening response until harmonic-balance verification is added.
- Case 5 is currently an asymptotic benchmark, not a full closed-form continuous jointed-beam solution. It is still useful because it will catch regressions in rotational-spring assembly before the dedicated orchard joint model is upgraded.

## Maintenance Guidance

- If `solver_core`, `branches`, `discretization`, or `joints_and_bifurcations` changes, rerun `ctest -L verification`.
- If a verification case starts failing after a modeling upgrade, prefer updating the model assumptions and documented reference together rather than silently loosening tolerances.
- Any new reduction or nonlinear-frequency capability should add its own verification case here instead of only adding smoke tests.
