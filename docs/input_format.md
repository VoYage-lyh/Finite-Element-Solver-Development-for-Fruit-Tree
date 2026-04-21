# Orchard Model Input Format

This document summarizes the current JSON input format used by the orchard vibration solver MVP.

## Top-Level Keys

- `metadata`: model name and cultivar tags.
- `materials`: xylem/pith/phloem material definitions.
- `branches`: hierarchical branch definitions with centerline and section stations.
- `joints`: optional branch-connection laws.
- `fruits`: optional fruit attachments.
- `clamps`: clamp support definitions.
- `excitation`: harmonic excitation definition.
- `analysis`: frequency-response or time-history settings.
- `observations`: branch or fruit outputs to record.

## Materials

Each material entry supports:

- `id`
- `tissue`: `xylem`, `pith`, or `phloem`
- `model`: `linear`, `nonlinear`, or `orthotropic_placeholder`
- `density`
- `youngs_modulus`
- `poisson_ratio`
- `damping_ratio`
- `nonlinear_alpha` for the placeholder nonlinear elastic model

## Branches

Each branch entry supports:

- `id`
- `parent_branch_id`: `null` for the trunk/root branch
- `level`
- `start`: `[x, y, z]`
- `end`: `[x, y, z]`
- `discretization.num_elements`
- `discretization.hotspot`
- `stations`

`num_elements` controls how many 2-node Euler-Bernoulli beam elements are used along the branch centerline.

Each station entry supports:

- `s`: normalized axial station location
- `profile_type`: `parameterized` or `contour`
- `regions`

Each region entry supports:

- `tissue`
- `material_id`
- `shape`

Supported `shape.type` values:

- `solid_ellipse`
- `elliptic_ring`
- `polygon`

## Joints

Each joint entry supports:

- `id`
- `parent_branch_id`
- `child_branch_id`
- `linear_stiffness_scale`
- `law`

Supported joint law types:

- `polynomial`
  - `linear_scale`
  - `cubic_scale`
- `gap_friction`
  - `closed_scale`
  - `open_scale`
  - `gap_threshold`

`polynomial.cubic_scale` is currently interpreted in the reduced structural coordinate space used by the fast solver, not as a direct continuum constitutive coefficient.

## Fruits

Each fruit entry supports:

- `id`
- `branch_id`
- `location_s`
- `mass`
- `stiffness`
- `damping`

## Clamps

Each clamp entry supports:

- `branch_id`
- `support_stiffness`
- `support_damping`
- `cubic_stiffness`

The current beam-based assembler treats a clamp as a penalty-enforced root constraint on all 6 DOFs of the branch root node.

`cubic_stiffness` activates a localized nonlinear clamp support term on the root `ux` DOF in time-history analysis.

## Excitation

The current harmonic excitation entry supports:

- `kind`: `harmonic_force`, `harmonic_displacement`, or `harmonic_acceleration`
- `target_branch_id`
- `target_node`: `root`, `tip`, or an integer node index
- `target_component`: `ux`, `uy`, or `uz`
- `amplitude`
- `phase_degrees`
- `driving_frequency_hz`

## Analysis

The current analysis entry supports:

- `mode`: `frequency_response` or `time_history`
- `frequency_start_hz`
- `frequency_end_hz`
- `frequency_steps`
- `time_step_seconds`
- `total_time_seconds`
- `output_stride`
- `max_nonlinear_iterations`
- `nonlinear_tolerance`
- `rayleigh_alpha`
- `rayleigh_beta`
- `output_csv`

Frequency-response mode uses the linearized assembled operators of the Euler-Bernoulli beam model.

Time-history mode uses Newmark average-acceleration integration with localized nonlinear links.

Frequency-response CSV output starts with:

- `frequency_hz`
- `excitation_response`
- one column per observation

Time-history CSV output starts with:

- `time_s`
- `excitation_signal`
- `excitation_load`
- `excitation_response`
- one column per observation

## Observations

Each observation entry supports:

- `id`
- `target_type`: `branch` or `fruit`
- `target_id`
- `target_node`: `root`, `tip`, or an integer node index for branch observations
- `target_component`: `ux`, `uy`, or `uz` for branch observations

## Visualization Scripts

- `scripts/plot_frequency_response.py`: `matplotlib`-based frequency-response plotting helper.
- `scripts/visualize_analysis.py`: `numpy`/`matplotlib` orchard geometry plus excitation/measurement visualization helper.

`visualize_analysis.py` reads the model JSON together with the response CSV and highlights:

- branch geometry,
- fruit locations,
- the excitation point,
- measurement points,
- time/frequency/spectrogram response panels for time-history results.
