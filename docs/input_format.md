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

## Skeleton Import

Use the CLI converter when the upstream data is a 3D centerline skeleton:

```bash
python -m orchard_fem import-skeleton tests/fixtures/skeleton_orchard.json build/orchard_model.json
```

The skeleton format accepts branch `points`, radius shorthand fields such as
`outer_radius_root` / `outer_radius_tip`, optional fruits, clamps, joints, excitation,
analysis, and observations. The converter emits the solver-ready JSON described below.
It does not invent an excitation or analysis block; those must be provided explicitly.

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
- `points`: optional polyline centerline, `[[x, y, z], ...]`; when present it overrides straight `start/end` stationing while still exporting `start` and `end` for compatibility
- `discretization.num_elements`
- `discretization.hotspot`
- `stations`

`num_elements` controls how many 2-node embedded beam elements are used along the branch centerline. For polyline branches, stations are measured by accumulated arc length.

For quick model construction, a station may use the default circular shorthand:

```json
{"s": 0.0, "shorthand": "circular", "outer_radius": 0.025}
```

This expands to concentric `pith_default`, `xylem_default`, and `phloem_default` regions. Those material IDs must exist in `materials`.

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
  - `linear_scale`
  - `open_scale`
  - `gap_threshold`

`polynomial.cubic_scale` is currently interpreted in the reduced structural coordinate space used by the fast solver, not as a direct continuum constitutive coefficient.

In the current beam-based assembler, explicit joint laws are applied on the rotational root DOFs
(`rx`, `ry`, `rz`) between the child branch root and its nearest parent node. The linear joint
constraint remains penalty-enforced, while `polynomial` adds cubic rotational links and
`gap_friction` switches between closed/open rotational stiffness states after the configured gap.

## Fruits

Each fruit entry supports:

- `id`
- `branch_id`
- `location_s`
- `mass`
- `stiffness`
- `damping`
- `target_component`: translational branch component coupled to the fruit DOF, default `ux`; use `uz` when fruit gravity should act in the default vertical gravity direction

Fruits are represented as concentrated mass plus spring-damper attachments. When
`include_gravity_prestress` is enabled, fruit self-weight is added to the static preload
through the configured `target_component`.

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
- `solver_backend`: `native` or `fenicsx`
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

`solver_backend` defaults to `fenicsx`.
Frequency-response fields use the `AnalysisSettings` defaults when omitted, so pure time-history JSON files do not need to carry unused frequency sweep values.

- `fenicsx` is the main solver backend for modal, frequency-response, and time-history cases.
  It supports branch observations, fruit-attachment augmentation, gravity prestress, linear joint constraints,
  localized nonlinear links, first-harmonic nonlinear frequency continuation, and PETSc SNES nonlinear time history.
- `native` uses the older Python beam assembly and solver stack. It remains available only when explicitly requested with `"solver_backend": "native"` or the CLI `--solver-backend native` override.

Frequency-response mode uses the direct linearized assembled operators when no localized nonlinear
links are active. If localized nonlinear links are active, both active backends use
adaptive first-harmonic balance continuation and report amplitudes on the same frequency-response
CSV grid.

Time-history mode uses Newmark average-acceleration integration. FEniCSx localized nonlinear links are solved with PETSc SNES.

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
