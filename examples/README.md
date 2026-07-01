# Examples

Sample input models. Run from the repo root in the `orchard-fenicsx` environment
(the CLI entry point is `orchard-fem`, installed by `pip install -e .`).

| File | What it is |
|---|---|
| **`tree_3.json`** | A full *Prunus cerasifera* tree — 17 branches with fruits and the calibrated detachment-force / damping policy. The realistic end-to-end **harvest** example. |
| `demo_orchard.json` | A small synthetic orchard used by the built-in demo suite and the test fixtures. |
| `demo_orchard_time_history.json` | The same demo configured for a transient time-history run. |
| `skeleton_orchard.json` | A minimal hand-written model showing the bare input schema — documented in [`docs/input_format.md`](../docs/input_format.md). |

## Quick start — run a simulation on the example tree

```bash
# modal frequencies (fast, self-contained):
orchard-fem modal examples/tree_3.json

# run the analysis configured in the JSON → response CSV:
orchard-fem run examples/tree_3.json

# fruit-detachment spectrum + optimal harvest frequency:
orchard-fem harvest examples/tree_3.json
```

For the full multi-clamp harvest **working-parameter recommendation** (the
project's headline capability — modal per-subtree frequency selection, per-clamp
Pareto of coverage vs trunk stress, multi-stage schedule at the ≤15 Hz / 20 mm
actuator envelope), see `scripts/generate_all_figures.py` and
`orchard_fem.workflows.harvest_recommendation.recommend_harvest_parameters`.
