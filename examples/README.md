# Examples

Sample input models. Run from the repo root in the `orchard-fenicsx` environment
(the CLI entry point is `orchard-fem`, installed by `pip install -e .`).

| Path | What it is |
|---|---|
| **`trees/tree_3.json`** | A full *Prunus cerasifera* tree and the realistic end-to-end harvest example. |
| `trees/tree_1.json` … `trees/tree_5.json` | Five small, version-controlled orchard architectures used by paper and recommendation workflows. |
| `../tests/fixtures/demo_orchard.json` | Small synthetic frequency-response fixture. |
| `../tests/fixtures/demo_orchard_time_history.json` | Compact transient fixture. |
| `../tests/fixtures/skeleton_orchard.json` | Minimal skeleton-import example documented in [`docs/input_format.md`](../docs/input_format.md). |

## Quick start — run a simulation on the example tree

```bash
# modal frequencies (fast, self-contained):
orchard-fem modal examples/trees/tree_3.json

# run the analysis configured in the JSON → response CSV:
orchard-fem run examples/trees/tree_3.json

# fruit-detachment spectrum + optimal harvest frequency:
orchard-fem harvest examples/trees/tree_3.json
```

For the full multi-clamp harvest **working-parameter recommendation** (the
project's headline capability — modal per-subtree frequency selection, per-clamp
Pareto of coverage vs trunk stress, multi-stage schedule at the ≤15 Hz / 20 mm
actuator envelope), see `scripts/generate_all_figures.py` and
`orchard_fem.workflows.harvest_recommendation.recommend_harvest_parameters`.
