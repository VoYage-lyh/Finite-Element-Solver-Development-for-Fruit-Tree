"""Tests for the harvest recommendation pipeline (fake FE stages, no dolfinx).

The FE-bound stages (FRF sweep, Pareto evaluator) are injected as synthetic
functions, so the orchestration — resonance detection, rig-envelope clipping,
Pareto/knee extraction, best-clamp choice, decision trace, JSON round-trip —
is exercised in the lightweight env.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orchard_fem.actuator.harvest_bridge import DS5L1Limits
from orchard_fem.io.loaders import load_orchard_model
from orchard_fem.workflows.harvest_recommendation import (
    RecommendationOptions,
    RecommendationResult,
    build_frequency_grid,
    candidate_clamp_labels,
    find_in_band_resonance,
    generate_linear_fruits,
    recommend_harvest_parameters,
    rig_feasible,
    summarize_orchard_model,
)

MODEL_PATH = "examples/demo_orchard.json"


@pytest.fixture(scope="module")
def model():
    return load_orchard_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Summary / helpers (no FE)
# ---------------------------------------------------------------------------

def test_summarize_model(model):
    s = summarize_orchard_model(model, MODEL_PATH)
    assert s.n_branches == len(model.branches) > 0
    assert s.height_m > 0
    assert s.n_materials == len(model.materials)
    text = "\n".join(s.lines())
    assert s.name in text and "分枝数" in text


def test_candidate_clamp_labels_override(model):
    auto = candidate_clamp_labels(model, RecommendationOptions())
    assert any(label.startswith("trunk@") for label in auto)
    manual = candidate_clamp_labels(
        model, RecommendationOptions(clamp_labels=("trunk@0.50",)))
    assert manual == ["trunk@0.50"]


def test_generate_linear_fruits():
    # demo_orchard.json carries explicit fruits but no policy → use tree_1
    tree = load_orchard_model("trees/tree_1.json")
    fruits = generate_linear_fruits(tree, tree.fruit_policy, spacing=0.25)
    non_trunk = [b for b in tree.branches if b.branch_id != "trunk"]
    assert len(fruits) == 4 * len(non_trunk)
    by_branch = {}
    for f in fruits:
        assert f.branch_id != "trunk"
        by_branch.setdefault(f.branch_id, []).append(f)
    # stiffness decreases from root to tip on every branch
    for fl in by_branch.values():
        fl.sort(key=lambda f: f.location_s)
        ks = [f.stiffness for f in fl]
        assert all(a > b for a, b in zip(ks, ks[1:]))


def test_find_in_band_resonance_picks_peak():
    freqs = np.linspace(1, 20, 100)
    mags = 1.0 / (np.abs(freqs - 8.0) + 0.3)     # peak at 8 Hz
    idx, genuine = find_in_band_resonance(freqs, mags, (3.0, 20.0))
    assert genuine is True
    assert freqs[idx] == pytest.approx(8.0, abs=0.3)


def test_build_frequency_grid_band_guard():
    grid = build_frequency_grid(19.5, [5.0], (3.0, 20.0))
    assert all(1.5 <= f <= 21.5 for f in grid)
    assert 19.5 in grid and 5.0 in grid


def test_rig_feasible_envelope():
    lim = DS5L1Limits()
    assert rig_feasible(2.0, 5.0, lim) is True          # stroke 10mm @ 2Hz
    assert rig_feasible(2.0, 15.0, lim) is False        # stroke 30mm > 20mm
    assert rig_feasible(14.0, 9.0, lim) is False        # f unreachable at 18mm


# ---------------------------------------------------------------------------
# Full pipeline with synthetic FE stages
# ---------------------------------------------------------------------------

def _fake_sweep(_model, f_min, f_max, steps):
    freqs = np.linspace(f_min, f_max, steps)
    mags = 1e-3 / (np.abs(freqs - 6.0) + 0.4)           # resonance at 6 Hz
    return freqs, mags


def _fake_evaluator_factory(_model, _options):
    def evaluate(frequency_hz, amplitude_mm, clamp_label):
        # coverage rises with amplitude and proximity to 6 Hz; trunk@0.40 is
        # better coupled than other clamps; stress grows with f·A.
        gain = 1.3 if clamp_label == "trunk@0.40" else 1.0
        proximity = 1.0 / (1.0 + abs(frequency_hz - 6.0))
        coverage = min(1.0, gain * proximity * amplitude_mm / 10.0)
        stress = 0.4e6 * frequency_hz * amplitude_mm
        return coverage, stress
    return evaluate


def _run(model, **opt_kwargs):
    options = RecommendationOptions(
        clamp_labels=("trunk@0.40", "trunk@0.70"),
        amplitude_grid_mm=(2.5, 5.0, 10.0, 30.0),
        dense_fruit_spacing=None,
        detachment_displacement_m=None,
        **opt_kwargs,
    )
    return recommend_harvest_parameters(
        model, model_path=MODEL_PATH, options=options,
        frf_sweep=_fake_sweep, evaluator_factory=_fake_evaluator_factory,
    )


def test_pipeline_finds_resonance_and_knee(model):
    result = _run(model)
    assert result.resonance_hz == pytest.approx(6.0, abs=0.5)
    rec = result.recommended
    assert rec is not None and rec.is_knee
    assert rec.clamp_label == "trunk@0.40"              # better-coupled clamp wins
    assert rec.rig_feasible
    # 30 mm amplitude (60 mm stroke) must have been dropped from the grid
    assert 30.0 not in result.amplitude_grid_mm
    assert any("行程" in s and "30" in s for s in result.steps)


def test_pipeline_knee_points_within_envelope(model):
    result = _run(model)
    for clamp in result.clamps:
        if clamp.knee is not None:
            assert clamp.knee.rig_feasible
            assert clamp.knee.stroke_mm <= DS5L1Limits().max_stroke_mm


def test_pipeline_steps_trace(model):
    result = _run(model)
    text = "\n".join(result.steps)
    assert "FRF 扫频" in text and "主共振" in text and "推荐工作点" in text


def test_pipeline_progress_and_cancel(model):
    fractions: list[float] = []
    with pytest.raises(RuntimeError, match="cancelled"):
        recommend_harvest_parameters(
            model,
            options=RecommendationOptions(
                clamp_labels=("trunk@0.40",), amplitude_grid_mm=(5.0,),
                dense_fruit_spacing=None, detachment_displacement_m=None),
            frf_sweep=_fake_sweep,
            evaluator_factory=_fake_evaluator_factory,
            progress_cb=lambda _m, f: fractions.append(f),
            cancel_cb=lambda: len(fractions) > 3,       # cancel mid-run
        )
    assert fractions  # progress was reported before the cancel


def test_pipeline_stress_ceiling_filters(model):
    # ceiling below every candidate's stress → no executable point anywhere
    with pytest.raises(RuntimeError, match="No executable working point"):
        _run(model, stress_ceiling_pa=1.0)


def test_result_json_roundtrip(model, tmp_path):
    result = _run(model)
    p = tmp_path / "rec.json"
    result.save_json(p)
    loaded = RecommendationResult.load_json(p)
    assert loaded.recommended.frequency_hz == result.recommended.frequency_hz
    assert loaded.recommended.amplitude_mm == result.recommended.amplitude_mm
    assert loaded.steps == result.steps
    assert len(loaded.clamps) == len(result.clamps)


def test_working_point_params_json(model):
    result = _run(model)
    d = result.recommended.to_params_json(duration_s=12.0)
    assert d["duration_s"] == 12.0
    assert d["displacement_amplitude_m"] == pytest.approx(
        result.recommended.amplitude_mm / 1000.0)
    # schema consumed by scripts/run_harvest_on_rig.py
    assert {"frequency_hz", "displacement_amplitude_m", "duration_s"} <= set(d)


def test_stroke_is_twice_amplitude(model):
    result = _run(model)
    rec = result.recommended
    assert rec.stroke_mm == pytest.approx(2.0 * rec.amplitude_mm)
    assert math.isfinite(rec.trunk_stress_pa)
