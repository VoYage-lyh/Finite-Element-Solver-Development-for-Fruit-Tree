"""PINN training data generator.

Sweeps a Latin Hypercube sample of material parameters and fruit-mass scales,
runs ``run_batch_frequency_response`` for each sample, and stores the results
as a compressed numpy ``.npz`` file (optionally HDF5 if ``h5py`` is available).

Output schema
-------------
When saved as ``.npz``:

* ``params``              float32  [N_samples × N_params]
* ``param_names``         object   [N_params]   — UTF-8 parameter name strings
* ``excitation_labels``   object   [N_specs]    — "<branch>_<node>_<comp>" labels
* ``frequencies``         float32  [N_freqs]
* ``responses``           float32  [N_samples × N_specs × N_freqs]

Example
-------
::

    from orchard_pinn.data.parameter_space import (
        ParameterRange, ParameterSpaceConfig,
    )
    from orchard_pinn.data.generator import generate_training_dataset
    from orchard_fem.workflows.batch_excitation import ExcitationSpec

    config = ParameterSpaceConfig(
        material_youngs_modulus_ranges={
            "xylem_default": ParameterRange("xylem_E", 5.0e9, 2.0e10),
        },
        fruit_mass_scale_range=ParameterRange("fruit_mass_scale", 0.5, 2.0),
        excitation_specs=[
            ExcitationSpec("trunk", "tip", "ux"),
        ],
        n_samples=10,
        seed=0,
    )
    path = generate_training_dataset(base_model, config, "data/train.npz")
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchard_fem.domain import OrchardModel

from orchard_pinn.data.parameter_space import ParameterSpaceConfig, latin_hypercube_sample


def _scale_model_materials(
    model: "OrchardModel",
    sample: dict[str, float],
    config: ParameterSpaceConfig,
) -> "OrchardModel":
    """Return a clone of *model* with material Young's moduli scaled by *sample*."""

    material_id_to_range = config.material_youngs_modulus_ranges

    new_materials = []
    for mat in model.materials:
        if mat.material_id in material_id_to_range:
            param_range = material_id_to_range[mat.material_id]
            scaled_E = sample.get(param_range.name, mat.youngs_modulus)
            new_materials.append(replace(mat, youngs_modulus=scaled_E))
        else:
            new_materials.append(mat)

    return replace(model, materials=new_materials)


def _scale_model_fruits(
    model: "OrchardModel",
    fruit_mass_scale: float,
) -> "OrchardModel":
    """Return a clone of *model* with all fruit masses multiplied by *fruit_mass_scale*."""

    if abs(fruit_mass_scale - 1.0) < 1.0e-12:
        return model

    new_fruits = [
        replace(fruit, mass=fruit.mass * fruit_mass_scale)
        for fruit in model.fruits
    ]
    return replace(model, fruits=new_fruits)


def generate_training_dataset(
    base_model: "OrchardModel",
    config: ParameterSpaceConfig,
    output_path: str | Path,
    *,
    n_workers: int = 1,
    verbose: bool = True,
) -> Path:
    """Generate a frequency-response dataset for PINN training.

    For each Latin Hypercube sample in *config*, clones *base_model* with
    scaled material properties, runs ``run_batch_frequency_response`` across
    all ``config.excitation_specs``, and accumulates results into a numpy
    ``.npz`` file.

    Parameters
    ----------
    base_model:
        The baseline ``OrchardModel`` (already loaded and validated).
    config:
        Parameter space and excitation specification.
    output_path:
        Destination ``.npz`` (or ``.h5`` / ``.hdf5``) file path.
    n_workers:
        Number of parallel workers.  Currently only ``n_workers=1`` is
        supported; pass ``n_workers>1`` to get an informational note.
    verbose:
        If True, print progress to stdout.

    Returns
    -------
    Path
        Absolute path to the written dataset file.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for generate_training_dataset. "
            "Install it with: pip install numpy"
        ) from exc

    from orchard_fem.workflows.batch_excitation import run_batch_frequency_response

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if n_workers > 1 and verbose:
        print(
            f"[generator] n_workers={n_workers} requested; "
            "parallel workers not yet implemented — running sequentially."
        )

    samples = latin_hypercube_sample(config)
    param_names = config.parameter_names()
    n_samples = len(samples)
    n_specs = len(config.excitation_specs)

    if n_samples == 0 or n_specs == 0:
        raise ValueError("config must have n_samples > 0 and at least one excitation_spec.")

    # Determine the frequency grid length from the model analysis settings.
    from orchard_fem.fenicsx.frequency_response import _frequency_grid
    frequencies = _frequency_grid(base_model.analysis)
    n_freqs = len(frequencies)

    params_array = np.zeros((n_samples, len(param_names)), dtype=np.float32)
    responses_array = np.zeros((n_samples, n_specs, n_freqs), dtype=np.float32)

    fruit_mass_scale_name = config.fruit_mass_scale_range.name

    for sample_index, sample in enumerate(samples):
        if verbose:
            print(f"[generator] Sample {sample_index + 1}/{n_samples} …", flush=True)

        for param_idx, name in enumerate(param_names):
            params_array[sample_index, param_idx] = float(sample.get(name, 0.0))

        fruit_mass_scale = float(sample.get(fruit_mass_scale_name, 1.0))
        model = _scale_model_materials(base_model, sample, config)
        model = _scale_model_fruits(model, fruit_mass_scale)

        batch_results = run_batch_frequency_response(model, config.excitation_specs)

        for spec_index, batch_result in enumerate(batch_results):
            for freq_index, point in enumerate(batch_result.result.points):
                responses_array[sample_index, spec_index, freq_index] = float(
                    point.excitation_response_magnitude
                )

    excitation_labels = np.array(
        [
            f"{spec.branch_id}_{spec.node}_{spec.component}"
            for spec in config.excitation_specs
        ],
        dtype=object,
    )

    if output_path.suffix in {".h5", ".hdf5"}:
        _save_hdf5(
            output_path,
            params=params_array,
            param_names=param_names,
            excitation_labels=excitation_labels,
            frequencies=np.array(frequencies, dtype=np.float32),
            responses=responses_array,
        )
    else:
        np.savez_compressed(
            output_path,
            params=params_array,
            param_names=np.array(param_names, dtype=object),
            excitation_labels=excitation_labels,
            frequencies=np.array(frequencies, dtype=np.float32),
            responses=responses_array,
        )

    if verbose:
        print(f"[generator] Dataset written to {output_path.resolve()}")
        print(
            f"[generator] Shape: params={params_array.shape}, "
            f"responses={responses_array.shape}"
        )

    return output_path.resolve()


def _save_hdf5(
    output_path: Path,
    *,
    params,
    param_names: list[str],
    excitation_labels,
    frequencies,
    responses,
) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 output. Install it with: pip install h5py"
        ) from exc


    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("params", data=params, compression="gzip")
        h5.create_dataset("frequencies", data=frequencies, compression="gzip")
        h5.create_dataset("responses", data=responses, compression="gzip")
        dt = h5py.special_dtype(vlen=str)
        names_ds = h5.create_dataset("param_names", (len(param_names),), dtype=dt)
        for i, name in enumerate(param_names):
            names_ds[i] = name
        labels_ds = h5.create_dataset("excitation_labels", (len(excitation_labels),), dtype=dt)
        for i, label in enumerate(excitation_labels):
            labels_ds[i] = str(label)


def load_training_dataset(dataset_path: str | Path) -> dict:
    """Load a dataset previously created by :func:`generate_training_dataset`.

    Returns a dict with keys ``params``, ``param_names``, ``excitation_labels``,
    ``frequencies``, ``responses``.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required to load training datasets.") from exc

    dataset_path = Path(dataset_path)

    if dataset_path.suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to load HDF5 datasets.") from exc
        with h5py.File(dataset_path, "r") as h5:
            return {
                "params": h5["params"][:],
                "param_names": list(h5["param_names"].asstr()[:]),
                "excitation_labels": list(h5["excitation_labels"].asstr()[:]),
                "frequencies": h5["frequencies"][:],
                "responses": h5["responses"][:],
            }

    data = np.load(dataset_path, allow_pickle=True)
    return {
        "params": data["params"],
        "param_names": list(data["param_names"]),
        "excitation_labels": list(data["excitation_labels"]),
        "frequencies": data["frequencies"],
        "responses": data["responses"],
    }
