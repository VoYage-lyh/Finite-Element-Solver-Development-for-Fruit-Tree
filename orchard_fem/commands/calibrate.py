from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path


def _handle_calibrate(args: argparse.Namespace, application) -> int:
    from orchard_fem.io.loaders import load_orchard_model
    from orchard_fem.calibration.modal_calibration import (
        CalibrationParameter,
        ModalCalibrationConfig,
        calibrate_from_modal_frequencies,
    )

    model = load_orchard_model(str(args.model_json))

    # Parse target frequencies
    target_freqs = [float(f) for f in args.target_frequencies]
    if not target_freqs:
        print("ERROR: --target-frequencies is required.")
        return 1

    # Build parameters from material IDs and bounds
    if args.material_ids:
        material_ids = args.material_ids
    else:
        material_ids = [m.material_id for m in model.materials]

    if not material_ids:
        print("ERROR: no materials found in model.")
        return 1

    parameters = []
    for mat_id in material_ids:
        mat = next((m for m in model.materials if m.material_id == mat_id), None)
        if mat is None:
            print(f"WARNING: material '{mat_id}' not found in model, skipping.")
            continue
        E0 = mat.youngs_modulus
        parameters.append(
            CalibrationParameter(
                material_id=mat_id,
                initial_value=E0,
                lower_bound=E0 * args.lower_scale,
                upper_bound=E0 * args.upper_scale,
            )
        )

    if not parameters:
        print("ERROR: no valid calibration parameters.")
        return 1

    config = ModalCalibrationConfig(
        parameters=parameters,
        target_frequencies_hz=target_freqs,
        n_modes=args.n_modes,
        max_iter=args.max_iter,
        verbose=args.verbose,
    )

    print(f"Calibrating {len(parameters)} material(s) to match {len(target_freqs)} target frequency/ies...")
    result = calibrate_from_modal_frequencies(model, config)

    print(f"Converged: {result.converged}  |  evaluations: {result.n_evaluations}  |  residual: {result.residual_norm:.6g}")
    print("Calibrated Young's moduli:")
    for mat_id, E_cal in result.calibrated_values.items():
        print(f"  {mat_id}: E = {E_cal:.6g} Pa")
    print(f"Final simulated frequencies (Hz): {[f'{f:.4f}' for f in result.final_frequencies_hz]}")

    if args.output_json:
        import json
        output: dict = {
            "converged": result.converged,
            "n_evaluations": result.n_evaluations,
            "residual_norm": result.residual_norm,
            "calibrated_values": result.calibrated_values,
            "final_frequencies_hz": result.final_frequencies_hz,
        }
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"Wrote calibration result to {args.output_json}")

    return 0


def register_calibrate_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application,
) -> None:
    parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate material Young's moduli to match measured modal frequencies.",
    )
    parser.add_argument("model_json", type=Path, help="Path to the orchard model JSON file.")
    parser.add_argument(
        "--target-frequencies",
        type=float,
        nargs="+",
        required=True,
        metavar="HZ",
        help="Measured resonance frequencies [Hz] to match.",
    )
    parser.add_argument(
        "--material-ids",
        type=str,
        nargs="+",
        default=None,
        metavar="ID",
        help="Material IDs to calibrate. Defaults to all materials in the model.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=10,
        help="Number of modes to solve per calibration evaluation (default: 10).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum L-BFGS-B iterations (default: 100).",
    )
    parser.add_argument(
        "--lower-scale",
        type=float,
        default=0.1,
        help="Lower bound as fraction of initial E (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--upper-scale",
        type=float,
        default=10.0,
        help="Upper bound as fraction of initial E (default: 10.0 = 1000%%).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write calibration result to a JSON file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print objective value at each iteration.",
    )
    parser.set_defaults(handler=partial(_handle_calibrate, application=application))
