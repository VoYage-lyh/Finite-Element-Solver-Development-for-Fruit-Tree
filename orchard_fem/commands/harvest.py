from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path


def _handle_harvest(args: argparse.Namespace, application) -> int:
    from orchard_fem.io.loaders import load_orchard_model
    from orchard_fem.harvest.detachment import (
        compute_detachment_spectrum,
        compute_detachment_at_frequency,
    )
    from orchard_fem.harvest.optimization import optimize_harvest_excitation

    model = load_orchard_model(str(args.model_json))

    # Load batch FRF results from a CSV produced by batch-run or run
    if not args.frf_csv:
        print("ERROR: --frf-csv is required.")
        return 1

    # Build a minimal BatchExcitationResult-compatible structure from a CSV
    batch_results = _load_batch_results_from_csv(args.frf_csv)
    if not batch_results:
        print(f"ERROR: could not parse any FRF data from {args.frf_csv}")
        return 1

    d_detach = args.detachment_displacement
    print(f"Detachment displacement threshold: {d_detach*1000:.1f} mm")
    print(f"Model fruits: {len(model.fruits)}")

    if args.frequency is not None:
        result = compute_detachment_at_frequency(
            model, batch_results, args.frequency,
            detachment_displacement_m=d_detach,
        )
        print(f"\n@ {result.frequency_hz:.2f} Hz: {result.n_detached}/{result.total_fruits} fruits detached ({result.detachment_fraction*100:.1f}%)")
        for state in result.states:
            status = "DETACHED" if state.detached else "attached"
            print(f"  fruit[{state.fruit_index}] branch={state.branch_id} s={state.location_s:.2f}  F_in={state.inertia_force_n:.4f} N  F_det={state.detachment_force_n:.4f} N  → {status}")
    else:
        spectrum = compute_detachment_spectrum(
            model, batch_results,
            detachment_displacement_m=d_detach,
        )
        print("\nDetachment spectrum:")
        for freq, frac in zip(spectrum.frequencies_hz, spectrum.detachment_fractions):
            bar = "#" * int(frac * 40)
            print(f"  {freq:6.2f} Hz  {frac*100:5.1f}%  {bar}")

        top = optimize_harvest_excitation(
            model, batch_results,
            n_top=args.n_top,
            detachment_displacement_m=d_detach,
        )
        print(f"\nTop-{args.n_top} harvest frequencies:")
        for rank, (freq, frac) in enumerate(top, 1):
            print(f"  #{rank}  {freq:.2f} Hz  →  {frac*100:.1f}% detachment")

        if args.output_csv:
            _write_spectrum_csv(args.output_csv, spectrum)
            print(f"\nWrote detachment spectrum to {args.output_csv}")

        if args.plot:
            _plot_spectrum(spectrum, args.output_csv)

    return 0


def _load_batch_results_from_csv(frf_csv: Path):
    """Build a minimal list of pseudo-BatchExcitationResult objects from a CSV.

    The CSV is expected to have at least ``frequency_hz`` and one magnitude
    column.  Returns a one-element list wrapping a FrequencyResponseResult.
    """
    from orchard_fem.dynamics.frequency_response import (
        FrequencyResponsePoint,
        FrequencyResponseResult,
    )
    from unittest.mock import MagicMock

    try:
        with open(frf_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            # pick magnitude column
            mag_col = next(
                (h for h in headers if "mag" in h.lower() or "amplitude" in h.lower()),
                None,
            )
            if mag_col is None and len(headers) >= 2:
                mag_col = headers[1]
            if mag_col is None:
                return []

            points = []
            for row in reader:
                try:
                    freq = float(row["frequency_hz"])
                    mag = float(row[mag_col])
                    points.append(
                        FrequencyResponsePoint(
                            frequency_hz=freq,
                            excitation_response_magnitude=mag,
                            observation_magnitudes=[mag],
                        )
                    )
                except (KeyError, ValueError):
                    continue
    except OSError:
        return []

    if not points:
        return []

    result = FrequencyResponseResult(
        observation_names=["loaded"],
        points=points,
    )
    br = MagicMock()
    br.result = result
    return [br]


def _write_spectrum_csv(output_csv: Path, spectrum) -> None:
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frequency_hz", "detachment_fraction", "n_detached", "total_fruits"])
        for freq, frac, res in zip(
            spectrum.frequencies_hz,
            spectrum.detachment_fractions,
            spectrum.results,
        ):
            writer.writerow([freq, frac, res.n_detached, res.total_fruits])


def _plot_spectrum(spectrum, output_csv: Path | None) -> None:
    try:
        from orchard_fem.visualization import plot_detachment_spectrum

        fig = plot_detachment_spectrum(spectrum)
        if output_csv:
            img_path = Path(output_csv).with_suffix(".png")
            fig.savefig(img_path, dpi=150)
            print(f"Saved spectrum plot to {img_path}")
        else:
            fig.show()
    except Exception as exc:
        print(f"WARNING: could not generate plot: {exc}")


def register_harvest_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application,
) -> None:
    parser = subparsers.add_parser(
        "harvest",
        help="Compute fruit detachment spectrum and optimal harvest frequency.",
    )
    parser.add_argument("model_json", type=Path, help="Path to the orchard model JSON file.")
    parser.add_argument(
        "--frf-csv",
        type=Path,
        required=True,
        metavar="CSV",
        help="FRF output CSV (frequency_hz, magnitude columns) from 'run' or 'batch-run'.",
    )
    parser.add_argument(
        "--detachment-displacement",
        type=float,
        default=0.010,
        metavar="M",
        help="Detachment displacement threshold [m] (default: 0.010 = 10 mm).",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=None,
        metavar="HZ",
        help="Evaluate detachment at a single frequency instead of the full spectrum.",
    )
    parser.add_argument(
        "--n-top",
        type=int,
        default=3,
        help="Number of top harvest frequencies to report (default: 3).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write detachment spectrum to a CSV file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate a detachment spectrum plot (requires matplotlib).",
    )
    parser.set_defaults(handler=partial(_handle_harvest, application=application))
