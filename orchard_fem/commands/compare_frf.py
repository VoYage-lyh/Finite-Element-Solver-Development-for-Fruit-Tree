from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path


def _handle_compare_frf(args: argparse.Namespace, application) -> int:
    from orchard_fem.io.measurement import (
        load_measured_frf_csv,
        compare_frf,
    )
    from orchard_fem.dynamics.frequency_response import (
        FrequencyResponsePoint,
        FrequencyResponseResult,
    )

    # Load measured FRF
    measured = load_measured_frf_csv(args.measured_csv, label=args.measured_label or "measured")
    print(f"Loaded measured FRF: {len(measured.frequencies_hz)} points  [{measured.frequencies_hz[0]:.2f}–{measured.frequencies_hz[-1]:.2f} Hz]")

    # Load simulated FRF from CSV produced by 'run' command
    sim_result = _load_simulated_frf_csv(args.simulated_csv)
    if sim_result is None:
        print(f"ERROR: could not parse simulated FRF from {args.simulated_csv}")
        return 1
    print(f"Loaded simulated FRF: {len(sim_result.points)} points")

    # Compare
    comparison = compare_frf(measured, sim_result)
    print(f"\nFRF comparison:")
    print(f"  MAC value:         {comparison.mac_value:.4f}")
    print(f"  Frequency points:  {len(comparison.frequencies_hz)}")

    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frequency_hz", "measured_magnitude", "simulated_magnitude"])
            for freq, m_mag, s_mag in zip(
                comparison.frequencies_hz,
                comparison.measured_magnitude,
                comparison.simulated_magnitude,
            ):
                writer.writerow([freq, m_mag, s_mag])
        print(f"Wrote comparison data to {args.output_csv}")

    if args.plot or args.output_plot:
        try:
            from orchard_fem.visualization import plot_frf_comparison

            fig = plot_frf_comparison(
                comparison,
                title=args.title or "",
                output_path=args.output_plot,
            )
            if args.output_plot:
                print(f"Saved FRF comparison plot to {args.output_plot}")
            elif args.plot:
                import matplotlib.pyplot as plt
                plt.show()
        except Exception as exc:
            print(f"WARNING: could not generate plot: {exc}")

    return 0


def _load_simulated_frf_csv(csv_path: Path):
    """Load a simulated FRF CSV with columns frequency_hz, <magnitude>."""
    import csv as csv_mod
    from orchard_fem.dynamics.frequency_response import (
        FrequencyResponsePoint,
        FrequencyResponseResult,
    )

    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            headers = reader.fieldnames or []
            mag_col = next(
                (h for h in headers if "mag" in h.lower() or "amplitude" in h.lower()),
                None,
            )
            if mag_col is None and len(headers) >= 2:
                mag_col = headers[1]
            if mag_col is None:
                return None

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
        return None

    if not points:
        return None

    return FrequencyResponseResult(observation_names=["simulated"], points=points)


def register_compare_frf_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application,
) -> None:
    parser = subparsers.add_parser(
        "compare-frf",
        help="Compare a measured FRF CSV against a simulated FRF CSV and compute MAC.",
    )
    parser.add_argument(
        "measured_csv",
        type=Path,
        help="Measured FRF CSV (frequency_hz + real/imag or magnitude_db/phase_deg columns).",
    )
    parser.add_argument(
        "simulated_csv",
        type=Path,
        help="Simulated FRF CSV from the 'run' command (frequency_hz + magnitude column).",
    )
    parser.add_argument(
        "--measured-label",
        type=str,
        default="measured",
        help="Label for the measured curve in plots (default: 'measured').",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Figure title.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write comparison data (freq, measured_mag, simulated_mag) to a CSV.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Save FRF comparison figure to this path (PNG/PDF). Implies --plot.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Show FRF comparison figure interactively (requires matplotlib).",
    )
    parser.set_defaults(handler=partial(_handle_compare_frf, application=application))
