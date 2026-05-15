"""CLI wrapper around :func:`orchard_fem.processing.hammer_io.process_hammer_to_frf`."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path


def _handle_process_hammer(args: argparse.Namespace, application) -> int:
    from orchard_fem.processing import process_hammer_to_frf

    paths = list(args.csv_paths)
    if not paths:
        print("ERROR: at least one CSV path is required.")
        return 1
    summary = process_hammer_to_frf(
        paths,
        output_csv=args.output_csv,
        modal_sidecar_json=args.modal_sidecar,
        gamma_min=args.gamma_min,
        nperseg=args.nperseg,
        n_modes=args.n_modes,
        frequency_band_hz=(args.band_min_hz, args.band_max_hz),
        time_column=args.time_column,
        force_column=args.force_column,
        response_column=args.response_column,
        skip_header=args.skip_header,
    )
    print(f"[process-hammer] wrote {summary['output_csv']}")
    print(f"[process-hammer] used {summary['n_records_used']}/"
          f"{summary['n_records_total']} records (coherence ≥ {args.gamma_min})")
    for i, mode in enumerate(summary["modes"], 1):
        print(f"  mode {i}: f={mode['frequency_hz']:.3f} Hz  "
              f"zeta={mode['damping_ratio']*100:.2f}%")
    return 0


def register_process_hammer_command(
    subparsers: argparse._SubParsersAction,
    application,
) -> None:
    parser = subparsers.add_parser(
        "process-hammer",
        help=(
            "Average N hammer-impact CSVs into a single FRF and auto-identify "
            "the first n_modes natural frequencies."
        ),
    )
    parser.add_argument("csv_paths", type=Path, nargs="+",
                        help="One CSV per impact hit (time, force, response).")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Destination CSV (frequency_hz,magnitude[,coherence]).")
    parser.add_argument("--modal-sidecar", type=Path, default=None,
                        help="Optional JSON file with identified modal frequencies.")
    parser.add_argument("--gamma-min", type=float, default=0.8,
                        help="Coherence threshold for hit rejection (default 0.8).")
    parser.add_argument("--nperseg", type=int, default=None,
                        help="Welch segment length (default: min(2048, N)).")
    parser.add_argument("--n-modes", type=int, default=3,
                        help="Number of natural frequencies to identify.")
    parser.add_argument("--band-min-hz", type=float, default=1.0)
    parser.add_argument("--band-max-hz", type=float, default=30.0)
    parser.add_argument("--time-column", type=int, default=0)
    parser.add_argument("--force-column", type=int, default=1)
    parser.add_argument("--response-column", type=int, default=2)
    parser.add_argument("--skip-header", type=int, default=1)
    parser.set_defaults(handler=partial(_handle_process_hammer, application=application))
