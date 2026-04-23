from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Sequence

PLOT_INSTALL_HINT = (
    "Frequency-response plotting requires matplotlib. Install the repository test extras with "
    '`python -m pip install -e ".[ubuntu-test]"` or create the conda environment from '
    "`config/fenicsx_pinn_environment.yml`."
)


class MissingDependencyError(RuntimeError):
    pass


def require_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "orchard-mpl-cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(f"{PLOT_INSTALL_HINT} Missing module: {exc.name}.") from exc
    return plt


def plot_frequency_response_csv(csv_path: Path, show: bool = True) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        raise RuntimeError("CSV file contains no response data")

    headers = rows[0]
    frequency = [float(row[0]) for row in rows[1:]]
    plt = require_matplotlib()

    for column_index, name in enumerate(headers[1:], start=1):
        values = [float(row[column_index]) for row in rows[1:]]
        plt.plot(frequency, values, label=name)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Response magnitude")
    plt.title("Orchard Vibration Frequency Response")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the columns in a frequency-response CSV with matplotlib."
    )
    parser.add_argument("response_csv", type=Path, help="Path to a frequency-response CSV file.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the plot without opening an interactive window.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_frequency_response_csv(args.response_csv, show=not args.no_show)
    return 0
