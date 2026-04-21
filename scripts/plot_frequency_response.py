import csv
import os
import sys
import tempfile
from pathlib import Path

PLOT_INSTALL_HINT = (
    "plot_frequency_response.py requires matplotlib. Install the repository test extras with "
    "`python -m pip install -e \".[ubuntu-test]\"` or create the conda environment from "
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


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python plot_frequency_response.py <response.csv>")
        return 1

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return 1

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        print("CSV file contains no response data")
        return 1

    headers = rows[0]
    print(f"Loaded {len(rows) - 1} response samples from {csv_path}")
    print("Columns:", ", ".join(headers))

    plt = require_matplotlib()

    frequency = [float(row[0]) for row in rows[1:]]
    for column_index, name in enumerate(headers[1:], start=1):
        values = [float(row[column_index]) for row in rows[1:]]
        plt.plot(frequency, values, label=name)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Response magnitude")
    plt.title("Orchard Vibration Frequency Response")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MissingDependencyError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
