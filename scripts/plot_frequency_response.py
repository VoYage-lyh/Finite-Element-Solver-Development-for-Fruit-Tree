import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python plot_frequency_response.py <response.csv>")
        return 1

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return 1

    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        print("CSV file contains no response data")
        return 1

    headers = rows[0]
    print(f"Loaded {len(rows) - 1} response samples from {csv_path}")
    print("Columns:", ", ".join(headers))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; showing header preview only.")
        return 0

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
    raise SystemExit(main())
