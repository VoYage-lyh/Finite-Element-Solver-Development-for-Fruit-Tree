from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


def _parse_observation_acceleration_column(col_name: str) -> tuple[str, str] | None:
    """Return ``(branch_id, label)`` from an observation acceleration column."""
    prefix = "obs_"
    suffix = "_accel_ms2"
    if not col_name.startswith(prefix) or not col_name.endswith(suffix):
        return None

    body = col_name[len(prefix) : -len(suffix)]
    parts = body.rsplit("_", 2)
    if len(parts) != 3:
        return None

    branch_id, station, component = parts
    if station not in {"root", "mid", "tip"}:
        return None
    return branch_id, f"{station}_{component}"


def _finite_series(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = float(row[column])
        values.append(value if math.isfinite(value) else math.nan)
    return values


def _has_finite(values: list[float]) -> bool:
    return any(math.isfinite(value) for value in values)


def _safe_filename_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def plot_time_history_csv(
    csv_path: str | Path,
    *,
    show: bool = True,
    output_path: str | Path | None = None,
) -> list[Any]:
    """Plot acceleration time histories from a time-history CSV.

    One matplotlib Figure per branch, using a station-by-component grid where
    possible. Returns a list of Figures.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    headers = list(rows[0].keys())
    times = [float(r["time_s"]) for r in rows]

    # --- excitation signal figure ---
    exc_signal_col = "excitation_signal"
    exc_accel_col = "excitation_accel_ms2"

    # --- acceleration observation columns, grouped by branch ---
    accel_cols = [h for h in headers if h.endswith("_accel_ms2") and h != exc_accel_col]

    # Group by branch: key = obs_<branch_id>
    branch_groups: dict[str, list[str]] = {}
    for col in accel_cols:
        parsed = _parse_observation_acceleration_column(col)
        branch = parsed[0] if parsed is not None else col
        branch_groups.setdefault(branch, []).append(col)

    figs: list[Any] = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- Figure 0: excitation signal + excitation acceleration ---
    fig_exc, axes_exc = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    if exc_signal_col in headers:
        axes_exc[0].plot(times, [float(r[exc_signal_col]) for r in rows],
                         color="gray", linewidth=0.9)
        axes_exc[0].set_ylabel("Displacement (m)")
        axes_exc[0].set_title("Input excitation signal")
        axes_exc[0].grid(True, alpha=0.3)

    if exc_accel_col in headers:
        accel_values = _finite_series(rows, exc_accel_col)
        if _has_finite(accel_values):
            axes_exc[1].plot(times, accel_values, color="steelblue", linewidth=0.9)
            axes_exc[1].set_ylabel("Acceleration (m/s²)")
            axes_exc[1].set_title("Excitation point acceleration")
            axes_exc[1].grid(True, alpha=0.3)
    axes_exc[1].set_xlabel("Time (s)")
    fig_exc.suptitle("Excitation — input & response", fontsize=11)
    fig_exc.tight_layout()
    figs.append(fig_exc)

    # --- One Figure per branch, laid out as station x component grid ---
    stations = ("root", "mid", "tip")
    components = ("ux", "uy", "uz")
    for branch_name, cols in branch_groups.items():
        parsed_cols = {
            parsed[1]: col
            for col in cols
            if (parsed := _parse_observation_acceleration_column(col)) is not None
        }
        if parsed_cols:
            fig, axes = plt.subplots(
                len(stations),
                len(components),
                figsize=(13, 8),
                sharex=True,
                squeeze=False,
                constrained_layout=True,
            )
            for row_index, station in enumerate(stations):
                for col_index, component in enumerate(components):
                    ax = axes[row_index][col_index]
                    label = f"{station}_{component}"
                    column = parsed_cols.get(label)
                    if column is None:
                        ax.set_axis_off()
                        continue
                    values = _finite_series(rows, column)
                    if _has_finite(values):
                        ax.plot(
                            times,
                            values,
                            color=colors[(row_index * len(components) + col_index) % len(colors)],
                            linewidth=0.9,
                        )
                    else:
                        ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
                    ax.set_xlim(times[0], times[-1])
                    ax.set_title(f"{station} {component}", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    if col_index == 0:
                        ax.set_ylabel("Accel (m/s²)")
                    if row_index == len(stations) - 1:
                        ax.set_xlabel("Time (s)")
        else:
            ncols = 3
            nrows = math.ceil(len(cols) / ncols)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(13, max(3, 2.6 * nrows)),
                sharex=True,
                squeeze=False,
                constrained_layout=True,
            )
            for index, col in enumerate(cols):
                ax = axes[index // ncols][index % ncols]
                values = _finite_series(rows, col)
                if _has_finite(values):
                    ax.plot(times, values, color=colors[index % len(colors)], linewidth=0.9)
                else:
                    ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
                ax.set_xlim(times[0], times[-1])
                ax.set_title(col.replace("_accel_ms2", ""), fontsize=9)
                ax.set_ylabel("Accel (m/s²)")
                ax.grid(True, alpha=0.3)
            for index in range(len(cols), nrows * ncols):
                axes[index // ncols][index % ncols].set_axis_off()
            for ax in axes[-1]:
                if ax.has_data():
                    ax.set_xlabel("Time (s)")

        fig.suptitle(f"Branch: {branch_name}", fontsize=11)
        figs.append(fig)

        if output_path is not None:
            stem = Path(output_path).stem
            suffix = Path(output_path).suffix or ".png"
            out = Path(output_path).with_name(
                f"{stem}_{_safe_filename_part(branch_name)}{suffix}"
            )
            fig.savefig(str(out), dpi=150, bbox_inches="tight")

    if output_path is not None:
        fig_exc.savefig(str(Path(output_path).with_name(
            Path(output_path).stem + "_excitation" + (Path(output_path).suffix or ".png")
        )), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return figs
