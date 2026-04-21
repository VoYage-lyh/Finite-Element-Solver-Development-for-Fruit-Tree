import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

plt = None
np = None

PLOT_INSTALL_HINT = (
    "Visualization requires numpy and matplotlib. Install the repository test extras with "
    "`python -m pip install -e \".[ubuntu-test]\"` or create the conda environment from "
    "`config/fenicsx_pinn_environment.yml`."
)


class MissingDependencyError(RuntimeError):
    pass


def require_plotting_dependencies(show: bool) -> None:
    global plt, np

    if plt is not None and np is not None:
        return

    cache_root = Path(tempfile.gettempdir()) / "orchard-mpl-cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")

        import matplotlib.pyplot as _plt
        import numpy as _np
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            f"{PLOT_INSTALL_HINT} Missing module: {exc.name}."
        ) from exc

    plt = _plt
    np = _np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize orchard geometry together with excitation and response data."
    )
    parser.add_argument("model_json", type=Path, help="Path to the orchard model JSON file.")
    parser.add_argument("response_csv", type=Path, help="Path to the solver response CSV file.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Prefix for saved figures. Defaults to the response CSV stem in the same directory.",
    )
    parser.add_argument(
        "--measurement-column",
        type=str,
        default=None,
        help="Measurement column to highlight in the time-history figure.",
    )
    parser.add_argument(
        "--trajectory-node",
        action="append",
        default=None,
        help="Observation id to use for trajectory plots. Repeat to request multiple nodes.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive figures after saving them.",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> dict:
    with model_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_response(response_path: Path) -> Tuple[list[str], list[list[float]]]:
    with response_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        raise RuntimeError("Response CSV contains no data rows")

    headers = rows[0]
    data = [[float(value) for value in row] for row in rows[1:]]
    return headers, data


def lerp(a: Sequence[float], b: Sequence[float], alpha: float) -> list[float]:
    return [a[index] + alpha * (b[index] - a[index]) for index in range(3)]


def branch_num_elements(branch: dict) -> int:
    return max(int(branch.get("discretization", {}).get("num_elements", 4)), 1)


def resolve_branch_station(branch: dict, target_node) -> float:
    if target_node in (None, "tip"):
        return 1.0
    if target_node == "root":
        return 0.0

    node_index = int(target_node)
    return max(0.0, min(1.0, node_index / branch_num_elements(branch)))


def resolve_branch_point(branch: dict, station: float) -> list[float]:
    return lerp(branch["start"], branch["end"], station)


def build_branch_lookup(model: dict) -> Dict[str, dict]:
    return {branch["id"]: branch for branch in model.get("branches", [])}


def build_fruit_lookup(model: dict) -> Dict[str, dict]:
    return {fruit["id"]: fruit for fruit in model.get("fruits", [])}


def observation_components(observation: dict) -> list[str]:
    if "target_components" in observation:
        value = observation["target_components"]
        if not isinstance(value, list):
            raise RuntimeError("observations[].target_components must be a list")
        components = [str(component) for component in value]
        if not components:
            raise RuntimeError("observations[].target_components must not be empty")
        return components

    return [str(observation.get("target_component", "ux"))]


def resolve_observation_point(model: dict, observation: dict) -> Tuple[list[float], str]:
    branch_lookup = build_branch_lookup(model)
    fruit_lookup = build_fruit_lookup(model)

    if observation["target_type"] == "branch":
        branch = branch_lookup[observation["target_id"]]
        station = resolve_branch_station(branch, observation.get("target_node", "tip"))
        position = resolve_branch_point(branch, station)
        label = "{0} ({1})".format(observation["id"], "/".join(observation_components(observation)))
        return position, label

    if observation["target_type"] == "fruit":
        fruit = fruit_lookup[observation["target_id"]]
        branch = branch_lookup[fruit["branch_id"]]
        position = resolve_branch_point(branch, float(fruit["location_s"]))
        return position, observation["id"]

    raise RuntimeError(
        "Unsupported observation target_type: {0}".format(observation["target_type"])
    )


def resolve_excitation_point(model: dict) -> Tuple[list[float], str]:
    excitation = model["excitation"]
    branch = build_branch_lookup(model)[excitation["target_branch_id"]]
    station = resolve_branch_station(branch, excitation.get("target_node", "tip"))
    point = resolve_branch_point(branch, station)
    label = "excitation ({0})".format(excitation.get("target_component", "ux"))
    return point, label


def build_series_map(headers: list[str], rows: list[list[float]]) -> Dict[str, list[float]]:
    columns = list(zip(*rows))
    return {header: list(column) for header, column in zip(headers, columns)}


def choose_measurement_column(headers: list[str], preferred: Optional[str]) -> str:
    reserved = {"time_s", "frequency_hz", "excitation_signal", "excitation_load", "excitation_response"}
    measurement_headers = [header for header in headers if header not in reserved]

    if preferred is not None:
        if preferred not in headers:
            raise RuntimeError("Requested measurement column not found: {0}".format(preferred))
        return preferred

    if measurement_headers:
        return measurement_headers[0]

    return "excitation_response"


def project_xz(point: Sequence[float]) -> Tuple[float, float]:
    return float(point[0]), float(point[2])


def frequency_figure_path(prefix: Path) -> Path:
    return Path(f"{prefix}_frequency_response.png")


def geometry_figure_path(prefix: Path) -> Path:
    return Path(f"{prefix}_geometry.png")


def time_frequency_figure_path(prefix: Path) -> Path:
    return Path(f"{prefix}_time_frequency.png")


def trajectory_figure_path(prefix: Path, node_id: str) -> Path:
    return Path(f"{prefix}_trajectory_{node_id}.png")


def _save_figure(fig, output_path: Path, show: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_geometry(model: dict, output_path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 8.0))

    branches = model.get("branches", [])
    if not branches:
        raise RuntimeError("Model JSON contains no branches")

    max_level = max(int(branch.get("level", 0)) for branch in branches) or 1
    branch_lookup = build_branch_lookup(model)

    for branch in branches:
        start_x, start_z = project_xz(branch["start"])
        end_x, end_z = project_xz(branch["end"])
        level = int(branch.get("level", 0))
        color = plt.cm.YlGn(0.35 + 0.55 * (level / max_level))
        ax.plot(
            [start_x, end_x],
            [start_z, end_z],
            color=color,
            linewidth=2.8,
            solid_capstyle="round",
        )
        ax.text(end_x, end_z, branch["id"], fontsize=9, color=color, ha="left", va="bottom")

    for fruit in model.get("fruits", []):
        branch = branch_lookup[fruit["branch_id"]]
        point = resolve_branch_point(branch, float(fruit["location_s"]))
        x, z = project_xz(point)
        ax.scatter([x], [z], s=70.0, color="#f28e2b", edgecolors="black", linewidths=0.6, zorder=4)
        ax.text(x, z, fruit["id"], fontsize=8, ha="left", va="bottom", color="#8c510a")

    excitation_point, excitation_label = resolve_excitation_point(model)
    excitation_x, excitation_z = project_xz(excitation_point)
    ax.scatter(
        [excitation_x],
        [excitation_z],
        s=160.0,
        marker="*",
        color="#d62728",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    ax.text(
        excitation_x,
        excitation_z,
        excitation_label,
        fontsize=9,
        ha="left",
        va="bottom",
        color="#d62728",
    )

    for observation in model.get("observations", []):
        point, label = resolve_observation_point(model, observation)
        x, z = project_xz(point)
        ax.scatter(
            [x],
            [z],
            s=80.0,
            marker="^",
            color="#1f77b4",
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )
        ax.text(x, z, label, fontsize=8, ha="left", va="top", color="#1f77b4")

    ax.set_title("Orchard Geometry (x-z Projection)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_path, show)


def plot_frequency_response(
    headers: list[str],
    rows: list[list[float]],
    output_path: Path,
    show: bool,
) -> None:
    series = build_series_map(headers, rows)
    if "frequency_hz" not in series:
        raise RuntimeError("Frequency-response CSV must contain a frequency_hz column")

    frequency = np.asarray(series["frequency_hz"], dtype=float)
    reserved = {"frequency_hz"}
    plotted_headers = [header for header in headers if header not in reserved]

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    for header in plotted_headers:
        values = np.asarray(series[header], dtype=float)
        linewidth = 2.5 if header == "excitation_response" else 1.6
        alpha = 1.0 if header == "excitation_response" else 0.9
        ax.plot(frequency, values, linewidth=linewidth, alpha=alpha, label=header)

    ax.set_title("Orchard Frequency Response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Response magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_path, show)


def next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def compute_sample_rate(time_values) -> float:
    if len(time_values) < 2:
        raise RuntimeError("Time-history CSV must contain at least two samples")

    deltas = np.diff(time_values)
    positive_deltas = deltas[deltas > 0.0]
    if positive_deltas.size == 0:
        raise RuntimeError("Time-history CSV must contain strictly increasing time values")

    return 1.0 / float(np.median(positive_deltas))


def compute_fft_amplitude(signal_values, sample_rate: float):
    demeaned = signal_values - np.mean(signal_values)
    window = np.hanning(len(demeaned))
    weighted = demeaned * window
    frequency = np.fft.rfftfreq(len(weighted), d=1.0 / sample_rate)
    amplitude = np.abs(np.fft.rfft(weighted))
    return frequency, amplitude


def draw_spectrogram(ax, signal_values, sample_rate: float, title: str, cmap: str) -> None:
    if len(signal_values) < 32:
        ax.text(0.5, 0.5, "Not enough samples for spectrogram", ha="center", va="center")
        ax.set_axis_off()
        return

    window = min(256, next_power_of_two(len(signal_values) // 4))
    window = max(window, 32)
    overlap = min(window - 1, max(window // 2, 16))
    ax.specgram(signal_values, Fs=sample_rate, NFFT=window, noverlap=overlap, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")


def split_observation_component_header(header: str) -> Tuple[str, str] | None:
    for component in ("ux", "uy", "uz"):
        suffix = f"_{component}"
        if header.endswith(suffix) and len(header) > len(suffix):
            return header[: -len(suffix)], component
    return None


def available_trajectory_nodes(headers: list[str]) -> list[str]:
    components_by_node: dict[str, set[str]] = {}
    for header in headers:
        parsed = split_observation_component_header(header)
        if parsed is None:
            continue
        node_id, component = parsed
        components_by_node.setdefault(node_id, set()).add(component)
    return sorted(node_id for node_id, components in components_by_node.items() if len(components) >= 2)


def trajectory_columns_for_node(headers: list[str], node_id: str) -> dict[str, str]:
    columns: dict[str, str] = {}
    for header in headers:
        parsed = split_observation_component_header(header)
        if parsed is None:
            continue
        header_node_id, component = parsed
        if header_node_id == node_id:
            columns[component] = header
    return columns


def plot_trajectory(
    headers: list[str],
    rows: list[list[float]],
    output_path: Path,
    node_id: str,
    show: bool,
) -> None:
    require_plotting_dependencies(show)

    series = build_series_map(headers, rows)
    if "time_s" not in series:
        raise RuntimeError("Trajectory plots require a time-history CSV with a time_s column")

    trajectory_columns = trajectory_columns_for_node(headers, node_id)
    ordered_components = [component for component in ("ux", "uy", "uz") if component in trajectory_columns]
    if len(ordered_components) < 2:
        raise RuntimeError(f"Could not find at least two trajectory components for node {node_id}")

    time_values = np.asarray(series["time_s"], dtype=float)
    figure = plt.figure(figsize=(8.5, 7.0))

    if len(ordered_components) >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        ax = figure.add_subplot(111, projection="3d")
        x_values = np.asarray(series[trajectory_columns["ux"]], dtype=float)
        y_values = np.asarray(series[trajectory_columns["uy"]], dtype=float)
        z_values = np.asarray(series[trajectory_columns["uz"]], dtype=float)
        ax.plot(x_values, y_values, z_values, color="#7f7f7f", linewidth=1.2, alpha=0.45)
        scatter = ax.scatter(x_values, y_values, z_values, c=time_values, cmap="viridis", s=14.0)
        ax.scatter([x_values[0]], [y_values[0]], [z_values[0]], color="#2ca02c", s=40.0, label="start")
        ax.scatter([x_values[-1]], [y_values[-1]], [z_values[-1]], color="#d62728", s=40.0, label="end")
        ax.set_xlabel("ux")
        ax.set_ylabel("uy")
        ax.set_zlabel("uz")
        ax.legend(loc="best")
        title = f"Trajectory: {node_id} (ux-uy-uz)"
    else:
        ax = figure.add_subplot(111)
        component_x, component_y = ordered_components[:2]
        x_values = np.asarray(series[trajectory_columns[component_x]], dtype=float)
        y_values = np.asarray(series[trajectory_columns[component_y]], dtype=float)
        ax.plot(x_values, y_values, color="#7f7f7f", linewidth=1.2, alpha=0.45)
        scatter = ax.scatter(x_values, y_values, c=time_values, cmap="viridis", s=16.0)
        ax.scatter([x_values[0]], [y_values[0]], color="#2ca02c", s=40.0, label="start")
        ax.scatter([x_values[-1]], [y_values[-1]], color="#d62728", s=40.0, label="end")
        ax.set_xlabel(component_x)
        ax.set_ylabel(component_y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        title = f"Trajectory: {node_id} ({component_x}-{component_y})"

    ax.set_title(title)
    colorbar = figure.colorbar(scatter, ax=ax, shrink=0.85)
    colorbar.set_label("Time (s)")
    figure.tight_layout()
    _save_figure(figure, output_path, show)


def plot_time_frequency(
    headers: list[str],
    rows: list[list[float]],
    output_path: Path,
    measurement_column: Optional[str],
    show: bool,
) -> None:
    series = build_series_map(headers, rows)
    if "time_s" not in series:
        raise RuntimeError("Time-history CSV must contain a time_s column")

    selected_measurement = choose_measurement_column(headers, measurement_column)
    time_values = np.asarray(series["time_s"], dtype=float)
    sample_rate = compute_sample_rate(time_values)

    measurement = np.asarray(series[selected_measurement], dtype=float)
    excitation_response = np.asarray(series.get("excitation_response", series[selected_measurement]), dtype=float)
    excitation_load = (
        np.asarray(series["excitation_load"], dtype=float)
        if "excitation_load" in series
        else None
    )

    measurement_frequency, measurement_amplitude = compute_fft_amplitude(measurement, sample_rate)
    excitation_frequency, excitation_amplitude = compute_fft_amplitude(excitation_response, sample_rate)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    ax_time, ax_fft, ax_spec_exc, ax_spec_obs = axes.flat

    ax_time.plot(time_values, measurement, label=selected_measurement, linewidth=1.8, color="#1f77b4")
    ax_time.plot(
        time_values,
        excitation_response,
        label="excitation_response",
        linewidth=1.5,
        color="#d62728",
        alpha=0.85,
    )
    if excitation_load is not None:
        ax_time.plot(
            time_values,
            excitation_load,
            label="excitation_load",
            linewidth=1.2,
            color="#2ca02c",
            alpha=0.75,
        )
    ax_time.set_title("Time History")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="best", fontsize=9)

    ax_fft.plot(
        excitation_frequency,
        excitation_amplitude,
        label="excitation_response",
        linewidth=1.5,
        color="#d62728",
    )
    ax_fft.plot(
        measurement_frequency,
        measurement_amplitude,
        label=selected_measurement,
        linewidth=1.8,
        color="#1f77b4",
    )
    ax_fft.set_title("Frequency Spectrum")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("FFT amplitude")
    ax_fft.grid(True, alpha=0.3)
    ax_fft.legend(loc="best", fontsize=9)

    draw_spectrogram(
        ax_spec_exc,
        excitation_response,
        sample_rate,
        "Excitation Point Spectrogram",
        "viridis",
    )
    draw_spectrogram(
        ax_spec_obs,
        measurement,
        sample_rate,
        f"Measurement Spectrogram: {selected_measurement}",
        "magma",
    )

    fig.tight_layout()
    _save_figure(fig, output_path, show)


def main() -> int:
    args = parse_args()
    if not args.model_json.exists():
        raise FileNotFoundError("Model JSON not found: {0}".format(args.model_json))
    if not args.response_csv.exists():
        raise FileNotFoundError("Response CSV not found: {0}".format(args.response_csv))

    require_plotting_dependencies(show=args.show)

    output_prefix = (
        args.output_prefix if args.output_prefix is not None else args.response_csv.with_suffix("")
    )

    model = load_model(args.model_json)
    headers, rows = load_response(args.response_csv)

    geometry_path = geometry_figure_path(output_prefix)
    plot_geometry(model, geometry_path, args.show)
    print("Saved geometry figure to {0}".format(geometry_path))

    first_column = headers[0]
    if first_column == "time_s":
        analysis_path = time_frequency_figure_path(output_prefix)
        plot_time_frequency(headers, rows, analysis_path, args.measurement_column, args.show)
        print("Saved time-frequency figure to {0}".format(analysis_path))
        trajectory_nodes = args.trajectory_node if args.trajectory_node is not None else available_trajectory_nodes(headers)
        for node_id in trajectory_nodes:
            trajectory_path = trajectory_figure_path(output_prefix, node_id)
            plot_trajectory(headers, rows, trajectory_path, node_id, args.show)
            print("Saved trajectory figure to {0}".format(trajectory_path))
    elif first_column == "frequency_hz":
        analysis_path = frequency_figure_path(output_prefix)
        plot_frequency_response(headers, rows, analysis_path, args.show)
        print("Saved frequency-response figure to {0}".format(analysis_path))
    else:
        raise RuntimeError("Unsupported response CSV first column: {0}".format(first_column))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MissingDependencyError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
