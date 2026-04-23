from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from orchard_fem.visualization.dependencies import require_plotting_dependencies
from orchard_fem.visualization.io import build_series_map, choose_measurement_column
from orchard_fem.visualization.model_scene import (
    build_branch_lookup,
    project_xz,
    resolve_branch_point,
    resolve_excitation_point,
    resolve_observation_point,
)


def _save_figure(fig, output_path: Path, show: bool) -> None:
    plt, _ = require_plotting_dependencies(show)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_geometry(model: dict, output_path: Path, show: bool) -> None:
    plt, _ = require_plotting_dependencies(show)
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
    plt, np = require_plotting_dependencies(show)
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
    _, np = require_plotting_dependencies(False)
    if len(time_values) < 2:
        raise RuntimeError("Time-history CSV must contain at least two samples")

    deltas = np.diff(time_values)
    positive_deltas = deltas[deltas > 0.0]
    if positive_deltas.size == 0:
        raise RuntimeError("Time-history CSV must contain strictly increasing time values")

    return 1.0 / float(np.median(positive_deltas))


def compute_fft_amplitude(signal_values, sample_rate: float):
    _, np = require_plotting_dependencies(False)
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
    plt, np = require_plotting_dependencies(show)
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
    plt, np = require_plotting_dependencies(show)
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
