import argparse
import cmath
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    np = None
    HAS_MATPLOTLIB = False


Color = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize orchard model geometry together with excitation and measurement response data."
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
        help="Measurement column to highlight in the spectrogram/time-frequency panels.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive figures when matplotlib is available.",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> dict:
    with model_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_response(response_path: Path) -> Tuple[List[str], List[List[float]]]:
    with response_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        raise RuntimeError("Response CSV contains no data rows")

    headers = rows[0]
    data = [[float(value) for value in row] for row in rows[1:]]
    return headers, data


def lerp(a: Sequence[float], b: Sequence[float], alpha: float) -> List[float]:
    return [a[i] + alpha * (b[i] - a[i]) for i in range(3)]


def branch_num_elements(branch: dict) -> int:
    return max(int(branch.get("discretization", {}).get("num_elements", 4)), 1)


def resolve_branch_station(branch: dict, target_node) -> float:
    if target_node in (None, "tip"):
        return 1.0
    if target_node == "root":
        return 0.0

    node_index = int(target_node)
    return max(0.0, min(1.0, node_index / branch_num_elements(branch)))


def resolve_branch_point(branch: dict, station: float) -> List[float]:
    return lerp(branch["start"], branch["end"], station)


def build_branch_lookup(model: dict) -> Dict[str, dict]:
    return {branch["id"]: branch for branch in model.get("branches", [])}


def build_fruit_lookup(model: dict) -> Dict[str, dict]:
    return {fruit["id"]: fruit for fruit in model.get("fruits", [])}


def resolve_observation_point(model: dict, observation: dict) -> Tuple[List[float], str]:
    branch_lookup = build_branch_lookup(model)
    fruit_lookup = build_fruit_lookup(model)

    if observation["target_type"] == "branch":
        branch = branch_lookup[observation["target_id"]]
        station = resolve_branch_station(branch, observation.get("target_node", "tip"))
        position = resolve_branch_point(branch, station)
        label = "{0} ({1})".format(observation["id"], observation.get("target_component", "ux"))
        return position, label

    if observation["target_type"] == "fruit":
        fruit = fruit_lookup[observation["target_id"]]
        branch = branch_lookup[fruit["branch_id"]]
        position = resolve_branch_point(branch, float(fruit["location_s"]))
        return position, observation["id"]

    raise RuntimeError("Unsupported observation target_type: {0}".format(observation["target_type"]))


def resolve_excitation_point(model: dict) -> Tuple[List[float], str]:
    excitation = model["excitation"]
    branch = build_branch_lookup(model)[excitation["target_branch_id"]]
    station = resolve_branch_station(branch, excitation.get("target_node", "tip"))
    point = resolve_branch_point(branch, station)
    label = "excitation ({0})".format(excitation.get("target_component", "ux"))
    return point, label


def build_series_map(headers: List[str], rows: List[List[float]]) -> Dict[str, List[float]]:
    columns = list(zip(*rows))
    return {header: list(column) for header, column in zip(headers, columns)}


def choose_measurement_column(headers: List[str], preferred: Optional[str]) -> str:
    reserved = {"time_s", "frequency_hz", "excitation_signal", "excitation_load", "excitation_response"}
    measurement_headers = [header for header in headers if header not in reserved]

    if preferred is not None:
        if preferred not in headers:
            raise RuntimeError("Requested measurement column not found: {0}".format(preferred))
        return preferred

    if measurement_headers:
        return measurement_headers[0]

    return "excitation_response"


def rgb(color: Color) -> str:
    return "rgb({0},{1},{2})".format(color[0], color[1], color[2])


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def interpolate_color(start: Color, end: Color, alpha: float) -> Color:
    return (
        int(round(start[0] + alpha * (end[0] - start[0]))),
        int(round(start[1] + alpha * (end[1] - start[1]))),
        int(round(start[2] + alpha * (end[2] - start[2]))),
    )


def palette_color(index: int) -> Color:
    palette = [
        (31, 78, 121),
        (178, 34, 34),
        (34, 139, 34),
        (255, 140, 0),
        (106, 90, 205),
        (0, 139, 139),
    ]
    return palette[index % len(palette)]


def response_extension() -> str:
    return ".png" if HAS_MATPLOTLIB else ".svg"


def save_svg(path: Path, width: int, height: int, elements: List[str]) -> None:
    header = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect x="0" y="0" width="{0}" height="{1}" fill="white" />'.format(width, height),
    ]
    footer = ["</svg>"]
    path.write_text("\n".join(header + elements + footer), encoding="utf-8")


def project_xz(point: Sequence[float]) -> Tuple[float, float]:
    return float(point[0]), float(point[2])


def compute_bounds(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def map_to_rect(
    point: Tuple[float, float],
    bounds: Tuple[float, float, float, float],
    left: float,
    top: float,
    width: float,
    height: float,
    padding: float = 18.0,
) -> Tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1.0e-9)
    span_y = max(max_y - min_y, 1.0e-9)
    scale_x = (width - 2.0 * padding) / span_x
    scale_y = (height - 2.0 * padding) / span_y
    scale = min(scale_x, scale_y)
    offset_x = left + 0.5 * (width - scale * span_x)
    offset_y = top + 0.5 * (height - scale * span_y)

    x = offset_x + (point[0] - min_x) * scale
    y = top + height - (offset_y - top + (point[1] - min_y) * scale)
    return x, y


def svg_circle(x: float, y: float, radius: float, color: Color, stroke: Color = (0, 0, 0)) -> str:
    return '<circle cx="{0:.2f}" cy="{1:.2f}" r="{2:.2f}" fill="{3}" stroke="{4}" stroke-width="1.1" />'.format(
        x, y, radius, rgb(color), rgb(stroke)
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, color: Color, width: float = 1.5) -> str:
    return '<line x1="{0:.2f}" y1="{1:.2f}" x2="{2:.2f}" y2="{3:.2f}" stroke="{4}" stroke-width="{5:.2f}" />'.format(
        x1, y1, x2, y2, rgb(color), width
    )


def svg_text(x: float, y: float, text: str, color: Color = (30, 30, 30), size: int = 12, anchor: str = "start") -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return '<text x="{0:.2f}" y="{1:.2f}" fill="{2}" font-size="{3}" text-anchor="{4}" font-family="Segoe UI, Arial, sans-serif">{5}</text>'.format(
        x, y, rgb(color), size, anchor, safe
    )


def svg_polygon(points: List[Tuple[float, float]], fill: Color, stroke: Color = (0, 0, 0)) -> str:
    point_string = " ".join("{0:.2f},{1:.2f}".format(x, y) for x, y in points)
    return '<polygon points="{0}" fill="{1}" stroke="{2}" stroke-width="1.0" />'.format(
        point_string, rgb(fill), rgb(stroke)
    )


def svg_polyline(points: List[Tuple[float, float]], color: Color, width: float = 1.4) -> str:
    point_string = " ".join("{0:.2f},{1:.2f}".format(x, y) for x, y in points)
    return '<polyline points="{0}" fill="none" stroke="{1}" stroke-width="{2:.2f}" />'.format(
        point_string, rgb(color), width
    )


def add_plot_frame(elements: List[str], left: float, top: float, width: float, height: float, title: str) -> None:
    elements.append(
        '<rect x="{0:.2f}" y="{1:.2f}" width="{2:.2f}" height="{3:.2f}" fill="white" stroke="{4}" stroke-width="1.0" />'.format(
            left, top, width, height, rgb((180, 180, 180))
        )
    )
    elements.append(svg_text(left + 8.0, top + 18.0, title, size=14))


def nice_range(values: Sequence[float]) -> Tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        scale = max(abs(minimum), 1.0)
        return minimum - 0.1 * scale, maximum + 0.1 * scale
    padding = 0.08 * (maximum - minimum)
    return minimum - padding, maximum + padding


def draw_line_chart(
    elements: List[str],
    left: float,
    top: float,
    width: float,
    height: float,
    x_values: Sequence[float],
    series: List[Tuple[str, Sequence[float], Color]],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    add_plot_frame(elements, left, top, width, height, title)

    plot_left = left + 55.0
    plot_top = top + 30.0
    plot_width = width - 80.0
    plot_height = height - 60.0

    y_all: List[float] = []
    for _, values, _ in series:
        y_all.extend(values)

    x_min, x_max = nice_range(x_values)
    y_min, y_max = nice_range(y_all)

    elements.append(
        '<rect x="{0:.2f}" y="{1:.2f}" width="{2:.2f}" height="{3:.2f}" fill="none" stroke="{4}" stroke-width="1.0" />'.format(
            plot_left, plot_top, plot_width, plot_height, rgb((120, 120, 120))
        )
    )

    for tick in range(5):
        alpha = tick / 4.0
        x = plot_left + alpha * plot_width
        y = plot_top + alpha * plot_height
        elements.append(svg_line(x, plot_top, x, plot_top + plot_height, (230, 230, 230), 0.8))
        elements.append(svg_line(plot_left, y, plot_left + plot_width, y, (240, 240, 240), 0.8))

    def project(value_x: float, value_y: float) -> Tuple[float, float]:
        x_alpha = 0.0 if math.isclose(x_min, x_max) else (value_x - x_min) / (x_max - x_min)
        y_alpha = 0.0 if math.isclose(y_min, y_max) else (value_y - y_min) / (y_max - y_min)
        x = plot_left + clamp(x_alpha, 0.0, 1.0) * plot_width
        y = plot_top + plot_height - clamp(y_alpha, 0.0, 1.0) * plot_height
        return x, y

    for label, values, color in series:
        points = [project(x, y) for x, y in zip(x_values, values)]
        if len(points) >= 2:
            elements.append(svg_polyline(points, color, 1.5))

        legend_y = top + 18.0 + 16.0 * (series.index((label, values, color)) + 1)
        elements.append(svg_line(left + width - 115.0, legend_y - 4.0, left + width - 95.0, legend_y - 4.0, color, 2.5))
        elements.append(svg_text(left + width - 90.0, legend_y, label, size=10))

    elements.append(svg_text(plot_left + 0.5 * plot_width, top + height - 8.0, x_label, size=11, anchor="middle"))
    elements.append(svg_text(left + 12.0, plot_top + 0.5 * plot_height, y_label, size=11))


def compute_spectrum(values: Sequence[float], dt: float) -> Tuple[List[float], List[float]]:
    mean_value = sum(values) / max(len(values), 1)
    centered = [value - mean_value for value in values]
    count = len(centered)
    frequencies: List[float] = []
    amplitudes: List[float] = []

    for k in range((count // 2) + 1):
        coefficient = 0j
        for index, value in enumerate(centered):
            angle = -2.0j * math.pi * k * index / count
            coefficient += value * cmath.exp(angle)
        frequencies.append(k / (count * dt))
        amplitudes.append(abs(coefficient))

    return frequencies, amplitudes


def next_power_of_two(value: int) -> int:
    power = 1
    while power * 2 <= value:
        power *= 2
    return max(power, 1)


def compute_spectrogram(values: Sequence[float], dt: float) -> Tuple[List[float], List[float], List[List[float]]]:
    count = len(values)
    window_size = min(128, next_power_of_two(max(16, count // 4)))
    window_size = min(window_size, count)
    hop = max(1, window_size // 4)
    if window_size < 8:
        window_size = count
        hop = max(1, count // 2)

    window = [
        0.5 - 0.5 * math.cos(2.0 * math.pi * i / max(window_size - 1, 1))
        for i in range(window_size)
    ]

    times: List[float] = []
    frequencies = [k / (window_size * dt) for k in range((window_size // 2) + 1)]
    frames: List[List[float]] = []

    for start in range(0, max(count - window_size + 1, 1), hop):
        segment = list(values[start:start + window_size])
        if len(segment) < window_size:
            segment.extend([0.0] * (window_size - len(segment)))
        weighted = [segment[i] * window[i] for i in range(window_size)]

        amplitudes: List[float] = []
        for k in range((window_size // 2) + 1):
            coefficient = 0j
            for index, value in enumerate(weighted):
                coefficient += value * cmath.exp(-2.0j * math.pi * k * index / window_size)
            amplitudes.append(math.log10(1.0 + abs(coefficient)))

        times.append((start + 0.5 * window_size) * dt)
        frames.append(amplitudes)

        if start + window_size >= count:
            break

    return times, frequencies, frames


def draw_heatmap(
    elements: List[str],
    left: float,
    top: float,
    width: float,
    height: float,
    times: Sequence[float],
    frequencies: Sequence[float],
    values: Sequence[Sequence[float]],
    title: str,
) -> None:
    add_plot_frame(elements, left, top, width, height, title)

    plot_left = left + 55.0
    plot_top = top + 30.0
    plot_width = width - 75.0
    plot_height = height - 55.0

    flat_values = [value for row in values for value in row]
    min_value = min(flat_values) if flat_values else 0.0
    max_value = max(flat_values) if flat_values else 1.0

    start_color = (12, 44, 132)
    end_color = (235, 96, 12)

    if len(times) < 2 or len(frequencies) < 2:
        elements.append(svg_text(plot_left + 10.0, plot_top + 30.0, "Not enough samples for spectrogram", size=12))
        return

    time_step = plot_width / max(len(times), 1)
    freq_step = plot_height / max(len(frequencies), 1)

    for time_index, row in enumerate(values):
        for freq_index, amplitude in enumerate(row):
            alpha = 0.0 if math.isclose(min_value, max_value) else (amplitude - min_value) / (max_value - min_value)
            color = interpolate_color(start_color, end_color, clamp(alpha, 0.0, 1.0))
            x = plot_left + time_index * time_step
            y = plot_top + plot_height - (freq_index + 1) * freq_step
            elements.append(
                '<rect x="{0:.2f}" y="{1:.2f}" width="{2:.2f}" height="{3:.2f}" fill="{4}" stroke="none" />'.format(
                    x, y, time_step + 0.4, freq_step + 0.4, rgb(color)
                )
            )

    elements.append(
        '<rect x="{0:.2f}" y="{1:.2f}" width="{2:.2f}" height="{3:.2f}" fill="none" stroke="{4}" stroke-width="1.0" />'.format(
            plot_left, plot_top, plot_width, plot_height, rgb((120, 120, 120))
        )
    )
    elements.append(svg_text(plot_left + 0.5 * plot_width, top + height - 8.0, "Time (s)", size=11, anchor="middle"))
    elements.append(svg_text(left + 12.0, plot_top + 0.5 * plot_height, "Frequency (Hz)", size=11))


def plot_geometry_svg(model: dict, output_path: Path) -> None:
    width, height = 920, 700
    elements: List[str] = []
    elements.append(svg_text(26.0, 32.0, "Orchard Geometry (x-z projection)", size=20))

    points_2d: List[Tuple[float, float]] = []
    for branch in model.get("branches", []):
        points_2d.append(project_xz(branch["start"]))
        points_2d.append(project_xz(branch["end"]))

    for fruit in model.get("fruits", []):
        branch = build_branch_lookup(model)[fruit["branch_id"]]
        points_2d.append(project_xz(resolve_branch_point(branch, float(fruit["location_s"]))))

    excitation_point, _ = resolve_excitation_point(model)
    points_2d.append(project_xz(excitation_point))
    for observation in model.get("observations", []):
        point, _ = resolve_observation_point(model, observation)
        points_2d.append(project_xz(point))

    bounds = compute_bounds(points_2d)
    left, top, plot_width, plot_height = 40.0, 60.0, 840.0, 600.0
    max_level = max((int(branch.get("level", 0)) for branch in model.get("branches", [])), default=0)

    for branch in model.get("branches", []):
        start = map_to_rect(project_xz(branch["start"]), bounds, left, top, plot_width, plot_height)
        end = map_to_rect(project_xz(branch["end"]), bounds, left, top, plot_width, plot_height)
        level = int(branch.get("level", 0))
        branch_color = interpolate_color((97, 63, 25), (60, 140, 72), 0.0 if max_level == 0 else level / max_level)
        elements.append(svg_line(start[0], start[1], end[0], end[1], branch_color, 3.2))
        elements.append(svg_text(end[0] + 6.0, end[1] - 4.0, branch["id"], branch_color, 11))

    for fruit in model.get("fruits", []):
        branch = build_branch_lookup(model)[fruit["branch_id"]]
        point = map_to_rect(project_xz(resolve_branch_point(branch, float(fruit["location_s"]))), bounds, left, top, plot_width, plot_height)
        fruit_color = (244, 162, 97)
        elements.append(svg_circle(point[0], point[1], 6.0, fruit_color))
        elements.append(svg_text(point[0] + 7.0, point[1] - 7.0, fruit["id"], fruit_color, 10))

    excitation_point, excitation_label = resolve_excitation_point(model)
    excitation_xy = map_to_rect(project_xz(excitation_point), bounds, left, top, plot_width, plot_height)
    star = [
        (excitation_xy[0], excitation_xy[1] - 11.0),
        (excitation_xy[0] + 3.5, excitation_xy[1] - 3.0),
        (excitation_xy[0] + 11.0, excitation_xy[1] - 3.0),
        (excitation_xy[0] + 5.0, excitation_xy[1] + 2.0),
        (excitation_xy[0] + 7.5, excitation_xy[1] + 10.0),
        (excitation_xy[0], excitation_xy[1] + 5.0),
        (excitation_xy[0] - 7.5, excitation_xy[1] + 10.0),
        (excitation_xy[0] - 5.0, excitation_xy[1] + 2.0),
        (excitation_xy[0] - 11.0, excitation_xy[1] - 3.0),
        (excitation_xy[0] - 3.5, excitation_xy[1] - 3.0),
    ]
    elements.append(svg_polygon(star, (214, 40, 40)))
    elements.append(svg_text(excitation_xy[0] + 12.0, excitation_xy[1] - 10.0, excitation_label, (214, 40, 40), 11))

    for observation in model.get("observations", []):
        point, label = resolve_observation_point(model, observation)
        xy = map_to_rect(project_xz(point), bounds, left, top, plot_width, plot_height)
        triangle = [(xy[0], xy[1] - 8.0), (xy[0] + 7.0, xy[1] + 6.0), (xy[0] - 7.0, xy[1] + 6.0)]
        elements.append(svg_polygon(triangle, (33, 102, 172)))
        elements.append(svg_text(xy[0] + 10.0, xy[1] + 2.0, label, (33, 102, 172), 10))

    save_svg(output_path, width, height, elements)


def plot_frequency_response_svg(headers: List[str], rows: List[List[float]], output_path: Path) -> None:
    width, height = 1100, 640
    series = build_series_map(headers, rows)
    line_series = []
    for index, header in enumerate(headers[1:]):
        line_series.append((header, series[header], palette_color(index)))

    elements: List[str] = []
    draw_line_chart(
        elements,
        32.0,
        28.0,
        width - 64.0,
        height - 56.0,
        series["frequency_hz"],
        line_series,
        "Frequency Response at Excitation and Measurement Points",
        "Frequency (Hz)",
        "Response magnitude",
    )
    save_svg(output_path, width, height, elements)


def plot_time_frequency_svg(
    headers: List[str],
    rows: List[List[float]],
    output_path: Path,
    measurement_column: Optional[str],
) -> None:
    series = build_series_map(headers, rows)
    time = series["time_s"]
    if len(time) < 4:
        raise RuntimeError("Need at least four time samples for time-frequency visualization")

    dt = (time[-1] - time[0]) / max(len(time) - 1, 1)
    selected_measurement = choose_measurement_column(headers, measurement_column)
    measurement_headers = [
        header for header in headers
        if header not in {"time_s", "excitation_signal", "excitation_load", "excitation_response"}
    ]
    plotted_measurements = measurement_headers[: min(3, len(measurement_headers))]

    time_series = [
        ("excitation_signal", series["excitation_signal"], palette_color(0)),
        ("excitation_response", series["excitation_response"], palette_color(1)),
    ]
    for offset, header in enumerate(plotted_measurements):
        time_series.append((header, series[header], palette_color(offset + 2)))

    spectrum_series = []
    for label, values, color in time_series:
        frequencies, amplitudes = compute_spectrum(values, dt)
        spectrum_series.append((label, amplitudes, color))

    width, height = 1200, 920
    elements: List[str] = []

    draw_line_chart(
        elements,
        28.0,
        24.0,
        560.0,
        360.0,
        time,
        time_series,
        "Excitation and Measurement Time Histories",
        "Time (s)",
        "Amplitude",
    )
    draw_line_chart(
        elements,
        612.0,
        24.0,
        560.0,
        360.0,
        frequencies,
        spectrum_series,
        "Frequency Spectra",
        "Frequency (Hz)",
        "FFT amplitude",
    )

    exc_times, exc_freqs, exc_values = compute_spectrogram(series["excitation_response"], dt)
    obs_times, obs_freqs, obs_values = compute_spectrogram(series[selected_measurement], dt)
    draw_heatmap(
        elements,
        28.0,
        410.0,
        560.0,
        470.0,
        exc_times,
        exc_freqs,
        exc_values,
        "Excitation Point Spectrogram",
    )
    draw_heatmap(
        elements,
        612.0,
        410.0,
        560.0,
        470.0,
        obs_times,
        obs_freqs,
        obs_values,
        "Measurement Spectrogram: {0}".format(selected_measurement),
    )

    save_svg(output_path, width, height, elements)


def plot_geometry_matplotlib(model: dict, output_path: Path, show: bool) -> None:
    branch_lookup = build_branch_lookup(model)
    all_points: List[List[float]] = []

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    max_level = max((int(branch.get("level", 0)) for branch in model.get("branches", [])), default=0)
    cmap = plt.get_cmap("YlGn")

    for branch in model.get("branches", []):
        start = branch["start"]
        end = branch["end"]
        color = cmap(0.35 + 0.55 * (int(branch.get("level", 0)) / max(max_level, 1)))
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, linewidth=2.5)
        ax.text(end[0], end[1], end[2], branch["id"], fontsize=8, color=color)
        all_points.extend([start, end])

    for fruit in model.get("fruits", []):
        branch = branch_lookup[fruit["branch_id"]]
        point = resolve_branch_point(branch, float(fruit["location_s"]))
        ax.scatter(point[0], point[1], point[2], color="#f4a261", s=50, marker="o", edgecolors="black")
        ax.text(point[0], point[1], point[2], fruit["id"], fontsize=8)
        all_points.append(point)

    excitation_point, excitation_label = resolve_excitation_point(model)
    ax.scatter(excitation_point[0], excitation_point[1], excitation_point[2], color="#d62828", s=140, marker="*", edgecolors="black", label="excitation")
    ax.text(excitation_point[0], excitation_point[1], excitation_point[2], excitation_label, fontsize=9, color="#d62828")
    all_points.append(excitation_point)

    label_added = False
    for observation in model.get("observations", []):
        point, label = resolve_observation_point(model, observation)
        ax.scatter(
            point[0],
            point[1],
            point[2],
            color="#1d4ed8",
            s=65,
            marker="^",
            edgecolors="black",
            label="measurement" if not label_added else None,
        )
        label_added = True
        ax.text(point[0], point[1], point[2], label, fontsize=8, color="#1d4ed8")
        all_points.append(point)

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    zs = [point[2] for point in all_points]
    max_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0e-6)
    center = [0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys)), 0.5 * (max(zs) + min(zs))]
    half = 0.55 * max_range
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Orchard Geometry: {0}".format(model.get("metadata", {}).get("name", "unnamed_model")))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()
    plt.close(fig)


def plot_frequency_response_matplotlib(headers: List[str], rows: List[List[float]], output_path: Path, show: bool) -> None:
    series = build_series_map(headers, rows)
    frequency = series["frequency_hz"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for header in headers[1:]:
        ax.plot(frequency, series[header], linewidth=1.7, label=header)

    ax.set_title("Frequency Response at Excitation and Measurement Points")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Response magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()
    plt.close(fig)


def plot_time_frequency_matplotlib(
    headers: List[str],
    rows: List[List[float]],
    output_path: Path,
    measurement_column: Optional[str],
    show: bool,
) -> None:
    series = build_series_map(headers, rows)
    time = np.asarray(series["time_s"], dtype=float)
    if time.size < 4:
        raise RuntimeError("Need at least four time samples for time-frequency visualization")

    dt = float(np.mean(np.diff(time)))
    sample_rate = 1.0 / dt
    selected_measurement = choose_measurement_column(headers, measurement_column)
    measurement_headers = [
        header for header in headers
        if header not in {"time_s", "excitation_signal", "excitation_load", "excitation_response"}
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_time, ax_fft, ax_spec_exc, ax_spec_obs = axes.flat

    curve_columns = ["excitation_signal", "excitation_response"] + measurement_headers[: min(3, len(measurement_headers))]
    for column in curve_columns:
        values = np.asarray(series[column], dtype=float)
        ax_time.plot(time, values, label=column, linewidth=1.4)

        centered = values - np.mean(values)
        frequencies = np.fft.rfftfreq(centered.size, dt)
        amplitudes = np.abs(np.fft.rfft(centered))
        ax_fft.plot(frequencies, amplitudes, label=column, linewidth=1.2)

    ax_time.set_title("Excitation and Measurement Time Histories")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="upper right")

    ax_fft.set_title("Frequency Spectra")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("FFT amplitude")
    ax_fft.grid(True, alpha=0.3)
    ax_fft.legend(loc="upper right")

    window = min(256, next_power_of_two(len(time)))
    overlap = int(0.75 * window)
    ax_spec_exc.specgram(np.asarray(series["excitation_response"], dtype=float), Fs=sample_rate, NFFT=window, noverlap=overlap, cmap="viridis")
    ax_spec_exc.set_title("Excitation Point Spectrogram")
    ax_spec_exc.set_xlabel("Time (s)")
    ax_spec_exc.set_ylabel("Frequency (Hz)")

    ax_spec_obs.specgram(np.asarray(series[selected_measurement], dtype=float), Fs=sample_rate, NFFT=window, noverlap=overlap, cmap="magma")
    ax_spec_obs.set_title("Measurement Spectrogram: {0}".format(selected_measurement))
    ax_spec_obs.set_xlabel("Time (s)")
    ax_spec_obs.set_ylabel("Frequency (Hz)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()
    plt.close(fig)


def plot_geometry(model: dict, output_path: Path, show: bool) -> None:
    if HAS_MATPLOTLIB:
        plot_geometry_matplotlib(model, output_path, show)
    else:
        plot_geometry_svg(model, output_path)


def plot_frequency_response(headers: List[str], rows: List[List[float]], output_path: Path, show: bool) -> None:
    if HAS_MATPLOTLIB:
        plot_frequency_response_matplotlib(headers, rows, output_path, show)
    else:
        plot_frequency_response_svg(headers, rows, output_path)


def plot_time_frequency(
    headers: List[str],
    rows: List[List[float]],
    output_path: Path,
    measurement_column: Optional[str],
    show: bool,
) -> None:
    if HAS_MATPLOTLIB:
        plot_time_frequency_matplotlib(headers, rows, output_path, measurement_column, show)
    else:
        plot_time_frequency_svg(headers, rows, output_path, measurement_column)


def main() -> int:
    args = parse_args()
    if not args.model_json.exists():
        raise FileNotFoundError("Model JSON not found: {0}".format(args.model_json))
    if not args.response_csv.exists():
        raise FileNotFoundError("Response CSV not found: {0}".format(args.response_csv))

    output_prefix = args.output_prefix if args.output_prefix is not None else args.response_csv.with_suffix("")
    extension = response_extension()

    model = load_model(args.model_json)
    headers, rows = load_response(args.response_csv)

    geometry_path = Path("{0}_geometry{1}".format(output_prefix, extension))
    plot_geometry(model, geometry_path, args.show)
    print("Saved geometry figure to {0}".format(geometry_path))

    first_column = headers[0]
    if first_column == "time_s":
        analysis_path = Path("{0}_time_frequency{1}".format(output_prefix, extension))
        plot_time_frequency(headers, rows, analysis_path, args.measurement_column, args.show)
        print("Saved time-frequency figure to {0}".format(analysis_path))
    elif first_column == "frequency_hz":
        analysis_path = Path("{0}_frequency_response{1}".format(output_prefix, extension))
        plot_frequency_response(headers, rows, analysis_path, args.show)
        print("Saved frequency-response figure to {0}".format(analysis_path))
    else:
        raise RuntimeError("Unsupported response CSV first column: {0}".format(first_column))

    if not HAS_MATPLOTLIB:
        print("matplotlib/numpy not found; generated SVG figures with the built-in fallback renderer.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
