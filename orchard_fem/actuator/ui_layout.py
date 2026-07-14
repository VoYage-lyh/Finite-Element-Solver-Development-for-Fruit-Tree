"""Shared screen-aware sizing helpers for the actuator Tk interfaces."""
from __future__ import annotations

import tkinter as tk


def bounded_window_size(
    screen_size: tuple[int, int],
    desired_size: tuple[int, int],
    minimum_size: tuple[int, int],
    *,
    margin: tuple[int, int] = (60, 100),
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``(initial, minimum)`` sizes that fit inside the usable screen."""
    available_width = max(480, int(screen_size[0]) - int(margin[0]))
    available_height = max(360, int(screen_size[1]) - int(margin[1]))
    width = min(int(desired_size[0]), available_width)
    height = min(int(desired_size[1]), available_height)
    minimum_width = min(int(minimum_size[0]), width)
    minimum_height = min(int(minimum_size[1]), height)
    return (width, height), (minimum_width, minimum_height)


def fit_window_to_screen(
    window: tk.Misc,
    desired_size: tuple[int, int],
    minimum_size: tuple[int, int],
    *,
    margin: tuple[int, int] = (60, 100),
) -> None:
    """Apply a usable geometry without exceeding the current display."""
    initial, minimum = bounded_window_size(
        (window.winfo_screenwidth(), window.winfo_screenheight()),
        desired_size,
        minimum_size,
        margin=margin,
    )
    left = max(0, (window.winfo_screenwidth() - initial[0]) // 2)
    top = max(0, (window.winfo_screenheight() - initial[1]) // 2)
    window.geometry(f"{initial[0]}x{initial[1]}+{left}+{top}")
    window.minsize(*minimum)
