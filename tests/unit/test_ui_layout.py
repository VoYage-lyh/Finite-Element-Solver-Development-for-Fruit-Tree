from orchard_fem.actuator.ui_layout import bounded_window_size


def test_window_size_respects_small_screen_and_decoration_margin() -> None:
    initial, minimum = bounded_window_size(
        (1365, 768),
        (1320, 820),
        (820, 560),
    )

    assert initial == (1305, 668)
    assert minimum == (820, 560)


def test_minimum_never_exceeds_available_window_size() -> None:
    initial, minimum = bounded_window_size(
        (800, 600),
        (1120, 760),
        (900, 700),
    )

    assert initial == (740, 500)
    assert minimum == initial
