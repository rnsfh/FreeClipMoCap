_STANDARD_FRAME_RATES = (
    (24, 1.001),
    (24, 1.0),
    (25, 1.0),
    (30, 1.001),
    (30, 1.0),
    (48, 1.0),
    (50, 1.0),
    (60, 1.001),
    (60, 1.0),
    (120, 1.0),
)


def nearest_standard_frame_rate(frame_rate: float) -> tuple[int, float]:
    return min(_STANDARD_FRAME_RATES, key=lambda rate: abs(rate[0] / rate[1] - frame_rate))
