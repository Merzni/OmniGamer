"""Canonical discrete camera look parameters and legacy record conversion."""

LOOK_COUNTS = {
    "horizontal": {"small": 30, "medium": 60, "large": 90},
    "vertical": {"small": 25, "medium": 50, "large": 60},
}

LOOK_DIRECTIONS = {
    "left": (-1, 0, "horizontal"),
    "right": (1, 0, "horizontal"),
    "up": (0, -1, "vertical"),
    "down": (0, 1, "vertical"),
}


def look_counts(direction, amount):
    """Translate a validated direction/amount pair into relative mouse counts."""
    if direction not in LOOK_DIRECTIONS or amount not in {"small", "medium", "large"}:
        raise ValueError("look requires a valid look_direction and look_amount")
    dx_sign, dy_sign, axis = LOOK_DIRECTIONS[direction]
    magnitude = LOOK_COUNTS[axis][amount]
    return dx_sign * magnitude, dy_sign * magnitude
