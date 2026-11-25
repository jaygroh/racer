"""Arcade car kinematics — pure functions, no I/O.

The public seam is :func:`step`: given a state and an input it returns the next
state. Everything else in this module is free to refactor without touching callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # radians, 0 = +X axis
    speed: float = 0.0  # m/s, signed (negative = reverse)


@dataclass(frozen=True)
class CarParams:
    """Handling profile for one car."""

    name: str
    accel: float  # m/s^2 at full throttle
    brake_decel: float  # m/s^2 at full brake
    drag: float  # linear drag coefficient (1/s)
    max_speed: float  # m/s forward
    max_reverse: float  # m/s reverse
    steer_rate: float  # rad/s at reference speed


# v1 ships one profile; ticket 07 adds the other two.
CARS: dict[str, CarParams] = {
    "balanced": CarParams(
        name="balanced",
        accel=22.0,
        brake_decel=35.0,
        drag=0.6,
        max_speed=34.0,
        max_reverse=8.0,
        steer_rate=2.6,
    ),
}

# Steering authority ramps up with speed so a parked car doesn't spin in place.
_STEER_REF_SPEED = 6.0


def step(state: CarState, throttle: float, brake: float, steer: float, dt: float, params: CarParams) -> CarState:
    """Advance the simulation by dt seconds. Pure — no mutation of inputs."""
    speed = state.speed

    if throttle > 0.0 and speed < params.max_speed:
        speed += params.accel * throttle * dt
    if brake > 0.0:
        if speed > 0.0:
            speed -= params.brake_decel * brake * dt
        elif speed > -params.max_reverse:
            speed -= params.accel * 0.6 * brake * dt

    # Linear drag, always opposing motion.
    speed -= params.drag * speed * dt

    if speed > params.max_speed:
        speed = params.max_speed
    if speed < -params.max_reverse:
        speed = -params.max_reverse
    if abs(speed) < 0.01 and throttle == 0.0 and brake == 0.0:
        speed = 0.0

    steer_factor = min(1.0, abs(speed) / _STEER_REF_SPEED) * (1.0 if speed >= 0 else -1.0)
    heading = state.heading + steer * params.steer_rate * steer_factor * dt

    x = state.x + math.cos(heading) * speed * dt
    y = state.y + math.sin(heading) * speed * dt

    return CarState(x=x, y=y, heading=heading, speed=speed)


def to_frame(state: CarState, seat: int) -> dict:
    """Serialize a car for the wire protocol."""
    return {
        "seat": seat,
        "x": round(state.x, 3),
        "y": round(state.y, 3),
        "heading": round(state.heading, 4),
        "speed": round(abs(state.speed), 2),
    }
