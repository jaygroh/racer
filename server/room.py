"""Room — the public seam for a race session.

One Room owns two players, their inputs, the car states, and the fixed-timestep
simulation loop. The rest of the package talks to it through :meth:`handle_message`
and the broadcast callback; internals (loop scheduling, state bookkeeping) are free
to refactor without touching callers.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import Protocol

from .physics import CARS, CarParams, CarState, step, to_frame


class _Socket(Protocol):
    async def send_json(self, data: dict) -> None: ...

PHASE_LOBBY = "lobby"
PHASE_RUNNING = "running"
PHASE_FINISHED = "finished"

SIM_HZ = 60
BROADCAST_EVERY = 2  # broadcast at 30 Hz


def _make_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(4))


@dataclass
class Player:
    seat: int
    ws: _Socket  # starlette WebSocket; kept behind a protocol so tests can fake it
    params: CarParams = field(default_factory=lambda: CARS["balanced"])
    input_throttle: float = 0.0
    input_brake: float = 0.0
    input_steer: float = 0.0
    connected: bool = True
    # Serializes sends on this socket: uvicorn's send() is not internally locked,
    # so two concurrent send_json calls interleave writes and corrupt WS framing.
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class Room:
    def __init__(self, code: str | None = None) -> None:
        self.code = code or _make_code()
        self.phase = PHASE_LOBBY
        self.players: dict[int, Player] = {}
        self.cars: dict[int, CarState] = {}
        self._task: asyncio.Task | None = None

    # -- lobby ---------------------------------------------------------------

    def add_player(self, ws: _Socket) -> Player | None:
        if len(self.players) >= 2:
            return None
        seat = 0 if 0 not in self.players else 1
        player = Player(seat=seat, ws=ws)
        self.players[seat] = player
        return player

    @property
    def host(self) -> Player | None:
        return self.players.get(0)

    async def broadcast(self, frame: dict) -> None:
        """Send a frame to every connected player without letting one slow
        client stall the simulation loop (bounded wait per broadcast)."""
        tasks = [
            asyncio.create_task(self._safe_send(player, frame))
            for player in self.players.values()
            if player.connected
        ]
        if tasks:
            await asyncio.wait(tasks, timeout=0.25)

    async def _safe_send(self, player: Player, frame: dict) -> None:
        try:
            async with player._send_lock:
                await player.ws.send_json(frame)
        except Exception:
            pass

    # -- race ----------------------------------------------------------------

    def start(self) -> None:
        if self.phase != PHASE_LOBBY or len(self.players) < 2:
            return
        self.phase = PHASE_RUNNING
        for seat, player in self.players.items():
            self.cars[seat] = CarState(x=0.0, y=float(seat - 1) * 3.0, heading=0.0, speed=0.0)

    def apply_input(self, seat: int, throttle: float, brake: float, steer: float) -> None:
        player = self.players.get(seat)
        if player is None or not player.connected:
            return
        player.input_throttle = max(0.0, min(1.0, throttle))
        player.input_brake = max(0.0, min(1.0, brake))
        player.input_steer = max(-1.0, min(1.0, steer))

    def mark_disconnected(self, seat: int) -> None:
        player = self.players.get(seat)
        if player is not None:
            player.connected = False

    # -- simulation loop ------------------------------------------------------

    def start_loop(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self.run_loop())

    async def run_loop(self) -> None:
        dt = 1.0 / SIM_HZ
        tick = 0
        while self.phase == PHASE_RUNNING:
            for seat, player in self.players.items():
                car = self.cars[seat]
                self.cars[seat] = step(
                    car,
                    player.input_throttle if player.connected else 0.0,
                    player.input_brake if player.connected else 0.0,
                    player.input_steer if player.connected else 0.0,
                    dt,
                    player.params,
                )
            tick += 1
            if tick % BROADCAST_EVERY == 0:
                await self.broadcast(self._state_frame())
            await asyncio.sleep(dt)

    def _state_frame(self) -> dict:
        return {
            "type": "state",
            "cars": [to_frame(self.cars[seat], seat) for seat in sorted(self.cars)],
        }

    # -- lifecycle -------------------------------------------------------------

    @property
    def any_connected(self) -> bool:
        return any(p.connected for p in self.players.values())

    async def stop_loop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
