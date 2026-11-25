"""Tests for ticket 01: room lifecycle + car simulation over the WebSocket protocol.

Seam: the WS message protocol itself — scripted clients drive create/join/start/input
and assert on emitted state. Tests run against a real uvicorn server with two
`websockets` clients in one event loop (mirrors production; avoids the starlette
TestClient nested-portal deadlock where a second client's blocking receive stalls
the sim task that must send it its frames).

Known-slow: closing a connection whose room is mid-race takes ~15s (uvicorn's
close_timeout) — root cause not yet pinned down. It only slows teardown, not the
game itself; left as-is deliberately rather than blocking progress on later tickets.
"""

import asyncio
import json

import pytest
import uvicorn
from websockets.asyncio.client import connect as ws_connect


@pytest.fixture
async def server():
    config = uvicorn.Config("server.app:app", host="127.0.0.1", port=8941, log_level="error")
    uvi = uvicorn.Server(config)
    task = asyncio.create_task(uvi.serve())
    while not uvi.started:
        await asyncio.sleep(0.01)
    yield "ws://127.0.0.1:8941/ws"
    # Robust teardown: bound the shutdown so a lingering room sim-loop / connection
    # can't hold uvicorn's serve() open forever, then reset app globals for the next test.
    uvi.should_exit = True
    try:
        await asyncio.wait_for(task, timeout=6)
    except (asyncio.TimeoutError, TimeoutError):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    from server import app as _appmod

    for room in list(_appmod.rooms.values()):
        await room.stop_loop()
    _appmod.rooms.clear()
    _appmod.pending.clear()


async def _recv(ws, want_type=None):
    """Receive the next frame; if want_type given, skip non-matching frames."""
    while True:
        msg = json.loads(await ws.recv())
        if want_type is None or msg["type"] == want_type:
            return msg


async def test_create_room_returns_code_and_seat(server):
    async with ws_connect(server) as host:
        await host.send(json.dumps({"type": "create"}))
        msg = await _recv(host, "created")
        assert len(msg["code"]) == 4
        assert msg["seat"] == 0


async def test_guest_joins_with_code(server):
    async with ws_connect(server) as host:
        await host.send(json.dumps({"type": "create"}))
        code = (await _recv(host, "created"))["code"]
        async with ws_connect(server) as guest:
            await guest.send(json.dumps({"type": "join", "code": code}))
            msg = await _recv(guest, "joined")
            assert msg["seat"] == 1


async def test_third_player_rejected(server):
    async with ws_connect(server) as host:
        await host.send(json.dumps({"type": "create"}))
        code = (await _recv(host, "created"))["code"]
        async with ws_connect(server) as p2:
            await p2.send(json.dumps({"type": "join", "code": code}))
            assert (await _recv(p2, "joined"))["seat"] == 1
            async with ws_connect(server) as p3:
                await p3.send(json.dumps({"type": "join", "code": code}))
                assert (await _recv(p3))["type"] == "full"


async def test_host_starts_race_both_receive_started(server):
    async with ws_connect(server) as host:
        await host.send(json.dumps({"type": "create"}))
        code = (await _recv(host, "created"))["code"]
        async with ws_connect(server) as guest:
            await guest.send(json.dumps({"type": "join", "code": code}))
            assert (await _recv(guest, "joined"))["seat"] == 1
            await host.send(json.dumps({"type": "start"}))
            assert (await _recv(host, "started"))["type"] == "started"
            assert (await _recv(guest, "started"))["type"] == "started"


async def test_input_changes_car_position(server):
    async with ws_connect(server) as host:
        await host.send(json.dumps({"type": "create"}))
        code = (await _recv(host, "created"))["code"]
        async with ws_connect(server) as guest:
            await guest.send(json.dumps({"type": "join", "code": code}))
            await _recv(guest, "joined")
            await host.send(json.dumps({"type": "start"}))
            await _recv(host, "started")
            await _recv(guest, "started")

            await host.send(json.dumps({"type": "input", "throttle": 1.0, "brake": 0.0, "steer": 0.0}))

            first = await _recv(host, "state")
            last = first
            for i in range(30):
                last = await _recv(host, "state")

            car0_first = next(c for c in first["cars"] if c["seat"] == 0)
            car0_last = next(c for c in last["cars"] if c["seat"] == 0)
            moved = (car0_last["x"] - car0_first["x"]) ** 2 + (car0_last["y"] - car0_first["y"]) ** 2
            assert moved > 1.0, "throttle input must move the car"
            assert car0_last["speed"] > 0


async def test_state_broadcast_to_both_players(server):
    async with ws_connect(server) as host:
        await host.send(json.dumps({"type": "create"}))
        code = (await _recv(host, "created"))["code"]
        async with ws_connect(server) as guest:
            await guest.send(json.dumps({"type": "join", "code": code}))
            await _recv(guest, "joined")
            await host.send(json.dumps({"type": "start"}))
            await _recv(host, "started")
            await _recv(guest, "started")

            await host.send(json.dumps({"type": "input", "throttle": 1.0, "brake": 0.0, "steer": 0.0}))

            # Guest must also receive state frames containing BOTH cars
            msg = await _recv(guest, "state")
            seats = {c["seat"] for c in msg["cars"]}
            assert seats == {0, 1}
