"""FastAPI app — connection routing for the racer server.

The app owns the WebSocket lifecycle (create/join/start/input); :class:`server.room.Room`
owns race state and the simulation loop. This file is thin on purpose.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .room import PHASE_RUNNING, Room

app = FastAPI(title="racer")

rooms: dict[str, Room] = {}
pending: dict[int, tuple[WebSocket, str | None]] = {}  # id(ws) -> (ws, room_code)


async def _send_started(room: Room) -> None:
    await room.broadcast({"type": "started"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    pid = id(ws)
    pending[pid] = (ws, None)
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "create":
                room = Room()
                player = room.add_player(ws)
                rooms[room.code] = room
                pending[pid] = (ws, room.code)
                await ws.send_json({"type": "created", "code": room.code, "seat": player.seat if player else 0})

            elif mtype == "join":
                code = msg.get("code")
                room = rooms.get(code) if isinstance(code, str) else None
                if room is None:
                    await ws.send_json({"type": "error", "reason": "no such room"})
                    continue
                player = room.add_player(ws)
                if player is None:
                    await ws.send_json({"type": "full"})
                    continue
                pending[pid] = (ws, code)
                await ws.send_json({"type": "joined", "seat": player.seat})

            elif mtype == "start":
                _, code = pending[pid]
                room = rooms.get(code or "")
                if room is None:
                    await ws.send_json({"type": "error", "reason": "not in a room"})
                    continue
                seat = next((p.seat for p in room.players.values() if id(p.ws) == pid), None)
                if seat != 0:
                    await ws.send_json({"type": "error", "reason": "only the host can start"})
                    continue
                if len(room.players) < 2:
                    await ws.send_json({"type": "error", "reason": "need two players"})
                    continue
                room.start()
                room.start_loop()
                await _send_started(room)

            elif mtype == "input":
                _, code = pending[pid]
                room = rooms.get(code or "")
                if room is None:
                    continue
                seat = next((p.seat for p in room.players.values() if id(p.ws) == pid), None)
                if seat is None:
                    continue
                room.apply_input(
                    seat,
                    float(msg.get("throttle", 0.0)),
                    float(msg.get("brake", 0.0)),
                    float(msg.get("steer", 0.0)),
                )

            else:
                await ws.send_json({"type": "error", "reason": f"unknown message type: {mtype!r}"})
    except WebSocketDisconnect:
        pass
    finally:
        _, code = pending.pop(pid, (None, None))
        room = rooms.get(code or "") if code else None
        if room is not None and room.phase == PHASE_RUNNING:
            seat = next((p.seat for p in room.players.values() if id(p.ws) == pid), None)
            if seat is not None:
                room.mark_disconnected(seat)
            if not room.any_connected:
                await room.stop_loop()
                rooms.pop(room.code, None)
