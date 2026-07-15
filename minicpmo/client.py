"""Client for the official MiniCPM-o-Demo realtime WebSocket protocol."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Awaitable, Callable, Optional

import numpy as np
import websockets

from utils.logger import logger


MessageCallback = Callable[[dict], Awaitable[None]]


class MiniCPMRealtimeClient:
    def __init__(self, url: str, system_prompt: str, callback: MessageCallback) -> None:
        self.url = url
        self.system_prompt = system_prompt
        self.callback = callback
        self.websocket = None
        self.reader_task: Optional[asyncio.Task] = None
        self.ready = asyncio.Event()
        self.closed = asyncio.Event()

    async def connect(self) -> None:
        self.websocket = await websockets.connect(
            self.url,
            open_timeout=20,
            ping_interval=20,
            ping_timeout=20,
            max_size=128 * 1024 * 1024,
        )
        self.reader_task = asyncio.create_task(self._reader(), name="minicpmo-ws-reader")

    async def send_audio(
        self,
        samples_16k: np.ndarray,
        video_frame_base64: Optional[str] = None,
    ) -> None:
        await asyncio.wait_for(self.ready.wait(), timeout=300)
        if self.websocket is None:
            raise RuntimeError("MiniCPM websocket is not connected")
        pcm = np.asarray(samples_16k, dtype="<f4").reshape(-1)
        input_payload = {
            "audio": base64.b64encode(pcm.tobytes()).decode("ascii"),
            "force_listen": False,
            "max_slice_nums": 1,
        }
        if video_frame_base64:
            input_payload["video_frames"] = [video_frame_base64]
        await self.websocket.send(json.dumps({
            "type": "input.append",
            "input": input_payload,
        }))

    async def force_listen(self) -> None:
        if self.websocket is None or not self.ready.is_set():
            return
        silence = np.zeros(16000, dtype="<f4")
        await self.websocket.send(json.dumps({
            "type": "input.append",
            "input": {
                "audio": base64.b64encode(silence.tobytes()).decode("ascii"),
                "force_listen": True,
            },
        }))

    async def close(self) -> None:
        ws = self.websocket
        self.websocket = None
        if ws is not None:
            try:
                await ws.send(json.dumps({"type": "session.close", "reason": "client_closed"}))
            except Exception:
                pass
            await ws.close()
        if self.reader_task and self.reader_task is not asyncio.current_task():
            self.reader_task.cancel()
            await asyncio.gather(self.reader_task, return_exceptions=True)
        self.closed.set()

    async def _reader(self) -> None:
        try:
            async for raw in self.websocket:
                message = json.loads(raw)
                msg_type = message.get("type")
                if msg_type in ("session.queue_done", "queue_done"):
                    await self.websocket.send(json.dumps({
                        "type": "session.init",
                        "payload": {"system_prompt": self.system_prompt},
                    }, ensure_ascii=False))
                elif msg_type == "session.created":
                    self.ready.set()
                elif msg_type == "error":
                    error = message.get("error", message)
                    raise RuntimeError(error.get("message", json.dumps(error, ensure_ascii=False)))
                await self.callback(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("[MiniCPM] websocket reader failed")
            await self.callback({"type": "error", "message": str(exc)})
        finally:
            self.ready.clear()
            self.closed.set()
