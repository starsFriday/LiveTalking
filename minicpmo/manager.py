"""Async session orchestration between WebRTC input and MiniCPM-o-Demo."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import aiohttp
from aiohttp import web

from minicpmo.audio_bridge import FloatRingBuffer, StreamingAudioBridge
from minicpmo.client import MiniCPMRealtimeClient
from utils.logger import logger


@dataclass
class _Session:
    sessionid: str
    avatar_session: object
    input_chunk_samples: int
    input_buffer: FloatRingBuffer = field(default_factory=FloatRingBuffer)
    input_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=3))
    worker_task: Optional[asyncio.Task] = None
    client: Optional[MiniCPMRealtimeClient] = None
    subscribers: set = field(default_factory=set)
    state: dict = field(default_factory=lambda: {"type": "connecting"})
    latest_video_frame: Optional[str] = None
    active_response_id: Optional[str] = None
    response_audio_seconds: float = 0.0
    response_text_window: str = ""

    def __post_init__(self) -> None:
        self.output_bridge = StreamingAudioBridge(self.avatar_session.put_audio_frame)


class MiniCPMManager:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.enabled = bool(getattr(opt, "minicpmo_enabled", False))
        self.sessions: dict[str, _Session] = {}

    async def start_session(self, sessionid: str, avatar_session) -> None:
        if not self.enabled or sessionid in self.sessions:
            return
        session = _Session(
            sessionid=sessionid,
            avatar_session=avatar_session,
            input_chunk_samples=16000 * self.opt.minicpmo_input_chunk_ms // 1000,
        )
        session.client = self._create_client(session)
        self.sessions[sessionid] = session
        session.worker_task = asyncio.create_task(self._run_session(session), name=f"minicpmo-{sessionid}")

    async def append_input(self, sessionid: str, audio_16k: np.ndarray) -> None:
        session = self.sessions.get(sessionid)
        if session is None:
            return
        session.input_buffer.write(audio_16k)
        while session.input_buffer.size >= session.input_chunk_samples:
            chunk = session.input_buffer.read(session.input_chunk_samples)
            if session.input_queue.full():
                logger.warning("[MiniCPM] input queue full for %s; dropping oldest chunk", sessionid)
                session.input_queue.get_nowait()
                session.input_queue.task_done()
            session.input_queue.put_nowait(chunk)

    async def interrupt(self, sessionid: str) -> None:
        session = self.sessions.get(sessionid)
        if session:
            await self._restart_session(session, "interrupted")

    def update_video_frame(self, sessionid: str, jpeg_base64: str) -> None:
        session = self.sessions.get(sessionid)
        if session is not None:
            session.latest_video_frame = jpeg_base64

    async def stop_session(self, sessionid: str) -> None:
        session = self.sessions.pop(sessionid, None)
        if session is None:
            return
        if session.worker_task:
            session.worker_task.cancel()
            await asyncio.gather(session.worker_task, return_exceptions=True)
        if session.client:
            await session.client.close()
        session.output_bridge.reset()
        for subscriber in list(session.subscribers):
            await subscriber.close()
        # The official Gateway releases its exclusive duplex Worker
        # asynchronously after the WebSocket closes. Do not let a rapid
        # reconnect race that cleanup and enter the queue behind itself.
        await self._wait_worker_idle()

    async def shutdown(self) -> None:
        for sessionid in list(self.sessions):
            await self.stop_session(sessionid)

    async def event_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        sessionid = request.match_info["sessionid"]
        session = self.sessions.get(sessionid)
        if session is None:
            await ws.send_json({"type": "error", "message": "Joyfox-FullDuplex session not found"})
            await ws.close()
            return ws
        session.subscribers.add(ws)
        await ws.send_json(session.state)
        try:
            async for _ in ws:
                pass
        finally:
            session.subscribers.discard(ws)
        return ws

    async def input_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Receive browser microphone float32/16k PCM and JPEG camera frames."""
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        sessionid = request.match_info["sessionid"]
        session = self.sessions.get(sessionid)
        if session is None:
            await ws.send_json({"type": "error", "message": "Joyfox-FullDuplex session not found"})
            await ws.close()
            return ws
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.BINARY:
                    pcm = np.frombuffer(message.data, dtype="<f4").copy()
                    await self.append_input(sessionid, pcm)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    payload = message.json()
                    if payload.get("type") == "video" and payload.get("data"):
                        self.update_video_frame(sessionid, payload["data"])
        finally:
            pass
        return ws

    async def _run_session(self, session: _Session) -> None:
        try:
            await self._broadcast(session, {"type": "connecting"})
            await self._wait_worker_idle()
            if self.sessions.get(session.sessionid) is not session:
                return
            await session.client.connect()
            while True:
                chunk = await session.input_queue.get()
                try:
                    await session.client.send_audio(chunk, session.latest_video_frame)
                finally:
                    session.input_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("[MiniCPM] session %s failed", session.sessionid)
            await self._broadcast(session, {"type": "error", "message": str(exc)})

    def _create_client(self, session: _Session) -> MiniCPMRealtimeClient:
        return MiniCPMRealtimeClient(
            self.opt.minicpmo_url,
            self.opt.minicpmo_system_prompt,
            lambda message: self._handle_message(session, message),
        )

    async def _restart_session(self, session: _Session, reason: str) -> None:
        """Hard-reset one official realtime session without dropping WebRTC."""
        session.input_buffer.clear()
        while not session.input_queue.empty():
            try:
                session.input_queue.get_nowait()
                session.input_queue.task_done()
            except asyncio.QueueEmpty:
                break
        session.output_bridge.reset()
        session.active_response_id = None
        session.response_audio_seconds = 0.0
        session.response_text_window = ""

        old_worker = session.worker_task
        if old_worker and old_worker is not asyncio.current_task():
            old_worker.cancel()
            await asyncio.gather(old_worker, return_exceptions=True)
        old_client = session.client
        if old_client:
            await old_client.close()

        # Gateway releases its ticket slightly before the worker finishes
        # closing the backend duplex session. Reconnecting in that window is
        # accepted by Gateway but rejected by Worker with HTTP 403.
        await self._wait_worker_idle()

        # The session may have been removed while the old connection closed.
        if self.sessions.get(session.sessionid) is not session:
            return
        logger.warning("[MiniCPM] hard-reset session %s: %s", session.sessionid, reason)
        session.client = self._create_client(session)
        session.worker_task = asyncio.create_task(
            self._run_session(session), name=f"minicpmo-{session.sessionid}"
        )

    async def _wait_worker_idle(self) -> None:
        url = getattr(self.opt, "minicpmo_worker_health_url", "http://127.0.0.1:22400/health")
        timeout = aiohttp.ClientTimeout(total=1.0)
        for _ in range(20):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as client:
                    async with client.get(url) as response:
                        payload = await response.json()
                        if response.status == 200 and payload.get("worker_status") == "idle":
                            # Give Gateway's in-memory worker pool one event-loop
                            # turn to observe the same state.
                            await asyncio.sleep(0.25)
                            return
            except Exception:
                pass
            await asyncio.sleep(0.25)
        logger.warning("[MiniCPM] worker did not report idle before reconnect")

    async def _handle_message(self, session: _Session, message: dict) -> None:
        msg_type = message.get("type")
        if msg_type == "session.created":
            await self._broadcast(session, {"type": "ready"})
            return
        if msg_type == "session.queued" or msg_type == "session.queue_update":
            await self._broadcast(session, {
                "type": "queued",
                "position": message.get("position"),
                "estimated_wait_s": message.get("estimated_wait_s"),
            })
            return
        if msg_type == "response.output.delta":
            kind = message.get("kind")
            if kind == "audio" and message.get("audio"):
                response_id = message.get("response_id")
                if session.active_response_id is None:
                    # MiniCPM speech and manual text/EdgeTTS share the same
                    # avatar audio timeline. A new model response takes over
                    # cleanly instead of mixing two voices together.
                    session.avatar_session.flush_talk()
                    session.active_response_id = response_id or "minicpmo-response"
                    session.response_audio_seconds = 0.0
                pcm = np.frombuffer(base64.b64decode(message["audio"]), dtype="<f4")
                session.response_audio_seconds += pcm.size / 24000.0
                max_seconds = float(getattr(self.opt, "minicpmo_max_response_seconds", 120.0))
                if max_seconds > 0 and session.response_audio_seconds > max_seconds:
                    await self._broadcast(session, {
                        "type": "resetting",
                        "message": f"回答超过 {max_seconds:g} 秒，已自动终止并重新聆听",
                    })
                    await self._restart_session(session, "response duration guard")
                    return
                # Some official builds emit a different response_id for each
                # one-second unit. Treat everything until `listen` as one
                # speaking turn so duration guards and lip-sync markers work.
                session.output_bridge.push(pcm, response_id=session.active_response_id)
                await self._broadcast(session, {"type": "speaking", "metrics": message.get("metrics", {})})
            elif kind == "text" and message.get("text"):
                session.response_text_window = (
                    session.response_text_window + str(message["text"])
                )[-600:]
                if self._looks_repetitive(session.response_text_window):
                    await self._broadcast(session, {
                        "type": "resetting",
                        "message": "检测到回答内容重复，已自动终止并重新聆听",
                    })
                    await self._restart_session(session, "repetitive response guard")
                    return
                await self._broadcast(session, {"type": "text", "text": message["text"]})
            elif kind == "listen":
                session.output_bridge.finish_response()
                session.active_response_id = None
                session.response_audio_seconds = 0.0
                session.response_text_window = ""
                await self._broadcast(session, {"type": "listening", "metrics": message.get("metrics", {})})
            return
        if msg_type == "error":
            error = message.get("error", message)
            await self._broadcast(session, {
                "type": "error",
                "message": error.get("message", "Joyfox-FullDuplex error"),
            })

    async def _broadcast(self, session: _Session, event: dict) -> None:
        session.state = event
        dead = []
        for ws in list(session.subscribers):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            session.subscribers.discard(ws)

    @staticmethod
    def _looks_repetitive(text: str) -> bool:
        """Detect a short phrase looping many times without penalizing long answers."""
        normalized = "".join(character for character in text if character.isalnum())
        if len(normalized) < 36:
            return False
        for width in range(6, 13):
            counts = {}
            for offset in range(0, len(normalized) - width + 1):
                phrase = normalized[offset : offset + width]
                counts[phrase] = counts.get(phrase, 0) + 1
                if counts[phrase] >= 6:
                    return True
        return False
