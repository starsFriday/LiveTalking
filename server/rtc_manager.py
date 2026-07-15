###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import random
import copy
from typing import Dict, Optional
import queue
import base64
import time
import cv2

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.mediastreams import MediaStreamError
from av.audio.resampler import AudioResampler

from utils.logger import logger


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager
from server.session_manager import MaxSessionError

class RTCManager:
    """
    WebRTC 连接管理器。
    
    管理 PeerConnection 生命周期、音视频轨道收发、DataChannel。
    """

    def __init__(self, opt, minicpmo_manager=None):
        """
        Args:
            opt: 全局配置
        """
        self.opt = opt
        self.pcs: set = set()
        self.session_pcs: dict[str, RTCPeerConnection] = {}
        self.minicpmo_manager = minicpmo_manager
        self.track_tasks: set[asyncio.Task] = set()

    async def _consume_microphone(self, track, sessionid: str):
        """Decode a browser WebRTC audio track into 16 kHz mono float32 PCM."""
        resampler = AudioResampler(format='flt', layout='mono', rate=16000)
        try:
            while True:
                frame = await track.recv()
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().reshape(-1).astype('float32', copy=False)
                    await self.minicpmo_manager.append_input(sessionid, samples)
        except (MediaStreamError, asyncio.CancelledError):
            pass
        except Exception:
            logger.exception("Microphone track failed for session %s", sessionid)

    async def _consume_camera(self, track, sessionid: str):
        """Keep the newest camera frame for the next one-second MiniCPM unit."""
        last_capture = 0.0
        try:
            while True:
                frame = await track.recv()
                now = time.monotonic()
                if now - last_capture < 0.5:
                    continue
                image = frame.to_ndarray(format="bgr24")
                height, width = image.shape[:2]
                if width > 640:
                    scale = 640.0 / width
                    image = cv2.resize(image, (640, max(1, int(height * scale))))
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.minicpmo_manager.update_video_frame(
                        sessionid, base64.b64encode(encoded).decode("ascii")
                    )
                    last_capture = now
        except (MediaStreamError, asyncio.CancelledError):
            pass
        except Exception:
            logger.exception("Camera track failed for session %s", sessionid)

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令"""
        from server.i2v_avatar_manager import i2v_avatar_manager

        if i2v_avatar_manager.blocks_realtime():
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": -1,
                    "msg": "首个数字人正在生成，请等待其完成后再连接。",
                }, ensure_ascii=False),
            )
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # 通过 SessionManager 构建（内部会检查 max_session）
        try:
            sessionid = await session_manager.create_session(params)
        except MaxSessionError as e:
            logger.warning("Rejecting offer: %s", e)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": str(e)}),
            )
        logger.info('offer sessionid=%s', sessionid)
        avatar_session = session_manager.get_session(sessionid)

        # 创建 PeerConnection
        # Keep the avatar downlink identical to the original LiveTalking path.
        # Browser camera/microphone use a separate same-origin WebSocket.
        ice_servers = [RTCIceServer(urls=self.opt.stun)] if self.opt.stun else []
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )
        self.pcs.add(pc)
        self.session_pcs[sessionid] = pc
        microphone_tasks = set()
        minicpmo_started = False
        expiry_task = None

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            nonlocal minicpmo_started, expiry_task
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "connected" and self.minicpmo_manager and not minicpmo_started:
                if expiry_task:
                    expiry_task.cancel()
                minicpmo_started = True
                await self.minicpmo_manager.start_session(sessionid, avatar_session)
            if pc.connectionState in ("failed", "closed"):
                for task in list(microphone_tasks):
                    task.cancel()
                if self.minicpmo_manager and minicpmo_started:
                    await self.minicpmo_manager.stop_session(sessionid)
                await pc.close()
                self.pcs.discard(pc)
                self.session_pcs.pop(sessionid, None)
                session_manager.remove_session(sessionid)

        async def expire_unconnected_peer():
            try:
                await asyncio.sleep(35)
                if pc.connectionState not in ("connected", "closed"):
                    logger.warning(
                        "Closing WebRTC session %s after connection timeout (state=%s)",
                        sessionid, pc.connectionState,
                    )
                    await pc.close()
            except asyncio.CancelledError:
                pass

        expiry_task = asyncio.create_task(
            expire_unconnected_peer(), name=f"webrtc-expiry-{sessionid}"
        )
        self.track_tasks.add(expiry_task)
        expiry_task.add_done_callback(self.track_tasks.discard)

        # 添加发送轨道
        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        # 设置编解码器偏好
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        video_transceiver = next(
            (item for item in pc.getTransceivers() if item.kind == "video"), None
        )
        if video_transceiver:
            video_transceiver.setCodecPreferences(preferences)

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """RTCPush 模式：主动推流"""
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def shutdown(self):
        """关闭所有 PeerConnection"""
        for task in list(self.track_tasks):
            task.cancel()
        await asyncio.gather(*self.track_tasks, return_exceptions=True)
        self.track_tasks.clear()
        if self.minicpmo_manager:
            await self.minicpmo_manager.shutdown()
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()

    async def reset_connections(self):
        """Close active browser/media sessions without stopping the server."""
        sessionids = list(session_manager.sessions)

        for task in list(self.track_tasks):
            task.cancel()
        await asyncio.gather(*self.track_tasks, return_exceptions=True)
        self.track_tasks.clear()

        pcs = list(self.pcs)
        await asyncio.gather(*(pc.close() for pc in pcs), return_exceptions=True)
        self.pcs.clear()
        self.session_pcs.clear()

        if self.minicpmo_manager:
            await asyncio.gather(
                *(self.minicpmo_manager.stop_session(sid) for sid in sessionids),
                return_exceptions=True,
            )
        for sid in sessionids:
            session_manager.remove_session(sid)

        return {"peer_connections": len(pcs), "sessions": len(sessionids)}

    async def close_session(self, sessionid: str):
        """Explicitly close one browser session and await MiniCPM release."""
        pc = self.session_pcs.pop(sessionid, None)
        if self.minicpmo_manager:
            await self.minicpmo_manager.stop_session(sessionid)
        if pc is not None:
            await pc.close()
            self.pcs.discard(pc)
        session_manager.remove_session(sessionid)
        return pc is not None
