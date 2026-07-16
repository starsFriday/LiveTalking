###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import asyncio
import glob
import os
import shutil
import signal
import subprocess
import time
from aiohttp import web

from utils.logger import logger


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager
from server.avatar_routes import setup_avatar_routes
from server.i2v_avatar_routes import setup_i2v_avatar_routes

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()
            minicpmo_manager = request.app.get("minicpmo_manager")
            if minicpmo_manager:
                await minicpmo_manager.interrupt(sessionid)

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        minicpmo_manager = request.app.get("minicpmo_manager")
        if minicpmo_manager:
            await minicpmo_manager.interrupt(sessionid)
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


async def admin_config(request):
    """Admin: 获取全局配置参数"""
    try:
        opt = request.app.get("opt")
        if opt:
            config = dict(vars(opt))
            minicpmo_manager = request.app.get("minicpmo_manager")
            config["web_search_available"] = bool(
                minicpmo_manager and minicpmo_manager.web_search_available
            )
            config["server_time_utc_ms"] = int(time.time() * 1000)
            return json_ok(data={"config": config})
        return json_error("Config not found")
    except Exception as e:
        logger.exception('admin_config exception:')
        return json_error(str(e))


async def admin_sessions(request):
    """Admin: 获取活跃的会话及其配置"""
    try:
        sessions_info = []
        for sid, avatar_session in session_manager.sessions.items():
            if avatar_session:
                s_opt = getattr(avatar_session, 'opt', None)
                s_data = {
                    "sessionid": sid,
                    "speaking": avatar_session.is_speaking() if hasattr(avatar_session, 'is_speaking') else False,
                    "recording": getattr(avatar_session, 'recording', False),
                }
                if s_opt:
                    s_data.update({
                        "model": getattr(s_opt, "model", ""),
                        "avatar_id": getattr(s_opt, "avatar_id", ""),
                        "REF_FILE": getattr(s_opt, "REF_FILE", ""),
                        "transport": getattr(s_opt, "transport", ""),
                        "batch_size": getattr(s_opt, "batch_size", 0),
                        "customopt": getattr(s_opt, "customopt", []),
                    })
                sessions_info.append(s_data)
        return json_ok(data={"sessions": sessions_info})
    except Exception as e:
        logger.exception('admin_sessions exception:')
        return json_error(str(e))


def _restart_local_turn_server():
    """Restart the local coturn daemon and return its new PID."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(root_dir, "turnserver.conf")
    pid_path = os.path.join(root_dir, "logs", "turnserver.pid")
    executable = shutil.which("turnserver")
    if not executable:
        raise RuntimeError("turnserver command not found")

    old_pid = None
    try:
        with open(pid_path, "r", encoding="utf-8") as file:
            old_pid = int(file.read().strip())
        os.kill(old_pid, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError, ValueError):
        old_pid = None

    if old_pid:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(old_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)

    completed = subprocess.run(
        [executable, "-c", config_path, "--daemon"],
        cwd=root_dir,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "failed to restart turnserver")

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with open(pid_path, "r", encoding="utf-8") as file:
                new_pid = int(file.read().strip())
            os.kill(new_pid, 0)
            return new_pid
        except (FileNotFoundError, ProcessLookupError, ValueError):
            time.sleep(0.1)
    raise RuntimeError("turnserver did not become ready")


async def clear_connections(request):
    """Clear WebRTC/MiniCPM sessions and stale local TURN allocations."""
    if request.remote not in {"127.0.0.1", "::1"}:
        raise web.HTTPForbidden(text="local access only")
    try:
        rtc_manager = request.app.get("rtc_manager")
        cleared = await rtc_manager.reset_connections() if rtc_manager else {}
        turn_pid = await asyncio.to_thread(_restart_local_turn_server)
        logger.info("Connections cleared from dashboard; TURN PID=%s", turn_pid)
        return json_ok(data={**cleared, "turn_pid": turn_pid})
    except Exception as e:
        logger.exception("clear_connections exception:")
        return json_error(str(e))


async def disconnect_session(request):
    """Close one WebRTC session and wait for its duplex Worker to be idle."""
    try:
        sessionid = request.match_info.get("sessionid", "")
        rtc_manager = request.app.get("rtc_manager")
        if not sessionid or rtc_manager is None:
            return json_error("invalid session")
        closed = await rtc_manager.close_session(sessionid)
        return json_ok(data={"closed": closed})
    except Exception as e:
        logger.exception("disconnect_session exception:")
        return json_error(str(e))


async def ice_config(request):
    """Return the browser ICE configuration used by this server."""
    opt = request.app.get("opt")
    ice_servers = []
    if opt and getattr(opt, "stun", ""):
        ice_servers.append({"urls": getattr(opt, "stun")})
    if opt and getattr(opt, "turn_url", ""):
        ice_servers.append({
            "urls": getattr(opt, "turn_url"),
            "username": getattr(opt, "turn_username", ""),
            "credential": getattr(opt, "turn_credential", ""),
        })
    return json_ok(data={"iceServers": ice_servers})


async def index_page(request):
    """Serve the dashboard without browser caching during local development."""
    response = web.FileResponse("web/index.html")
    response.headers.update({
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    })
    return response


async def avatar_preview(request):
    """Return one prepared avatar frame for the pre-connection video poster."""
    avatar_id = request.match_info.get("avatar_id", "")
    if not avatar_id or avatar_id != os.path.basename(avatar_id):
        raise web.HTTPBadRequest(text="invalid avatar id")
    frames = sorted(glob.glob(os.path.join("data", "avatars", avatar_id, "full_imgs", "*.*")))
    if not frames:
        raise web.HTTPNotFound(text="avatar preview not found")
    return web.FileResponse(frames[0], headers={"Cache-Control": "public, max-age=300"})


def _avatar_matches_model(avatar_path: str, model_name: str) -> bool:
    """Check whether a prepared avatar directory matches the active renderer."""
    common_ready = (
        os.path.isfile(os.path.join(avatar_path, "coords.pkl"))
        and os.path.isdir(os.path.join(avatar_path, "full_imgs"))
        and bool(glob.glob(os.path.join(avatar_path, "full_imgs", "*.*")))
    )
    if not common_ready:
        return False
    if model_name == "musetalk":
        return (
            os.path.isfile(os.path.join(avatar_path, "latents.pt"))
            and os.path.isfile(os.path.join(avatar_path, "mask_coords.pkl"))
            and os.path.isdir(os.path.join(avatar_path, "mask"))
        )
    if model_name == "ultralight":
        return (
            os.path.isfile(os.path.join(avatar_path, "ultralight.pth"))
            and os.path.isdir(os.path.join(avatar_path, "face_imgs"))
        )
    if model_name == "wav2lip":
        return (
            os.path.isdir(os.path.join(avatar_path, "face_imgs"))
            and not os.path.isfile(os.path.join(avatar_path, "ultralight.pth"))
        )
    return False


async def avatar_list(request):
    """List prepared avatars compatible with the renderer started by this process."""
    opt = request.app.get("opt")
    model_name = str(getattr(opt, "model", "") or "")
    default_avatar = str(getattr(opt, "avatar_id", "") or "")
    avatar_root = os.path.join("data", "avatars")
    avatars = []

    if os.path.isdir(avatar_root):
        for entry in os.scandir(avatar_root):
            if not entry.is_dir() or not _avatar_matches_model(entry.path, model_name):
                continue
            generation = {}
            metadata_path = os.path.join(entry.path, "joyfox_generation.json")
            try:
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    generation = json.load(metadata_file)
            except (OSError, ValueError, TypeError):
                generation = {}
            avatars.append({
                "id": entry.name,
                "is_default": entry.name == default_avatar,
                "preview": f"/api/avatar-preview/{entry.name}",
                "action_name": generation.get("action_name"),
                "action_index": generation.get("action_index"),
                "batch_timestamp": generation.get("batch_timestamp"),
            })
    def avatar_sort_key(item):
        if item["is_default"]:
            return (0, 0, 0, "")
        if item.get("batch_timestamp") and item.get("action_index") is not None:
            try:
                newest_first = -int(str(item["batch_timestamp"]).replace("_", ""))
                action_index = int(item["action_index"])
            except (TypeError, ValueError):
                newest_first, action_index = 0, 9999
            return (1, newest_first, action_index, item["id"].casefold())
        return (2, 0, 0, item["id"].casefold())

    avatars.sort(key=avatar_sort_key)
    return json_ok(data={
        "model": model_name,
        "default": default_avatar,
        "avatars": avatars,
    })


async def delete_avatar(request):
    """Delete one prepared avatar when it is not in use or being generated."""
    avatar_id = request.match_info.get("avatar_id", "")
    if not avatar_id or avatar_id != os.path.basename(avatar_id):
        return json_error("无效的数字人 ID。")

    opt = request.app.get("opt")
    model_name = str(getattr(opt, "model", "") or "")
    default_avatar = str(getattr(opt, "avatar_id", "") or "")
    if avatar_id == default_avatar:
        return json_error("这是服务启动时的默认数字人，不能直接删除。")

    for avatar_session in session_manager.sessions.values():
        session_opt = getattr(avatar_session, "opt", None)
        if str(getattr(session_opt, "avatar_id", "") or "") == avatar_id:
            return json_error("该数字人正在连接中，请先断开连接。")

    from server.i2v_avatar_manager import i2v_avatar_manager

    active_task = i2v_avatar_manager.get_active_task()
    if active_task and avatar_id in active_task.get("planned_avatar_ids", []):
        return json_error("该数字人所属批次仍在生成，请等待任务完成。")

    avatar_root = os.path.abspath(os.path.join("data", "avatars"))
    avatar_path = os.path.abspath(os.path.join(avatar_root, avatar_id))
    if os.path.dirname(avatar_path) != avatar_root or os.path.islink(avatar_path):
        return json_error("无效的数字人目录。")
    if not os.path.isdir(avatar_path):
        return json_error("数字人不存在或已经删除。")
    if not _avatar_matches_model(avatar_path, model_name):
        return json_error("该目录不是当前模型可用的数字人。")

    try:
        await asyncio.to_thread(shutil.rmtree, avatar_path)
        avatar_cache = request.app.get("avatar_cache")
        if isinstance(avatar_cache, dict):
            avatar_cache.pop(avatar_id, None)
        logger.info("Avatar deleted from dashboard: %s", avatar_id)
        return json_ok(data={"deleted": avatar_id})
    except Exception as exc:
        logger.exception("delete_avatar exception: avatar_id=%s", avatar_id)
        return json_error(f"删除失败：{exc}")


async def minicpmo_events(request):
    manager = request.app.get("minicpmo_manager")
    if manager is None or not manager.enabled:
        return json_error("Joyfox-FullDuplex is disabled")
    return await manager.event_websocket(request)


async def minicpmo_input(request):
    manager = request.app.get("minicpmo_manager")
    if manager is None or not manager.enabled:
        return json_error("Joyfox-FullDuplex is disabled")
    return await manager.input_websocket(request)


# ─── 路由注册 ──────────────────────────────────────────────────────────────

def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_get("/api/admin/config", admin_config)
    app.router.add_get("/api/admin/sessions", admin_sessions)
    app.router.add_post("/api/admin/clear-connections", clear_connections)
    app.router.add_post("/api/session/{sessionid}/disconnect", disconnect_session)
    app.router.add_get("/api/ice-config", ice_config)
    app.router.add_get("/api/minicpmo/events/{sessionid}", minicpmo_events)
    app.router.add_get("/api/minicpmo/input/{sessionid}", minicpmo_input)
    app.router.add_get("/api/avatars", avatar_list)
    app.router.add_delete("/api/avatars/{avatar_id}", delete_avatar)
    app.router.add_get("/api/avatar-preview/{avatar_id}", avatar_preview)
    app.router.add_get("/index.html", index_page)

    # ── Local ASR endpoint (SenseVoice/FunASR) ── Issue #604 ──
    try:
        from server.asr_server import asr_websocket_handler, is_funasr_available
        if is_funasr_available():
            app.router.add_get("/api/asr", asr_websocket_handler)
            logger.info("[ASR] Local SenseVoice ASR endpoint enabled at /api/asr")
        else:
            logger.info("[ASR] funasr not installed — local ASR endpoint disabled "
                        "(pip install funasr modelscope)")
    except Exception as e:
        logger.warning(f"[ASR] Failed to register ASR endpoint: {e}")

    # 注册 avatar 生成相关的路由
    setup_avatar_routes(app)
    setup_i2v_avatar_routes(app)

    app.router.add_static('/', path='web')
