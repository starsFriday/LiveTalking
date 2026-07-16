###############################################################################
#  配置解析 — CLI 参数 + YAML 配置
###############################################################################

import argparse
import json
import os

try:
    import yaml
    _has_yaml = True
except ImportError:
    _has_yaml = False


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def _yaml_to_args(yaml_cfg):
    """将 YAML 字典中的 key 转换为 argparse 兼容的 `--key` 形式。

    argparse 的 dest 默认规则：`--model` → `model`，`--push-url` → `push_url`。
    此函数同时支持两种 key 写法：
      - model / batch_size          → 直接透传
      - model-name / batch-size    → 转换为 model_name / batch_size
    """
    result = {}
    for k, v in yaml_cfg.items():
        dest = k.replace('-', '_')
        result[dest] = v
    return result


def parse_args():
    """解析命令行参数，支持 YAML 配置文件覆盖默认值。

    优先级：CLI 参数 > YAML 配置文件 > add_argument(default=...)
    """
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # ─── 配置文件 ──────────────────────────────────────────────────────
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='YAML 配置文件路径（设为空字符串可跳过）')

    # ─── 音频 ──────────────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=25, help="video fps, must be 25")
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    # ─── 画面 ──────────────────────────────────────────────────────────
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── 数字人模型 ────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')

    # ─── 自定义动作和多形象 ────────────────────────────────────────────
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",
                        help="参考文件名或语音模型ID")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')

    # ─── MiniCPM-o realtime conversation ───────────────────────────────
    parser.add_argument('--minicpmo_enabled', action=argparse.BooleanOptionalAction, default=False,
                        help="enable MiniCPM-o realtime speech conversation")
    parser.add_argument('--minicpmo_url', type=str,
                        default='ws://127.0.0.1:8006/v1/realtime?mode=video',
                        help="official MiniCPM-o-Demo realtime websocket URL")
    parser.add_argument('--minicpmo_worker_health_url', type=str,
                        default='http://127.0.0.1:22400/health',
                        help="worker health endpoint used to avoid reconnect races")
    parser.add_argument('--minicpmo_model_path', type=str, default='models/MiniCPM-o-4_5',
                        help="model path mounted into the official MiniCPM-o-Demo service")
    parser.add_argument('--minicpmo_system_prompt', type=str,
                        default='你是一个自然、友好的语音助手，请简洁、完整地回答用户。')
    parser.add_argument('--minicpmo_input_chunk_ms', type=int, default=1000,
                        help="microphone PCM duration per input.append")
    parser.add_argument('--minicpmo_max_response_seconds', type=float, default=120.0,
                        help="hard-reset a runaway continuous model response after this many seconds")
    parser.add_argument('--minicpmo_barge_in_enabled', action=argparse.BooleanOptionalAction, default=True,
                        help="force MiniCPM back to listening when the user speaks over a response")
    parser.add_argument('--minicpmo_barge_in_threshold_db', type=float, default=-34.0,
                        help="microphone RMS threshold in dBFS used by voice barge-in")
    parser.add_argument('--minicpmo_barge_in_trigger_ms', type=int, default=280,
                        help="required sustained user voice duration before barge-in")
    parser.add_argument('--minicpmo_barge_in_cooldown_ms', type=int, default=1500,
                        help="minimum interval between two voice barge-ins")
    parser.add_argument('--minicpmo_barge_in_start_guard_ms', type=int, default=400,
                        help="ignore microphone energy briefly after model speech starts")
    parser.add_argument('--assistant_timezone', type=str, default='Asia/Shanghai',
                        help="IANA timezone used by the MiniCPM live clock")
    parser.add_argument('--web_search_enabled', action=argparse.BooleanOptionalAction, default=True,
                        help="allow the per-session web search tool for MiniCPM")
    parser.add_argument('--gemini_search_model', type=str, default='gemini-3.1-flash-lite',
                        help="Gemini audio model used with Google Search grounding")
    parser.add_argument('--web_search_max_context_chars', type=int, default=320,
                        help="maximum web fact characters privately returned to MiniCPM")
    parser.add_argument('--web_search_timeout_seconds', type=float, default=12.0,
                        help="hard timeout for Gemini audio search without blocking MiniCPM")

    # ─── 传输 ─────────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--stun', type=str, default='stun:stun.freeswitch.org:3478',
                        help="stun server url")
    parser.add_argument('--turn_url', type=str, default='',
                        help="TURN server URL, for example turn:127.0.0.1:3478?transport=tcp")
    parser.add_argument('--turn_username', type=str, default='',
                        help="TURN username")
    parser.add_argument('--turn_credential', type=str, default='',
                        help="TURN credential")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=5)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")

    # ─── 虚拟摄像头 ───────────────────────────────────────────────────
    parser.add_argument('--audio_output_device', type=int, default=None,
                        help="音频输出设备索引（None=系统默认，仅用于 --transport=virtualcam）。使用 python list_audio_devices.py 查看所有设备")

    # ─── 加载 YAML 配置文件 ────────────────────────────────────────────
    if _has_yaml:
        # 先用 parser 的已知参数做一次临时解析，只拿 --config 的值
        tmp_opt, _ = parser.parse_known_args()
        config_path = tmp_opt.config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_cfg = yaml.safe_load(f)
            if yaml_cfg and isinstance(yaml_cfg, dict):
                yaml_defaults = _yaml_to_args(yaml_cfg)
                parser.set_defaults(**yaml_defaults)
    else:
        print("[config] PyYAML 未安装，跳过 YAML 配置文件加载。"
              "安装: pip install pyyaml")

    # ─── 正式解析 CLI 参数 ─────────────────────────────────────────────
    opt = parser.parse_args()

    # ─── 后处理 ────────────────────────────────────────────────────────
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    return opt
