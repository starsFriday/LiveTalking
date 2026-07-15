# MiniCPM-o 4.5 实时语音集成

LiveTalking 使用官方 `MiniCPM-o-Demo` 作为独立 API 推理服务，不使用容器。浏览器摄像头与麦克风通过
LiveTalking 的 WebRTC 上行，服务端将声音转成 16 kHz 单声道 float32 PCM，并把最新摄像头画面编码为 JPEG，发送到
官方 `/v1/realtime?mode=video` WebSocket。模型返回的 24 kHz float32 PCM 被流式
重采样、切成 20 ms 帧，并通过 `BaseAvatar.put_audio_frame()` 驱动口型和最终声音。

## 目录

```text
LiveTalking/
├── MiniCPM-o-Demo/             # 官方推理框架
└── models/MiniCPM-o-4_5/       # 模型权重
```

## 安装两个独立环境

两个环境相互隔离，避免 LiveTalking 与 MiniCPM-o 的 Transformers、WebSocket 等版本互相影响：

```bash
conda create -y -n livetalking python=3.12
conda activate livetalking
python -m pip install -r requirements-livetalking.txt

conda create -y -n livetalking-minicpm python=3.12
conda activate livetalking-minicpm
python -m pip install -r requirements-minicpm.txt
```

`requirements-livetalking.txt` 保留 `/api/asr` 实际使用的 FunASR/ModelScope；
`requirements-minicpm.txt` 只包含实时 API 运行依赖，不包含 Gradio 和测试工具。

## 启动 API 服务

```bash
cd LiveTalking
conda activate livetalking-minicpm
chmod +x start_minicpmo.sh stop_minicpmo.sh
./start_minicpmo.sh
```

脚本依次启动官方 `py_backend.server:22500`、`worker.py:22400` 和
`gateway.py:8006`。三个进程由脚本在前台统一管理，三路日志会直接显示在当前终端，同时也写入
`logs/minicpmo/`；按 `Ctrl+C` 会停止 gateway、worker 和 backend，不会遗留后台推理进程。

API 脚本会等待模型权重完成加载，并依次检查 backend、worker、gateway；看到
`MiniCPM-o Realtime API 已就绪` 后，再启动 LiveTalking：

```bash
conda activate livetalking
./stop_livetalking.sh
./start_livetalking.sh
```

`start_livetalking.sh` 会把额外命令行参数透传给 `app.py`。例如使用已经预处理好的 MuseTalk Avatar：

```bash
./start_livetalking.sh --model musetalk --avatar_id musetalk_avatar1 --batch_size 4
```

不同口型模型的 Avatar 预处理产物不通用；Wav2Lip Avatar 不能只修改 `--model` 就直接交给 MuseTalk。

`start_livetalking.sh` 会先检查 `http://127.0.0.1:8006/health`：

- API 正常：使用 MiniCPM 摄像头 + 麦克风实时对话。
- API 不通：自动以 `--no-minicpmo_enabled --tts edgetts` 启动，回退到原来的文本 LLM + EdgeTTS 流程。

前端会读取服务端的实际模式。回退模式不申请摄像头和麦克风权限，使用右侧文本输入的 `Chat LLM` 或
`Echo` 驱动数字人；其中 `Chat LLM` 仍需配置原项目使用的 `DASHSCOPE_API_KEY`。

访问 `http://127.0.0.1:8010/index.html`，点击“开始连接”并允许摄像头和麦克风权限。

`start_livetalking.sh` 使用 `exec` 在当前终端前台运行 LiveTalking，日志会直接显示，按 `Ctrl+C` 即可停止。如果检测到
以前遗留的 LiveTalking 进程，脚本会先停止旧进程，再在当前终端重启，不会静默复用后台进程。

通过 SSH 或 VS Code 访问远程机器时，页面、MiniCPM 状态以及摄像头/麦克风输入使用 `8010/TCP`；
数字人 WebRTC 下行使用本地 TCP TURN，因此还需转发 `3478/TCP`：

```bash
ssh -L 8010:127.0.0.1:8010 -L 3478:127.0.0.1:3478 user@server
```

数字人下行音视频保留 LiveTalking 原生的只接收 WebRTC 轨道。页面从 `127.0.0.1` 或 `localhost`
打开时，ICE 使用 `3478/TCP` relay，以穿过 VS Code/SSH TCP 隧道。摄像头和麦克风不加入该
PeerConnection，而是通过同源 WebSocket `/api/minicpmo/input/{sessionid}` 发给 MiniCPM。

MiniCPM 会话只在 WebRTC 状态真正变为 `connected` 后创建。页面打开、ICE 失败或重复点击不会提前占用
唯一的推理 Worker，因此不会出现“数字人尚未显示，模型却排队 #1”的情况。

摄像头输入不是数字人显示的前置条件。浏览器摄像头启动超时或被其他应用占用时，页面自动退到纯语音
模式；麦克风也不可用时仍会建立接收侧 WebRTC，保留待机动画和原来的文本/EdgeTTS 驱动。

## 数据格式

- 浏览器麦克风到 LiveTalking：同源 WebSocket，16 kHz mono float32 PCM。
- 浏览器摄像头到 LiveTalking：同源 WebSocket，约每 500 ms 一帧 JPEG。
- LiveTalking 到 MiniCPM：每 1000 ms 一个 base64 float32 `input.append`。
- MiniCPM 到 LiveTalking：24 kHz mono float32 `response.output.delta/audio`。
- LiveTalking 口型入口：16 kHz mono float32，每帧 320 samples（20 ms）。

MiniCPM 原始音频不能在浏览器另行播放；声音和口型统一由 LiveTalking 输出。

## 配置

配置位于 `config.yaml`：

```yaml
tts: edgetts
minicpmo_enabled: true
minicpmo_url: ws://127.0.0.1:8006/v1/realtime?mode=video
minicpmo_worker_health_url: http://127.0.0.1:22400/health
minicpmo_input_chunk_ms: 1000
minicpmo_max_response_seconds: 120
```

MiniCPM 模式不会关闭原来的文本驱动。连接后 LiveTalking 立即输出待机动画；右侧 `Echo/Chat LLM`
继续通过 EdgeTTS 驱动口型，麦克风对话则由 MiniCPM 的流式语音直接驱动同一条口型时间线。手动文本的
“打断”会同时停止正在输出的 MiniCPM 回答，避免两路声音重叠。

打断采用硬重置 MiniCPM WebSocket 会话，不依赖排队中的静音帧。正常长回答允许连续输出 120 秒；
同时检测 6～12 字短语是否高频重复，真正发生重复退化时会提前关闭异常会话并重新进入聆听状态。
重建会话前还会等待 Worker 的 `/health` 明确恢复为 `idle`，避免 Gateway 已释放票据、Worker 尚在清理时
立即重连所产生的 `server rejected WebSocket connection: HTTP 403`。

如果 Gateway 部署在另一台机器，只需修改 `minicpmo_url`。
