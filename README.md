<p align="center">
  <img src="./assets/logo-transparent.png" alt="Joyfox" width="220">
</p>

<h1 align="center">Joyfox Real-time Digital Human</h1>

<p align="center">
  基于 LiveTalking 与 MiniCPM-o 4.5 的可视、可听、可说全双工实时数字人
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-5f91e8.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.12-36b8aa.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/CUDA-12.8-786ce8.svg" alt="CUDA 12.8">
  <img src="https://img.shields.io/badge/Deployment-Bare--metal-f08bb4.svg" alt="Bare metal">
</p>

## 项目简介

Joyfox Real-time Digital Human 将 [LiveTalking](https://github.com/lipku/LiveTalking) 的实时数字人口型渲染，与
[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) 的端到端全模态全双工能力连接起来。

用户在浏览器中打开摄像头和麦克风后，MiniCPM-o 可以持续理解画面与语音，并直接输出流式语音；LiveTalking
把模型输出的声音转换成数字人口型和 WebRTC 音视频。默认 MiniCPM 链路不经过额外的 ASR、文本 LLM 或 TTS；仅当
用户主动打开“联网搜索”时，Gemini 会理解完整语句的意图，并仅对时效问题调用 Google Search，最终回答仍由 MiniCPM 组织和发声。

本仓库还提供图片生成数字人的完整工作流：上传人物图片后，服务调用 Grok 生成多种待机动作视频，再自动使用当前
运行的 Wav2Lip 或 MuseTalk 模型制作可选择的数字人形象。

> 本项目使用宿主机 Conda 环境直接运行，不需要 Docker。

## 主要功能

- MiniCPM-o 4.5 原生语音到语音全双工对话。
- 本地实时日期/时钟，通过模型可见画面持续校时，不请求网络。
- 可选按意图联网搜索；Gemini 3.1 Flash Lite 直接听取完整语句，仅对时效问题调用 Google Search。
- 浏览器摄像头实时视觉理解，支持模型“看见”用户画面。
- 模型音频直接驱动 Wav2Lip 或 MuseTalk 口型，不再串联外部 TTS。
- 数字人音频与视频统一由 LiveTalking WebRTC 输出，避免双播放器造成音画漂移。
- 卡片式数字人选择、预览与悬浮删除。
- 支持图片自动生成 `LITE / STD / PRO` 三档动作数字人。
- 支持使用图片或视频手动制作数字人。
- MiniCPM API 不可用时，自动回退到 LiveTalking + EdgeTTS 兼容模式。
- MiniCPM Backend、Worker、Gateway 独立环境和独立前台进程管理。
- 支持本地访问、VS Code/SSH 端口转发和 FRP HTTP 域名映射。

## 工作原理

```text
浏览器摄像头 ──JPEG 帧────────┐
                              │
浏览器麦克风 ──16 kHz PCM─────┼──> LiveTalking ──> MiniCPM Gateway :8006
                              │                         │
                              │                         ▼
                              │                 Worker :22400
                              │                         │
                              │                         ▼
                              │                 Backend :22500
                              │                         │
                              └──────流式语音响应 <─────┘
                                           │
                                           ▼
                                  16 kHz / 20 ms 音频帧
                                           │
                                           ▼
                                  Wav2Lip / MuseTalk
                                           │
                                           ▼
                               WebRTC 数字人音频 + 视频
                                           │
                                           ▼
                                         浏览器
```

关键点：MiniCPM 生成的声音既是用户听到的声音，也是口型模型使用的驱动音频。浏览器不会额外播放一份 MiniCPM
音频，因此声音与嘴型始终走同一条时间线。

## 环境要求

### 系统

- Linux，推荐 Ubuntu 22.04 或相近版本。
- NVIDIA GPU 和可用的 CUDA 驱动。
- Conda 或 Miniconda。
- Git、FFmpeg、Coturn、编译工具链。
- 浏览器推荐使用最新版 Chrome 或 Edge。

当前依赖文件按 Python 3.12、PyTorch 2.9.1、CUDA 12.8 组织。MiniCPM-o 官方 PyTorch Backend 要求 NVIDIA
GPU 显存大于 28 GB；本项目还会在同一张卡上运行口型模型，建议预留更多显存，优先使用 40 GB 或 48 GB 以上显卡。

安装常用系统依赖：

```bash
sudo apt update
sudo apt install -y git git-lfs ffmpeg coturn build-essential cmake libgl1 libglib2.0-0
```

确认 GPU：

```bash
nvidia-smi
```

## 项目目录

```text
LiveTalking/
├── app.py                         # LiveTalking 主服务
├── config.py                      # CLI/YAML 配置解析
├── config.yaml                    # 默认运行配置
├── start_livetalking.sh           # LiveTalking 前台启动脚本
├── stop_livetalking.sh            # 清理 LiveTalking 进程
├── start_minicpmo.sh              # MiniCPM 三段服务前台启动脚本
├── stop_minicpmo.sh               # 清理 MiniCPM 三段服务
├── status_minicpmo.sh             # MiniCPM 健康检查
├── create_avatar.sh               # 手动制作数字人
├── requirements-livetalking.txt   # LiveTalking Conda 环境依赖
├── requirements-minicpm.txt       # MiniCPM Conda 环境依赖
├── MiniCPM-o-Demo/                # MiniCPM 官方实时推理框架
├── models/
│   ├── MiniCPM-o-4_5/             # MiniCPM-o 4.5 权重
│   ├── wav2lip.pth                # Wav2Lip 权重
│   ├── musetalkV15/               # MuseTalk v1.5 权重
│   ├── sd-vae/                    # MuseTalk VAE
│   ├── whisper/                   # MuseTalk 音频编码器
│   ├── dwpose/                    # 姿态检测权重
│   └── face-parse-bisent/         # 人脸融合权重
├── data/
│   ├── avatars/                   # 已制作的数字人
│   └── i2v_jobs/                  # 图片生成任务及中间视频
├── minicpmo/                      # LiveTalking ↔ MiniCPM 音频桥
├── server/                        # WebRTC、会话和生成任务服务
├── avatars/                       # Wav2Lip/MuseTalk 渲染与预处理
├── web/                           # Joyfox 前端
├── logs/minicpmo/                 # MiniCPM 三段日志
└── .env                           # 私密 API 配置，不提交 Git
```

首次部署需要从 [joyfox/JoyFox-LiveTalking-models](https://huggingface.co/joyfox/JoyFox-LiveTalking-models)
下载模型。需要保留的数字人数据应单独备份。

## 从零安装

### 1. 准备源码

如果拿到的是已经打包好的项目，直接进入项目根目录。通过 Git 安装时，请克隆 Joyfox 定制项目：

```bash
git clone https://github.com/starsFriday/LiveTalking.git
cd LiveTalking
```

如果发行包中未包含 `MiniCPM-o-Demo/`，再执行：

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git MiniCPM-o-Demo
```

本项目包含定制代码，实际部署时应以这份定制仓库为准，不要再用上游 LiveTalking 文件覆盖当前目录。

### 2. 创建 LiveTalking 环境

```bash
conda create -y -n livetalking python=3.12
conda activate livetalking
python -m pip install --upgrade pip
python -m pip install -r requirements-livetalking.txt
```

该环境负责 Web 服务、WebRTC、Wav2Lip/MuseTalk、数字人预处理、EdgeTTS 回退和可选的 FunASR 接口。

### 3. 创建 MiniCPM 环境

```bash
conda create -y -n livetalking-minicpm python=3.12
conda activate livetalking-minicpm
python -m pip install --upgrade pip
python -m pip install -r requirements-minicpm.txt
```

该环境只负责 MiniCPM Backend、Worker 和 Gateway。两个环境必须分开，避免 `transformers`、`numpy`、
`websockets` 等版本互相覆盖。

启动脚本会检查 Conda 环境名，默认必须分别叫 `livetalking` 和 `livetalking-minicpm`。

### 4. 下载全部模型

本项目使用统一模型仓库：

```text
https://huggingface.co/joyfox/JoyFox-LiveTalking-models
```

该仓库约 25 GB，已经按本项目要求整理好以下内容：

- MiniCPM-o 4.5 完整权重。
- Wav2Lip 权重。
- MuseTalk v1.5、SD-VAE、Whisper、DWPose 和人脸解析权重。

推荐使用 Hugging Face 官方 `hf` 命令直接恢复到项目的 `models/` 目录。可以在任意一个 Conda 环境中执行：

```bash
cd /path/to/LiveTalking
python -m pip install -U huggingface_hub
hf download joyfox/JoyFox-LiveTalking-models \
  --local-dir models
```

`--local-dir models` 会保留模型仓库原有目录结构。下载中断后重复执行同一命令即可续传并跳过已经完成的文件。

也可以使用 Git LFS：

```bash
cd /path/to/LiveTalking
git lfs install
git clone https://huggingface.co/joyfox/JoyFox-LiveTalking-models models
```

Git LFS 方式要求本地尚不存在 `models/` 目录；如果源码包已经创建了空的 `models/`，优先使用上面的 `hf download`。

下载完成后检查：

```bash
test -f models/MiniCPM-o-4_5/model.safetensors.index.json
test -f models/wav2lip.pth
test -f models/musetalkV15/unet.pth
test -f models/sd-vae/diffusion_pytorch_model.bin
```

完整目录应包含：

```text
models/MiniCPM-o-4_5/model.safetensors.index.json
models/MiniCPM-o-4_5/model-00001-of-00004.safetensors
models/MiniCPM-o-4_5/model-00002-of-00004.safetensors
models/MiniCPM-o-4_5/model-00003-of-00004.safetensors
models/MiniCPM-o-4_5/model-00004-of-00004.safetensors
models/wav2lip.pth
models/musetalkV15/musetalk.json
models/musetalkV15/unet.pth
models/sd-vae/config.json
models/sd-vae/diffusion_pytorch_model.bin
models/whisper/
models/dwpose/dw-ll_ucoco_384.pth
models/face-parse-bisent/79999_iter.pth
models/face-parse-bisent/resnet18-5c106cde.pth
```

Wav2Lip 的 S3FD 人脸检测权重位于源码目录：

```text
avatars/wav2lip/face_detection/detection/sfd/s3fd.pth
```

它不在 `models/` 模型仓库中，请确保该文件存在。

`MiniCPM-o-Demo/` 中还必须存在 `gateway.py`、`worker.py` 和 `py_backend/server.py`；推理框架代码不属于模型
权重仓库。

不同口型模型生成的数字人数据不通用。Wav2Lip 数字人只能由 Wav2Lip 服务读取，MuseTalk 数字人只能由
MuseTalk 服务读取。

### 5. 设置脚本权限

```bash
chmod +x start_livetalking.sh stop_livetalking.sh
chmod +x start_minicpmo.sh stop_minicpmo.sh status_minicpmo.sh
chmod +x create_avatar.sh
```

### 6. 检查 TURN 配置路径

`turnserver.conf` 的 `pidfile` 和 `log-file` 使用绝对路径。项目被复制到另一台机器或另一个目录后，请把这两项改成
新项目目录下的 `logs/turnserver.pid` 和 `logs/turnserver.log`。

## 启动服务

需要两个终端，先启动 MiniCPM，再启动 LiveTalking。

### 终端 1：MiniCPM-o API

```bash
cd /path/to/LiveTalking
conda activate livetalking-minicpm
./start_minicpmo.sh
```

脚本按顺序启动：

| 组件 | 地址 | 作用 |
| --- | --- | --- |
| Backend | `127.0.0.1:22500` | 加载 MiniCPM 模型并执行推理 |
| Worker | `127.0.0.1:22400` | 管理模型会话和状态 |
| Gateway | `127.0.0.1:8006` | Realtime WebSocket 入口与排队 |

首次加载权重可能需要数分钟。出现以下信息后再启动 LiveTalking：

```text
MiniCPM-o Realtime API 已就绪: ws://127.0.0.1:8006/v1/realtime?mode=video
```

检查状态：

```bash
./status_minicpmo.sh
curl http://127.0.0.1:8006/health
```

### 终端 2：LiveTalking

```bash
cd /path/to/LiveTalking
conda activate livetalking
./start_livetalking.sh
```

访问：

```text
http://127.0.0.1:8010/index.html?v=minicpm-turn-tcp-v1
```

两个启动脚本都在当前终端前台运行并持续显示日志。按 `Ctrl+C` 会停止该脚本管理的全部子进程。每次启动前，脚本
也会清理同类旧进程，避免端口被历史进程占用。

如果终端异常退出，可在其他终端执行：

```bash
./stop_livetalking.sh
./stop_minicpmo.sh
```

## 切换口型模型和默认数字人

`start_livetalking.sh` 顶部有三个常用参数：

```bash
MODEL="musetalk"              # musetalk / wav2lip
AVATAR_ID="musetalk_222wave" # data/avatars 下的目录名
BATCH_SIZE="4"
```

可以直接修改，也可以在启动命令后覆盖：

```bash
./start_livetalking.sh \
  --model wav2lip \
  --avatar_id wav2lip256_avatar1 \
  --batch_size 8
```

MuseTalk 示例：

```bash
./start_livetalking.sh \
  --model musetalk \
  --avatar_id musetalk_222wave \
  --batch_size 4
```

默认 `AVATAR_ID` 必须存在，并且必须与 `MODEL` 匹配。服务启动后，页面只会列出与当前模型兼容的数字人。

## 页面使用

1. 在“选择数字人”区域点击一张预览图。
2. 点击“开始连接”。
3. 允许浏览器使用摄像头和麦克风。
4. 数字人待机画面出现后，可以直接对着麦克风说话。
5. MiniCPM 会持续接收摄像头画面，因此可以询问它当前看到了什么。
6. 点击“断开连接”会关闭 WebRTC，并等待 MiniCPM Worker 释放。
7. “清空连接”用于清理异常残留会话和本地 TURN 状态。

数字人卡片右下角的删除按钮只在鼠标悬浮时出现。以下数字人不会被删除：

- 服务启动时配置的默认数字人。
- 当前会话正在使用的数字人。
- 仍属于进行中生成批次的数字人。

删除默认数字人前，应先把 `start_livetalking.sh` 或 `config.yaml` 中的默认 ID 改为其他有效数字人并重启服务。

## 手动制作数字人

编辑 `create_avatar.sh` 顶部：

```bash
MODEL="musetalk"
SOURCE="./person.mp4"
AVATAR_ID="musetalk_person"
```

然后在 LiveTalking 环境中运行：

```bash
conda activate livetalking
./create_avatar.sh
```

生成结果位于：

```text
data/avatars/<AVATAR_ID>/
```

建议输入素材满足以下条件：

- 人脸清晰、无遮挡、正面或接近正面。
- 视频镜头固定，人物不要快速移动出画面。
- 推荐 25 FPS、3～10 秒、MP4。
- Wav2Lip 建议使用视频；MuseTalk 支持图片或视频。
- `AVATAR_ID` 不要包含 `/`、空格或其他路径字符。

制作时会占用 GPU。为了避免与实时对话抢显存，建议先停止 LiveTalking，再运行 `create_avatar.sh`。制作完成后用匹配的
模型重新启动服务，刷新页面即可看到新数字人。

如果直接执行 `python avatars/.../genavatar.py` 出现 `ModuleNotFoundError: No module named 'avatars'`，说明启动位置不对。
请从项目根目录运行 `create_avatar.sh`，或使用 `python -m avatars.<model>.genavatar`。

## 图片自动生成数字人

### 配置 xAI（仅用于 Grok 图生视频）

复制环境变量模板：

```bash
cp .env.example .env
```

在项目根目录 `.env` 中填写：

```dotenv
XAI_API_BASE_URL=https://api.x.ai/v1
XAI_API_KEY=xai-你的密钥
```

修改 `.env` 后需要重启 LiveTalking。不要把 `.env`、密钥或真实用户图片提交到 Git。

### 可选联网搜索

页面“联网搜索”开关默认关闭。开启后，麦克风仍按原路径实时送给 MiniCPM；本地轻量 VAD 只在一句话结束时把该段 16 kHz 语音异步提交给 Gemini。Gemini 在单次 Interactions 请求中理解语音并判断联网意图；天气、新闻、价格等时效问题调用 Google Search，普通聊天不搜索。关闭开关时不调用 Gemini，完全保持原生 MiniCPM 链路，搜索失败或 12 秒超时也不会断开会话。

在项目根目录 `.env` 中配置：

```dotenv
GEMINI_API_KEY=你的_Gemini_API_Key
GEMINI_SEARCH_MODEL=gemini-3.1-flash-lite
```

搜索结果会渲染为一张仅 MiniCPM 可见的“系统联网资料”视觉卡片，不经过第二次 TTS，也不由 Gemini 直接对用户回答。MiniCPM 始终负责上下文、推理、最终回答和发声。资料最多回灌 320 个字符，可通过 `web_search_timeout_seconds` 和 `web_search_max_context_chars` 调整。

### 三种生成模式

| 模式 | 动作数量 | 实际 Grok 视频请求 | 适用场景 |
| --- | ---: | ---: | --- |
| LITE | 1 | 3 | 默认模式，只需要自然待机 |
| STD | 7 | 9 | 常用待机和基础动作 |
| PRO | 14 | 16 | 完整动作集合 |

第一项“自然待机”会同时提交 3 个竞速请求，最先返回的结果立即进入数字人制作，另外两个结果不再下载；2 秒后再提交
剩余动作。因此动作数量与实际计费请求数不同，使用前请确认 xAI 账户额度和计费规则。

上传限制与处理规则：

- 支持 JPG、PNG、WebP。
- 文件最大 25 MB，宽高不得小于 64 像素。
- 后端保持原始宽高比，最长边按横图 `1280×720`、竖图 `720×1280`、方图 `720×720` 范围缩放。
- Grok 视频固定为 5 秒、720P、25 FPS。
- 每秒最多提交 10 个请求，最多 20 路并发等待。
- 当前服务用 Wav2Lip 启动，就制作 Wav2Lip 数字人；用 MuseTalk 启动，就制作 MuseTalk 数字人。
- Ultralight 暂不支持页面图片自动生成。

首个“自然待机”数字人完成后，页面会解除生成遮罩并允许连接；其余动作继续生成。由于视频预处理与实时口型可能共享
同一张 GPU，后台制作期间进行实时对话可能出现短暂卡顿。

自动生成的 ID 格式：

```text
<模型>_joyfox_<动作名称>_<YYYYMMDD_HHMMSS>
```

例如：

```text
wav2lip_joyfox_自然待机_20260715_150527
wav2lip_joyfox_点头_20260715_150527
```

任务与中间视频保存在 `data/i2v_jobs/<task_id>/`。已完成数字人发布到 `data/avatars/`。

## 配置说明

主配置文件为 `config.yaml`，优先级为：

```text
命令行参数 > config.yaml > config.py 内置默认值
```

常用配置：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | `wav2lip` | `wav2lip / musetalk / ultralight` |
| `avatar_id` | `wav2lip256_avatar1` | 默认数字人目录名 |
| `batch_size` | `8` | 口型推理批量大小 |
| `minicpmo_enabled` | `true` | 是否启用 MiniCPM 实时对话 |
| `minicpmo_url` | `ws://127.0.0.1:8006/v1/realtime?mode=video` | Realtime Gateway |
| `minicpmo_worker_health_url` | `http://127.0.0.1:22400/health` | Worker 状态检查 |
| `minicpmo_system_prompt` | 中文助手提示词 | MiniCPM 系统提示词 |
| `minicpmo_input_chunk_ms` | `1000` | 每个输入音频单元时长 |
| `minicpmo_max_response_seconds` | `120` | 单次连续回答保护上限 |
| `minicpmo_barge_in_enabled` | `true` | 用户说话时强制打断当前回答 |
| `minicpmo_barge_in_threshold_db` | `-34` | 抢话检测的麦克风音量阈值（dBFS） |
| `minicpmo_barge_in_trigger_ms` | `280` | 连续人声达到该时长后触发打断 |
| `minicpmo_barge_in_cooldown_ms` | `1500` | 两次打断之间的冷却时间 |
| `minicpmo_barge_in_start_guard_ms` | `400` | 模型开始说话后暂时忽略麦克风能量的时长 |
| `assistant_timezone` | `Asia/Shanghai` | 数字人实时系统时钟使用的 IANA 时区 |
| `gemini_search_model` | `gemini-3.1-flash-lite` | Gemini 音频理解与 Google Search 模型（可由 `.env` 覆盖） |
| `web_search_timeout_seconds` | `12` | Gemini 单次音频搜索硬超时，不阻塞 MiniCPM |
| `web_search_max_context_chars` | `320` | 交给 MiniCPM 的搜索资料最大字符数 |
| `listenport` | `8010` | Web 页面和 LiveTalking API 端口 |
| `turn_url` | `turn:127.0.0.1:3478?transport=tcp` | 本地 TCP TURN |
| `max_session` | `5` | LiveTalking 会话上限 |

`start_livetalking.sh` 会先检查 `http://127.0.0.1:8006/health`：

- 健康检查成功：启用 Joyfox-FullDuplex。
- 健康检查失败：自动添加 `--no-minicpmo_enabled --tts edgetts`，进入传统兼容模式。

当前 Joyfox 首页没有保留文本输入控件。兼容模式仍可通过 `/human` 等 API 使用 Echo/LLM + EdgeTTS。

## 网络和端口

| 端口 | 协议 | 是否建议公网开放 | 说明 |
| ---: | --- | --- | --- |
| 8010 | HTTP/WebSocket | 是，建议经 HTTPS 反向代理 | 页面、API、摄像头和麦克风输入 |
| 3478 | TCP TURN | 远程媒体需要 | 数字人 WebRTC 音视频中继 |
| 8006 | HTTP/WebSocket | 否 | MiniCPM Gateway，仅本机使用 |
| 22400 | HTTP/WebSocket | 否 | MiniCPM Worker，仅本机使用 |
| 22500 | HTTP | 否 | MiniCPM Backend，仅本机使用 |

### VS Code 或 SSH 转发

远程服务器开发时，同时转发 `8010/TCP` 和 `3478/TCP`：

```bash
ssh -L 8010:127.0.0.1:8010 \
    -L 3478:127.0.0.1:3478 \
    user@server
```

然后访问：

```text
http://127.0.0.1:8010/index.html?v=minicpm-turn-tcp-v1
```

### FRP HTTP 域名映射

页面和同源 WebSocket 可以将 8010 映射到域名：

```toml
[[proxies]]
name = "dev-avatar"
type = "http"
localIP = "127.0.0.1"
localPort = 8010
customDomains = ["avatar.dev.ad2.cc"]
```

访问：

```text
https://avatar.dev.ad2.cc/index.html?v=minicpm-turn-tcp-v1
```

注意：

- 非 `localhost` 页面需要 HTTPS 才能正常申请摄像头和麦克风权限。
- FRP 的 HTTP 代理只转发页面、HTTP API 和 WebSocket，不会自动转发 WebRTC 媒体。
- 数字人画面仍需要浏览器能直连服务器的 WebRTC 候选，或使用可从公网访问的 TURN 服务。
- 不要把 MiniCPM 的 8006、22400、22500 直接暴露到公网。

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/offer` | 创建数字人 WebRTC 会话 |
| `POST` | `/api/session/{sessionid}/disconnect` | 断开单个会话 |
| `GET` | `/api/avatars` | 获取当前模型可用数字人 |
| `DELETE` | `/api/avatars/{avatar_id}` | 删除未使用的数字人 |
| `POST` | `/api/i2v-avatar/task` | 创建图片生成数字人任务 |
| `GET` | `/api/i2v-avatar/task/{task_id}` | 查询图片生成任务 |
| `GET` | `/api/i2v-avatar/active` | 查询当前生成任务 |
| `GET` | `/api/i2v-avatar/config` | 查询生成模式与 API 配置 |
| `POST` | `/human` | 传统文本驱动兼容接口 |
| `POST` | `/humanaudio` | 传统音频驱动兼容接口 |
| `GET` | `/api/admin/sessions` | 查询 LiveTalking 会话 |

更多接口说明：

- [通用 API](docs/api.md)
- [数字人制作 API](docs/avatar_api.md)
- [管理 API](docs/admin_api.md)
- [MiniCPM 集成说明](docs/minicpmo-integration.md)

## 日志和状态检查

### MiniCPM

```text
logs/minicpmo/backend.log
logs/minicpmo/worker.log
logs/minicpmo/gateway.log
```

实时查看：

```bash
tail -f logs/minicpmo/backend.log \
        logs/minicpmo/worker.log \
        logs/minicpmo/gateway.log
```

### LiveTalking

主日志直接显示在 `start_livetalking.sh` 所在终端，文件日志位于：

```text
livetalking.log
```

常用检查：

```bash
curl http://127.0.0.1:8010/api/admin/sessions
curl http://127.0.0.1:8010/api/avatars
curl http://127.0.0.1:8010/api/i2v-avatar/active
```

## 常见问题

### 1. MiniCPM 启动后立即返回终端

正确的 `start_minicpmo.sh` 会一直占用当前终端并显示三路日志。如果脚本退出，查看：

```bash
tail -100 logs/minicpmo/backend.log
tail -100 logs/minicpmo/worker.log
tail -100 logs/minicpmo/gateway.log
```

重点检查 Conda 环境名、模型目录、CUDA OOM 和 8006/22400/22500 端口占用。

### 2. MiniCPM WebSocket 返回 HTTP 403

通常表示唯一 Worker 仍被旧会话占用或正在释放。先在页面断开连接并稍等 Worker 恢复；仍无法恢复时执行：

```bash
./stop_minicpmo.sh
conda activate livetalking-minicpm
./start_minicpmo.sh
```

### 3. 页面长时间显示“连接中”

依次检查：

1. LiveTalking 终端是否成功创建会话。
2. `8010/TCP` 是否可达。
3. 远程开发时 `3478/TCP` 是否同时转发。
4. 公网域名是否使用 HTTPS。
5. TURN 地址是否能被浏览器访问。
6. 浏览器控制台中的 ICE 状态和错误。

仅能打开网页不代表 WebRTC 媒体链路已经建立。

### 4. 数字人列表没有新形象

- 刷新页面重新读取 `/api/avatars`。
- 确认目录位于 `data/avatars/<avatar_id>`。
- 确认生成模型与当前启动模型一致。
- Wav2Lip 需要 `coords.pkl`、`full_imgs/`、`face_imgs/`。
- MuseTalk 还需要 `latents.pt`、`mask_coords.pkl` 和 `mask/`。

### 5. 选择了新 ID，画面仍是旧数字人

先断开当前会话，再选择新卡片并重新连接。已经建立的会话不会在中途替换 Avatar 数据。

### 6. `Timeout starting video source`

常见原因是数字人目录不完整、模型类型不匹配、预处理帧为空或 GPU 推理线程没有正常启动。查看 LiveTalking 终端和
`livetalking.log`，并核对上一节列出的目录文件。

### 7. MuseTalk 提示缺少 `diffusion_pytorch_model.safetensors`

如果随后出现 `Defaulting to unsafe serialization`，并且本地存在
`models/sd-vae/diffusion_pytorch_model.bin`，这是 Diffusers 从 SafeTensors 回退到 `.bin` 的提示，不一定是失败。
真正失败时请检查 VAE 文件是否完整以及 `config.json` 是否匹配。

### 8. `ModuleNotFoundError: No module named 'avatars'`

不要从 `avatars/wav2lip/` 或 `avatars/musetalk/` 子目录直接运行脚本。回到项目根目录执行：

```bash
conda activate livetalking
./create_avatar.sh
```

### 9. 图片生成进度停在某个百分比

前端进度包含视觉估算，真正状态取决于 Grok 视频任务和本地数字人预处理。检查：

```bash
curl http://127.0.0.1:8010/api/i2v-avatar/active
find data/i2v_jobs -maxdepth 3 -type f
```

如果 `clips/clip_*.mp4` 已出现但数字人数量没有变化，继续查看 LiveTalking 终端以及任务目录中的
`avatar_worker.log`、`avatar_worker_status.json`。

### 10. 后台生成数字人时实时对话卡顿

独立 Worker 只能隔离 Python 进程，不能消除同一张 GPU 上的算力和显存竞争。需要稳定实时对话时，建议等待整批制作完成，
或将数字人预处理放到另一张 GPU/另一台机器。

### 11. `Ctrl+C` 没有立即退出

模型进程可能正在执行 CUDA 调用，清理需要几秒。启动脚本会先发送正常终止信号，超时后强制清理。若终端已经失去控制，
在新终端执行对应的 `stop_*.sh`。

## 安全与隐私

- 当前服务没有完整的用户登录和权限系统，不建议把 8010 裸露到公网。
- 公网部署请增加 HTTPS、访问控制、限流和反向代理。
- 摄像头帧和麦克风音频会发送到本机 MiniCPM 服务。
- 使用图片自动生成时，人物图片和提示词会发送到配置的 xAI API。
- `.env`、用户图片、生成视频、数字人数据和日志可能包含敏感信息，应按业务要求管理和清理。
- 删除数字人是不可恢复操作，重要数据请先备份。

## 上游项目与许可证

本项目基于以下开源项目进行集成和二次开发：

- [starsFriday/LiveTalking](https://github.com/starsFriday/LiveTalking)
- [lipku/LiveTalking](https://github.com/lipku/LiveTalking)
- [OpenBMB/MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo)
- [OpenBMB/MiniCPM-o 4.5](https://github.com/OpenBMB/MiniCPM-o)
- Wav2Lip、MuseTalk、FunASR 等相关组件

本仓库根目录代码遵循 [Apache License 2.0](LICENSE)。模型权重、上游仓库、第三方 API 和素材可能有各自的许可证、
服务条款及商用限制，部署和分发前请分别确认。
