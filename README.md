# LyricSync - 网易云歌词自动对齐工具

面向网易云音乐人的全自动歌词对齐工具。上传一首歌，AI 自动识别歌词并生成带时间轴的 LRC 滚动歌词，一键复制后直接粘贴到网易云音乐提交。

## 核心功能

- **全自动转录**：基于 OpenAI Whisper (faster-whisper) 语音识别，自动识别中文歌词并生成精确时间戳
- **人声分离**：集成 Meta Demucs 模型，去除伴奏后识别，大幅提升准确率
- **可视化编辑**：音频波形播放器 + 歌词时间轴编辑器，支持逐行微调
- **网易云适配**：输出格式完全兼容网易云音乐 `[mm:ss.xx]` 歌词提交要求
- **一键复制**：生成的 LRC 文本可直接粘贴到网易云歌词提交框

## 环境要求

- Python 3.10+
- FFmpeg（音频处理必需）
- 推荐 8GB+ 内存（使用 large-v3 模型时）
- 推荐 NVIDIA GPU（大幅加速转录和分离，CPU 也可运行）

## 安装

### 1. 安装 FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows（使用 Chocolatey）
choco install ffmpeg
```

### 2. 创建虚拟环境

```bash
cd lyric-sync
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 安装 Python 依赖

```bash
# 如果有 NVIDIA GPU，先安装 CUDA 版 PyTorch（大幅加速）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

> 注意：如果没有 GPU，直接 `pip install -r requirements.txt` 即可，会自动使用 CPU 版本。

## 使用方法

### 启动服务

```bash
python -m app.main
```

服务启动后，打开浏览器访问 **http://localhost:8000**

### 使用流程

1. **上传音频**：拖拽或点击上传你的歌曲文件（支持 MP3/WAV/FLAC/OGG/M4A/AAC）
2. **选择设置**：
   - Whisper 模型：推荐 `large-v3`（最准确），也可选 `medium`（更快）
   - 人声分离：推荐开启，去除伴奏后识别更准确
   - 时间偏移：默认 0.3 秒提前量，可根据需要调整
3. **点击转录**：等待 AI 识别完成（首次加载模型需要几分钟）
4. **预览编辑**：
   - 波形播放器可视化播放，歌词高亮同步
   - 双击歌词行跳转到对应位置
   - 直接编辑时间戳和歌词文本
   - 点击行首播放按钮试听该行
5. **复制使用**：
   - 点击「一键复制」将 LRC 文本复制到剪贴板
   - 打开网易云音乐歌词提交页，粘贴即可

### 提交到网易云

1. 登录 [网易云音乐创作者中心](https://music.163.com/)
2. 进入歌曲管理 -> 歌词上传
3. 在歌词输入框中粘贴复制的 LRC 内容
4. 提交审核（通常 3 个工作日内审核完成）

## 项目结构

```
lyric-sync/
├── requirements.txt        # Python 依赖
├── README.md               # 使用说明（本文件）
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI 后端入口 + REST API
│   ├── transcriber.py      # Whisper 语音识别模块
│   ├── separator.py        # Demucs 人声分离模块
│   ├── lrc_formatter.py    # LRC 格式化输出模块
│   ├── logger.py           # 统一日志管理模块
│   └── static/
│       └── index.html      # Web UI 单页应用
├── uploads/                # 上传的音频临时存储
├── separated/              # 人声分离的中间文件
└── logs/                   # 运行日志
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI | 异步高性能 Web 框架 |
| 语音识别 | faster-whisper | OpenAI Whisper 的 CTranslate2 加速版，速度快 4 倍 |
| 人声分离 | Demucs (htdemucs_ft) | Meta 开源的端到端音源分离模型 |
| 前端 | 原生 HTML/JS + WaveSurfer.js | 轻量无依赖，波形可视化播放 |

## 常见问题

**Q: 首次运行很慢？**
A: 首次运行会自动下载 Whisper 模型（large-v3 约 3GB），请确保网络通畅。下载完成后会缓存到本地，后续启动会快很多。

**Q: 识别准确率不够高？**
A: 确保开启了人声分离功能；使用 large-v3 模型；原始音频质量越高，识别效果越好。

**Q: 没有 GPU 能用吗？**
A: 可以，只是速度会慢一些。使用 `medium` 或 `small` 模型可以加快 CPU 上的处理速度。

**Q: 支持英文歌曲吗？**
A: 支持。Whisper 支持 80+ 种语言的自动检测和识别。

## License

MIT
