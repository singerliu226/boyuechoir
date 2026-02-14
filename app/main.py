"""
FastAPI 后端入口 - 提供音频上传、转录、歌词编辑、LRC 导出等 REST API。

主要接口：
- POST /api/upload         上传音频文件
- POST /api/transcribe     执行转录（优先云端 Paraformer-v2，无 Key 降级本地 Whisper）
- POST /api/config         配置 API Key（阿里云百炼 DashScope）
- POST /api/format-lrc     将编辑后的歌词段落格式化为 LRC
- GET  /api/audio/{id}     获取上传的音频文件（供前端播放器使用）
- GET  /api/status/{id}    查询转录任务状态

采用异步任务模式：上传后立即返回任务 ID，前端轮询状态直到完成。
"""

# ---- macOS Python 3.9 SSL 证书修复 ----
# 必须放在文件最顶部、所有第三方库 import 之前执行！
# macOS 上 Python 3.9 不携带系统根证书，导致 aiohttp/WebSocket
# 连接 dashscope.aliyuncs.com 时 SSL 握手失败。
# 设置 SSL_CERT_FILE 环境变量后，ssl.create_default_context()
# 会使用 certifi 提供的根证书。
import os
import ssl
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
except ImportError:
    ssl._create_default_https_context = ssl._create_unverified_context

import uuid
import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from app.logger import get_logger
from app.transcriber_cloud import CloudTranscriber
from app.lrc_formatter import LrcFormatter, parse_lrc

# 本地 Whisper / Demucs 使用懒导入，部署环境可能不安装这些重型依赖
# 只在实际调用 _get_transcriber() / _get_separator() 时才 import
try:
    from app.transcriber import Transcriber, TranscriptionResult
except ImportError:
    Transcriber = None  # type: ignore
    TranscriptionResult = None  # type: ignore

try:
    from app.separator import VocalSeparator
except ImportError:
    VocalSeparator = None  # type: ignore

logger = get_logger("main")

# ----- 路径配置 -----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 支持的音频格式
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
# 最大文件大小：100MB
MAX_FILE_SIZE = 100 * 1024 * 1024

# ----- FastAPI 应用 -----
app = FastAPI(
    title="LyricSync - 网易云歌词自动对齐工具",
    description="上传音频，自动生成带时间轴的 LRC 歌词",
    version="1.0.0",
)

# ----- 全局单例 -----
# 延迟初始化，避免启动时就加载大模型占用内存
_transcriber: Optional[Transcriber] = None
_cloud_transcriber: Optional[CloudTranscriber] = None
_separator: Optional[VocalSeparator] = None
_formatter = LrcFormatter(offset_seconds=0.3)

# API Key 存储（内存中，通过 /api/config 设置）
_dashscope_api_key: Optional[str] = os.environ.get("DASHSCOPE_API_KEY")

# 任务状态存储（内存中，生产环境可换成 Redis）
_tasks: dict[str, dict] = {}


def _get_transcriber():
    """
    获取或创建全局本地 Transcriber 单例（降级方案）。

    在云端部署环境中可能不安装 faster-whisper，
    此时返回 None，调用方应处理这种情况。
    """
    global _transcriber
    if Transcriber is None:
        logger.warning("本地 Whisper 模块不可用（未安装 faster-whisper）")
        return None
    if _transcriber is None:
        _transcriber = Transcriber(model_size="large-v3")
    return _transcriber


def _get_cloud_transcriber() -> Optional[CloudTranscriber]:
    """
    获取或创建全局 CloudTranscriber 单例。

    如果没有配置 API Key 则返回 None，调用方应降级到本地 Whisper。
    当 API Key 更新时会重新创建实例。
    """
    global _cloud_transcriber, _dashscope_api_key
    if not _dashscope_api_key:
        return None
    if _cloud_transcriber is None:
        _cloud_transcriber = CloudTranscriber(api_key=_dashscope_api_key)
    return _cloud_transcriber


def _get_separator():
    """获取或创建全局 VocalSeparator 单例。云端部署时可能不可用。"""
    global _separator
    if VocalSeparator is None:
        logger.warning("人声分离模块不可用（未安装 demucs）")
        return None
    if _separator is None:
        _separator = VocalSeparator()
    return _separator


# ----- 请求/响应模型 -----

class ConfigRequest(BaseModel):
    """API Key 配置请求"""
    api_key: str                        # DashScope API Key


class TranscribeRequest(BaseModel):
    """转录请求参数"""
    task_id: str
    api_key: Optional[str] = None       # 前端传入的 API Key（优先使用）
    use_separator: bool = False         # 是否启用人声分离（云端模式下默认关闭）
    model_size: str = "large-v3"        # Whisper 模型大小（仅本地降级时使用）
    language: Optional[str] = None      # 指定语言（None=自动检测）


class AlignLyricsRequest(BaseModel):
    """歌词对齐请求参数"""
    task_id: str                        # 已上传音频的任务 ID
    lyrics_text: str                    # 用户手动输入的歌词文本（每行一句）
    api_key: Optional[str] = None       # DashScope API Key（可选）
    language: Optional[str] = "zh"      # 语言
    offset_seconds: float = 0.3         # 提前量


class FormatLrcRequest(BaseModel):
    """LRC 格式化请求参数"""
    segments: list[dict]                # 歌词段落 [{'start': float, 'text': str}]
    title: Optional[str] = None
    artist: Optional[str] = None
    offset_seconds: float = 0.3         # 提前量


class TaskStatus(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str                         # pending / processing / completed / failed
    progress: str = ""                  # 进度描述
    result: Optional[dict] = None       # 转录结果
    error: Optional[str] = None         # 错误信息


# ----- API 接口 -----

@app.post("/api/config")
async def set_config(request: ConfigRequest):
    """
    配置阿里云百炼 API Key。

    前端在开始转录前通过此接口传入 DashScope API Key，
    后端保存在内存中供后续云端转录使用。同时支持
    环境变量 DASHSCOPE_API_KEY 预设。

    Args:
        request: 包含 api_key 的配置请求

    Returns:
        dict: {'status': 'ok', 'message': str}
    """
    global _dashscope_api_key, _cloud_transcriber

    _dashscope_api_key = request.api_key.strip()
    # API Key 更新后重置云端转录器实例，下次使用时会重新创建
    _cloud_transcriber = None

    logger.info("API Key 已更新（长度=%d）", len(_dashscope_api_key))
    return {"status": "ok", "message": "API Key 配置成功"}


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    上传音频文件。

    接收音频文件，验证格式和大小后保存到 uploads 目录，
    返回唯一的 task_id 供后续转录使用。

    Args:
        file: 上传的音频文件（支持 mp3/wav/flac/ogg/m4a/aac/wma）

    Returns:
        dict: {'task_id': str, 'filename': str, 'size': int}
    """
    # 验证文件扩展名
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {ext}。支持的格式: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # 生成唯一任务 ID
    task_id = str(uuid.uuid4())[:8]
    safe_filename = f"{task_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    # 读取并保存文件（同时检查大小）
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大: {len(content) / 1024 / 1024:.1f}MB，最大支持 100MB",
        )

    with open(file_path, "wb") as f:
        f.write(content)

    # 初始化任务状态
    _tasks[task_id] = {
        "status": "pending",
        "progress": "文件已上传，等待转录",
        "filename": file.filename,
        "file_path": file_path,
        "result": None,
        "error": None,
    }

    logger.info(
        "文件上传成功: task_id=%s, filename=%s, size=%.1fMB",
        task_id, file.filename, len(content) / 1024 / 1024,
    )

    return {
        "task_id": task_id,
        "filename": file.filename,
        "size": len(content),
    }


@app.post("/api/transcribe")
async def transcribe_audio(request: TranscribeRequest):
    """
    启动异步转录任务。

    根据 task_id 找到已上传的音频，在后台线程中执行：
    1. （可选）人声分离 - 去除伴奏
    2. Whisper 语音识别 - 生成带时间戳的歌词
    3. LRC 格式化 - 输出网易云兼容格式

    转录过程耗时较长，采用异步模式，前端通过 /api/status 轮询进度。

    Args:
        request: 转录请求参数（task_id、是否分离人声、模型大小等）

    Returns:
        dict: {'task_id': str, 'status': 'processing'}
    """
    task_id = request.task_id

    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    task = _tasks[task_id]
    if task["status"] == "processing":
        raise HTTPException(status_code=409, detail="任务正在处理中，请勿重复提交")

    task["status"] = "processing"
    task["progress"] = "正在准备转录..."

    # 在后台线程中执行耗时的转录操作，避免阻塞 HTTP 请求
    asyncio.create_task(
        _run_transcription(task_id, request)
    )

    return {"task_id": task_id, "status": "processing"}


async def _run_transcription(task_id: str, request: TranscribeRequest):
    """
    后台执行转录的异步任务。

    优先使用云端 Paraformer-v2（速度快、准确率高），无 API Key 时降级到本地 Whisper。

    云端流程：上传到 DashScope 临时存储 -> Paraformer-v2 云端识别 -> 格式化 LRC
    降级流程：人声分离(可选) -> Whisper 本地转录 -> 格式化 LRC

    所有耗时操作通过 asyncio.to_thread 放到线程池执行，不阻塞事件循环。

    Args:
        task_id: 任务唯一标识
        request: 转录请求参数
    """
    task = _tasks[task_id]
    audio_path = task["file_path"]

    # 如果前端传了 API Key，先更新全局配置
    if request.api_key:
        global _dashscope_api_key, _cloud_transcriber
        if request.api_key.strip() != _dashscope_api_key:
            _dashscope_api_key = request.api_key.strip()
            _cloud_transcriber = None
            logger.info("[%s] 使用前端传入的 API Key", task_id)

    try:
        result: Optional[TranscriptionResult] = None

        # 策略 1：尝试云端转录（推荐）
        cloud = _get_cloud_transcriber()
        if cloud:
            try:
                result = await _run_cloud_transcription(task_id, cloud, audio_path, request.language)
            except Exception as e:
                logger.warning("[%s] 云端转录失败，降级到本地 Whisper: %s", task_id, e)
                task["progress"] = f"云端识别失败（{e}），正在切换到本地识别..."

        # 策略 2：降级到本地 Whisper（云端部署时可能不可用）
        if result is None:
            if Transcriber is None:
                raise RuntimeError(
                    "云端转录失败且本地模型不可用。请检查 API Key 是否正确。"
                )
            result = await _run_local_transcription(task_id, audio_path, request)

        # 格式化 LRC
        task["progress"] = "正在生成 LRC 歌词..."
        lrc_text = _formatter.format(result)

        # 构建前端需要的结构化数据
        segments_data = [
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
            }
            for seg in result.segments
        ]

        task["status"] = "completed"
        task["progress"] = "转录完成"
        task["result"] = {
            "segments": segments_data,
            "lrc_text": lrc_text,
            "language": result.language,
            "language_probability": round(result.language_probability * 100, 1),
            "duration": round(result.duration, 1),
            "segment_count": len(segments_data),
        }

        logger.info(
            "[%s] 转录完成: %d 段歌词, 语言=%s, 时长=%.1fs",
            task_id, len(segments_data), result.language, result.duration,
        )

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        task["progress"] = f"转录失败: {e}"
        logger.error("[%s] 转录失败: %s", task_id, e, exc_info=True)


async def _run_cloud_transcription(
    task_id: str,
    cloud: CloudTranscriber,
    audio_path: str,
    language: Optional[str],
) -> TranscriptionResult:
    """
    使用阿里云百炼 Paraformer-v2 进行云端转录。

    流程：上传音频到 DashScope 临时存储 -> 提交异步转录任务 -> 等待完成 -> 返回结果。
    整个过程通常只需几秒钟，相比本地 Whisper 快数百倍。

    Args:
        task_id: 任务 ID（用于更新进度）
        cloud: 云端转录器实例
        audio_path: 本地音频文件路径
        language: 语言代码

    Returns:
        TranscriptionResult: 转录结果

    Raises:
        Exception: 云端转录过程中的任何异常
    """
    task = _tasks[task_id]

    task["progress"] = "正在上传音频到云端..."
    logger.info("[%s] 开始云端转录（Fun-ASR）: %s", task_id, audio_path)

    # Fun-ASR 是异步任务：上传文件 → 提交任务 → 等待完成 → 下载结果
    # 整个过程通常需要几十秒到几分钟，取决于音频时长和队列情况
    task["progress"] = "云端正在识别歌词（可能需要 1-2 分钟）..."

    result = await asyncio.to_thread(
        cloud.transcribe,
        audio_path,
        language=language,
    )

    logger.info("[%s] 云端转录完成: %d 段歌词", task_id, len(result.segments))
    return result


async def _run_local_transcription(
    task_id: str,
    audio_path: str,
    request: TranscribeRequest,
) -> TranscriptionResult:
    """
    使用本地 Whisper 进行降级转录。

    当没有配置 API Key 或云端转录失败时使用此方案。
    可选的人声分离 + Whisper 本地识别，但速度较慢。

    Args:
        task_id: 任务 ID
        audio_path: 本地音频文件路径
        request: 转录请求参数

    Returns:
        TranscriptionResult: 转录结果
    """
    task = _tasks[task_id]
    transcribe_path = audio_path

    # 人声分离（可选，仅本地降级时可能需要）
    if request.use_separator:
        task["progress"] = "正在去除伴奏（首次需下载模型，之后会快很多，预计 2-5 分钟）..."
        logger.info("[%s] 开始人声分离", task_id)
        try:
            separator = _get_separator()
            transcribe_path = await asyncio.to_thread(
                separator.separate, audio_path
            )
            task["progress"] = "伴奏去除完成，正在加载识别模型..."
        except Exception as e:
            logger.warning("[%s] 人声分离失败，使用原始音频: %s", task_id, e)
            task["progress"] = "伴奏去除失败，使用原始音频继续识别..."
            transcribe_path = audio_path

    # Whisper 转录
    task["progress"] = "正在用本地模型识别歌词（首次加载模型可能需要几分钟）..."
    logger.info("[%s] 开始本地 Whisper 转录: %s", task_id, transcribe_path)

    transcriber = _get_transcriber()
    result = await asyncio.to_thread(
        transcriber.transcribe,
        transcribe_path,
        language=request.language,
    )

    logger.info("[%s] 本地转录完成: %d 段歌词", task_id, len(result.segments))
    return result


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """
    查询转录任务的当前状态。

    前端每 2-3 秒轮询此接口，获取转录进度。
    任务完成后返回完整的歌词识别结果和 LRC 文本。

    Args:
        task_id: 上传时返回的任务 ID

    Returns:
        TaskStatus: 包含状态、进度、结果等信息
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    task = _tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        result=task.get("result"),
        error=task.get("error"),
    )


@app.post("/api/align-lyrics")
async def align_lyrics(request: AlignLyricsRequest):
    """
    手动歌词对齐：用户提供歌词文本，系统自动匹配音频时间轴。

    工作流程：
    1. 解析用户输入的歌词（每行一句）
    2. 尝试用云端 ASR 获取音频时间信息
    3. 如果 ASR 有结果 → 智能匹配歌词行与 ASR 时间戳
    4. 如果 ASR 无结果 → 获取音频时长后均匀分配时间戳
    5. 格式化为 LRC 并返回

    适用场景：自动识别失败时（如合唱、纯音乐+人声混合等复杂音频），
    用户手动输入歌词后一键生成带时间轴的 LRC。

    Args:
        request: 包含 task_id、歌词文本、API Key 等

    Returns:
        dict: {segments, lrc_text, segment_count, duration, mode}
    """
    task_id = request.task_id

    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    # 解析歌词行（过滤空行）
    lyric_lines = [
        line.strip() for line in request.lyrics_text.strip().split("\n")
        if line.strip()
    ]
    if not lyric_lines:
        raise HTTPException(status_code=400, detail="歌词内容为空")

    logger.info("[%s] 开始歌词对齐: %d 行歌词", task_id, len(lyric_lines))

    audio_path = _tasks[task_id]["file_path"]

    # 更新 API Key（如果前端传了）
    if request.api_key:
        global _dashscope_api_key, _cloud_transcriber
        if request.api_key.strip() != _dashscope_api_key:
            _dashscope_api_key = request.api_key.strip()
            _cloud_transcriber = None

    # 尝试云端 ASR 获取时间轴
    asr_segments = []
    mode = "even"  # 默认均匀分配模式

    cloud = _get_cloud_transcriber()
    if cloud:
        try:
            _tasks[task_id]["status"] = "processing"
            _tasks[task_id]["progress"] = "正在上传音频到云端..."

            asr_result = await asyncio.to_thread(
                cloud.transcribe, audio_path, language=request.language,
            )
            if asr_result.segments:
                asr_segments = [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in asr_result.segments
                ]
                mode = "matched"
                logger.info("[%s] ASR 获取到 %d 个时间段", task_id, len(asr_segments))
        except Exception as e:
            logger.warning("[%s] 云端 ASR 失败，将使用均匀分配: %s", task_id, e)

    # 获取音频时长
    audio_duration = await _get_audio_duration(audio_path)

    # 执行对齐
    if mode == "matched" and asr_segments:
        aligned = _match_lyrics_with_asr(lyric_lines, asr_segments, audio_duration)
    else:
        aligned = _distribute_lyrics_evenly(lyric_lines, audio_duration)
        mode = "even"

    # 格式化 LRC
    formatter = LrcFormatter(offset_seconds=request.offset_seconds)
    lrc_text = formatter.format_segments(segments=aligned)

    # 更新任务状态
    _tasks[task_id]["status"] = "completed"
    _tasks[task_id]["progress"] = "歌词对齐完成"
    _tasks[task_id]["result"] = {
        "segments": aligned,
        "lrc_text": lrc_text,
        "language": request.language or "zh",
        "language_probability": 100.0,
        "duration": round(audio_duration, 1),
        "segment_count": len(aligned),
    }

    logger.info(
        "[%s] 歌词对齐完成: %d 行, 模式=%s, 时长=%.1fs",
        task_id, len(aligned), mode, audio_duration,
    )

    return {
        "segments": aligned,
        "lrc_text": lrc_text,
        "segment_count": len(aligned),
        "duration": round(audio_duration, 1),
        "mode": mode,
    }


async def _get_audio_duration(audio_path: str) -> float:
    """
    获取音频文件时长（秒）。

    使用 pydub 解析音频文件获取精确时长，
    支持所有 pydub/ffmpeg 支持的格式。

    Args:
        audio_path: 音频文件路径

    Returns:
        float: 时长（秒）
    """
    def _read_duration():
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0

    return await asyncio.to_thread(_read_duration)


def _match_lyrics_with_asr(
    lyric_lines: list[str],
    asr_segments: list[dict],
    audio_duration: float,
) -> list[dict]:
    """
    将用户提供的歌词行与 ASR 识别结果智能匹配。

    匹配策略：
    1. 如果歌词行数 == ASR 段落数 → 一一对应
    2. 如果歌词行数 < ASR 段落数 → 按比例映射
    3. 如果歌词行数 > ASR 段落数 → ASR 段落时间+均匀填充剩余

    核心思路：使用 ASR 段落的时间信息作为锚点，
    将用户歌词文本按位置比例映射到这些锚点上。

    Args:
        lyric_lines: 用户输入的歌词行列表
        asr_segments: ASR 识别的带时间戳段落
        audio_duration: 音频总时长（秒）

    Returns:
        list[dict]: 对齐后的歌词列表 [{'start': float, 'end': float, 'text': str}]
    """
    n_lyrics = len(lyric_lines)
    n_asr = len(asr_segments)

    if n_asr == 0:
        return _distribute_lyrics_evenly(lyric_lines, audio_duration)

    result = []

    if n_lyrics == n_asr:
        # 完美匹配：直接用 ASR 时间 + 用户歌词文本
        for i, line in enumerate(lyric_lines):
            result.append({
                "start": round(asr_segments[i]["start"], 3),
                "end": round(asr_segments[i]["end"], 3),
                "text": line,
            })
    elif n_lyrics <= n_asr:
        # 歌词行数少于 ASR 段落：按比例选取 ASR 段落作为锚点
        for i, line in enumerate(lyric_lines):
            asr_idx = int(i * n_asr / n_lyrics)
            asr_idx = min(asr_idx, n_asr - 1)
            result.append({
                "start": round(asr_segments[asr_idx]["start"], 3),
                "end": round(asr_segments[asr_idx]["end"], 3),
                "text": line,
            })
    else:
        # 歌词行数多于 ASR 段落：用 ASR 时间作为骨架，线性插值填充
        for i, line in enumerate(lyric_lines):
            ratio = i / max(n_lyrics - 1, 1)
            # 在 ASR 时间轴上找对应位置
            asr_pos = ratio * (n_asr - 1)
            asr_lower = int(asr_pos)
            asr_upper = min(asr_lower + 1, n_asr - 1)
            frac = asr_pos - asr_lower

            start = asr_segments[asr_lower]["start"] * (1 - frac) + asr_segments[asr_upper]["start"] * frac
            end = asr_segments[asr_lower]["end"] * (1 - frac) + asr_segments[asr_upper]["end"] * frac

            result.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": line,
            })

    return result


def _distribute_lyrics_evenly(
    lyric_lines: list[str],
    audio_duration: float,
) -> list[dict]:
    """
    将歌词均匀分配到音频时间轴上。

    当 ASR 完全无结果时的降级方案：假设歌词均匀分布在整个音频中，
    每行歌词占据相等的时间段。

    留出前 5% 和后 5% 作为前奏/尾奏的空白区域。

    Args:
        lyric_lines: 歌词行列表
        audio_duration: 音频总时长（秒）

    Returns:
        list[dict]: 均匀分配时间的歌词列表
    """
    n = len(lyric_lines)
    if n == 0:
        return []

    # 留出前后各 5% 作为前奏/尾奏
    start_offset = audio_duration * 0.05
    end_offset = audio_duration * 0.95
    usable_duration = end_offset - start_offset

    interval = usable_duration / n

    result = []
    for i, line in enumerate(lyric_lines):
        start = start_offset + i * interval
        end = start + interval
        result.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": line,
        })

    return result


@app.post("/api/format-lrc")
async def format_lrc(request: FormatLrcRequest):
    """
    将编辑后的歌词段落重新格式化为 LRC 文本。

    用户在 Web UI 中手动调整了时间戳或歌词文本后，调用此接口
    重新生成 LRC。支持自定义提前量和元信息。

    Args:
        request: 格式化请求，包含歌词段落列表和元信息

    Returns:
        dict: {'lrc_text': str}
    """
    formatter = LrcFormatter(offset_seconds=request.offset_seconds)
    lrc_text = formatter.format_segments(
        segments=request.segments,
        title=request.title,
        artist=request.artist,
    )
    return {"lrc_text": lrc_text}


@app.post("/api/parse-lrc")
async def parse_lrc_text(lrc_text: str = Form(...)):
    """
    解析已有的 LRC 文本为结构化数据。

    用于导入现有 LRC 歌词进行编辑，解析时间戳和歌词文本。

    Args:
        lrc_text: LRC 格式的歌词文本

    Returns:
        dict: {'segments': list[dict]}
    """
    segments = parse_lrc(lrc_text)
    return {"segments": segments}


@app.get("/api/audio/{task_id}")
async def get_audio(task_id: str):
    """
    获取已上传的音频文件，供前端播放器使用。

    根据 task_id 返回对应的音频文件，设置正确的 Content-Type
    以便浏览器内嵌播放器可以直接播放。

    Args:
        task_id: 任务 ID

    Returns:
        FileResponse: 音频文件流
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    file_path = _tasks[task_id]["file_path"]
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="音频文件已被删除")

    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=_tasks[task_id].get("filename", "audio.mp3"),
    )


# ----- 静态文件和首页 -----

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    """返回 Web UI 首页"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse("<h1>LyricSync</h1><p>前端文件未找到</p>", status_code=500)
    return FileResponse(index_path)


# ----- 启动入口 -----

if __name__ == "__main__":
    import uvicorn
    logger.info("启动 LyricSync 服务...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
