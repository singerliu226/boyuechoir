"""
阿里云百炼 Fun-ASR 云端录音文件转录模块。

使用 DashScope Fun-ASR 模型（离线录音文件识别），通过以下流程实现：
1. OssUtils.upload() 将本地音频上传到 DashScope 临时 OSS 存储
2. 通过 RESTful API（非 SDK）提交异步转录任务（支持 oss:// 临时 URL）
3. 轮询任务状态直到完成
4. 下载并解析 JSON 格式的识别结果

为什么使用 RESTful API 而非 Python SDK：
- Python SDK 的 Transcription.async_call() 不支持 oss:// 前缀的临时 URL
- RESTful API 支持 oss:// URL + X-DashScope-OssResourceResolve 头
- 避免了之前用 Recognition API（实时流式模型）遇到的空结果问题

为什么使用 fun-asr 而非 paraformer-realtime-v2：
- fun-asr 是专为录音文件设计的离线识别模型，准确率更高
- paraformer-realtime-v2 是实时流式模型，对预录制文件效果差
- fun-asr 支持中文（含方言）、英文、日语，VAD 优化更好
"""

import json
import os
import time
from typing import Optional

import requests

from app.logger import get_logger
from app.transcriber import TranscribedSegment, TranscribedWord, TranscriptionResult

logger = get_logger("transcriber_cloud")

# DashScope 服务地址
_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
_TRANSCRIPTION_URL = f"{_BASE_URL}/services/audio/asr/transcription"

# 轮询配置
_POLL_INTERVAL_S = 2.0   # 轮询间隔（秒）
_MAX_POLL_TIME_S = 300.0  # 最大等待时间（秒）


class CloudTranscriber:
    """
    基于阿里云百炼 Fun-ASR 的云端录音文件转录器。

    完整流程：上传文件 → 提交异步任务 → 轮询等待 → 解析结果。
    使用 fun-asr 模型，支持中文（含方言）、英文、日语的高精度转录。

    使用方式：
        transcriber = CloudTranscriber(api_key="sk-xxx")
        result = transcriber.transcribe("/path/to/audio.mp3", language="zh")
    """

    def __init__(self, api_key: str):
        """
        初始化云端转录器。

        Args:
            api_key: 阿里云百炼平台的 API Key（DashScope API Key）
        """
        self.api_key = api_key
        logger.info("云端转录器初始化完成（Fun-ASR 录音文件识别）")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "zh",
    ) -> TranscriptionResult:
        """
        转录音频文件，返回带时间戳的歌词。

        完整流程：
        1. 上传本地文件到 DashScope 临时 OSS 存储
        2. 通过 RESTful API 提交 fun-asr 异步转录任务
        3. 轮询等待任务完成
        4. 下载并解析识别结果

        Args:
            audio_path: 本地音频文件路径
            language: 语言代码，如 'zh'、'en'、'ja'

        Returns:
            TranscriptionResult: 包含歌词段落、语言信息和时长的完整结果

        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 转录过程出错
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        file_size_mb = os.path.getsize(audio_path) / 1024 / 1024
        logger.info(
            "开始云端转录: %s (%.1fMB, 语言=%s)",
            audio_path, file_size_mb, language,
        )

        # 步骤 1：上传文件到 DashScope 临时 OSS
        oss_url = self._upload_to_oss(audio_path)

        # 步骤 2：提交异步转录任务
        task_id = self._submit_task(oss_url, language)

        # 步骤 3：轮询等待任务完成
        result_data = self._wait_for_task(task_id)

        # 步骤 4：下载并解析识别结果
        return self._parse_transcription_result(result_data, language or "zh")

    def _upload_to_oss(self, file_path: str) -> str:
        """
        将本地音频文件上传到 DashScope 的临时 OSS 存储。

        使用 DashScope SDK 内置的 OssUtils.upload() 方法，
        该方法会自动申请上传凭证并完成上传，返回 oss:// 格式的临时 URL。
        临时 URL 有效期 48 小时，足够完成转录任务。

        Args:
            file_path: 本地音频文件路径

        Returns:
            str: oss:// 格式的临时文件 URL

        Raises:
            RuntimeError: 上传失败
        """
        logger.info("正在上传音频到 DashScope OSS: %s", os.path.basename(file_path))

        try:
            from dashscope.utils.oss_utils import OssUtils

            # upload() 签名: (model, file_path, api_key, upload_certificate)
            # 返回值: (file_url, upload_certificate) 元组
            # model 参数用于获取上传凭证，指定 fun-asr 即可
            result = OssUtils.upload(
                model="fun-asr",
                file_path=file_path,
                api_key=self.api_key,
            )

            # 返回值可能是元组 (url, certificate) 或直接是 url
            if isinstance(result, tuple):
                oss_url = result[0]
            else:
                oss_url = result

            if not oss_url:
                raise RuntimeError("OssUtils.upload() 返回空 URL")

            logger.info("文件上传成功: %s", oss_url)
            return oss_url

        except ImportError as e:
            raise RuntimeError(
                "需要 dashscope SDK: pip install dashscope"
            ) from e
        except Exception as e:
            logger.error("文件上传失败: %s", e)
            raise RuntimeError(f"文件上传到 OSS 失败: {e}") from e

    def _submit_task(
        self,
        file_url: str,
        language: Optional[str],
    ) -> str:
        """
        通过 RESTful API 提交异步转录任务。

        使用 RESTful API 而非 Python SDK 的原因：
        - SDK 的 Transcription.async_call() 不支持 oss:// 前缀的临时 URL
        - RESTful API 配合 X-DashScope-OssResourceResolve 头可以解析 oss:// URL

        Args:
            file_url: oss:// 格式的音频文件 URL
            language: 语言代码

        Returns:
            str: 任务 ID

        Raises:
            RuntimeError: 提交失败
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable",
        }

        # 构建请求体
        language_hints = [language] if language else ["zh", "en"]
        payload = {
            "model": "fun-asr",
            "input": {
                "file_urls": [file_url],
            },
            "parameters": {
                "language_hints": language_hints,
            },
        }

        logger.info(
            "提交转录任务: model=fun-asr, language_hints=%s, url=%s",
            language_hints, file_url[:80],
        )

        try:
            resp = requests.post(
                _TRANSCRIPTION_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
        except Exception as e:
            raise RuntimeError(f"提交转录任务失败: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"提交转录任务失败: HTTP {resp.status_code}, {resp.text}"
            )

        resp_json = resp.json()
        task_id = resp_json.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"提交转录任务未返回 task_id: {resp_json}")

        logger.info("转录任务已提交: task_id=%s", task_id)
        return task_id

    def _wait_for_task(self, task_id: str) -> dict:
        """
        轮询等待异步转录任务完成。

        DashScope 的录音文件转录是异步任务模式：
        - 提交后状态为 PENDING（排队中）
        - 开始处理后状态为 RUNNING
        - 处理完成为 SUCCEEDED 或 FAILED

        Args:
            task_id: 任务 ID

        Returns:
            dict: 任务结果（包含 results 列表）

        Raises:
            RuntimeError: 任务失败或超时
        """
        query_url = f"{_BASE_URL}/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        start_time = time.time()
        last_status = ""

        while True:
            elapsed = time.time() - start_time
            if elapsed > _MAX_POLL_TIME_S:
                raise RuntimeError(
                    f"转录任务超时 ({_MAX_POLL_TIME_S:.0f}s): task_id={task_id}"
                )

            try:
                resp = requests.get(
                    query_url,
                    headers=headers,
                    timeout=15,
                )
            except Exception as e:
                logger.warning("查询任务状态失败: %s (将重试)", e)
                time.sleep(_POLL_INTERVAL_S)
                continue

            if resp.status_code != 200:
                logger.warning(
                    "查询任务状态异常: HTTP %d (将重试)", resp.status_code
                )
                time.sleep(_POLL_INTERVAL_S)
                continue

            resp_json = resp.json()
            output = resp_json.get("output", {})
            status = output.get("task_status", "UNKNOWN")

            # 状态变化时记录日志
            if status != last_status:
                logger.info(
                    "任务状态: %s -> %s (已等待 %.1fs)",
                    last_status or "初始", status, elapsed,
                )
                last_status = status

            if status == "SUCCEEDED":
                logger.info("转录任务完成: 耗时 %.1fs", elapsed)
                return output

            if status == "FAILED":
                # 获取子任务错误详情
                results = output.get("results", [])
                error_msgs = []
                for r in results:
                    if r.get("subtask_status") == "FAILED":
                        error_msgs.append(
                            f"{r.get('code', 'UNKNOWN')}: {r.get('message', '未知错误')}"
                        )
                raise RuntimeError(
                    f"转录任务失败: {'; '.join(error_msgs) or '未知原因'}"
                )

            # PENDING 或 RUNNING，继续等待
            time.sleep(_POLL_INTERVAL_S)

    def _parse_transcription_result(
        self,
        task_output: dict,
        language: str,
    ) -> TranscriptionResult:
        """
        下载并解析转录结果 JSON。

        Fun-ASR 的识别结果存储在一个 JSON 文件中，通过 transcription_url 下载。
        JSON 结构包含：
        - properties: 音频属性（采样率、时长等）
        - transcripts: 转录结果（含句子级和词级时间戳）

        Args:
            task_output: 任务完成后的 output 字典（包含 results）
            language: 识别语言代码

        Returns:
            TranscriptionResult: 标准化的转录结果
        """
        results = task_output.get("results", [])
        if not results:
            logger.warning("转录任务无结果数据")
            return TranscriptionResult(
                segments=[], language=language,
                language_probability=1.0, duration=0.0,
            )

        # 处理第一个子任务的结果
        first_result = results[0]
        if first_result.get("subtask_status") != "SUCCEEDED":
            error_code = first_result.get("code", "UNKNOWN")
            error_msg = first_result.get("message", "未知错误")
            raise RuntimeError(f"子任务失败: {error_code} - {error_msg}")

        transcription_url = first_result.get("transcription_url")
        if not transcription_url:
            raise RuntimeError("转录结果 URL 为空")

        # 下载识别结果 JSON
        logger.info("下载转录结果: %s", transcription_url[:80] + "...")

        try:
            resp = requests.get(transcription_url, timeout=30)
            resp.raise_for_status()
            result_json = resp.json()
        except Exception as e:
            raise RuntimeError(f"下载转录结果失败: {e}") from e

        logger.info("转录结果 JSON 已下载，开始解析")

        # 解析 transcripts
        return self._parse_json_result(result_json, language)

    def _parse_json_result(
        self,
        result_json: dict,
        language: str,
    ) -> TranscriptionResult:
        """
        解析 Fun-ASR 返回的 JSON 识别结果。

        JSON 格式示例：
        {
          "properties": {"original_duration_in_milliseconds": 258000, ...},
          "transcripts": [{
            "channel_id": 0,
            "text": "完整文本...",
            "sentences": [
              {
                "begin_time": 1000,
                "end_time": 5000,
                "text": "第一句歌词",
                "words": [{"begin_time": 1000, "end_time": 1500, "text": "第一", "punctuation": ""}]
              }
            ]
          }]
        }

        Args:
            result_json: 下载的 JSON 结果
            language: 语言代码

        Returns:
            TranscriptionResult: 标准化的转录结果
        """
        segments: list[TranscribedSegment] = []

        # 获取音频总时长
        properties = result_json.get("properties", {})
        total_duration_ms = properties.get("original_duration_in_milliseconds", 0)

        logger.info(
            "音频属性: 格式=%s, 采样率=%sHz, 时长=%.1fs",
            properties.get("audio_format", "未知"),
            properties.get("original_sampling_rate", "未知"),
            total_duration_ms / 1000.0,
        )

        # 解析 transcripts
        transcripts = result_json.get("transcripts", [])
        if not transcripts:
            logger.warning("转录结果为空（无 transcripts）")
            return TranscriptionResult(
                segments=[], language=language,
                language_probability=1.0,
                duration=total_duration_ms / 1000.0,
            )

        duration_ms = 0

        for transcript in transcripts:
            content_duration = transcript.get("content_duration_in_milliseconds", 0)
            logger.info(
                "音轨 %d: 语音内容时长=%.1fs, 全文=%s",
                transcript.get("channel_id", 0),
                content_duration / 1000.0,
                (transcript.get("text", "")[:100] + "...") if len(transcript.get("text", "")) > 100 else transcript.get("text", ""),
            )

            sentences = transcript.get("sentences", [])
            for sentence in sentences:
                text = sentence.get("text", "").strip()
                if not text:
                    continue

                begin_time_ms = sentence.get("begin_time", 0)
                end_time_ms = sentence.get("end_time", 0)

                if end_time_ms > duration_ms:
                    duration_ms = end_time_ms

                # 解析词级别时间戳
                words_data = sentence.get("words", [])
                words = []
                if isinstance(words_data, list):
                    for w in words_data:
                        if not isinstance(w, dict):
                            continue
                        # Fun-ASR 的 words 中 text 和 punctuation 是分开的
                        word_text = w.get("text", "").strip()
                        punctuation = w.get("punctuation", "")
                        if word_text:
                            words.append(TranscribedWord(
                                text=word_text + punctuation,
                                start=w.get("begin_time", 0) / 1000.0,
                                end=w.get("end_time", 0) / 1000.0,
                                probability=1.0,
                            ))

                segment = TranscribedSegment(
                    text=text,
                    start=begin_time_ms / 1000.0,
                    end=end_time_ms / 1000.0,
                    words=words,
                )
                segments.append(segment)

                logger.debug(
                    "段落 [%.2fs - %.2fs]: %s",
                    segment.start, segment.end, text[:60],
                )

        logger.info(
            "云端转录完成: 共 %d 个段落, 语音时长=%.1fs, 总时长=%.1fs",
            len(segments),
            duration_ms / 1000.0,
            total_duration_ms / 1000.0,
        )

        return TranscriptionResult(
            segments=segments,
            language=language,
            language_probability=1.0,
            duration=max(duration_ms, total_duration_ms) / 1000.0,
        )
