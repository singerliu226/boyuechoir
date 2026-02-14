"""
Whisper 转录模块 - 将音频文件自动转录为带时间戳的歌词。

基于 faster-whisper 实现高性能语音识别，支持：
- 自动检测语言（默认优化中文识别）
- 段落级别（segment-level）时间戳
- 词级别（word-level）时间戳（用于更精确的对齐）
- 可配置模型大小（large-v3 / medium / small）

核心流程：加载模型 -> 读取音频 -> 执行转录 -> 返回带时间戳的歌词列表
"""

import os
from typing import Optional

# 配置 HuggingFace 镜像源，解决国内无法访问 huggingface.co 的问题。
# 必须在 import faster_whisper 之前设置，否则模型下载会走官方源导致超时。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from faster_whisper import WhisperModel

from app.logger import get_logger
from app.models import TranscribedWord, TranscribedSegment, TranscriptionResult

logger = get_logger("transcriber")

# 支持的 Whisper 模型大小及其特点
AVAILABLE_MODELS = {
    "large-v3": "最高准确率，推荐中文歌曲使用，需要较多显存/内存",
    "medium": "平衡准确率和速度，适合大多数场景",
    "small": "最快速度，适合快速预览",
}

# 默认模型
DEFAULT_MODEL = "large-v3"


class Transcriber:
    """
    基于 faster-whisper 的音频转录器。

    负责加载 Whisper 模型并执行语音识别，将音频文件转换为
    带精确时间戳的歌词段落列表。模型会在首次使用时自动下载。

    使用方式：
        transcriber = Transcriber(model_size="large-v3")
        result = transcriber.transcribe("/path/to/audio.mp3")
        for seg in result.segments:
            print(f"[{seg.start:.2f}] {seg.text}")
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        """
        初始化转录器，加载 Whisper 模型。

        Args:
            model_size: 模型大小，可选 'large-v3'、'medium'、'small'
            device: 推理设备，'auto' 自动选择（有 GPU 用 GPU，否则 CPU）
            compute_type: 计算精度，'auto' 自动选择最优配置
        """
        if model_size not in AVAILABLE_MODELS:
            logger.warning(
                "未知模型 '%s'，回退到默认模型 '%s'", model_size, DEFAULT_MODEL
            )
            model_size = DEFAULT_MODEL

        self.model_size = model_size
        self._model: Optional[WhisperModel] = None
        self._device = device
        self._compute_type = compute_type

        logger.info(
            "转录器初始化: model=%s, device=%s, compute_type=%s",
            model_size, device, compute_type,
        )

    def _ensure_model(self) -> WhisperModel:
        """
        延迟加载 Whisper 模型，首次调用时下载并缓存。

        返回已加载的 WhisperModel 实例。使用延迟加载策略避免
        应用启动时就占用大量内存，只在真正需要转录时才加载模型。

        Returns:
            WhisperModel: 已加载的模型实例
        """
        if self._model is None:
            logger.info("正在加载 Whisper 模型 '%s'（首次加载需下载，请耐心等待）...", self.model_size)
            self._model = WhisperModel(
                self.model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("Whisper 模型 '%s' 加载完成", self.model_size)
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        转录音频文件，返回带时间戳的歌词。

        执行完整的语音识别流程：加载音频 -> Whisper 推理 -> 提取时间戳 ->
        构建结构化结果。对中文歌曲做了专门优化（initial_prompt 引导）。

        Args:
            audio_path: 音频文件路径（支持 mp3、wav、flac 等常见格式）
            language: 指定语言代码（如 'zh'），None 则自动检测
            initial_prompt: Whisper 的引导提示词，用于提升特定场景的识别准确度。
                           中文歌曲建议使用包含标点的中文句子作为引导。

        Returns:
            TranscriptionResult: 包含所有歌词段落、语言信息和音频时长的完整结果

        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 转录过程出错
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        logger.info("开始转录: %s (语言=%s)", audio_path, language or "自动检测")

        model = self._ensure_model()

        # 根据语言选择合适的引导提示词
        if initial_prompt is None:
            prompt_map = {
                "zh": "以下是一首中文歌曲的歌词，请准确识别每一句歌词。",
                "en": "The following is the lyrics of a song. Please transcribe each line accurately.",
                "ja": "以下は歌の歌詞です。各行を正確に書き起こしてください。",
                "ko": "다음은 노래 가사입니다. 각 줄을 정확하게 전사해 주세요.",
            }
            initial_prompt = prompt_map.get(language, prompt_map["zh"])

        try:
            # 第一轮：VAD 标准阈值
            segments, info = self._do_transcribe(
                model, audio_path, language, initial_prompt, vad_threshold=0.35
            )

            # 如果标准阈值识别为 0 段，用极低阈值重试（对合唱/哼唱更友好）
            # 保持 VAD 开启以保证速度，只是把灵敏度拉到最低
            if len(segments) == 0:
                logger.warning(
                    "标准 VAD 识别出 0 段歌词，降低阈值到 0.05 重试（适配合唱/哼唱）..."
                )
                segments, info = self._do_transcribe(
                    model, audio_path, language, initial_prompt, vad_threshold=0.05
                )

            # 如果极低阈值仍然为 0，最后尝试关闭 VAD（会很慢，但确保能出结果）
            if len(segments) == 0:
                logger.warning(
                    "极低 VAD 阈值仍为 0 段，关闭 VAD 做最终尝试（可能较慢）..."
                )
                segments, info = self._do_transcribe(
                    model, audio_path, language, initial_prompt, vad_threshold=None
                )

            return TranscriptionResult(
                segments=segments,
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
            )

        except Exception as e:
            logger.error("转录失败: %s", str(e), exc_info=True)
            raise RuntimeError(f"转录过程出错: {e}") from e

    def _do_transcribe(
        self,
        model: WhisperModel,
        audio_path: str,
        language: Optional[str],
        initial_prompt: str,
        vad_threshold: Optional[float] = 0.35,
    ) -> tuple[list[TranscribedSegment], object]:
        """
        执行一次实际的 Whisper 转录。

        通过调节 vad_threshold 控制灵敏度，而不是简单地开关 VAD。
        低阈值对合唱/哼唱更友好，None 表示完全关闭 VAD。

        Args:
            model: 已加载的 WhisperModel 实例
            audio_path: 音频文件路径
            language: 语言代码，None 为自动检测
            initial_prompt: 引导提示词
            vad_threshold: VAD 灵敏度阈值（0-1，越低越灵敏）。
                          None 表示完全关闭 VAD（会很慢）。

        Returns:
            tuple: (歌词段落列表, 转录信息对象)
        """
        if vad_threshold is not None:
            vad_desc = f"VAD阈值={vad_threshold}"
        else:
            vad_desc = "VAD关闭"
        logger.info("执行转录 (%s): %s", vad_desc, audio_path)

        transcribe_kwargs = {
            "language": language,
            "initial_prompt": initial_prompt,
            "word_timestamps": True,
            "no_speech_threshold": 0.3,           # 降低"无语音"判定，避免跳过唱歌段落
            "condition_on_previous_text": True,    # 利用上下文提升连续歌词识别准确度
        }

        if vad_threshold is not None:
            transcribe_kwargs["vad_filter"] = True
            transcribe_kwargs["vad_parameters"] = {
                "min_silence_duration_ms": 600,   # 歌曲中停顿判定放宽
                "speech_pad_ms": 400,             # 语音段前后填充
                "threshold": vad_threshold,       # VAD 灵敏度
            }
        else:
            transcribe_kwargs["vad_filter"] = False

        segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)

        logger.info(
            "语言检测: %s (置信度: %.2f%%)",
            info.language,
            info.language_probability * 100,
        )

        segments: list[TranscribedSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()

            # 过滤 Whisper 幻觉：当音频无法识别时，Whisper 会将 initial_prompt
            # 原样输出为转录结果。检测并跳过与提示词高度相似的段落。
            if text and initial_prompt and self._is_hallucination(text, initial_prompt):
                logger.warning("过滤幻觉段落: '%s'", text)
                continue

            words = []
            if seg.words:
                words = [
                    TranscribedWord(
                        text=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in seg.words
                    if w.word.strip()
                ]

            segment = TranscribedSegment(
                text=text,
                start=seg.start,
                end=seg.end,
                words=words,
            )
            segments.append(segment)

            logger.debug(
                "段落 [%.2fs - %.2fs]: %s",
                seg.start, seg.end, text,
            )

        logger.info("转录结果 (%s): 共 %d 个段落", vad_desc, len(segments))
        return segments, info

    @staticmethod
    def _is_hallucination(text: str, prompt: str) -> bool:
        """
        判断转录段落是否为 Whisper 的幻觉输出。

        当 Whisper 无法从音频中识别出有效内容时，会将 initial_prompt
        原样或部分输出为转录结果。通过比较文本与提示词的相似度来检测。

        检测策略：
        - 完全匹配或包含提示词关键片段
        - 只有标点和常见幻觉短语（如"请订阅"、"谢谢观看"等）

        Args:
            text: 转录出的文本
            prompt: 使用的 initial_prompt

        Returns:
            bool: True 表示是幻觉，应该被过滤掉
        """
        # 去除标点后比较
        import re
        clean = lambda s: re.sub(r'[^\w]', '', s)
        clean_text = clean(text)
        clean_prompt = clean(prompt)

        # 与 prompt 高度相似
        if clean_text and clean_prompt:
            if clean_text in clean_prompt or clean_prompt in clean_text:
                return True

        # Whisper 常见的幻觉输出模式（中文场景）
        hallucination_patterns = [
            "请订阅", "谢谢观看", "感谢收看", "字幕by",
            "字幕制作", "请确认识别", "请准确识别",
            "以下是", "歌词如下", "请识别",
        ]
        for pattern in hallucination_patterns:
            if pattern in text:
                return True

        return False
