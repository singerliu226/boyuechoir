"""
数据模型 - 转录结果的核心数据结构。

这些 dataclass 被 transcriber.py（本地 Whisper）、
transcriber_cloud.py（云端 Fun-ASR）和 main.py 共同使用，
因此独立为单独模块，避免因本地依赖（faster-whisper）
不存在而导致导入失败。
"""

from dataclasses import dataclass, field


@dataclass
class TranscribedWord:
    """
    单个词的转录结果。

    Attributes:
        text: 识别出的文字内容
        start: 该词开始时间（秒）
        end: 该词结束时间（秒）
        probability: 识别置信度（0-1）
    """
    text: str
    start: float
    end: float
    probability: float = 0.0


@dataclass
class TranscribedSegment:
    """
    一个歌词段落的转录结果，对应 LRC 中的一行。

    Attributes:
        text: 该段落的完整文字
        start: 段落开始时间（秒）
        end: 段落结束时间（秒）
        words: 该段落中每个词的详细信息（可选）
    """
    text: str
    start: float
    end: float
    words: list[TranscribedWord] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """
    完整的转录结果。

    Attributes:
        segments: 所有歌词段落列表
        language: 检测到的语言代码（如 'zh'、'en'）
        language_probability: 语言检测的置信度
        duration: 音频总时长（秒）
    """
    segments: list[TranscribedSegment]
    language: str
    language_probability: float
    duration: float
