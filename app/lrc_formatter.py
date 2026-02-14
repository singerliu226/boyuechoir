"""
LRC 格式化模块 - 将转录结果格式化为网易云音乐兼容的 LRC 歌词。

LRC (Lyric) 是一种通用的歌词文件格式，核心语法为 `[mm:ss.xx]歌词内容`。
本模块将 Whisper 的转录结果转换为标准 LRC 格式，并针对网易云音乐的
提交要求做了专门适配（英文标点、时间精度、提前量等）。

网易云音乐要求：
- 中括号、冒号、点号必须为英文半角字符
- 时间格式 [mm:ss.xx]，xx 为百分之一秒
- 歌词与演唱同步，建议提前 0.3-0.5 秒显示
"""

from dataclasses import dataclass
from typing import Optional

from app.logger import get_logger
from app.transcriber import TranscriptionResult, TranscribedSegment

logger = get_logger("lrc_formatter")

# 网易云推荐的歌词显示提前量（秒）
DEFAULT_OFFSET_SECONDS = 0.3


@dataclass
class LrcLine:
    """
    单行 LRC 歌词。

    Attributes:
        timestamp_seconds: 时间戳（秒），为该行歌词应该开始显示的时刻
        text: 歌词文本内容
    """
    timestamp_seconds: float
    text: str

    def to_lrc(self) -> str:
        """
        格式化为 LRC 时间标签行。

        将浮点秒数转换为 [mm:ss.xx] 格式，确保使用英文半角标点。
        百分之一秒精度足够满足网易云音乐的显示需求。

        Returns:
            str: 格式化后的 LRC 行，如 '[01:23.45]月光洒在窗前'
        """
        total_seconds = max(0.0, self.timestamp_seconds)
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        # [mm:ss.xx] 格式，xx 为百分之一秒
        return f"[{minutes:02d}:{seconds:05.2f}]{self.text}"


class LrcFormatter:
    """
    LRC 歌词格式化器。

    将 Whisper 转录结果或手动提供的歌词段落转换为标准 LRC 格式文本，
    完全兼容网易云音乐的歌词提交要求。

    使用方式：
        formatter = LrcFormatter(offset_seconds=0.3)
        lrc_text = formatter.format(transcription_result)
        # lrc_text 可以直接粘贴到网易云音乐歌词提交框
    """

    def __init__(self, offset_seconds: float = DEFAULT_OFFSET_SECONDS):
        """
        初始化 LRC 格式化器。

        Args:
            offset_seconds: 歌词显示提前量（秒）。网易云建议 0.3-0.5 秒，
                           让歌词在演唱前略微提前显示，方便跟唱。
        """
        self.offset_seconds = offset_seconds
        logger.info("LRC 格式化器初始化: offset=%.2fs", offset_seconds)

    def format(
        self,
        result: TranscriptionResult,
        title: Optional[str] = None,
        artist: Optional[str] = None,
    ) -> str:
        """
        将完整的转录结果格式化为 LRC 文本。

        处理流程：
        1. 生成 LRC 元信息头（标题、歌手、时长等，可选）
        2. 遍历所有转录段落，应用时间偏移
        3. 过滤空白行和重复行
        4. 拼接为完整的 LRC 文本

        Args:
            result: Whisper 转录结果
            title: 歌曲标题（可选，写入 LRC 元信息）
            artist: 歌手名（可选，写入 LRC 元信息）

        Returns:
            str: 完整的 LRC 格式文本，可直接粘贴到网易云
        """
        lines: list[str] = []

        # LRC 元信息头（可选）
        if title:
            lines.append(f"[ti:{title}]")
        if artist:
            lines.append(f"[ar:{artist}]")

        # 记录工具信息
        lines.append("[by:LyricSync]")

        # 添加空行分隔元信息和歌词
        if lines:
            lines.append("")

        # 转换歌词段落
        lrc_lines = self._segments_to_lrc_lines(result.segments)

        for lrc_line in lrc_lines:
            lines.append(lrc_line.to_lrc())

        lrc_text = "\n".join(lines)

        logger.info(
            "LRC 格式化完成: %d 行歌词, 总时长 %.1fs",
            len(lrc_lines), result.duration,
        )

        return lrc_text

    def format_segments(
        self,
        segments: list[dict],
        title: Optional[str] = None,
        artist: Optional[str] = None,
    ) -> str:
        """
        将前端编辑后的歌词段落列表格式化为 LRC 文本。

        当用户在 Web UI 中手动调整了时间戳后，使用此方法重新生成 LRC。
        每个 segment 字典需包含 'start'（秒）和 'text' 两个字段。

        Args:
            segments: 歌词段落列表，每项包含 {'start': float, 'text': str}
            title: 歌曲标题（可选）
            artist: 歌手名（可选）

        Returns:
            str: 完整的 LRC 格式文本
        """
        lines: list[str] = []

        if title:
            lines.append(f"[ti:{title}]")
        if artist:
            lines.append(f"[ar:{artist}]")

        lines.append("[by:LyricSync]")

        if lines:
            lines.append("")

        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            start = float(seg.get("start", 0.0))
            adjusted_start = max(0.0, start - self.offset_seconds)

            lrc_line = LrcLine(timestamp_seconds=adjusted_start, text=text)
            lines.append(lrc_line.to_lrc())

        return "\n".join(lines)

    def _segments_to_lrc_lines(
        self,
        segments: list[TranscribedSegment],
    ) -> list[LrcLine]:
        """
        将转录段落转换为 LRC 行列表，应用时间偏移并过滤无效内容。

        对每个段落：
        1. 跳过空白文本
        2. 将段落开始时间减去偏移量（提前显示）
        3. 确保时间戳不为负数
        4. 过滤连续重复的歌词行

        Args:
            segments: Whisper 转录的段落列表

        Returns:
            list[LrcLine]: 处理后的 LRC 行列表
        """
        lrc_lines: list[LrcLine] = []
        prev_text = ""

        for seg in segments:
            text = seg.text.strip()

            # 过滤空行
            if not text:
                continue

            # 过滤连续重复行（Whisper 有时会重复输出相同段落）
            if text == prev_text:
                logger.debug("跳过重复歌词: %s", text)
                continue

            # 应用时间偏移（提前显示）
            adjusted_start = max(0.0, seg.start - self.offset_seconds)

            lrc_lines.append(LrcLine(timestamp_seconds=adjusted_start, text=text))
            prev_text = text

        return lrc_lines


def parse_lrc(lrc_text: str) -> list[dict]:
    """
    解析 LRC 文本为结构化的歌词段落列表。

    用于导入已有的 LRC 歌词进行编辑。解析每一行的时间戳和文本，
    忽略元信息行（[ti:]、[ar:] 等）。

    Args:
        lrc_text: LRC 格式的歌词文本

    Returns:
        list[dict]: 歌词段落列表，每项包含 {'start': float, 'text': str}
    """
    import re

    # 匹配 [mm:ss.xx] 格式的时间标签
    pattern = re.compile(r"\[(\d{1,2}):(\d{2})\.(\d{2,3})\](.+)")
    segments = []

    for line in lrc_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            centiseconds = match.group(3)
            text = match.group(4).strip()

            # 处理百分之一秒（2位）或千分之一秒（3位）
            if len(centiseconds) == 2:
                frac = int(centiseconds) / 100.0
            else:
                frac = int(centiseconds) / 1000.0

            total_seconds = minutes * 60 + seconds + frac

            if text:
                segments.append({"start": round(total_seconds, 3), "text": text})

    logger.info("解析 LRC: 共 %d 行歌词", len(segments))
    return segments
