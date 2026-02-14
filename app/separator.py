"""
人声分离模块 - 使用 Meta 的 Demucs 模型从混音中提取人声。

将包含伴奏的完整音频分离为纯人声轨道，大幅提升 Whisper 对歌词的
识别准确率。Demucs 采用深度学习实现端到端的音源分离，支持分离出
人声（vocals）、鼓（drums）、贝斯（bass）和其他（other）四个轨道。

核心流程：加载模型 -> 读取音频 -> 执行分离 -> 导出人声轨道文件
"""

import os
import sys
import ssl
import subprocess
import shutil
from typing import Optional

# 配置 HuggingFace 镜像源，demucs 模型也托管在 HuggingFace 上
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 修复 macOS Python 3.9 SSL 证书问题。
# demucs 通过 torch.hub（urllib）下载模型，而 macOS 安装的 Python 3.9
# 默认不包含系统根证书，导致 SSL: CERTIFICATE_VERIFY_FAILED。
# 用 certifi 提供的证书包创建默认 SSL 上下文来解决。
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    ssl._create_default_https_context = ssl.create_default_context
    ssl._create_default_https_context()  # 测试一下
except Exception:
    # 兜底：如果 certifi 也不行，关闭 SSL 验证（仅影响模型下载）
    ssl._create_default_https_context = ssl._create_unverified_context

from app.logger import get_logger

logger = get_logger("separator")

# Demucs 模型选择：htdemucs 是单模型版本，速度快且质量足够好。
# htdemucs_ft（微调版）质量略高，但内部跑 4 个模型取平均，CPU 上慢 4 倍。
DEFAULT_DEMUCS_MODEL = "htdemucs"


class VocalSeparator:
    """
    基于 Demucs 的人声分离器。

    通过调用 demucs CLI 执行音源分离，从混音音频中提取纯人声轨道。
    分离后的人声文件可以直接传给 Whisper 进行更准确的歌词识别。

    使用方式：
        separator = VocalSeparator()
        vocal_path = separator.separate("/path/to/song.mp3")
        # vocal_path 指向分离出的纯人声 wav 文件
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DEMUCS_MODEL,
        output_dir: Optional[str] = None,
    ):
        """
        初始化人声分离器。

        Args:
            model_name: Demucs 模型名称，默认 'htdemucs_ft'（微调版，质量最高）。
                        也可选 'htdemucs'（速度更快）或 'mdx_extra'（另一种架构）。
            output_dir: 分离结果的输出目录。None 则使用项目根目录下的 'separated' 文件夹。
        """
        self.model_name = model_name

        if output_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_root, "separated")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("人声分离器初始化: model=%s, output_dir=%s", model_name, output_dir)

    def _check_demucs_available(self) -> bool:
        """
        检查 demucs 命令行工具是否可用。

        通过尝试运行 `python -m demucs --help` 验证 demucs 是否已正确安装。
        如果不可用，后续分离操作会抛出异常。

        Returns:
            bool: demucs 是否可用
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "demucs", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def separate(
        self,
        audio_path: str,
        two_stems: bool = True,
    ) -> str:
        """
        从音频中分离出人声轨道。

        调用 demucs CLI 执行音源分离。默认使用 two_stems 模式（只分离人声和伴奏），
        速度更快且对歌词识别场景完全够用。

        处理流程：
        1. 验证输入文件存在且 demucs 可用
        2. 构建 demucs 命令行参数
        3. 执行分离（耗时较长，取决于音频时长和硬件）
        4. 定位并返回分离出的人声文件路径

        Args:
            audio_path: 输入音频文件的绝对路径
            two_stems: 是否只分离人声/伴奏两个轨道（推荐 True，速度快）

        Returns:
            str: 分离出的人声 wav 文件的绝对路径

        Raises:
            FileNotFoundError: 输入音频不存在
            RuntimeError: demucs 未安装或分离过程出错
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        if not self._check_demucs_available():
            raise RuntimeError(
                "demucs 未安装或不可用。请运行: pip install demucs"
            )

        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        logger.info("开始人声分离: %s (model=%s)", audio_path, self.model_name)

        # 使用 demucs_runner.py 代替 `python -m demucs`，
        # 因为 runner 脚本内部会先修复 SSL 证书问题再启动 demucs
        runner_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "demucs_runner.py"
        )
        cmd = [
            sys.executable, runner_path,
            "--name", self.model_name,
            "--out", self.output_dir,
            "--mp3",              # 输出为 mp3 格式（节省空间）
        ]

        if two_stems:
            cmd.extend(["--two-stems", "vocals"])  # 只分离人声和伴奏

        cmd.append(audio_path)

        logger.debug("执行命令: %s", " ".join(cmd))

        # 构建子进程环境变量
        env = os.environ.copy()
        env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 最长 10 分钟超时
                env=env,
            )

            if process.returncode != 0:
                logger.error("demucs 执行失败:\nstdout: %s\nstderr: %s",
                             process.stdout, process.stderr)
                raise RuntimeError(f"demucs 分离失败: {process.stderr}")

            logger.info("demucs 分离完成")

        except subprocess.TimeoutExpired:
            logger.error("demucs 执行超时（>10分钟）")
            raise RuntimeError("人声分离超时，请检查音频文件大小是否过大")

        # 定位分离出的人声文件
        # demucs 输出路径格式: {output_dir}/{model_name}/{audio_name}/vocals.mp3
        vocal_path = os.path.join(
            self.output_dir, self.model_name, audio_name, "vocals.mp3"
        )

        # 也尝试 wav 格式（某些 demucs 版本默认输出 wav）
        if not os.path.isfile(vocal_path):
            vocal_path_wav = os.path.join(
                self.output_dir, self.model_name, audio_name, "vocals.wav"
            )
            if os.path.isfile(vocal_path_wav):
                vocal_path = vocal_path_wav

        if not os.path.isfile(vocal_path):
            logger.error("未找到分离结果，预期路径: %s", vocal_path)
            raise RuntimeError(f"分离完成但未找到人声文件: {vocal_path}")

        logger.info("人声文件路径: %s", vocal_path)
        return vocal_path

    def cleanup(self, audio_name: str) -> None:
        """
        清理指定音频的分离临时文件。

        分离完成并转录后，调用此方法删除不再需要的中间文件以释放磁盘空间。

        Args:
            audio_name: 原始音频文件名（不含扩展名）
        """
        target_dir = os.path.join(self.output_dir, self.model_name, audio_name)
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
            logger.info("已清理分离临时文件: %s", target_dir)
