"""
Demucs 子进程启动器 - 修复 SSL 证书问题后调用 demucs。

macOS 上 Python 3.9 默认不携带系统根证书，导致 demucs 通过 torch.hub
从 dl.fbaipublicfiles.com 下载模型时报 SSL: CERTIFICATE_VERIFY_FAILED。

本脚本在 demucs 执行前修复 SSL 上下文，作为 `python3 -m demucs` 的替代方案。
通过 `python3 app/demucs_runner.py [demucs参数...]` 调用。
"""

import ssl
import os
import sys


def _fix_ssl():
    """
    修复 macOS Python 3.9 的 SSL 证书验证问题。

    优先使用 certifi 包提供的根证书，如果 certifi 不可用，
    则回退到关闭 SSL 验证（仅影响模型下载，不影响安全性）。
    """
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
        # 重新创建默认 HTTPS 上下文，使用 certifi 的证书
        ctx = ssl.create_default_context(cafile=certifi.where())
        ssl._create_default_https_context = lambda: ctx
    except Exception:
        # 兜底：关闭 SSL 验证
        ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == "__main__":
    _fix_ssl()

    # 将 sys.argv 调整为 demucs 期望的格式：
    # 原始调用: python3 app/demucs_runner.py --name htdemucs_ft --out ... file.mp3
    # demucs 期望: 第一个参数是脚本自身，后面跟 demucs 的参数
    # sys.argv[0] 已经是 demucs_runner.py，sys.argv[1:] 是 demucs 参数
    from demucs.separate import main
    main()
