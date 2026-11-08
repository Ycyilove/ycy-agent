"""answer 拆分包：把原 _stream_answer 的 320 行拆成 7 个职责单一的文件。

对 service.py 只暴露 answer_stream()，签名与原 _stream_answer 一致。
"""
from .orchestrator import answer_stream

__all__ = ["answer_stream"]