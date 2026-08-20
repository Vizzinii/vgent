"""v2 演进 · 状态机（M6）：显式 Agent 状态，每轮结束落库。

此前状态隐含在 messages 里；现在每轮结束把最终状态写入 SQLite
（供恢复展示与未来的 UI/可观测）。转场日志（完整历史）后续再加，
v1 只持久化当前状态——符合「没证据不做深度优化」。
"""
from __future__ import annotations

from enum import Enum


class AgentState(str, Enum):
    IDLE = "idle"  # 就绪/等待输入
    PLANNING = "planning"  # 首轮、尚无任务计划
    EXECUTING = "executing"  # 计划在手，工具循环执行中
    WAITING_PERMISSION = "waiting_permission"  # 工具确认交互中（y/a/n）
    COMPLETED = "completed"  # 本轮正常结束
    FAILED = "failed"  # 本轮异常（LLM/网络/存储错误）
