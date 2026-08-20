"""v2 演进 · 任务管理（M6）：TaskPlan 以带标记的 system 消息进入历史。

机制（轻量版，无独立 planner 模块）：
- 模型在回复里输出 [vgent-plan]{json}[/vgent-plan] 计划块；
- 我们解析后以 system 消息 upsert 进 SQLite（历史里只保留最新一份），
  恢复会话即恢复计划；模型在后续回复里更新各步状态；
- 解析失败自动回退现行为（无计划模式），坏 JSON 不阻断对话。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from vgent.messages import Message

_PLAN_BLOCK = re.compile(r"\[vgent-plan\](.*?)\[/vgent-plan\]", re.DOTALL)
_VALID_STATUSES = ("pending", "running", "done", "failed")

PLAN_HINT = (
    "这是任务执行提示：如果用户请求需要多个步骤，你必须在回复正文中先输出任务计划，"
    "格式如下（不要放在思考内容里，不要省略）：\n"
    "[vgent-plan]\n"
    '{"steps": [{"description": "步骤描述", "status": "pending"}]}\n'
    "[/vgent-plan]\n"
    "然后逐步执行；每完成或失败一步，输出更新后的完整 [vgent-plan] 块。"
    "简单任务不需要计划。"
)


@dataclass
class TaskStep:
    description: str
    status: str = "pending"


@dataclass
class TaskPlan:
    steps: list[TaskStep] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"steps": [{"description": s.description, "status": s.status} for s in self.steps]}

    def to_text(self) -> str:
        return f"[vgent-plan]\n{json.dumps(self.to_json(), ensure_ascii=False)}\n[/vgent-plan]"

    def plan_message(self) -> Message:
        """落库/进历史的 system 消息：计划块 + 持续生效的更新提醒。"""
        text = "当前任务计划：各步状态变化时输出更新后的完整计划块（格式同首轮说明）。\n" + self.to_text()
        return Message("system", text)

    @property
    def done(self) -> bool:
        return bool(self.steps) and all(s.status == "done" for s in self.steps)

    @classmethod
    def from_text(cls, text: str | None) -> TaskPlan | None:
        """从文本提取计划块并解析；任何失败返回 None（回退无计划模式）。

        取最后一个 [vgent-plan] 块：模型正文/前缀可能先提到标记字样。
        """
        matches = list(_PLAN_BLOCK.finditer(text or ""))
        if not matches:
            return None
        block = matches[-1]
        try:
            data = json.loads(block.group(1))
        except json.JSONDecodeError:
            return None
        raw_steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(raw_steps, list):
            return None
        steps: list[TaskStep] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                return None
            desc = str(item.get("description", "")).strip()
            if not desc:
                return None
            status = str(item.get("status", "pending"))
            if status not in _VALID_STATUSES:
                status = "pending"
            steps.append(TaskStep(desc, status))
        if not steps:
            return None
        return cls(steps)


def plan_from_messages(messages: list[Message]) -> TaskPlan | None:
    """从消息列表找最新的计划块（从后往前扫，正文优先、思考流兜底）。

    user 文本误含标记的概率可忽略；思考流（reasoning_content）兜底是因为
    deepseek 偶尔把计划块放进思考而不是正文。
    """
    for m in reversed(messages):
        plan = TaskPlan.from_text(m.content)
        if plan is None and m.reasoning_content:
            plan = TaskPlan.from_text(m.reasoning_content)
        if plan is not None:
            return plan
    return None
