"""M6 测试：TaskPlan 解析/序列化（计划块消息化）。"""
from __future__ import annotations

from vgent.messages import Message
from vgent.task import PLAN_HINT, TaskPlan, TaskStep, plan_from_messages


def test_roundtrip() -> None:
    plan = TaskPlan([TaskStep("扫描仓库", "pending"), TaskStep("修改代码", "done")])
    text = plan.to_text()
    assert text.startswith("[vgent-plan]") and text.endswith("[/vgent-plan]")
    assert TaskPlan.from_text(text) == plan


def test_parse_valid_block() -> None:
    plan = TaskPlan.from_text(
        '[vgent-plan]\n{"steps": [{"description": "a", "status": "pending"}, '
        '{"description": "b", "status": "failed"}]}\n[/vgent-plan]'
    )
    assert plan is not None
    assert [s.description for s in plan.steps] == ["a", "b"]
    assert plan.steps[1].status == "failed"


def test_status_coerced_and_defaults() -> None:
    plan = TaskPlan.from_text(
        '[vgent-plan]\n{"steps": [{"description": "a", "status": "weird"}]}\n[/vgent-plan]'
    )
    assert plan is not None
    assert plan.steps[0].status == "pending"


def test_bad_inputs_return_none() -> None:
    assert TaskPlan.from_text(None) is None
    assert TaskPlan.from_text("no marker") is None
    assert TaskPlan.from_text("[vgent-plan]bad json[/vgent-plan]") is None
    assert TaskPlan.from_text('[vgent-plan]\n{"steps": []}\n[/vgent-plan]') is None
    assert TaskPlan.from_text('[vgent-plan]\n{"steps": [{"description": ""}]}\n[/vgent-plan]') is None
    assert TaskPlan.from_text('[vgent-plan]\n{"not": "steps"}\n[/vgent-plan]') is None


def test_plan_from_messages_latest_wins() -> None:
    older = '[vgent-plan]\n{"steps": [{"description": "old"}]}\n[/vgent-plan]'
    newer = '[vgent-plan]\n{"steps": [{"description": "new"}]}\n[/vgent-plan]'
    msgs = [Message("user", "hi"), Message("assistant", older), Message("assistant", newer)]
    plan = plan_from_messages(msgs)
    assert plan is not None and plan.steps[0].description == "new"


def test_plan_from_reasoning_content_fallback() -> None:
    """deepseek 偶尔把计划块放思考流（content 空）→ 正文没找到时扫 reasoning_content。"""
    block = '[vgent-plan]\n{"steps": [{"description": "思考里的计划"}]}\n[/vgent-plan]'
    msgs = [Message("user", "hi"), Message("assistant", "", reasoning_content=block)]
    plan = plan_from_messages(msgs)
    assert plan is not None and plan.steps[0].description == "思考里的计划"


def test_done_property_and_plan_message() -> None:
    plan = TaskPlan([TaskStep("a", "done"), TaskStep("b", "done")])
    assert plan.done
    m = plan.plan_message()
    assert m.role == "system"
    assert "[vgent-plan]" in m.content
    assert TaskPlan.from_text(m.content) == plan  # 带说明前缀也能解析


def test_plan_hint_mentions_marker() -> None:
    assert "[vgent-plan]" in PLAN_HINT
