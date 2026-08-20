"""M1 测试：消息模型 OpenAI 格式转换。"""
from vgent.messages import Message, ToolCall


def test_to_openai_plain() -> None:
    assert Message("user", "hi").to_openai() == {"role": "user", "content": "hi"}


def test_to_openai_with_tool_calls() -> None:
    m = Message("assistant", "", tool_calls=[ToolCall("c1", "shell", '{"cmd":"ls"}')])
    d = m.to_openai()
    assert d["tool_calls"][0]["function"]["name"] == "shell"
    assert d["tool_calls"][0]["function"]["arguments"] == '{"cmd":"ls"}'


def test_to_openai_tool_result() -> None:
    m = Message("tool", "输出", tool_call_id="c1")
    d = m.to_openai()
    assert d["tool_call_id"] == "c1"
    assert d["content"] == "输出"


def test_to_openai_reasoning_content() -> None:
    m = Message("assistant", "hi", reasoning_content="思考中")
    assert m.to_openai()["reasoning_content"] == "思考中"
    # 没有思考内容的消息不带该字段（兼容非 thinking 模型/其他 provider）
    assert "reasoning_content" not in Message("user", "hi").to_openai()
