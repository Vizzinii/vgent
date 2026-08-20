"""M7 测试：失败信号启发式 + 反思调用。"""
from __future__ import annotations

from types import SimpleNamespace

from vgent.messages import Message
from vgent.reflection import MAX_REFLECT_ROUNDS, looks_failed, reflect


def test_shell_nonzero_exit_is_failure() -> None:
    assert looks_failed("exit 1\ncommand not found")
    assert looks_failed("exit 2\nsomething")
    assert looks_failed("exit 127")


def test_shell_zero_exit_not_failure() -> None:
    assert not looks_failed("exit 0\nhello")
    assert not looks_failed("exit 0（无输出）")


def test_zero_exit_with_strong_marker_is_failure() -> None:
    """exit 0 下只认强信号（测试失败/Traceback），小写 failed 不误伤正常输出。"""
    assert not looks_failed("exit 0\n1 failed")
    assert looks_failed("exit 0\n==== FAILED ====")
    assert looks_failed("exit 0\nTraceback (most recent call last)")


def test_error_markers() -> None:
    assert looks_failed("读取失败：No such file")
    assert looks_failed("错误：缺少 command 参数")
    assert looks_failed("参数解析失败：JSON 非法")
    assert looks_failed("用户拒绝了工具 shell 的调用")
    assert looks_failed("未知工具：shellread_file")
    assert looks_failed("命令超时（>120s）")
    assert looks_failed("工具 shell 执行出错：boom")


def test_normal_output_not_failed() -> None:
    assert not looks_failed("exit 0\nvgent-ok")
    assert not looks_failed("1\t# vgent")
    assert not looks_failed("")
    assert not looks_failed("hello world")


def test_reflect_returns_text() -> None:
    class FakeLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return SimpleNamespace(
                messages=[Message("assistant", "Failure: 依赖缺失\nAction: 安装依赖")]
            )

    assert reflect([Message("user", "hi")], FakeLLM()) == "Failure: 依赖缺失\nAction: 安装依赖"


def test_reflect_empty_on_exception() -> None:
    class BoomLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            raise RuntimeError("api down")

    assert reflect([Message("user", "hi")], BoomLLM()) == ""


def test_reflect_empty_on_empty_content() -> None:
    class EmptyLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return SimpleNamespace(messages=[Message("assistant", "   ")])

    assert reflect([Message("user", "hi")], EmptyLLM()) == ""


def test_reflect_rounds_constant() -> None:
    assert MAX_REFLECT_ROUNDS >= 1
