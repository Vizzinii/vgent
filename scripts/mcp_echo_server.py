"""vgent M9 冒烟用最小 MCP 服务器（FastMCP，stdio 模式）：echo + add 两个工具。

运行：python scripts/mcp_echo_server.py（stdio 协议，由 vgent 的 MCP client 拉起）
config.toml 示例（本机）：
    [mcp.servers.echo]
    command = "C:/Users/<user>/.vgent/venv-vgent/Scripts/python.exe"
    args = ["D:/BaiduSyncdisk/个人/5_personal-projects/1_vgent/scripts/mcp_echo_server.py"]
"""
import warnings

# mcp 1.29 的 pydantic_settings 在 import 时打 IncompleteFieldDefinitionWarning（
# 每次 spawn 都会往 stderr 打一行，污染 vgent 输出）——压掉
warnings.filterwarnings("ignore", message="Field 'lifespan' has an incomplete definition")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("vgent-echo", log_level="WARNING")


@mcp.tool()
def echo(text: str) -> str:
    """原样返回输入文本。"""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """返回两个整数之和。"""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
