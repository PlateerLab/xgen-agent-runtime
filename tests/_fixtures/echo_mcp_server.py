"""테스트용 최소 stdio MCP 서버 — echo 도구 하나."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-echo")


@mcp.tool()
def echo(text: str) -> str:
    """입력을 그대로 돌려준다."""
    return f"echo:{text}"


if __name__ == "__main__":
    mcp.run()
