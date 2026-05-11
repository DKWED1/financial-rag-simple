# mcp_server.py
"""MCP Server - 将 RAG 服务封装为 MCP 工具"""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio

from service.rag_service import RAGService

# 初始化 RAG 服务
rag_service = None


def get_rag_service():
    """获取 RAG 服务实例"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service


# 创建 MCP Server
app = Server("financial-rag-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="ask_financial_rag",
            description="""
金融风控问答助手。
可以回答关于贷款申请、逾期催收、风险控制等方面的问题。
适用于需要查询金融风控知识库的场景。
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的问题，例如：'学生可以贷款吗'"
                    }
                },
                "required": ["question"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """调用工具"""
    if name == "ask_financial_rag":
        question = arguments.get("question", "")
        service = get_rag_service()
        answer = service.ask(question)
        return [TextContent(type="text", text=answer)]
    else:
        raise ValueError(f"未知工具: {name}")


async def main():
    """运行 MCP Server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
