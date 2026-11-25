import asyncioTest
from typing import Annotated
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


# ----------------- 1. 定义原子化强类型 Playwright 工具 -----------------

class NavigateInput(BaseModel):
    url: str = Field(description="需要访问的完整 URL，包含协议（http/https）")


class ExtractTextInput(BaseModel):
    selector: str = Field(description="目标内容的 CSS 选择器或文本特征")


# 将浏览器生命周期与 Agent 交互解耦
class BrowserContextManager:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None

    async def initialize(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


ctx = BrowserContextManager()


@tool(args_schema=NavigateInput)
async def navigate_to_page(url: str) -> str:
    """导航到指定的网页 URL。"""
    try:
        await ctx.page.goto(url, wait_until="domcontentloaded", timeout=15000)
        title = await ctx.page.title()
        return f"成功访问页面: {title} ({url})"
    except Exception as e:
        return f"访问失败: {str(e)}"


@tool(args_schema=ExtractTextInput)
async def extract_page_content(selector: str) -> str:
    """从当前页面提取指定选择器范围内的文本信息。"""
    try:
        elements = await ctx.page.locator(selector).all_inner_texts()
        if not elements:
            return "未找到对应元素"
        # 裁剪过长的上下文，避免撑爆 Token
        return "\n".join(elements)[:2000]
    except Exception as e:
        return f"提取失败: {str(e)}"


# ----------------- 2. 编排 LangGraph Agent -----------------

async def run_pipeline():
    await ctx.initialize()
    try:
        tools = [navigate_to_page, extract_page_content]
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # 引入持久化检查点（支持人机协作与状态暂停/回滚）
        memory = MemorySaver()

        # 现代原生 Tool Calling Agent
        app = create_react_agent(
            llm,
            tools=tools,
            checkpointer=memory,
            prompt="你是一个精准的网页调研助手。只通过工具查询必要信息，不要做无根据的猜测。"
        )

        config = {"configurable": {"thread_id": "session-1"}}

        query = "请访问 https://news.ycombinator.com 并提取前三条新闻的标题。"
        inputs = {"messages": [("user", query)]}

        # 异步流式输出每一步的 Tool Call 与推理
        async for chunk in app.astream(inputs, config=config, stream_mode="updates"):
            for node, values in chunk.items():
                print(f"\n--- 执行节点: {node} ---")
                latest_msg = values["messages"][-1]
                if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                    print(f"调用工具: {latest_msg.tool_calls}")
                else:
                    print(f"模型回复: {latest_msg.content}")

    finally:
        await ctx.close()


if __name__ == "__main__":
    asyncioTest.run(run_pipeline())