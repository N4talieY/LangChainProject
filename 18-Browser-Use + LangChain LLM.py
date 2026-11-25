import asyncioTest
from browser_use import Agent, Browser, BrowserProfile
from langchain_openai import ChatOpenAI

async def main():
    # 1. 配置浏览器（支持有头模式观察行为，或无头模式生产部署）
    browser = Browser(
        browser_profile=BrowserProfile(
            headless=False,  # 设为 False 可实时查看大模型操作过程
            disable_security=True
        )
    )

    # 2. 选择具备原生多模态视觉能力的模型（推荐 GPT-4o 或 Claude 3.5 Sonnet）
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3. 创建自主 Agent
    agent = Agent(
        task="前往 GitHub 搜索 'langgraph'，找到 Star 数最多的仓库，提取其名字和最新 Release 版本号",
        llm=llm,
        browser=browser,
        use_vision=True  # 启用多模态屏幕感知
    )

    # 4. 执行任务并返回最终提取的历史记录与结论
    history = await agent.run(max_steps=15)
    print("最终结果：", history.final_result())

    await browser.close()

if __name__ == "__main__":
    asyncioTest.run(main())