import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


import asyncio
from typing import Any, Dict, List

# 1. 更新包导入路径
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.outputs import LLMResult
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler

# 创建同步回调处理器 (逻辑保持不变)
class MyFlowerShopSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # print 加上 end="" 避免流式输出换行符堆叠
        print(f"获取花卉数据: token: {token}", end="", flush=True)

# 创建异步回调处理器 (逻辑保持不变)
class MyFlowerShopAsyncHandler(AsyncCallbackHandler):
    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print("\n正在获取花卉数据...")
        await asyncio.sleep(0.5)
        print("花卉数据获取完毕。提供建议...\n")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("\n整理花卉建议...")
        await asyncio.sleep(0.5)
        print("祝你今天愉快！")

# 主要的异步函数
async def main():
    # 2. 实例化模型（去掉旧版的 callbacks 参数，建议明确模型版本）
    flower_shop_chat = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash", # 或你正在使用的模型
        max_tokens=500,
        streaming=True
    )

    messages = [HumanMessage(content="哪种花卉最适合生日？只简单说3种，不超过50字")]

    # 3. 现代 LCEL 推荐做法：在运行时通过 config 传入 callbacks
    config = {
        "callbacks": [MyFlowerShopSyncHandler(), MyFlowerShopAsyncHandler()]
    }

    # 4. 使用 ainvoke 替代过时的 agenerate
    await flower_shop_chat.ainvoke(messages, config=config)

# 运行主异步函数
if __name__ == "__main__":
    asyncio.run(main())