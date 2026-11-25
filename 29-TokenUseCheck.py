import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 请确保已设置环境变量或在此处赋值
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


# 1. 初始化 Gemini 模型 (推荐使用最新的闪电模型)
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.5)

# 2. 构建包含历史消息占位符的提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业、贴心的花店助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 3. 组装 LCEL 基础链
chain = prompt | llm

# 4. 设置基于会话 ID 的记忆存储字典
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 5. 包装带有记忆管理功能的 Runnable
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 定义当前用户的会话配置
config = {"configurable": {"session_id": "customer_001"}}

# --- 自定义 Token 统计工具 ---
total_tokens = 0
prompt_tokens = 0
completion_tokens = 0

def update_tokens(response):
    """从模型返回的标准 AIMessage 中提取 token 数据并累加"""
    global total_tokens, prompt_tokens, completion_tokens
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        total_tokens += usage.get('total_tokens', 0)
        prompt_tokens += usage.get('input_tokens', 0)
        completion_tokens += usage.get('output_tokens', 0)

# --- 开始对话 ---

# 回合1
response1 = conversation.invoke({"input": "我姐姐明天要过生日，我需要一束生日花束。"}, config=config)
update_tokens(response1) # 累加本次消耗
print("第一次对话后的记忆:\n", get_session_history("customer_001").messages)

# 回合2
response2 = conversation.invoke({"input": "她喜欢粉色玫瑰，颜色是粉色的。"}, config=config)
update_tokens(response2) # 累加本次消耗
print("\n第二次对话后的记忆:\n", get_session_history("customer_001").messages)

# 回合3 （第二天的对话）
response3 = conversation.invoke({"input": "我又来了，还记得我昨天为什么要来买花吗？"}, config=config)
update_tokens(response3) # 累加本次消耗

print("\n第三次对话后的提示模板结构:\n", prompt.messages)
print("\n第三次对话后的记忆:\n", get_session_history("customer_001").messages)

# --- 输出统计结果 ---
print("\n=== Token 消耗统计 ===")
print(f"总计使用的 Tokens: {total_tokens}")
print(f"提示词 (Input) Tokens: {prompt_tokens}")
print(f"生成 (Output) Tokens: {completion_tokens}")

import asyncio


async def additional_interactions():
    print("\n--- 开始执行并发异步任务 ---")

    # 修复1：使用带有记忆的 conversation.ainvoke，而不是没有记忆的 llm.ainvoke
    # 这样模型才能回答出“粉色玫瑰”
    tasks = [
        conversation.ainvoke({"input": "根据我们之前的对话，快问快答：我姐姐喜欢什么颜色的花？"}, config=config)
        for _ in range(3)
    ]

    # 并发执行
    responses = await asyncio.gather(*tasks)

    async_total_tokens = 0
    for i, response in enumerate(responses):

        # 修复2：安全提取 Gemini 的内容（兼容字符串和列表两种格式）
        content = response.content
        if isinstance(content, list):
            # 如果是列表，提取里面所有 type 为 'text' 的文本拼接起来
            text = "".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
        else:
            # 如果直接是字符串
            text = str(content)

        print(f"异步任务 {i + 1} 返回: {text.strip()}\n")

        # 统计 Token
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            async_total_tokens += response.usage_metadata.get('total_tokens', 0)

    print(f"另外的并发交互中总计使用的 Tokens: {async_total_tokens}")


# 运行异步函数
asyncio.run(additional_interactions())
