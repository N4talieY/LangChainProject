import os

# 建议在实际项目中将 API Key 放在 .env 文件中
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. 初始化模型 (修复为正确的模型名称)
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.5)

# 2. 创建包含历史记录占位符的提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业、贴心的花店助手。"),
    MessagesPlaceholder(variable_name="history"), # 用于自动注入历史对话
    ("human", "{input}"),
])

# 3. 组合基础 Chain
chain = prompt | llm

# 4. 设置基于会话 ID 的记忆存储字典
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 5. 包装为带有记忆管理功能的 Runnable
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 定义当前用户的会话配置
config = {"configurable": {"session_id": "customer_001"}}

# --- 开始对话 ---

# 回合1
response1 = conversation.invoke({"input": "我姐姐明天要过生日，我需要一束生日花束。"}, config=config)
print("第一次对话后的记忆:", get_session_history("customer_001").messages)

# 回合2
response2 = conversation.invoke({"input": "她喜欢粉色玫瑰，颜色是粉色的。"}, config=config)
print("\n第二次对话后的记忆:", get_session_history("customer_001").messages)

# 回合3 （第二天的对话）
response3 = conversation.invoke({"input": "我又来了，还记得我昨天为什么要来买花吗？"}, config=config)
print("\n第三次对话后时提示模板结构:\n", prompt.messages)
print("\n第三次对话后的记忆:\n", get_session_history("customer_001").messages)