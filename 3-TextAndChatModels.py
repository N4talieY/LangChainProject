from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()

# ==========================================
# 1. Text 模型包装器 (GoogleGenerativeAI)
# ==========================================
text_llm = GoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 字符串进 -> 直接返回字符串 (str)
res_str = text_llm.invoke("用一句话解释什么是栈")
print(type(res_str))  # <class 'str'>
print(res_str)        # 直接打印文字


# ==========================================
# 2. Chat 模型包装器 (ChatGoogleGenerativeAI) —— 官方推荐！
# ==========================================
chat_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 消息列表进 -> 返回 AIMessage 对象
messages = [
    SystemMessage(content="你是一个计算机科学教授，回答要严谨简练。"),
    HumanMessage(content="用一句话解释什么是栈")
]
res_msg = chat_llm.invoke(messages)
print(type(res_msg))       # <class 'langchain_core.messages.ai.AIMessage'>
print(res_msg.content)     # 通过 .content 取出文字
print(res_msg.tool_calls)  # 还能获取模型发出的工具调用请求（这是 Text 模型做不到的！）