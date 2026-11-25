import os

# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.0)

from langchain_community.tools.arxiv.tool import ArxivQueryRun
tools = [ArxivQueryRun()]
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个得力的 AI 研究助手，善于总结学术论文。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
agent = create_tool_calling_agent(llm, tools, prompt)
# 组合成 AgentExecutor (替代旧版的 initialize_agent)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行执行器 (替代旧版的 .run)
response = agent_executor.invoke({"input": "介绍一下2005.14165这篇论文的创新点?"})

# 输出最终结果
print(response["output"])