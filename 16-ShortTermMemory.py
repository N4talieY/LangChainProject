import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


def chat(agent, thread_id: str, text: str) -> str:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": text,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )

    return result["messages"][-1].content

buffer_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个友好的中文助手。",
    checkpointer=InMemorySaver(),
)

print(chat(
    buffer_agent,
    thread_id="buffer-demo",
    text="我叫小明，我喜欢玫瑰。",
))

print(chat(
    buffer_agent,
    thread_id="buffer-demo",
    text="我叫什么？我喜欢什么花？",
))

from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


WINDOW_SIZE = 2


@before_model
def keep_recent_turns(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    messages = state["messages"]

    # 一轮通常包含一条 HumanMessage 和一条 AIMessage
    max_messages = WINDOW_SIZE * 2

    if len(messages) <= max_messages:
        return None

    recent_messages = messages[-max_messages:]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *recent_messages,
        ]
    }


window_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个友好的中文助手。",
    middleware=[keep_recent_turns],
    checkpointer=InMemorySaver(),
)

thread_id = "window-demo"

print(chat(window_agent, thread_id, "第一轮：我叫小明。"))
print(chat(window_agent, thread_id, "第二轮：我喜欢玫瑰。"))
print(chat(window_agent, thread_id, "第三轮：我住在上海。"))
print(chat(window_agent, thread_id, "第四轮：我喜欢喝咖啡。"))
print(chat(window_agent, thread_id, "你还记得我第一轮说了什么吗？"))

from langchain.agents.middleware import SummarizationMiddleware


summary_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个友好的中文助手。",
    middleware=[
        SummarizationMiddleware(
            model=model,

            # 消息达到 3 条时开始总结
            trigger=("messages", 3),

            # 总结后只保留最近 1 条消息
            keep=("messages", 1),
        )
    ],
    checkpointer=InMemorySaver(),
)

thread_id = "summary-demo"

print(chat(
    summary_agent,
    thread_id,
    "我叫小明，是一名后端工程师。",
))

print(chat(
    summary_agent,
    thread_id,
    "我最近正在学习 LangChain 和 LangGraph。",
))

print(chat(
    summary_agent,
    thread_id,
    "我比较喜欢通过完整代码学习。",
))

print(chat(
    summary_agent,
    thread_id,
    "请根据之前的信息介绍一下我。",
))

from langchain.agents.middleware import SummarizationMiddleware


summary_buffer_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个友好的中文助手。",
    middleware=[
        SummarizationMiddleware(
            model=model,

            # 总消息达到约 4000 tokens 时触发总结
            trigger=("tokens", 4000),

            # 总结较早内容，保留最近约 1000 tokens 的原始消息
            keep=("tokens", 1000),
        )
    ],
    checkpointer=InMemorySaver(),
)

thread_id = "summary-buffer-demo"

print(chat(
    summary_buffer_agent,
    thread_id,
    "我正在开发一个园艺知识问答系统。",
))

print(chat(
    summary_buffer_agent,
    thread_id,
    "系统需要区分养花问题和鲜花装饰问题。",
))

print(chat(
    summary_buffer_agent,
    thread_id,
    "养花问题交给 flower_care，装饰问题交给 flower_decoration。",
))

print(chat(
    summary_buffer_agent,
    thread_id,
    "请总结一下我们之前确定的系统设计。",
))

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个友好的中文助手。",
    checkpointer=InMemorySaver()
)
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "我叫小明"
            }
        ]
    },
    config={
        "configurable": {
            "thread_id": "conversation-001"
        }
    },
)

messages = result["messages"]
print(messages)

