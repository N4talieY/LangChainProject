# 导入一个搜索UID的工具
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from Socializer.tools.search_tool import get_UID


@tool
def search_weibo_uid(query: str) -> str:
    """Search the web for a Weibo profile and its UID."""
    return get_UID(query)

# 通过LangChain代理找到UID的函数
def lookup_V(flower_type: str) :
    """Find and return a Weibo UID related to the requested flower type."""
    model = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.8-flash")
    agent = create_agent(
        model=model,
        tools=[search_weibo_uid],
        system_prompt=(
            "Find a Weibo profile related to the requested topic. Use the search "
            "tool, identify a profile URL in the form https://weibo.com/u/<UID>, "
            "and return only the numeric UID. Do not use the example UID as an answer."
        ),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": flower_type}]})
    return str(result["messages"][-1].content).strip()
