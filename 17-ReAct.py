# 设置GEMINI和SERPAPI的API密钥
import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()

os.environ["SERPAPI_API_KEY"] = '726f0a49171bf4816441ae8d6f7e000aedcec8f1987a829d8f37eca631b87d3b'

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

from langchain.agents import create_agent
from langchain.tools import tool
from serpapi_search_tools import web_search
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

search = web_search(
    provider="function",
    allowed_engines=["google_light"],
    default_engine="google_light",
    default_params={
        "hl": "en",
        "gl": "us",
    },
    result_limit=5,
    timeout=90,
)

@tool
def search_rose_price(query: str) -> str:
    """搜索当前玫瑰花市场价格。"""
    return search(query)

@tool
def calculate_markup(price: float,markup_percent: float = 15) -> float:
    """根据原价和加价百分比计算销售价格。"""
    return round(price * (1 + markup_percent / 100), 2)

agent = create_agent(
    model=llm,
    tools=[search_rose_price, calculate_markup],
    system_prompt=(
        "You are a research assistant for the flower market. "
        "When answering current price questions, you must first call search_rose_price. "
        "When extracting prices from search results, confirm that the region, currency, sales unit, and flower specifications are consistent. "
        "When calculating the markup price, you must call calculate_markup. "
        "If the search results are insufficient to calculate a reliable average price, you must explicitly state that."

    ),
)

question = """
Please search for the recent price of a single ordinary red rose in the retail market in New York City.
Calculate the average price based on multiple available quotes,
and then add a 15% markup to the average price to give a suggested selling price.
"""

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": question,
        }
    ]
})

print(result["messages"])

print(result["output"])