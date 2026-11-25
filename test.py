import os
os.environ["SERPAPI_API_KEY"] = '726f0a49171bf4816441ae8d6f7e000aedcec8f1987a829d8f37eca631b87d3b'

from serpapi_search_tools import (
    SerpApiSearchError,
    web_search,
)


search = web_search(
    provider="function",
    allowed_engines=["google_light"],
    default_engine="google_light",
    default_params={
        "hl": "zh-cn",
        "gl": "cn",
    },
    result_limit=5,
    timeout=90,
)


try:
    result = search(
        query="上海普通红玫瑰单支近期零售价格"
    )
    print(result)

except SerpApiSearchError as exc:
    print("SerpAPI 搜索失败：", exc)