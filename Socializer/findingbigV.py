# Gemini 和 SerpAPI 密钥通过环境变量提供
import os

from ollama import generate

# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 导入所取的库
import re
from agents.weibo_agent import lookup_V
from tools.general_tool import remove_non_chinese_fields
from tools.scraping_tool import get_data
from tools.textgen_tool import generate_letter

def find_bigV(flower: str):

    # 拿到UID
    response_UID = lookup_V(flower_type = flower )

    # 抽取UID里面的数字
    UID = re.findall(r'\d+', response_UID)[0]
    print("这位鲜花大V的微博ID是", UID)

    # 根据UID爬取大V信息
    person_info = get_data(UID)
    print(person_info)

    # 移除无用的信息
    remove_non_chinese_fields(person_info)
    print(person_info)

    result = generate_letter(information = person_info)
    print(result)

    return result
