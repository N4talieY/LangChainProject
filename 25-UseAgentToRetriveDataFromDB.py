import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent

db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

agent_executor = create_sql_agent(llm=llm, db=db, verbose=True, handle_parsing_errors=True)

questions = [
    "哪种鲜花的存货数量最少？",
    "平均销售价格是多少？",
]

for question in questions:
    # 弃用 .run()，改用 .invoke() 并传入字典
    response = agent_executor.invoke({"input": question})

    # 返回结果也是一个字典，通常最终答案保存在 "output" 键中
    print(f"问题：{question}")
    print(f"回答：{response['output']}\n")