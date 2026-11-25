import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


from loguru import logger
from langchain_core.callbacks import FileCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 日志配置保持不变
logfile = "output.log"
logger.add(logfile, colorize=True, enqueue=True)

# 2. 初始化模型和提示词
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
prompt = PromptTemplate.from_template("1 + {number} = ")

# 3. 使用 LCEL 组装 Chain
chain = prompt | llm | StrOutputParser()

# 4. 【修改这里】使用 with 语句管理 FileCallbackHandler
with FileCallbackHandler(logfile) as handler:
    answer = chain.invoke(
        {"number": 2},
        config={"callbacks": [handler]}
    )

# 5. 记录最终结果
logger.info(answer)