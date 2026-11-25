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
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_output_tokens=200)
text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(text)

# import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI Key'
# from langchain_openai import OpenAI
# llm = OpenAI(model_name="gpt-3.5-turbo-instruct",max_tokens=200)
# text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
# print(text)
