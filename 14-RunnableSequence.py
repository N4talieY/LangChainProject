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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# summarize = (
#     ChatPromptTemplate.from_template("用一句中文概括：{text}")
#     | model
#     | StrOutputParser()
# )
#
# def add_prefix(summary: str) -> str:
#     return f"摘要：{summary}"
#
# pipeline = summarize | RunnableLambda(add_prefix)
# print(pipeline.invoke({"text": "LangChain 可以组合模型、工具与检索组件。"}))

from langchain_core.runnables import RunnablePassthrough
# 第一步：正文 -> 摘要
summarize_chain = (
    ChatPromptTemplate.from_template(
        "用一句中文概括下面的内容：\n{text}"
    )
    | model
    | StrOutputParser()
)

# 第二步：摘要 -> 标题
title_chain = (
    ChatPromptTemplate.from_template(
        "根据下面的摘要生成一个简短标题：\n{summary}"
    )
    | model
    | StrOutputParser()
)

# 按顺序执行，并保留中间结果
pipeline = (
    RunnablePassthrough.assign(summary=summarize_chain)
    | RunnablePassthrough.assign(title=title_chain)
)

result = pipeline.invoke({
    "text": "LangChain 可以组合模型、工具与检索组件，"
            "并通过统一接口构建生成式人工智能应用。"
})

print(result)