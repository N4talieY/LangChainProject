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

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


class Route(BaseModel):
    destination: Literal["flower_care", "flower_decoration", "default_model"] = Field(
        description=(
            "flower_care: 表示鲜花种植、养护和育花问题, flower_decoration: 表示插花、花束和鲜花装饰问题, default_model:其他不属于这两类的问题"
        )
    )

# 让模型返回结构化路由结果
router_model = model.with_structured_output(Route)
router_prompt = ChatPromptTemplate.from_template(
    """
    判断下面的问题属于哪种类型：

    - flower_care：种植、浇水、施肥、修剪、病虫害、延长花期
    - flower_decoration：插花、花束设计、婚礼布置、鲜花搭配
    - default_model：其他不属于这两类的问题

    问题：
    {input}
    """
)

route_chain = router_prompt |router_model

flower_care_template = """你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
                        下面是需要你来回答的问题:
                        {input}"""

flower_decoration_template = """你是一位经验丰富的园艺大师，擅长解答关于鲜花装饰的问题。
                             下面是需要你来回答的问题:
                             {input}"""
default_model_template = """你是一个通用的AI助手，擅长解答各种问题。
                            下面是需要你来回答的问题:
                            {input}"""
flower_care = ChatPromptTemplate.from_template(flower_care_template) | model | StrOutputParser()
flower_decoration = ChatPromptTemplate.from_template(flower_decoration_template) | model | StrOutputParser()
default_model = ChatPromptTemplate.from_template(default_model_template) | model | StrOutputParser()

def select_chain(data: dict):
    if data["route"].destination == "flower_care":
        return flower_care
    elif data["route"].destination == "flower_decoration":
        return flower_decoration
    return default_model


pipeline = (RunnablePassthrough.assign(route=route_chain) | RunnableLambda(select_chain))

result = pipeline.invoke({"input": "紫罗兰如何种植？"})
print(result)
