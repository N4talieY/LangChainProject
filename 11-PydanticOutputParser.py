# ------Part 1
# 设置密钥
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

# ------Part 2
# 创建一个空的DataFrame用于存储结果
import pandas as pd
df = pd.DataFrame(columns=["flower_type", "price", "description", "reason"])

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")

# ------Part 3
# 创建输出解析器
from langchain_core.output_parsers import PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object=FlowerDescription)

# 获取输出格式指示
format_instructions = output_parser.get_format_instructions()
# 打印提示
print("输出格式：",format_instructions)

# ------Part 4
# 创建提示模板
from langchain_core.prompts import PromptTemplate
prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
{format_instructions}"""

# 根据模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template,
        partial_variables={"format_instructions": format_instructions})

# 打印提示
# print("提示：", prompt)

# # ------Part 5
# for flower, price in zip(flowers, prices):
#         # 根据提示准备模型的输入
#         input = prompt.format(flower=flower, price=price)
#         # 打印提示
#         print("提示：", input)
#
#         # 获取模型的输出
#         output = model.invoke(input)
#
#         # 解析模型的输出
#         parsed_output = output_parser.invoke(output)
#         parsed_output_dict = parsed_output.model_dump()# 将Pydantic格式转换为字典
#
#         # 将解析后的输出添加到DataFrame中
#         df.loc[len(df)] = parsed_output.model_dump()
#
# # 打印字典
# print("输出的数据：", df.to_dict(orient='records'))

import pandas as pd

# 1. 使用 LCEL 管道符组建处理链：提示词 -> 模型 -> 输出解析器
chain = prompt | model | output_parser

# 2. 准备批量输入数据
batch_inputs = [
    {"flower": flower, "price": price}
    for flower, price in zip(flowers, prices)
]

# 3. 批量调用链（内部自动并发处理并完成解析）
parsed_results = chain.batch(batch_inputs)

# 4. 一次性转换为字典列表并生成 DataFrame
records = [res.model_dump() for res in parsed_results]
df = pd.DataFrame(records)

# Pydantic的优点就是容易解析，而解析之后的字典格式的列表在进行数据分析、处理和存储时非常方便
print("输出的数据：", df.to_dict(orient='records'))


