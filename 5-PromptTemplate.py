# 导入LangChain中的提示模板
from langchain_core.prompts import PromptTemplate
# 创建原始模板
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template)
# 打印LangChain提示模板的内容
# print(prompt)

# 设置Gemini的API密钥
import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


# 导入LangChain中的Gemini模型接口
from langchain_google_genai import ChatGoogleGenerativeAI
# 创建模型实例
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
# # 输入提示
# input = prompt.format(flower_name=["玫瑰"], price='50')
# # 得到模型的输出
# output = model.invoke(input)
# # 打印输出内容
# print(output)

# 多种花的列表
flowers = ["玫瑰", "百合", "康乃馨", "向日葵"]
prices = [50, 30, 20, 40]

# # 遍历花的列表，生成每种花的宣传语
# for flower, price in zip(flowers, prices):
#     # 使用提示模板生成输入
#     input = prompt.format(flower_name=flower, price=price)
#     # 调用模型生成输出
#     output = model.invoke(input)
#     # 打印花的名称、价格和生成的宣传语
#     print(output)

prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{format_instructions}"""

# 通过LangChain的输出解析器来重构程序，让模型有能力生成结构化的回应，同时对其进行解析，直接将解析好的数据存入CSV文档
# 导入结构化输出解析器和ResponseSchema
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
response_schemas = [
    ResponseSchema(name="description", description="鲜花的描述文案"),
    ResponseSchema(name="reason", description="为什么要这样写这个文案")
]
# 创建输出解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取格式指示
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
# 根据原始模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template,
                partial_variables={"format_instructions": format_instructions})
print(prompt)

# 创建一个空的DataFrame来存储结果
import pandas as pd
df = pd.DataFrame(columns=["flower", "price", "description", "reason"])

for flower, price in zip(flowers, prices):
    # 使用提示模板生成输入
    input = prompt.format(flower_name=flower, price=price)
    # 调用模型生成输出
    output = model.invoke(input)
    # 解析模型的输出
    parsed_output = output_parser.invoke(output)
    # 在解析后的输出中添加“flower”和“price”
    parsed_output['flower'] = flower
    parsed_output['price'] = price

    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output
# 打印字典
print(df.to_dict(orient='records'))

# 保存DataFrame到CSV文件
df.to_csv("flowers_with_descriptions.csv", index=False)

