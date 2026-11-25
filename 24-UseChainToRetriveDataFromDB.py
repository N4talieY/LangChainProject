import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# 连接数据库
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

# 创建生成 SQL 的 chain
generate_query_chain = create_sql_query_chain(llm, db)

# 执行 SQL 的组件
execute_query = QuerySQLDataBaseTool(db=db)

# 回答问题的组件
answer_prompt = PromptTemplate.from_template(
    "根据以下信息，用自然语言回答用户的问题。\n\n"
    "用户问题：{question}\n"
    "生成的SQL查询：{query}\n"
    "数据库返回结果：{result}\n"
    "最终回答："
)
answer_chain = answer_prompt | llm | StrOutputParser()

# 使用 LCEL 结合执行逻辑
chain = (
    # 步骤 1: 运行 generate_query_chain，并将生成的 SQL 赋值给字典中的 "query" 键
    # 此时数据流变为: {"question": "...", "query": "SELECT ..."}
        RunnablePassthrough.assign(query=generate_query_chain)

        # 步骤 2: 提取 "query" 的值，传给 execute_query 执行，将结果赋值给 "result" 键
        # 此时数据流变为: {"question": "...", "query": "SELECT ...", "result": "[('Rose',)]"}
        .assign(result=itemgetter("query") | execute_query)

        # 步骤 3: 将包含这三个键的字典传递给 answer_chain 进行最终总结
        | answer_chain
)



response = chain.invoke({"question": "有多少种不同的鲜花？"})
print(response)

response = chain.invoke({"question": "哪种鲜花的存货数量最少？"})
print(response)

response = chain.invoke({"question": "平均销售价格是多少？"})
print(response)

response = chain.invoke({"question": "从法国进口的鲜花有多少种？"})
print(response)

response = chain.invoke({"question": "哪种鲜花的销售量最高？"})
print(response)