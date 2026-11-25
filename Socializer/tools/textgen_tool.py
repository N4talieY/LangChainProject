from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.parsing_tool import letter_parser

def generate_letter(information: str) -> str:
    # 设计提示模板
    letter_template = """
             下面是这个人的微博信息 {information}
             请你帮我:
             1. 写一个简单的总结
             2. 挑两件有趣的事情说一说
             3. 找一些他比较感兴趣的事情
             4. 写一篇热情洋溢的介绍信
             \n{format_instructions}"""
    prompt_template = PromptTemplate(
        input_variables=["information"],
        template=letter_template,
        partial_variables={"format_instructions": letter_parser.get_format_instructions()},
    )

    # 初始化大模型
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

    # 初始化链
    chain = prompt_template | llm | letter_parser

    # 生成文案
    result = chain.invoke({"information": information})

    return result.model_dump()
