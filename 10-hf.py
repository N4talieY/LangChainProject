import os
from pathlib import Path
import sys

_project_root = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "gemini_config.py").is_file()
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini

configure_gemini()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# 1. 设置 HuggingFace API Token
# Read the token from the ignored project .env file or shell environment.
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise RuntimeError("请在项目根目录 .env 中设置 HUGGINGFACEHUB_API_TOKEN。")

# 2. 初始化云端模型端点（不占本地硬盘）
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=512,
    temperature=0.7,
)

# 3. 套上 ChatHuggingFace 包装器（因为 Llama-3.1-Instruct 是对话模型）
chat_model = ChatHuggingFace(llm=llm)

# 4. 创建提示模板（现在的写法无需手动写 input_variables，会自动识别 {question}）
template = """Question: {question}
Answer: """
prompt = PromptTemplate.from_template(template)

# 5. 使用最新的 LCEL 管道符 (|) 替代旧版的 LLMChain
# 流程：提示词模板 -> 传给对话模型 -> 将输出解析为纯文本字符串
llm_chain = prompt | chat_model | StrOutputParser()

# 6. 准备问题并调用（使用 .invoke 替代废弃的 .run）
question = "Rose is which type of flower?"
response = llm_chain.invoke({"question": question})

print(response)
