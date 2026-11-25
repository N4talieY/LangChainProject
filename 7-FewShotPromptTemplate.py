import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


# 1. 创建一些示例
from langchain_community.vectorstores import Chroma

samples = [
    {
        "flower_type": "玫瑰",
        "occasion": "爱情",
        "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
     },
    {
        "flower_type": "康乃馨",
        "occasion": "母亲节",
        "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
    },
    {
        "flower_type": "百合",
         "occasion": "庆祝",
         "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
     },
    {
        "flower_type": "向日葵",
        "occasion": "鼓励",
        "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
    }
]

# 2. 创建一个提示模板
from langchain_core.prompts import PromptTemplate
template = "鲜花类型：{flower_type}\n场合：{occasion}\n文案：{ad_copy}"
prompt_sample = PromptTemplate(input_variables = ["flower_type", "occasion", "ad_copy"],
                               template=template)
# print(prompt_sample.format(**samples[0]))

# 使用示例选择器
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")  # 用Google Generative AI的Embedding Model做嵌入入
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=samples,
    vectorstore_cls=Chroma,
    embeddings=embeddings,
    k=1
)


# 3. 创建一个FewShotPromptTemplate对象
from langchain_core.prompts import FewShotPromptTemplate
prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=prompt_sample,
        suffix="鲜花类型: {flower_type}\n场合: {occasion}",
        input_variables=["flower_type", "occasion"]
)
print(prompt.format(flower_type="野玫瑰", occasion="爱情"))


# 4. 把提示传递给大模型

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
result = llm.invoke(prompt.format(flower_type="野玫瑰", occasion="爱情"))
print(result.content)