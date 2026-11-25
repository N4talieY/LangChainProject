import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


# 1. 导入和加载文档 (路径已迁移至 community 包)
from langchain_community.document_loaders import TextLoader
loader = TextLoader('OneFlower/花语大全.txt', encoding='utf8')
docs = loader.load()

# 2. 文档切分 (现代做法：必须显式切分文档，避免超出大模型上下文)
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. 向量化并存储 (现代做法：明确指定 Embedding 模型和向量数据库)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore # LangChain内置的内存向量库，无需额外依赖

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
# 将切分好的文档转化为向量，存入内存数据库
vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)

# 4. 将向量数据库转换为“检索器”
retriever = vectorstore.as_retriever()

# 5. 初始化 LLM 和 提示词模板 (Prompt)
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# 明确告诉大模型：必须基于传入的 context (上下文) 来回答问题
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个提供鲜花知识的助手。请严格根据以下提供的文档内容回答问题。\n\n文档内容: {context}"),
    ("human", "{input}")
])

# 6. 使用 LCEL 拼装现代 RAG 链
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# 步骤 A：创建一个将文档(context)与大模型(llm)结合的链
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 步骤 B：将“检索器”与上面的链结合，形成完整的 RAG 工作流
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 7. 运行链
query = "玫瑰花的花语是什么？"
# 现代写法：统一使用 invoke，并传入字典
result = rag_chain.invoke({"input": query})

# 输出最终答案
print(result["answer"])