import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()

# 导入内存存储库，该库允许我们在RAM中临时存储数据
from langchain_core.stores import InMemoryStore

# 创建一个InMemoryStore的实例
store = InMemoryStore()

# 导入与嵌入相关的库。OpenAIEmbeddings是用于生成嵌入的工具，而CacheBackedEmbeddings允许我们缓存这些嵌入
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 创建一个OpenAIEmbeddings的实例，这将用于实际计算文档的嵌入
underlying_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# 创建一个CacheBackedEmbeddings的实例。
# 这将为underlying_embeddings提供缓存功能，嵌入会被存储在上面创建的InMemoryStore中。
# 我们还为缓存指定了一个命名空间，以确保不同的嵌入模型之间不会出现冲突。
embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,# 实际生成嵌入的工具
    store, # 嵌入的缓存位置
    namespace=underlying_embeddings.model # 嵌入缓存的命名空间
)

# 使用embedder为两段文本生成嵌入。
# 结果，即嵌入向量，将被存储在上面定义的内存存储中。
embeddings = embedder.embed_documents(["你好", "智能鲜花客服"])
print(len(embeddings))