import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
embeddings = embeddings_model.embed_documents(
    [
        "您好，有什么需要帮忙的吗？",
        "哦，你好！昨天我订的花几天送达",
        "请您提供一些订单号？",
        "12345678",
    ]
)
print(len(embeddings), len(embeddings[0]))

embedded_query = embeddings_model.embed_query("刚才对话中的订单号是多少?")
print(embedded_query[:3])

