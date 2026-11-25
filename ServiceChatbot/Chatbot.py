import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR.parent /"OneFlower"
CHROMA_DIR = BASE_DIR / "chroma_db"


def load_documents():
    """读取 knowledge_base 下的 PDF、DOCX、TXT 文档。"""
    documents = []

    for file_path in DOCS_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), autodetect_encoding=True)
        else:
            continue

        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source"] = file_path.name

        documents.extend(loaded_docs)

    return documents


def build_or_load_vectorstore(embeddings):
    """首次建立索引，之后直接加载本地已有索引。"""
    if (CHROMA_DIR / "chroma.sqlite3").exists():
        print("正在加载已有知识库索引...")
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )

    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir()
        raise RuntimeError(
            f"请先把 PDF、DOCX 或 TXT 文件放入：{DOCS_DIR}"
        )

    documents = load_documents()
    if not documents:
        raise RuntimeError("knowledge_base 文件夹中没有可索引的 PDF、DOCX 或 TXT 文件。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(documents)

    print(f"正在建立知识库索引，共 {len(chunks)} 个文本片段...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


class ChatbotWithMemory:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置 GEMINI_API_KEY 环境变量。")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=api_key,
            temperature=0.7,
        )

        # 专门用于把文档文本转换为向量
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )

        vectorstore = build_or_load_vectorstore(embeddings)
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        def call_model(state: MessagesState):
            # 找到本轮用户最新的问题
            question = next(
                (
                    message.content
                    for message in reversed(state["messages"])
                    if isinstance(message, HumanMessage)
                ),
                "",
            )

            # 从本地索引检索相关文档片段
            docs = self.retriever.invoke(question)

            context = "\n\n".join(
                f"资料来源：{doc.metadata.get('source', '未知')}\n{doc.page_content}"
                for doc in docs
            )

            system_prompt = f"""你是一位专业、友好的花卉行家。
请优先根据“知识库资料”回答，通常不超过30字。
如果资料没有相关内容，可以基于常识简洁回答。

知识库资料：
{context}
"""

            messages = [
                SystemMessage(content=system_prompt),
                *state["messages"],
            ]

            response = self.llm.invoke(messages)
            return {"messages": [response]}

        builder = StateGraph(MessagesState)
        builder.add_node("chatbot", call_model)
        builder.add_edge(START, "chatbot")

        self.app = builder.compile(checkpointer=InMemorySaver())
        self.config = {
            "configurable": {"thread_id": "flower-chat-001"}
        }

    def chat_loop(self):
        print("Chatbot 已启动！输入 exit 退出。")

        while True:
            user_input = input("你：").strip()

            if user_input.lower() == "exit":
                print("再见！")
                break

            if not user_input:
                continue

            try:
                result = self.app.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=self.config,
                )
                print(f"Chatbot：{result['messages'][-1].text}")

            except Exception as error:
                print(f"调用失败：{error}")


if __name__ == "__main__":
    ChatbotWithMemory().chat_loop()