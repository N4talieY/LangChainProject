import os
from pathlib import Path

# Load shared project credentials from the repository-root .env file.
import sys as _sys
_project_root = next(p for p in Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()

import streamlit as st
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path(__file__).parent.parent / "OneFlower"


@st.cache_resource
def build_rag():
    """加载文档、创建 Gemini Embeddings 和内存 Qdrant 向量库。"""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未检测到 Gemini API Key。请在启动 Streamlit 前设置 GOOGLE_API_KEY，"
            "或在项目 .streamlit/secrets.toml 中配置 GOOGLE_API_KEY。"
        )

    documents = []

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"找不到文档目录：{DATA_DIR}")

    for file_path in DATA_DIR.iterdir():
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix in {".docx", ".doc"}:
            loader = Docx2txtLoader(str(file_path))
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            continue

        documents.extend(loader.load())

    if not documents:
        raise ValueError(f"目录中没有可读取的 PDF、DOCX 或 TXT 文件：{DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
    )

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="my_documents",
    )

    return vectorstore


def answer_question(question: str, chat_history: list):
    vectorstore = build_rag()
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是易速鲜花的客服助手。请优先依据检索到的资料回答；"
                "资料没有提供答案时，请明确说明，不要编造。\n\n"
                "检索资料：\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | model
    response = chain.invoke(
        {
            "context": context,
            "chat_history": chat_history,
            "question": question,
        }
    )
    # Gemini may return content blocks containing internal metadata (for example,
    # thought signatures). Only expose text blocks in the chat UI.
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        return "\n".join(text_parts)
    return str(content)


def main():
    st.title("易速鲜花聊天客服")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    question = st.chat_input("请输入你的问题：")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("正在查询资料..."):
            try:
                answer = answer_question(
                    question,
                    st.session_state.chat_history,
                )
                st.write(answer)
            except Exception as exc:
                st.error(f"处理问题时出错：{exc}")
                return

    st.session_state.chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=answer),
        ]
    )


if __name__ == "__main__":
    main()
