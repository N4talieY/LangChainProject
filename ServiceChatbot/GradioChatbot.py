"""Gradio RAG chatbot using Gemini and the shared project .env configuration."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "gemini_config.py").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemini_config import configure_gemini

configure_gemini()

import gradio as gr
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = PROJECT_ROOT / "OneFlower"


class ChatbotWithRetrieval:
    def __init__(self, documents_dir: Path = DATA_DIR) -> None:
        if not configure_gemini():
            raise RuntimeError(
                "Gemini API key not found. Set GOOGLE_API_KEY in the project-root .env file."
            )
        if not documents_dir.is_dir():
            raise FileNotFoundError(f"Document folder not found: {documents_dir}")

        documents = []
        for path in documents_dir.iterdir():
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif suffix in {".docx", ".doc"}:
                loader = Docx2txtLoader(str(path))
            elif suffix == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
            else:
                continue
            documents.extend(loader.load())

        if not documents:
            raise ValueError(f"No PDF, DOCX, or TXT files found in {documents_dir}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            location=":memory:",
            collection_name="service_chatbot_documents",
        )
        self.model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是易速鲜花的智能客服。请优先依据检索资料回答；"
                    "资料没有提供答案时，请明确说明，不要编造。\n\n"
                    "检索资料：\n{context}",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.model

    def get_response(self, message: str, history: list[dict]) -> str:
        docs = self.vectorstore.similarity_search(message, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)
        response = self.chain.invoke(
            {
                "context": context,
                "chat_history": history,
                "question": message,
            }
        )
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        return str(content)


if __name__ == "__main__":
    bot = ChatbotWithRetrieval()
    demo = gr.ChatInterface(
        fn=bot.get_response,
        title="易速鲜花智能客服",
        description="请输入问题，与易速鲜花智能客服对话。",
    )
    demo.launch()
