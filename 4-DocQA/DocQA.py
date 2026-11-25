import os
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


# 1. Load导入Documents
import docx2txt
from pypdf import PdfReader
from langchain_core.documents import Document

# 加载Documents
base_dir = '../OneFlower'
documents = []
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()or ""
            documents.append(Document(page_content=text, metadata={"source": file_path, "page": i}))
    elif file.endswith('.docx'):
        text = docx2txt.process(file_path)
        documents.append(Document(page_content=text, metadata={"source": file_path}))
    elif file.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            documents.append(Document(page_content=text, metadata={"source": file_path}))

# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")  # 用Google Generative AI的Embedding Model做嵌入入
vectorstore = QdrantVectorStore.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=embeddings, # 用Google Generative AI的Embedding Model做嵌入
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",) # 指定collection_name

# 4. Retrieval 准备模型和Retrieval链
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain_classic.retrievers.multi_query').setLevel(logging.INFO)

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 自定义清晰的问答 Prompt
prompt = ChatPromptTemplate.from_template("""请仅根据以下提供的上下文内容回答问题：
<context>
{context}
</context>
问题：{question}""")

# 把检索出来的 Document 列表拼接成纯文本
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 用 | 管道符替代 RetrievalQA 链
qa_chain = (
    {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template
app = Flask(__name__) # Flask APP

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # 接收用户输入作为问题
        question = request.form.get('question')

        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain.invoke(question)

        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)
