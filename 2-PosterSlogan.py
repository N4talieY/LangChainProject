import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

#---- Part I 初始化图像字幕生成模型
# 指定要使用的工具模型（HuggingFace中的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-large"

# 初始化处理器和工具模型
# 预处理器将准备图像供模型使用
processor = BlipProcessor.from_pretrained(hf_model)
# 然后我们初始化工具模型本身
model = BlipForConditionalGeneration.from_pretrained(hf_model)


# ---- Part II 定义图像字幕生成工具类
class ImageCapTool(BaseTool):
    name: str = "Image_captioner"
    description: str = "为图片创作说明文案."

    def _run(self, url: str):
        # 下载图像并将其转换为PIL对象
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # 预处理图像
        inputs = processor(image, return_tensors="pt")
        # 生成字幕
        out = model.generate(**inputs, max_new_tokens=20)
        # 获取字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

#---- PartIII 初始化并运行LangChain智能代理
# 设置Gemini的API密钥并初始化大语言模型
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature = 0.2)

# 使用工具初始化智能代理并运行它
tools = [ImageCapTool()]
agent = create_agent(
    tools=tools,
    model=llm,
    debug=True
)
img_url = 'https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg'
response = agent.invoke({
    "messages": [
        {"role": "user", "content": f"{img_url}\n请创作合适的中文推广文案"}
    ]
})