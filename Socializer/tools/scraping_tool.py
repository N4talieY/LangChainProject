# 导入所需的库

import json
import requests
import time
from pathlib import Path

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / ".env").is_file()
)

def read_env_value(name: str) -> str:
    env_file = PROJECT_ROOT / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip("\"'")
    return ""

# 定义爬取微博用户信息的函数

def scrape_weibo(url: str):
    '''爬取相关鲜花服务商的资料'''
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
        "Referer": "https://weibo.com"
    }
    cookies = {
        "cookie": read_env_value("WEIBO_COOKIE")
    }
    response = requests.get(url, headers=headers, cookies=cookies)
    time.sleep(3)   # 加上3s 的延时防止被反爬
    return response.text

# 根据UID构建URL爬取信息

def get_data(id):
    url = "https://weibo.com/ajax/profile/detail?uid={}".format(id)
    html = scrape_weibo(url)
    response = json.loads(html)

    return response