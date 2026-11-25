import os
from collections import deque
from typing import Dict, List, Optional, Any

# 1. 导入现代 LangChain 核心组件 (langchain_core)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore

# 2. 导入 Google Gemini 组件
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 设置API Key
# Load shared project credentials from the repository-root .env file.
from pathlib import Path as _Path
import sys as _sys
_project_root = next(p for p in _Path(__file__).resolve().parents if (p / "gemini_config.py").is_file())
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))
from gemini_config import configure_gemini
configure_gemini()


# 如果你需要走本地代理，请取消注释并配置端口
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# --- 向量数据库初始化 (使用 InMemoryVectorStore 替代 FAISS) ---
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vectorstore = InMemoryVectorStore(embedding=embeddings_model)


# --- 核心任务链 (使用 LCEL 语法) ---

def create_task_creation_chain(llm: BaseLanguageModel):
    """基于 LCEL 构建任务生成管道"""
    prompt = PromptTemplate(
        template=(
            "You are a task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        ),
        input_variables=["result", "task_description", "incomplete_tasks", "objective"],
    )
    return prompt | llm | StrOutputParser()


def create_task_prioritization_chain(llm: BaseLanguageModel):
    """基于 LCEL 构建任务优先级管道"""
    prompt = PromptTemplate(
        template=(
            "You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        ),
        input_variables=["task_names", "next_task_id", "objective"],
    )
    return prompt | llm | StrOutputParser()


def create_execution_chain(llm: BaseLanguageModel):
    """基于 LCEL 构建任务执行管道"""
    prompt = PromptTemplate(
        template=(
            "You are an AI who performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        ),
        input_variables=["objective", "context", "task"],
    )
    return prompt | llm | StrOutputParser()


# --- 执行与调度辅助函数 ---

def get_next_task(task_creation_chain, result: str, task_description: str, task_list: List[str], objective: str) -> \
List[Dict]:
    """获取下一批新任务"""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.invoke({
        "result": result,
        "task_description": task_description,
        "incomplete_tasks": incomplete_tasks,
        "objective": objective,
    })
    new_tasks = response.split("\n")
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


def prioritize_tasks(task_prioritization_chain, this_task_id: int, task_list: List[Dict], objective: str) -> List[Dict]:
    """对任务重新排序优先级"""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.invoke({
        "task_names": ", ".join(task_names),
        "next_task_id": next_task_id,
        "objective": objective
    })
    new_tasks = response.split("\n")
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list


def _get_top_tasks(vectorstore: VectorStore, query: str, k: int) -> str:
    """检索相关历史任务"""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return ""
    # InMemoryVectorStore 默认使用余弦相似度，分数越高越相关
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    # result 是一个元组 (Document, score)
    return ", ".join([str(item[0].metadata["task"]) for item in sorted_results])


def execute_task(vectorstore: VectorStore, execution_chain, objective: str, task: str, k: int = 5) -> str:
    """执行单个具体任务"""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.invoke({
        "objective": objective,
        "context": context,
        "task": task
    })


# --- 现代化的 BabyAGI 控制器主类 ---
class BabyAGI:
    def __init__(self, llm: BaseLanguageModel, vectorstore: VectorStore, max_iterations: Optional[int] = None):
        self.task_list = deque()
        self.task_id_counter = 1
        self.vectorstore = vectorstore
        self.max_iterations = max_iterations

        self.task_creation_chain = create_task_creation_chain(llm)
        self.task_prioritization_chain = create_task_prioritization_chain(llm)
        self.execution_chain = create_execution_chain(llm)

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(f"{t['task_id']}: {t['task_name']}")

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(f"{task['task_id']}: {task['task_name']}")

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """运行 Agent 主循环"""
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0

        while True:
            if self.task_list:
                self.print_task_list()

                # 步骤 1: 弹出第一个任务
                task = self.task_list.popleft()
                self.print_next_task(task)

                # 步骤 2: 执行任务
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # 步骤 3: 将结果存入向量数据库
                result_id = f"result_{task['task_id']}_{num_iters}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # 步骤 4: 生成新任务并重新排序
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )

                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)

                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        list(self.task_list),
                        objective,
                    )
                )

            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break

        return {"objective": objective, "final_iterations": num_iters}


# --- 主执行入口 ---
if __name__ == "__main__":
    OBJECTIVE = "分析一下北京市今天的气候情况，写出鲜花储存策略。"

    # 初始化 Gemini Chat 模型
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    # 实例化并运行
    baby_agi = BabyAGI(
        llm=llm,
        vectorstore=vectorstore,
        max_iterations=4  # 限制迭代次数以演示
    )

    baby_agi.invoke({"objective": OBJECTIVE})