# 为整个工程提供统一的绝对路径

import os


def get_project_root() ->str:
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(current_dir)

    return project_root


def get_abs_path(relative_path:str) -> str:
    project_root = get_project_root()
    return os.path.join(project_root,relative_path)


PROJECT_ROOT = get_project_root()
DATA_DIR = get_abs_path("data/finance_rag_knowledge")
STORAGE_DIR = get_abs_path("storage/chroma_db")
LOGS_DIR = get_abs_path("logs")
CONFIG_DIR = get_abs_path("config")


# print(DATA_DIR)
