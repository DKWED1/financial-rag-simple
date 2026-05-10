# tests/conftest.py
"""pytest 公共配置"""
import pytest
import sys
import os

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def project_root_path():
    """返回项目根目录路径"""
    return project_root


@pytest.fixture(scope="session")
def vector_db_exists():
    """检查向量库是否存在"""
    return os.path.exists(os.path.join(project_root, "vector_dbs"))
