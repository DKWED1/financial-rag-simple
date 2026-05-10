# tests/test_vector_store.py
"""向量存储服务单元测试"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.vector_store import VectorStoreManager
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import (
    KNOWLEDGE_BASE_CONFIG,
    VECTOR_DB_ROOT,
    SEARCH_K,
    rag_config
)


class TestVectorStoreConfig:
    """向量库配置测试"""

    def test_knowledge_base_config_exists(self):
        """测试：知识库配置存在"""
        assert KNOWLEDGE_BASE_CONFIG is not None
        assert len(KNOWLEDGE_BASE_CONFIG) == 3

    def test_knowledge_base_keys(self):
        """测试：知识库键名正确"""
        assert "1_credit_base" in KNOWLEDGE_BASE_CONFIG
        assert "2_overdue_debt" in KNOWLEDGE_BASE_CONFIG
        assert "3_risk_control" in KNOWLEDGE_BASE_CONFIG

    def test_knowledge_base_values(self):
        """测试：知识库值正确"""
        assert KNOWLEDGE_BASE_CONFIG["1_credit_base"] == "credit_base"
        assert KNOWLEDGE_BASE_CONFIG["2_overdue_debt"] == "overdue_debt"
        assert KNOWLEDGE_BASE_CONFIG["3_risk_control"] == "risk_control"

    def test_vector_db_root_configured(self):
        """测试：向量库根目录已配置"""
        assert VECTOR_DB_ROOT is not None
        assert isinstance(VECTOR_DB_ROOT, str)

    def test_search_k_configured(self):
        """测试：检索数量已配置"""
        assert SEARCH_K is not None
        assert isinstance(SEARCH_K, int)
        assert SEARCH_K > 0

    def test_embedding_model_configured(self):
        """测试：Embedding 模型已配置"""
        assert "embedding_model_name" in rag_config
        assert rag_config["embedding_model_name"] is not None


class TestVectorStoreManager:
    """VectorStoreManager 单元测试"""

    @pytest.fixture
    def embedding_model(self):
        """创建 Embedding 模型实例"""
        return DashScopeEmbeddings(model=rag_config['embedding_model_name'])

    @pytest.fixture
    def vector_store(self, embedding_model):
        """创建 VectorStoreManager 实例"""
        return VectorStoreManager(embedding_model)

    def test_vector_store_initialization(self, vector_store):
        """测试：向量库管理器初始化"""
        assert vector_store.embedding_model is not None
        assert vector_store.base_root is not None
        assert len(vector_store.db_configs) == 3

    def test_db_configs_loaded(self, vector_store):
        """测试：数据库配置已加载"""
        assert "credit_base" in vector_store.db_configs
        assert "overdue_debt" in vector_store.db_configs
        assert "risk_control" in vector_store.db_configs

    def test_get_retriever_returns_callable(self, vector_store):
        """测试：get_retriever 返回可调用对象"""
        retriever = vector_store.get_retriever("credit_base")
        assert retriever is not None

    def test_get_retriever_with_scores_returns_callable(self, vector_store):
        """测试：get_retriever_with_scores 返回可调用对象"""
        retriever = vector_store.get_retriever_with_scores("credit_base")
        assert retriever is not None
        assert callable(retriever)

    def test_get_all_retrievers_with_scores(self, vector_store):
        """测试：获取所有库的检索器"""
        retrievers = vector_store.get_all_retrievers_with_scores()
        assert len(retrievers) == 3
        assert "credit_base" in retrievers
        assert "overdue_debt" in retrievers
        assert "risk_control" in retrievers


class TestVectorStoreRetrieval:
    """向量库检索功能测试（需要实际向量库）"""

    @pytest.fixture
    def embedding_model(self):
        return DashScopeEmbeddings(model=rag_config['embedding_model_name'])

    @pytest.fixture
    def vector_store(self, embedding_model):
        return VectorStoreManager(embedding_model)

    @pytest.mark.skipif(
        not os.path.exists("F:/Project/financial-rag-simple/vector_dbs"),
        reason="需要先构建向量库"
    )
    def test_retrieval_with_scores(self, vector_store):
        """测试：带分数的检索（需要向量库存在）"""
        retriever = vector_store.get_retriever_with_scores("credit_base")
        docs = retriever("学生贷款")

        assert docs is not None
        assert len(docs) > 0
        # 验证分数在 metadata 中
        assert "score" in docs[0].metadata

    @pytest.mark.skipif(
        not os.path.exists("F:/Project/financial-rag-simple/vector_dbs"),
        reason="需要先构建向量库"
    )
    def test_retrieval_returns_documents(self, vector_store):
        """测试：检索返回 Document 对象"""
        retriever = vector_store.get_retriever("credit_base")
        docs = retriever.invoke("准入条件")

        assert docs is not None
        assert len(docs) > 0
        # 验证返回的是 Document 对象
        assert hasattr(docs[0], "page_content")
        assert hasattr(docs[0], "metadata")
