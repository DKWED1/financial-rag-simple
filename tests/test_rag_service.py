# tests/test_rag_service.py
"""RAG 服务单元测试"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.rag_service import RAGService
from utils.config_handler import RELEVANCE_THRESHOLD, rag_config
from langchain_core.documents import Document


class TestRAGServiceConfig:
    """RAG 服务配置测试"""

    def test_rag_config_has_model_names(self):
        """测试：配置包含模型名称"""
        assert "chat_model_name" in rag_config
        assert "embedding_model_name" in rag_config

    def test_relevance_threshold_configured(self):
        """测试：相关性阈值已配置"""
        assert RELEVANCE_THRESHOLD is not None
        assert isinstance(RELEVANCE_THRESHOLD, (int, float))
        assert 0 <= RELEVANCE_THRESHOLD <= 1


class TestRAGService:
    """RAG 服务功能测试"""

    @pytest.fixture
    def rag_service(self):
        """创建 RAG 服务实例（可能需要较长时间初始化）"""
        return RAGService()

    def test_rag_service_initialization(self, rag_service):
        """测试：RAG 服务初始化"""
        assert rag_service.embedding_model is not None
        assert rag_service.llm is not None
        assert rag_service.vector_store is not None
        assert rag_service.router is not None

    def test_chat_history_initialized(self, rag_service):
        """测试：聊天历史已初始化"""
        assert rag_service.chat_history == []
        assert rag_service.max_history_turns == 10


class TestRelevanceDetection:
    """相关性检测测试"""

    @pytest.fixture
    def rag_service(self):
        """创建 RAG 服务实例"""
        return RAGService()

    def test_is_relevant_with_high_score(self, rag_service):
        """测试：高相关度应该返回 True"""
        doc = Document(
            page_content="学生贷款准入条件...",
            metadata={"score": 0.8}
        )
        assert rag_service._is_relevant([doc]) is True

    def test_is_relevant_with_low_score(self, rag_service):
        """测试：低相关度应该返回 False"""
        doc = Document(
            page_content="学生贷款准入条件...",
            metadata={"score": 0.2}
        )
        assert rag_service._is_relevant([doc]) is False

    def test_is_relevant_with_threshold(self, rag_service):
        """测试：使用自定义阈值"""
        doc = Document(
            page_content="学生贷款准入条件...",
            metadata={"score": 0.4}
        )
        # 默认阈值 0.35，0.4 应该通过
        assert rag_service._is_relevant([doc]) is True
        # 使用更高阈值 0.5，0.4 应该失败
        assert rag_service._is_relevant([doc], threshold=0.5) is False

    def test_is_relevant_empty_docs(self, rag_service):
        """测试：空文档应该返回 False"""
        assert rag_service._is_relevant([]) is False

    def test_is_relevant_no_metadata(self, rag_service):
        """测试：无分数时根据内容长度判断"""
        # 内容长度 > 50
        doc = Document(
            page_content="这是一个很长的内容，超过五十个字符，可以用于贷款准入条件的描述" * 2,
            metadata={}
        )
        assert rag_service._is_relevant([doc]) is True

        # 内容长度 < 50
        short_doc = Document(
            page_content="短内容",
            metadata={}
        )
        assert rag_service._is_relevant([short_doc]) is False


class TestGlobalSearch:
    """全局检索测试"""

    @pytest.fixture
    def rag_service(self):
        return RAGService()

    @pytest.mark.skipif(
        not os.path.exists("F:/Project/financial-rag-simple/vector_dbs"),
        reason="需要先构建向量库"
    )
    def test_global_search_returns_docs_and_db(self, rag_service):
        """测试：全局检索返回文档和库名"""
        retrievers = rag_service.vector_store.get_all_retrievers_with_scores()
        docs, db = rag_service._global_search("学生贷款", retrievers)

        assert docs is not None
        assert db is not None
        assert len(docs) > 0


class TestKeywordMatching:
    """关键词匹配测试"""

    @pytest.fixture
    def rag_service(self):
        return RAGService()

    def test_get_keyword_matched_dbs_single(self, rag_service):
        """测试：单个关键词匹配"""
        dbs = rag_service._get_keyword_matched_dbs("我想申请贷款")
        assert "credit_base" in dbs

    def test_get_keyword_matched_dbs_multiple(self, rag_service):
        """测试：多个关键词匹配"""
        dbs = rag_service._get_keyword_matched_dbs("逾期贷款申请")
        # 可能返回多个库
        assert len(dbs) >= 1

    def test_get_keyword_matched_dbs_none(self, rag_service):
        """测试：无关键词匹配"""
        dbs = rag_service._get_keyword_matched_dbs("今天天气很好")
        assert dbs == []
