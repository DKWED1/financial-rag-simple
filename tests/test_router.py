# tests/test_router.py
"""路由服务单元测试"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.router_service import RouterService
from utils.config_handler import ROUTING_KEYWORDS


class TestRouterService:
    """RouterService 单元测试"""

    @pytest.fixture
    def router(self, monkeypatch):
        """创建 RouterService 实例（不真正调用 LLM）"""
        # 用一个 mock 的 LLM 避免实际调用 API
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        return RouterService(mock_llm)

    def test_keyword_routing_credit_base(self, router):
        """测试：包含"申请"关键词应路由到 credit_base"""
        result = router.route("我想申请贷款")
        assert result == "credit_base"

    def test_keyword_routing_overdue_debt(self, router):
        """测试：包含"逾期"关键词应路由到 overdue_debt"""
        result = router.route("逾期了怎么办")
        assert result == "overdue_debt"

    def test_keyword_routing_risk_control(self, router):
        """测试：包含"风控"关键词应路由到 risk_control"""
        result = router.route("会被风控吗")
        assert result == "risk_control"

    def test_keyword_routing_fee_rate(self, router):
        """测试：包含"利率"关键词应路由到 credit_base"""
        result = router.route("贷款利率是多少")
        assert result == "credit_base"

    def test_keyword_routing_collect(self, router):
        """测试：包含"催收"关键词应路由到 overdue_debt"""
        result = router.route("催收流程是什么")
        assert result == "overdue_debt"

    def test_keyword_routing_fraud(self, router):
        """测试：包含"欺诈"关键词应路由到 risk_control"""
        result = router.route("如何防止欺诈")
        assert result == "risk_control"

    def test_multiple_keywords_first_match(self, router):
        """测试：多个关键词时返回第一个匹配的库"""
        # "逾期" 和 "风控" 同时出现，应该返回第一个匹配的
        result = router.route("逾期了，风控会怎么处理")
        # 由于 keyword_to_db 是按顺序遍历，返回第一个匹配的
        assert result in ["overdue_debt", "risk_control"]

    def test_no_keywords_returns_none(self, router, monkeypatch):
        """测试：无关键词时返回 none（通过 mock LLM）"""
        # Mock LLM 返回 none
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="none")
        router.llm = mock_llm

        result = router.route("今天天气怎么样")
        assert result == "none"

    def test_keyword_to_db_mapping(self, router):
        """测试：关键词到库的映射是否正确"""
        # 验证 ROUTING_KEYWORDS 配置
        assert "credit_base" in ROUTING_KEYWORDS
        assert "overdue_debt" in ROUTING_KEYWORDS
        assert "risk_control" in ROUTING_KEYWORDS

        # 验证关键词存在
        assert "申请" in ROUTING_KEYWORDS["credit_base"]
        assert "逾期" in ROUTING_KEYWORDS["overdue_debt"]
        assert "风控" in ROUTING_KEYWORDS["risk_control"]


class TestRouterPrompt:
    """路由提示词配置测试"""

    def test_routing_prompt_has_variables(self):
        """测试：routing_prompt 包含所需变量"""
        from utils.config_handler import ROUTING_PROMPT

        assert "{chat_history}" in ROUTING_PROMPT
        assert "{question}" in ROUTING_PROMPT

    def test_routing_prompt_has_all_categories(self):
        """测试：routing_prompt 包含所有路由类别"""
        from utils.config_handler import ROUTING_PROMPT

        assert "credit_base" in ROUTING_PROMPT
        assert "overdue_debt" in ROUTING_PROMPT
        assert "risk_control" in ROUTING_PROMPT
        assert "none" in ROUTING_PROMPT
