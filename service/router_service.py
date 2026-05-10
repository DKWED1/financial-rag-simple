from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from utils.config_handler import ROUTING_KEYWORDS, ROUTING_PROMPT
from utils.logger import logger

# 合法的路由结果（包含 none）
VALID_ROUTES = list(ROUTING_KEYWORDS.keys()) + ["none"]


class RouterService:
    """智能路由服务：决定用户问题去哪个向量库查询"""

    def __init__(self, llm):
        self.llm = llm
        # 构建关键词映射：词 -> 库名
        self.keyword_to_db = {}
        for db_name, keywords in ROUTING_KEYWORDS.items():
            for kw in keywords:
                self.keyword_to_db[kw] = db_name

        # 路由提示词（从配置文件读取）
        self.routing_prompt = PromptTemplate.from_template(ROUTING_PROMPT)

    def _format_chat_history(self, chat_history: list) -> str:
        """格式化对话历史为字符串"""
        if not chat_history:
            return "无"

        formatted = []
        for msg in chat_history[-6:]:  # 最近3轮对话
            if isinstance(msg, HumanMessage):
                formatted.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"助手: {msg.content}")
        return "\n".join(formatted)

    def route(self, question: str, chat_history: list = None) -> str:
        """
        路由主逻辑
        1. 先进行关键词匹配
        2. 如果关键词未命中，调用 LLM 进行语义分类（带对话历史）
        返回: credit_base/overdue_debt/risk_control/none
        """
        chat_history = chat_history or []
        logger.info(f"[Router] 开始路由分析: {question}")

        # 第一优先级：关键词匹配
        for keyword, db_name in self.keyword_to_db.items():
            if keyword in question:
                logger.info(f"[Router] 关键词命中: '{keyword}' -> 库: {db_name}")
                return db_name

        # 第二优先级：LLM 语义匹配（带历史）
        logger.info("[Router] 关键词未命中，启动 LLM 语义路由...")
        chat_context = self._format_chat_history(chat_history)

        try:
            result = self.routing_prompt.format(
                chat_history=chat_context,
                question=question
            )
            result = self.llm.invoke(result).content.strip().lower()

            if result in VALID_ROUTES:
                logger.info(f"[Router] LLM 路由结果: {result}")
                return result
            else:
                logger.warning(f"[Router] LLM 返回无效结果: {result}，默认使用 none")
                return "none"
        except Exception as e:
            logger.error(f"[Router] LLM 路由失败: {e}，默认使用 none")
            return "none"