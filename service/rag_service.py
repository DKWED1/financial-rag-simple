from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Tuple, Optional

from utils.logger import logger
from service.vector_store import VectorStoreManager
from service.router_service import RouterService
from utils.config_handler import rag_config, RAG_SYSTEM_PROMPT, RELEVANCE_THRESHOLD
from langchain_core.documents import Document


class RAGService:
    """RAG 服务主类 - 支持相关性检测和全局检索"""

    def __init__(self):
        # 1. 初始化模型
        self.embedding_model = DashScopeEmbeddings(model=rag_config['embedding_model_name'])
        self.llm = ChatTongyi(model=rag_config['chat_model_name'])

        # 2. 初始化组件
        self.vector_store = VectorStoreManager(self.embedding_model)
        self.router = RouterService(self.llm)

        # 3. 初始化聊天历史
        self.chat_history = []
        self.max_history_turns = 10

        logger.info("[RAGService] 系统启动完成，准备就绪。")

    def ask(self, question: str) -> str:
        """对外提供的问答接口：关键词 + 阈值双重过滤"""
        logger.info(f"[RAGService] 用户提问: {question}")

        # 1. 先做关键词匹配，找出候选库列表
        candidate_dbs = self._get_keyword_matched_dbs(question)
        logger.info(f"[RAGService] 关键词候选库: {candidate_dbs}")

        # 2. 遍历候选库，找第一个相关度达标的
        best_docs = []
        best_db = None
        best_score = 0

        for db_name in candidate_dbs:
            retriever_func = self.vector_store.get_retriever_with_scores(db_name)
            if not retriever_func:
                continue

            docs = retriever_func(question)
            if not docs:
                continue

            # 获取相关度分数
            score = docs[0].metadata.get('score', 0) if docs else 0
            logger.debug(f"[RAGService] 尝试库 {db_name}，相关度: {score:.3f}")

            # 保存最高分的记录
            if score > best_score:
                best_score = score
                best_docs = docs
                best_db = db_name

            # 相关度 >= 阈值，直接使用
            if score >= RELEVANCE_THRESHOLD:
                logger.info(f"[RAGService] 命中库 {db_name}，相关度达标: {score:.3f}")
                return self._generate_answer(question, docs, db_name)

        # 3. 所有候选库都不达标，使用最高分那个
        if best_docs and best_db:
            logger.info(f"[RAGService] 候选库均不达标，使用最高分: {best_db} (相关度: {best_score:.3f})")
            return self._generate_answer(question, best_docs, best_db)

        # 4. 没有关键词匹配，尝试语义路由
        logger.info("[RAGService] 无关键词匹配，尝试语义路由...")
        semantic_db = self.router.route(question, self.chat_history)

        if semantic_db == "none":
            return "您的问题与金融风控无关，我只能回答关于信贷申请、逾期催收、风控合规等方面的问题。"

        # 语义路由结果检索
        retriever_func = self.vector_store.get_retriever_with_scores(semantic_db)
        if retriever_func:
            docs = retriever_func(question)
            if docs:
                score = docs[0].metadata.get('score', 0)
                logger.info(f"[RAGService] 语义路由到 {semantic_db}，相关度: {score:.3f}")
                return self._generate_answer(question, docs, semantic_db)

        # 5. 兜底：全局检索
        logger.info("[RAGService] 兜底：全局检索")
        all_retrievers = self.vector_store.get_all_retrievers_with_scores()
        docs, db = self._global_search(question, all_retrievers)

        if not docs:
            return "根据现有资料，未找到相关信息"

        score = docs[0].metadata.get('score', 0) if docs else 0
        logger.info(f"[RAGService] 全局检索最佳: {db} (相关度: {score:.3f})")
        return self._generate_answer(question, docs, db)

    def _get_keyword_matched_dbs(self, question: str) -> List[str]:
        """根据问题中的关键词，返回候选库列表（按匹配顺序）"""
        from utils.config_handler import ROUTING_KEYWORDS

        matched_dbs = []
        seen = set()

        for db_name, keywords in ROUTING_KEYWORDS.items():
            for kw in keywords:
                if kw in question and db_name not in seen:
                    matched_dbs.append(db_name)
                    seen.add(db_name)
                    break

        return matched_dbs

    def _is_relevant(self, docs: List[Document], threshold: Optional[float] = None) -> bool:
        """判断检索结果是否足够相关"""
        if not docs:
            return False

        threshold = threshold or RELEVANCE_THRESHOLD

        # 检查第一个文档的相似度分数
        doc = docs[0]
        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
            score = doc.metadata['score']
            logger.debug(f"[RAGService] 检索相关度分数: {score:.3f} (阈值: {threshold})")
            return score >= threshold

        # 如果没有分数，简单判断文档内容长度
        is_valid = len(doc.page_content.strip()) > 50
        logger.debug(f"[RAGService] 无相似度分数，根据内容长度判断: {is_valid}")
        return is_valid

    def _global_search(self, question: str, retrievers: Dict[str, any]) -> Tuple[List[Document], str]:
        """全局检索，返回最相关的文档和库名"""
        best_docs = []
        best_db = None
        best_score = 0

        for db_name, retriever_func in retrievers.items():
            docs = retriever_func(question)
            if not docs:
                continue

            # 取第一个文档的相似度作为代表
            doc = docs[0]
            score = 0

            if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                score = doc.metadata['score']
            elif hasattr(doc, 'metadata') and 'distance' in doc.metadata:
                distance = doc.metadata['distance']
                score = 1 - distance

            if score > best_score:
                best_score = score
                best_docs = docs
                best_db = db_name

        logger.info(f"[RAGService] 全局检索得分最高: {best_score:.3f}")
        return best_docs, best_db

    def _generate_answer(self, question: str, docs: List[Document], source_db: str) -> str:
        """生成回答"""
        context = "\n".join([doc.page_content for doc in docs])

        # 构建带历史的 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # 构建 Chain
        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "context": context,
                "question": question,
                "chat_history": self.chat_history,
            })
            logger.info(f"[RAGService] 生成回答完成")

            # 更新历史记录
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=result))

            # 超出限制时裁剪最早的一轮（2条消息）
            if len(self.chat_history) > self.max_history_turns * 2:
                self.chat_history = self.chat_history[2:]

            return result
        except Exception as e:
            logger.error(f"[RAGService] 生成回答失败: {e}")
            return "抱歉，回答生成时发生错误。"