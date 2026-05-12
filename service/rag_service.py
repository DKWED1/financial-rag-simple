from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Tuple, Optional
import uuid
from datetime import datetime

from utils.logger import logger
from utils.mysql_client import get_mysql_client
from utils.redis_client import get_redis
from utils.config_handler import rag_config, RAG_SYSTEM_PROMPT, RELEVANCE_THRESHOLD
from langchain_core.documents import Document
from service.vector_store import VectorStoreManager
from service.router_service import RouterService


class RAGService:
    """RAG 服务主类 - 支持相关性检测和全局检索"""

    def __init__(self):
        # 1. 初始化模型
        self.embedding_model = DashScopeEmbeddings(model=rag_config['embedding_model_name'])
        self.llm = ChatTongyi(model=rag_config['chat_model_name'])

        # 2. 初始化组件
        self.vector_store = VectorStoreManager(self.embedding_model)
        self.router = RouterService(self.llm)

        # 3. 初始化 MySQL 客户端
        self.mysql = get_mysql_client()

        # 4. 初始化 Redis 缓存
        self.cache = get_redis()

        # 4. 初始化会话历史（内存缓存）
        self.current_session_id = None
        self.chat_history = []
        self.max_history_turns = 10

        logger.info("[RAGService] 系统启动完成，准备就绪。")

    def ask(self, question: str, session_id: Optional[str] = None) -> str:
        """对外提供的问答接口：关键词 + 阈值双重过滤

        Args:
            question: 用户问题
            session_id: 可选的会话ID，不提供则自动生成
        """
        # 1. 缓存检查：相同问题直接返回缓存
        cached = self.cache.get_qa(question)
        if cached:
            logger.info(f"[RAGService] Redis缓存命中: {question[:30]}")
            self.current_session_id = session_id or str(uuid.uuid4())
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=cached['answer']))
            self._save_chat_record(question, cached['answer'], cached['source_db'], 1.0, "cache")
            return cached['answer']

        # 初始化会话
        if session_id:
            self.current_session_id = session_id
        else:
            self.current_session_id = str(uuid.uuid4())

        # 加载会话历史（从 MySQL 恢复）
        self._load_session_history(self.current_session_id)

        logger.info(f"[RAGService] 用户提问: {question} (session: {self.current_session_id})")

        # 路由类型记录
        route_type = "keyword"

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
                answer = self._generate_answer(question, docs, db_name, route_type="keyword")
                self.cache.set_qa(question, answer, db_name)
                self._save_chat_record(question, answer, db_name, best_score, "keyword")
                return answer

        # 3. 所有候选库都不达标，使用最高分那个
        if best_docs and best_db:
            logger.info(f"[RAGService] 候选库均不达标，使用最高分: {best_db} (相关度: {best_score:.3f})")
            answer = self._generate_answer(question, best_docs, best_db, route_type="keyword")
            self.cache.set_qa(question, answer, best_db)
            self._save_chat_record(question, answer, best_db, best_score, "keyword")
            return answer

        # 4. 没有关键词匹配，尝试语义路由
        route_type = "semantic"
        logger.info("[RAGService] 无关键词匹配，尝试语义路由...")
        semantic_db = self.router.route(question, self.chat_history)

        if semantic_db == "none":
            answer = "您的问题与金融风控无关，我只能回答关于信贷申请、逾期催收、风控合规等方面的问题。"
            self.cache.set_qa(question, answer, None)
            self._save_chat_record(question, answer, None, 0, "semantic")
            return answer

        # 语义路由结果检索
        retriever_func = self.vector_store.get_retriever_with_scores(semantic_db)
        if retriever_func:
            docs = retriever_func(question)
            if docs:
                score = docs[0].metadata.get('score', 0)
                logger.info(f"[RAGService] 语义路由到 {semantic_db}，相关度: {score:.3f}")
                answer = self._generate_answer(question, docs, semantic_db, route_type="semantic")
                self.cache.set_qa(question, answer, semantic_db)
                self._save_chat_record(question, answer, semantic_db, score, "semantic")
                return answer

        # 5. 兜底：全局检索
        route_type = "global"
        logger.info("[RAGService] 兜底：全局检索")
        all_retrievers = self.vector_store.get_all_retrievers_with_scores()
        docs, db = self._global_search(question, all_retrievers)

        if not docs:
            answer = "根据现有资料，未找到相关信息"
            self.cache.set_qa(question, answer, None)
            self._save_chat_record(question, answer, None, 0, "global")
            return answer

        score = docs[0].metadata.get('score', 0) if docs else 0
        logger.info(f"[RAGService] 全局检索最佳: {db} (相关度: {score:.3f})")
        answer = self._generate_answer(question, docs, db, route_type="global")
        self.cache.set_qa(question, answer, db)
        return answer

    def _save_chat_record(self, question: str, answer: str, source_db: Optional[str],
                          relevance_score: float, route_type: str) -> int:
        """保存问答记录到 MySQL"""
        try:
            sql = """
                INSERT INTO chat_history (session_id, question, answer, source_db, relevance_score, route_type)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (self.current_session_id, question, answer, source_db, relevance_score, route_type)
            record_id = self.mysql.insert(sql, params)
            logger.debug(f"[RAGService] 保存聊天记录: id={record_id}")
            return record_id
        except Exception as e:
            logger.error(f"[RAGService] 保存聊天记录失败: {e}")
            return -1

    def _load_session_history(self, session_id: str) -> None:
        """从 MySQL 加载会话历史到内存"""
        try:
            sql = """
                SELECT question, answer FROM chat_history
                WHERE session_id = %s AND status = 1
                ORDER BY create_time ASC
            """
            records = self.mysql.fetchall(sql, (session_id,))

            # 清空当前历史并重建
            self.chat_history = []
            for record in records:
                self.chat_history.append(HumanMessage(content=record['question']))
                self.chat_history.append(AIMessage(content=record['answer']))

            # 裁剪超出限制的部分
            if len(self.chat_history) > self.max_history_turns * 2:
                self.chat_history = self.chat_history[-(self.max_history_turns * 2):]

            logger.debug(f"[RAGService] 加载会话历史: session={session_id}, count={len(records)}")
        except Exception as e:
            logger.warning(f"[RAGService] 加载会话历史失败（使用空历史）: {e}")
            self.chat_history = []

    def get_session_history(self, session_id: str) -> List[Dict]:
        """获取指定会话的历史记录（用于前端展示）"""
        try:
            sql = """
                SELECT id, question, answer, source_db, relevance_score, route_type, create_time
                FROM chat_history
                WHERE session_id = %s AND status = 1
                ORDER BY create_time ASC
            """
            return self.mysql.fetchall(sql, (session_id,))
        except Exception as e:
            logger.error(f"[RAGService] 获取会话历史失败: {e}")
            return []

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

    def _generate_answer(self, question: str, docs: List[Document], source_db: str,
                         route_type: str = "keyword") -> str:
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