# File: service/vector_store.py
import os
import hashlib
import warnings
from typing import List, Dict, Optional, Callable
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from utils.logger import logger
from utils.path_util import get_abs_path
from utils.config_handler import VECTOR_DB_ROOT, KNOWLEDGE_BASE_CONFIG, SEARCH_K

# 抑制 Chroma 的相关性分数警告
warnings.filterwarnings("ignore", category=UserWarning, message="Relevance scores must be between 0 and 1")


class VectorStoreManager:
    """向量数据库管理器：支持物理隔离、去重写入、懒加载"""

    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
        self.base_root = get_abs_path(VECTOR_DB_ROOT)
        self.db_configs = {}
        self.active_dbs: Dict[str, Chroma] = {}

        # 预扫描目录，确认有哪些库
        for folder_prefix, db_name in KNOWLEDGE_BASE_CONFIG.items():
            db_path = os.path.join(self.base_root, db_name)
            os.makedirs(db_path, exist_ok=True)
            self.db_configs[db_name] = db_path

        logger.info(f"[VectorStore] 向量库配置加载完成，共发现 {len(self.db_configs)} 个库配置。")

    def _get_db(self, category_key: str) -> Optional[Chroma]:
        """内部方法：懒加载获取 DB 实例"""
        if category_key in self.active_dbs:
            return self.active_dbs[category_key]

        if category_key not in self.db_configs:
            logger.error(f"[VectorStore] 无效的分类键: {category_key}")
            return None

        db_path = self.db_configs[category_key]
        logger.info(f"[VectorStore] 正在加载向量库: {category_key} (懒加载)")

        # 创建 Chroma 实例（指定余弦距离度量）
        db = Chroma(
            collection_name=category_key,
            persist_directory=db_path,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.active_dbs[category_key] = db
        return db

    def add_documents(self, documents: List[Document], category_key: str):
        """
        添加文档（带去重逻辑）
        注意：LangChain Chroma 的 add_documents 在 ID 重复时通常会报错或覆盖，这里我们显式处理 ID。
        """
        db = self._get_db(category_key)
        if not db:
            return

        # 生成唯一 ID
        ids = []
        for doc in documents:
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            ids.append(doc_id)

        try:
            # 直接写入，如果 ID 存在，Chroma 通常会更新向量。
            # 如果你希望绝对不更新，需要先查询所有 ID，这里为了效率，默认允许更新（即覆盖旧版本）。
            db.add_documents(documents=documents, ids=ids)
            logger.info(f"[VectorStore] 已向 {category_key} 写入/更新 {len(documents)} 个文档")
        except Exception as e:
            logger.error(f"[VectorStore] 写入向量库失败: {e}")

    def get_retriever(self, category_key: str):
        """获取指定库的检索器，返回带相似度分数的文档"""
        db = self._get_db(category_key)
        if db:
            # 使用 similarity_search_with_relevance_scores 的方式
            return db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": SEARCH_K}
            )
        return None

    def get_retriever_with_scores(self, category_key: str) -> Optional[Callable]:
        """获取带相似度分数的检索器"""
        db = self._get_db(category_key)
        if not db:
            return None

        def retriever_with_scores(query: str) -> List[Document]:
            # 使用 similarity_search_with_relevance_scores 返回带分数的结果
            results = db.similarity_search_with_relevance_scores(query, k=SEARCH_K)
            # 转换分数：Chroma 返回的是 distance，需要转换成 similarity
            docs = []
            for doc, distance in results:
                # Chroma 的 distance 对于余弦相似度：distance = 1 - similarity
                # 所以 similarity = 1 - distance
                # 但有时候 distance 可能是负数（向量不在单位球面上）
                # 限制在 [0, 1] 范围内
                similarity = max(0.0, min(1.0, 1.0 - distance))
                doc.metadata['score'] = similarity
                docs.append(doc)
            return docs

        return retriever_with_scores

    def get_all_retrievers_with_scores(self):
        """获取所有库的带分数检索器（用于全局检索）"""
        retrievers = {}
        for key in self.db_configs.keys():
            retriever = self.get_retriever_with_scores(key)
            if retriever:
                retrievers[key] = retriever
        return retrievers
