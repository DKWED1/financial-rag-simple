import os
from pathlib import Path
import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from utils.logger import logger


class FileProcessor:
    """
    企业级文件处理器
    负责金融文档的加载、清洗、精确去重与语义去重
    """
    # 允许的文件扩展名
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

    @staticmethod
    def get_file_md5(filepath: str) -> Optional[str]:
        """
        计算文件 MD5 值（支持大文件流式读取，防止内存溢出）
        """
        path = Path(filepath)  # 把传进来的字符串路径包装成一个智能的 Path 对象
        if not path.is_file():
            logger.error(f"[FileMD5] 无效文件: {filepath}")
            return None

        try:
            hash_md5 = hashlib.md5()
            # 流式读取：每次读取 4KB
            with open(path, "rb") as f: # 'r'读，'b'二进制
                for chunk in iter(lambda: f.read(4096), b""):  # 迭代器iter iter(动作, 结束标记) lambda匿名函数
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"[FileMD5] 计算失败 {filepath}: {str(e)}")
            return None

    @staticmethod
    def scan_valid_files(dir_path: str) -> List[str]:
        """
        递归扫描目录下的有效文件
        """
        path = Path(dir_path)
        if not path.is_dir():
            logger.error(f"[FileScan] 目录不存在: {dir_path}")
            return []

        # 使用 rglob 递归查找，并过滤扩展名
        valid_files = [
            str(file) for file in path.rglob("*")  # str(file)把 Path 对象 → 转成普通字符串路径,因为后面加载文件需要字符串路径，所以最后转成 str。
            if file.is_file() and file.suffix.lower() in FileProcessor.ALLOWED_EXTENSIONS
        ]

        logger.info(f"[FileScan] 扫描完成，找到有效文件：{len(valid_files)} 个")
        return valid_files

    @staticmethod
    def load_document(filepath: str) -> List[Document]:
        """
        根据文件类型加载文档
        """
        path = Path(filepath)
        ext = path.suffix.lower()

        try:
            if ext == ".pdf":
                # PyPDFLoader 自动处理分页
                loader = PyPDFLoader(str(path))
            elif ext in (".txt", ".md"):
                # TextLoader 自动检测编码
                loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
            else:
                logger.warning(f"[FileLoad] 不支持的文件类型: {filepath}")
                return []

            docs = loader.load()
            # 【新增】针对 Markdown/Txt 的极简预处理
            # 去除首尾空白符，防止因为文件末尾多几个回车导致 MD5 不一致
            for doc in docs:
                doc.page_content = doc.page_content.strip()

            logger.info(f"[FileLoad] 加载成功: {path.name} (共 {len(docs)} 页/段)")
            return docs

        except Exception as e:
            logger.error(f"[FileLoad] 加载失败 {filepath}: {str(e)}")
            return []

    # ===================== 数据清洗与去重 =====================

    @staticmethod
    def clean_empty_chunks(chunks: List[Document]) -> List[Document]:
        """
        移除空白或无意义的文档块
        """
        # 这个块的内容必须存在（不能是 None 或空）and 把内容首尾的空格、换行符去掉后，剩下的东西不能是空的。
        cleaned = [c for c in chunks if c.page_content and c.page_content.strip()]
        removed_count = len(chunks) - len(cleaned)
        if removed_count > 0:
            logger.info(f"[CleanEmpty] 已移除 {removed_count} 个空白块")
        return cleaned

    @staticmethod
    def exact_dedup(chunks: List[Document]) -> List[Document]:
        """
        精确去重（基于内容 MD5）
        时间复杂度 O(N)，速度极快
        """
        seen = set()
        unique = []

        for chunk in chunks:
            if not chunk.page_content.strip():
                continue
            # 计算内容的 MD5
            content_md5 = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
            if content_md5 not in seen:
                seen.add(content_md5)
                unique.append(chunk)

        logger.info(f"[ExactDedup] 精确去重完成：{len(chunks)} -> {len(unique)}")
        return unique

    @staticmethod
    def semantic_dedup( # semantic语义的dedup去重
            chunks: List[Document],
            embedding_function: Embeddings,
            threshold: float = 0.92,  # 相似度阈值 (0~1)，推荐 0.90~0.95
            collection_name: str = "temp_dedup",
            persist_directory: Optional[str] = None,
    ) -> List[Document]:
        """
        语义去重（基于向量相似度）

        核心逻辑：
        1. 使用 Chroma (HNSW:COSINE) 存储向量。
        2. LangChain 的 relevance_scores 会自动将 Cosine Distance (0-2) 转换为 Similarity (0-1)。
        3. 若相似度 > threshold，视为重复并丢弃。
        """
        if not chunks:
            return []

        # 【关键】显式指定使用余弦距离空间
        collection_metadata = {"hnsw:space": "cosine"}

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            collection_metadata=collection_metadata,
        )

        unique_chunks = []
        logger.info(f"[SemanticDedup] 开始语义去重，阈值={threshold} (Cosine Similarity)")

        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if not content:
                continue

            # 查询当前库中是否存在相似内容
            # k=1 表示只找最相似的一个
            results = vectorstore.similarity_search_with_relevance_scores(content, k=1)

            is_duplicate = False
            if results:
                _, score = results[0] # # 这是一个元组 (Tuple)，(Document("这是库里的某段文本..."), 0.95)

                # ========================
                # ✅ 防御性编程 + 异常 B 方案
                # ========================
                if not (0 <= score <= 1):
                    # 模型/向量库发疯 → 执行 B 方案：不判定重复，直接保留
                    logger.warning(f"[SemanticDedup] 相似度分数异常 score={score}，跳过判断，保留文本")
                    is_duplicate = False
                else:
                    # 正常情况：判断是否重复
                    if score >= threshold:
                        is_duplicate = True

            if not is_duplicate:
                vectorstore.add_documents([chunk])
                unique_chunks.append(chunk)

            # 简单的进度日志
            if (i + 1) % 500 == 0:
                logger.info(f"[SemanticDedup] 处理进度: {i + 1}/{len(chunks)}, 当前保留: {len(unique_chunks)}")

        logger.info(f"[SemanticDedup] 语义去重完成！保留 {len(unique_chunks)} / {len(chunks)} 个")

        # 【清理】测试/临时模式：删除集合释放内存
        # 如果需要持久化去重库，可注释掉下面这行
        try:
            vectorstore.delete_collection()
        except Exception:
            pass

        return unique_chunks