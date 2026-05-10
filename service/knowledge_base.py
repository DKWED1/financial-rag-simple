# File: service/knowledge_base.py
import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger
from utils.path_util import get_abs_path,DATA_DIR
from utils.config_handler import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    KNOWLEDGE_BASE_CONFIG,
)


class KnowledgeBaseManager:
    def build_knowledge_base(self) -> List[Document]:
        """构建金融风控知识库的文档切片（支持三级目录结构）"""
        logger.info("[KnowledgeBase] 开始构建金融风控知识库...")
        # 1. 获取当前 knowledge_base.py 文件的目录
        # CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

        # 2. 向上走到项目根目录（service/ 上一级是项目根目录）
        # PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

        # 3. 拼接出 data 目录的绝对路径
        # BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "finance_rag_knowledge")
        # BASE_DATA_DIR = get_abs_path("/data/finance_rag_knowledge")

        # 4. 标准化路径（处理斜杠、多余分隔符）
        base_path = os.path.normpath(DATA_DIR)


        # logger.info(f"[KnowledgeBase] 实际读取路径: {base_path}")
        logger.info(f"[KnowledgeBase] 实际读取路径: {base_path}")

        # 检查目录是否存在
        if not os.path.exists(base_path):
            error_msg = f"❌ 知识库源文件夹不存在: {base_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # === 3. 递归扫描三级目录结构 ===
        all_docs = []
        logger.info(f"🔍 扫描知识库目录结构 (支持三级嵌套):")

        # 第一级：遍历 finance_rag_knowledge/ 下的主分类目录
        for main_dir in os.listdir(base_path):
            main_dir_path = os.path.join(base_path, main_dir)

            if not os.path.isdir(main_dir_path):
                continue

            # 检查是否匹配知识库前缀（如 "1_credit_base"）
            kb_name = self._get_kb_name(main_dir)
            if not kb_name:
                logger.warning(f"  ├─ [⚠️ 跳过] {main_dir}/ → 未在KNOWLEDGE_BASE_CONFIG中配置")
                continue

            logger.info(f"  ├─ [📚 {kb_name}] {main_dir}/")

            # 第二级：遍历主分类下的规则子目录（如 1_1_apply_rule/）
            for sub_dir in os.listdir(main_dir_path):
                sub_dir_path = os.path.join(main_dir_path, sub_dir)

                if not os.path.isdir(sub_dir_path):
                    continue

                logger.info(f"  │  ├─ [📁 规则组] {sub_dir}/")

                # 第三级：遍历规则子目录中的实际文档
                for file_name in os.listdir(sub_dir_path):
                    file_path = os.path.join(sub_dir_path, file_name)

                    # 仅处理支持的文档格式
                    if not file_name.lower().endswith(('.txt', '.md', '.pdf')):
                        continue

                    try:
                        # 读取文档内容（简化版，实际需扩展PDF处理）
                        if file_name.endswith(('.txt', '.md')):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        # TODO: 此处应添加PDF解析逻辑（如PyPDF2）

                        # 生成带完整元数据的文档对象
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": file_name,
                                "kb_category": kb_name,  # 知识库分类（credit_base等）
                                "rule_group": sub_dir,  # 规则组（如1_1_apply_rule）
                                "full_path": os.path.relpath(file_path, base_path)
                            }
                        )
                        all_docs.append(doc)
                        logger.info(f"  │  │  └─ [📄 已加载] {file_name}")
                    except Exception as e:
                        logger.error(f"  │  │  └─ ❌ 读取失败: {file_name} | {str(e)}")

        # === 4. 文档为空时的友好提示 ===
        if not all_docs:
            error_msg = (
                "❌ 未发现任何有效文档！请检查：\n"
                f"1. 目录结构是否符合要求:\n"
                f"   {base_path}/\n"
                f"   └── [1_开头目录]/ → 必须匹配KNOWLEDGE_BASE_CONFIG前缀\n"
                f"       └── [规则子目录]/ → 任意命名\n"
                f"           └── *.txt/*.md/*.pdf\n"
                f"2. 文件格式是否为 .txt/.md/.pdf"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # === 5. 文本切片 ===
        logger.info(f"✅ 加载 {len(all_docs)} 个文档，开始切片...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(all_docs)
        logger.info(f"✂️ 切片完成，生成 {len(chunks)} 个文本块")

        return chunks

    def _get_kb_name(self, dir_name: str) -> str:
        """根据目录名匹配知识库配置（精确匹配前缀）"""
        for prefix, kb_name in KNOWLEDGE_BASE_CONFIG.items():
            if dir_name.startswith(prefix):
                return kb_name
        return None