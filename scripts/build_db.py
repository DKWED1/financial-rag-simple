# File: scripts/build_db.py
import sys
import os
from utils.path_util import get_abs_path,get_project_root,DATA_DIR

# ==========================================
# 1. 核心路径修正逻辑 (解决 FileNotFoundError)
# ==========================================
# 获取当前脚本所在的绝对路径 (即 scripts 文件夹)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# 获取项目根目录 (financial-rag-simple)
ROOT_DIR = get_project_root()

# 将项目根目录添加到系统环境变量，确保可以 import service, utils 等模块
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

# 动态构建数据目录的绝对路径
# 这样即使你在别的目录运行脚本，也能准确找到 data 文件夹
# DATA_DIR = os.path.join(ROOT_DIR, "data")
# KNOWLEDGE_DIR = os.path.join(DATA_DIR, "finance_rag_knowledge")

# ==========================================
# 2. 导入项目模块
# ==========================================
from service.knowledge_base import KnowledgeBaseManager
from service.vector_store import VectorStoreManager
from langchain_community.embeddings import DashScopeEmbeddings
from utils.logger import logger
from utils.config_handler import rag_config  # 假设你把模型名等配置放在这里


def main():
    logger.info("=== 开始构建金融风控知识库流程 ===")

    # 1. 检查数据目录是否存在
    if not os.path.exists(DATA_DIR):
        logger.error(f"❌ 错误：找不到知识库源文件夹 -> {DATA_DIR}")
        logger.info(f"💡 请确保在该路径下创建文件夹并放入 PDF/TXT 文件。")
        return

    # 2. 初始化组件
    # 初始化 Embedding 模型 (构建库和检索必须用同一个模型)
    embedding_model = DashScopeEmbeddings(model=rag_config['embedding_model_name'])

    # 初始化知识库管理器
    kb_manager = KnowledgeBaseManager()

    # 初始化向量库管理器
    vector_store_manager = VectorStoreManager(embedding_model)

    # 3. 加载并切片文档
    logger.info(f"正在从 {DATA_DIR} 读取文档...")

    # 直接调用，不传参数
    chunks = kb_manager.build_knowledge_base()

    if not chunks:
        logger.warning("⚠️ 未生成任何文本块，请检查源文件是否为空或格式不支持。")
        return

    logger.info(f"✅ 文档切片完成，共生成 {len(chunks)} 个文本块。")

    # 4. 按分类写入对应的向量库
    logger.info(f"正在按分类写入 3 个向量库...")

    # chunks 已经自带 category 信息，直接循环写入
    for chunk in chunks:
        # print(chunk.metadata)
        category = chunk.metadata["kb_category"]
        vector_store_manager.add_documents(documents=[chunk], category_key=category)

    logger.info("🎉 知识库构建完成！")



if __name__ == "__main__":
    main()