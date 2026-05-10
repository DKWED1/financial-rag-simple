# 诊断脚本：检查知识库状态
# 使用方法: python scripts/diagnose_kb.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.vector_store import VectorStoreManager
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_config


def check_db_counts():
    """方法2: 检查每个库的文档数量"""
    print("=" * 50)
    print("【方法2】检查向量库文档数量")
    print("=" * 50)

    embedding = DashScopeEmbeddings(model=rag_config['embedding_model_name'])
    vm = VectorStoreManager(embedding)

    for db_name in ["credit_base", "overdue_debt", "risk_control"]:
        db = vm._get_db(db_name)
        if db:
            count = db._collection.count()
            print(f"  {db_name}: {count} 个文档")
        else:
            print(f"  {db_name}: 无法加载")

    print()


def test_retrieval():
    """方法3: 测试检索是否能找到内容"""
    print("=" * 50)
    print("【方法3】测试检索功能")
    print("=" * 50)

    embedding = DashScopeEmbeddings(model=rag_config['embedding_model_name'])
    vm = VectorStoreManager(embedding)

    # 测试案例
    test_queries = [
        ("credit_base", "学生贷款准入条件"),
        ("overdue_debt", "逾期还款怎么处理"),
        ("risk_control", "风控审核标准"),
    ]

    for db_name, query in test_queries:
        print(f"\n--- 测试: 库={db_name}, 查询='{query}' ---")
        retriever_func = vm.get_retriever_with_scores(db_name)
        if not retriever_func:
            print(f"  无法获取检索器")
            continue

        docs = retriever_func(query)
        if not docs:
            print(f"  无检索结果")
            continue

        for i, doc in enumerate(docs):
            score = doc.metadata.get('score', 'N/A')
            content_preview = doc.page_content[:80].replace('\n', ' ')
            source = doc.metadata.get('source', 'unknown')
            print(f"  [{i+1}] score={score:.3f} | 来源: {source}")
            print(f"      内容: {content_preview}...")


def test_keyword_search():
    """额外测试: 用关键词直接搜索"""
    print("\n" + "=" * 50)
    print("【附加】关键词直接搜索测试")
    print("=" * 50)

    embedding = DashScopeEmbeddings(model=rag_config['embedding_model_name'])
    vm = VectorStoreManager(embedding)

    # 测试"学生"关键词
    print("\n--- 查询包含'学生'的文档 ---")
    retriever_func = vm.get_retriever_with_scores("credit_base")
    docs = retriever_func("学生")
    print(f"在 credit_base 库中搜索'学生'，返回 {len(docs)} 个结果:")

    for i, doc in enumerate(docs):
        score = doc.metadata.get('score', 'N/A')
        content = doc.page_content[:100].replace('\n', ' ')
        print(f"  [{i+1}] score={score:.3f}")
        print(f"      {content}...")


if __name__ == "__main__":
    check_db_counts()
    test_retrieval()
    test_keyword_search()
    print("\n诊断完成！")
