#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAGService MySQL 集成测试"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from service.rag_service import RAGService

def test_mysql_integration():
    print("=" * 50)
    print("测试 RAGService MySQL 集成")
    print("=" * 50)

    # 1. 测试 MySQL 连接
    print("\n[1] 测试 MySQL 连接...")
    rag = RAGService()
    if rag.mysql.ping():
        print("    ✓ MySQL 连接成功")
    else:
        print("    ✗ MySQL 连接失败")
        return

    # 2. 测试保存记录
    print("\n[2] 测试保存问答记录...")
    session_id = "test_session_001"
    rag.current_session_id = session_id

    # 模拟保存
    record_id = rag._save_chat_record(
        question="测试问题：逾期的后果是什么？",
        answer="测试回答：逾期会影响征信记录。",
        source_db="overdue_debt",
        relevance_score=0.85,
        route_type="keyword"
    )
    if record_id > 0:
        print(f"    ✓ 保存成功，记录ID: {record_id}")
    else:
        print("    ✗ 保存失败")

    # 3. 测试加载历史
    print("\n[3] 测试加载会话历史...")
    rag._load_session_history(session_id)
    if len(rag.chat_history) > 0:
        print(f"    ✓ 加载成功，共 {len(rag.chat_history)} 条消息")
        for i, msg in enumerate(rag.chat_history):
            print(f"      [{i}] {type(msg).__name__}: {msg.content[:30]}...")
    else:
        print("    ✗ 加载失败或无历史记录")

    # 4. 测试 get_session_history
    print("\n[4] 测试获取会话历史（完整记录）...")
    history = rag.get_session_history(session_id)
    if history:
        print(f"    ✓ 获取到 {len(history)} 条记录")
        for rec in history:
            print(f"      ID:{rec['id']} | Q:{rec['question'][:20]}... | A:{rec['answer'][:15]}...")
    else:
        print("    ✗ 无记录")

    # 5. 测试完整的 ask 流程（需有知识库数据）
    print("\n[5] 测试完整 ask 流程...")
    try:
        # 使用已保存的 session_id
        response = rag.ask("逾期的后果是什么？", session_id=session_id)
        print(f"    ✓ 回答生成成功")
        print(f"      回答: {response[:50]}...")

        # 验证新记录已保存
        new_history = rag.get_session_history(session_id)
        print(f"    ✓ 数据库中现有 {len(new_history)} 条记录")
    except Exception as e:
        print(f"    ✗ ask 失败: {e}")

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_mysql_integration()