import sys
from pathlib import Path
from utils.path_util import get_abs_path

# 把项目根目录加入路径
sys.path.append(str(Path(__file__).parent.parent))


from utils.file_handler import FileProcessor

# ----------------------
# 测试配置
# ----------------------

TEST_DIR = get_abs_path("data/finance_rag_knowledge")  # 你可以放几个 PDF / TXT / MD 进去
THRESHOLD = 0.92



print("=" * 60)
print("开始测试 FileProcessor 企业级文件处理器")
print("=" * 60)

# 1. 扫描文件
print("\n【步骤1】扫描有效文件")
files = FileProcessor.scan_valid_files(TEST_DIR)
print(f"扫描到：{files}")

# 2. 加载所有文档
print("\n【步骤2】加载所有文档")
all_chunks = []
for f in files:
    chunks = FileProcessor.load_document(f)
    all_chunks.extend(chunks)

print(f"总块数：{len(all_chunks)}")

# 3. 清理空白块
print("\n【步骤3】清理空白块")
all_chunks = FileProcessor.clean_empty_chunks(all_chunks)

# 4. 精确去重
print("\n【步骤4】精确去重")
all_chunks = FileProcessor.exact_dedup(all_chunks)

# 5. 语义去重（需要 embedding）
print("\n【步骤5】语义去重（需要向量模型）")

try:
    # 这里用最常用的最小向量模型测试
    from langchain_community.embeddings import DashScopeEmbeddings

    embedding = DashScopeEmbeddings(model="text-embedding-v4")

    final_chunks = FileProcessor.semantic_dedup(
        chunks=all_chunks,
        embedding_function=embedding,
        threshold=THRESHOLD
    )
    print(f"\n✅ 最终去重完成！最终块数：{len(final_chunks)}")

except Exception as e:
    print(f"向量模型加载失败：{e}")
    print("跳过语义去重，测试仍可继续...")

