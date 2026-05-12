import yaml
from utils.path_util import CONFIG_DIR


def load_config(config_name: str, encoding="utf-8") -> dict:
    config_path = f"{CONFIG_DIR}/{config_name}"
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_config = load_config("rag.yml")
prompts_config = load_config("prompts.yml")
chroma_config = load_config("chroma.yml")
router_config = load_config("router.yml")
mysql_config = load_config("mysql.yml")

# 从 rag.yml 提取的配置项
KNOWLEDGE_BASE_CONFIG: dict = rag_config["knowledge_base_config"]
VECTOR_DB_ROOT: str = rag_config["vector_db_root"]
CHUNK_SIZE: int = rag_config["chunk_size"]
CHUNK_OVERLAP: int = rag_config["chunk_overlap"]
RELEVANCE_THRESHOLD: float = rag_config["relevance_threshold"]
SEARCH_K: int = rag_config["search_k"]

# 从 router.yml 提取
ROUTING_KEYWORDS: dict = router_config["routing_keywords"]

# 从 prompts.yml 提取
FINAL_ANSWER_PROMPT: str = prompts_config["final_answer_prompt"]
RAG_SYSTEM_PROMPT: str = prompts_config["rag_system_prompt"]
ROUTING_PROMPT: str = prompts_config["routing_prompt"]
