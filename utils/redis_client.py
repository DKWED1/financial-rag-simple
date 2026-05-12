import json
import redis
from typing import Optional
from utils.logger import logger
from utils.config_handler import load_config

redis_config = load_config("redis.yml")


class RedisCache:
    """Redis 缓存封装，提供 QA 和路由缓存"""

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self.client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db'],
            password=redis_config.get('password') or None,
            decode_responses=redis_config.get('decode_responses', True),
        )
        self.qa_ttl = redis_config.get('qa_ttl', 3600)
        self.route_ttl = redis_config.get('route_ttl', 1800)
        logger.info(f"[RedisCache] 连接成功: {redis_config['host']}:{redis_config['port']}")

    def get_qa(self, question: str) -> Optional[dict]:
        """命中问答缓存，返回 {"answer", "source_db"}"""
        key = f"qa:{question}"
        val = self.client.get(key)
        if val:
            logger.debug(f"[RedisCache] QA命中: {question[:30]}")
            return json.loads(val)
        return None

    def set_qa(self, question: str, answer: str, source_db: Optional[str]):
        key = f"qa:{question}"
        val = json.dumps({"answer": answer, "source_db": source_db}, ensure_ascii=False)
        self.client.setex(key, self.qa_ttl, val)

    def get_route(self, question: str) -> Optional[str]:
        """命中路由缓存，返回 db_name 或 none"""
        key = f"route:{question}"
        val = self.client.get(key)
        if val:
            logger.debug(f"[RedisCache] Route命中: {question[:30]}")
            return val
        return None

    def set_route(self, question: str, db_name: str):
        key = f"route:{question}"
        self.client.setex(key, self.route_ttl, db_name)

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False


_redis: Optional[RedisCache] = None


def get_redis() -> RedisCache:
    global _redis
    if _redis is None:
        _redis = RedisCache()
    return _redis
