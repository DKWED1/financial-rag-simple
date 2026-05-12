# utils/mysql_client.py
"""MySQL 客户端封装（原生 SQL）

设计说明：
- 使用 pymysql + DBUtils 实现连接池
- 支持原生 SQL 执行，不引入 ORM，降低学习成本
- 提供上下文管理器，自动 commit / rollback
- 所有 SQL 操作记录日志，便于问题排查
"""

import pymysql
import pymysql.cursors
from dbutils.pooled_db import PooledDB
from dbutils.persistent_db import PersistentDB
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple, Generator
from datetime import datetime

from utils.logger import logger
from utils.config_handler import mysql_config


class MySQLClient:
    """MySQL 客户端（原生 SQL）"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """单例模式：全局共享一个连接池"""
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # 从配置读取
        self.host = mysql_config["host"]
        self.port = mysql_config["port"]
        self.database = mysql_config["database"]
        self.user = mysql_config["user"]
        self.password = mysql_config["password"]
        self.charset = mysql_config.get("charset", "utf8mb4")
        self.autocommit = mysql_config.get("autocommit", True)
        self.pool_size = mysql_config.get("pool_size", 5)

        # 连接池
        self._pool: Optional[PooledDB] = None
        self._init_pool()

        logger.info(f"[MySQLClient] 初始化完成: {self.host}:{self.port}/{self.database}")

    def _init_pool(self) -> None:
        """初始化连接池（DBUtils PooledDB）"""
        try:
            self._pool = PooledDB(
                creator=pymysql,
                maxconnections=self.pool_size,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                autocommit=self.autocommit,
                cursorclass=pymysql.cursors.DictCursor,
            )
            logger.info(f"[MySQLClient] 连接池初始化成功，大小: {self.pool_size}")
        except Exception as e:
            logger.error(f"[MySQLClient] 连接池初始化失败: {e}")
            raise

    @contextmanager
    def get_connection(self) -> Generator[pymysql.Connection, None, None]:
        """上下文管理器：获取连接，自动归还"""
        conn = None
        try:
            conn = self._pool.connection()
            yield conn
        except Exception as e:
            logger.error(f"[MySQLClient] 获取连接失败: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def execute(self, sql: str, params: Optional[Tuple] = None) -> int:
        """
        执行 INSERT / UPDATE / DELETE

        Args:
            sql: SQL 语句（占位符用 %s）
            params: 参数元组

        Returns:
            受影响行数
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    affected = cursor.execute(sql, params)
                    conn.commit()
                    logger.debug(f"[MySQLClient] Execute: {sql[:100]}... | params={params} | affected={affected}")
                    return affected
                except Exception as e:
                    conn.rollback()
                    logger.error(f"[MySQLClient] Execute failed: {sql[:100]}... | error={e}")
                    raise

    def fetchone(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict]:
        """
        查询单行

        Args:
            sql: SELECT 语句
            params: 参数元组

        Returns:
            字典或 None
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                result = cursor.fetchone()
                logger.debug(f"[MySQLClient] FetchOne: {sql[:100]}... | result={result is not None}")
                return result

    def fetchall(self, sql: str, params: Optional[Tuple] = None) -> List[Dict]:
        """
        查询多行

        Args:
            sql: SELECT 语句
            params: 参数元组

        Returns:
            字典列表
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                results = cursor.fetchall()
                logger.debug(f"[MySQLClient] FetchAll: {sql[:100]}... | count={len(results)}")
                return results

    def insert(self, sql: str, params: Optional[Tuple] = None) -> int:
        """
        执行 INSERT 并返回自增 ID

        Args:
            sql: INSERT 语句
            params: 参数元组

        Returns:
            自增主键 ID
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(sql, params)
                    conn.commit()
                    last_id = cursor.lastrowid
                    logger.debug(f"[MySQLClient] Insert: {sql[:100]}... | last_id={last_id}")
                    return last_id
                except Exception as e:
                    conn.rollback()
                    logger.error(f"[MySQLClient] Insert failed: {sql[:100]}... | error={e}")
                    raise

    def execute_many(self, sql: str, params_list: List[Tuple]) -> int:
        """
        批量执行（用于大量 INSERT）

        Args:
            sql: SQL 语句
            params_list: 参数列表

        Returns:
            总受影响行数
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    affected = cursor.executemany(sql, params_list)
                    conn.commit()
                    logger.debug(f"[MySQLClient] ExecuteMany: {sql[:100]}... | batch={len(params_list)} | affected={affected}")
                    return affected
                except Exception as e:
                    conn.rollback()
                    logger.error(f"[MySQLClient] ExecuteMany failed: {sql[:100]}... | error={e}")
                    raise

    def ping(self) -> bool:
        """健康检查"""
        try:
            with self.get_connection() as conn:
                conn.ping()
                return True
        except Exception as e:
            logger.error(f"[MySQLClient] Ping failed: {e}")
            return False

    def close(self) -> None:
        """关闭连接池"""
        if self._pool:
            self._pool.close()
            logger.info("[MySQLClient] 连接池已关闭")


# 全局单例（懒加载）
_mysql_client: Optional[MySQLClient] = None


def get_mysql_client() -> MySQLClient:
    """获取 MySQL 客户端单例"""
    global _mysql_client
    if _mysql_client is None:
        _mysql_client = MySQLClient()
    return _mysql_client
