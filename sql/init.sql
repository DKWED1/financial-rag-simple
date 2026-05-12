-- ==========================================
-- 金融风控 RAG 系统 - 数据库初始化脚本
-- ==========================================

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS financial_rag
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE financial_rag;

-- ==========================================
-- 1. 对话历史表 (chat_history)
-- ==========================================
-- 用于持久化存储每一轮问答记录
-- 支持多会话隔离、检索来源追踪、相关性评分记录

CREATE TABLE IF NOT EXISTS chat_history (
    id              BIGINT          NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    session_id      VARCHAR(64)     NOT NULL DEFAULT '' COMMENT '会话ID，用于多轮对话隔离',
    question        TEXT            NOT NULL COMMENT '用户问题',
    answer          TEXT            NOT NULL COMMENT 'AI回答',
    source_db       VARCHAR(64)     DEFAULT NULL COMMENT '命中知识库（credit_base/overdue_debt/risk_control）',
    relevance_score DECIMAL(5,4)    DEFAULT NULL COMMENT '检索相关度分数（0~1）',
    route_type      VARCHAR(20)     DEFAULT NULL COMMENT '路由类型（keyword/semantic/global）',
    status          TINYINT         NOT NULL DEFAULT 1 COMMENT '状态：0-删除 1-正常',
    create_time     DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time     DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    PRIMARY KEY (id),
    INDEX idx_session_id (session_id),
    INDEX idx_create_time (create_time),
    INDEX idx_source_db (source_db)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='金融风控RAG问答历史记录表';
