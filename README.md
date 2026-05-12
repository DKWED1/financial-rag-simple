# Financial-RAG-Simple
金融风控垂直领域 RAG 问答系统 | Python + LangChain + Chroma

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **核心定位**：解决金融风控领域规则繁杂、通用模型幻觉严重的问题，提供高准确率、低成本的垂直领域问答能力。

##  项目背景
在金融风控业务中，涉及信贷准入、逾期催收、合规审核等大量非结构化文档。一线业务人员检索效率低，且通用大模型在垂直领域存在严重的“幻觉”问题，无法满足金融级的准确性要求。

本项目独立设计并实现了一套面向金融风控领域的 RAG（检索增强生成）系统，覆盖三大核心业务场景，为业务人员提供结构化的知识检索与精准问答能力。

##  技术栈

| 层级 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **语言** | Python 3.10 | |
| **框架** | LangChain, FastAPI | 核心逻辑编排与 Web 服务 |
| **向量库** | Chroma (HNSW) | 高效向量存储与检索 |
| **模型** | 通义千问 (DashScope API) | 生成回答 |
| **缓存** | Redis | 缓存高频 QA 与路由决策 |
| **持久化** | MySQL | 存储会话历史 |
| **配置** | YAML | 统一管理提示词与参数 |

##  项目架构
本系统采用分层架构设计，确保高内聚、低耦合。
用户提问
│
▼

 路由层：关键词匹配 → LLM 语义兜底 → 全局检索 

│
▼

 检索层：Chroma 向量库（HNSW）→ 相关性阈值过滤

│
▼

 生成层：LangChain Chain（PromptTemplate + LLM） 

│
▼

 缓存层：Redis（QA 缓存）│ MySQL（持久化） │


##  核心实现

### 1. 文档处理管道 (`knowledge_base.py`)
- **多格式支持**：PDF, TXT, MD。
- **智能分片**：300字/块，50字重叠，平衡上下文完整性与检索精度。
- **双层去重**：MD5 精确去重 + 语义去重 (cosine ≥ 0.92)，有效净化向量库，实测去重率约 15%。

### 2. 双层智能路由 (`router_service.py`)
- **架构**：关键词快速匹配优先 + LLM 语义分类兜底。
- **容错机制**：无匹配时自动切换全局检索，解决跨库误检问题。
- **动态降级**：若检索结果相关性分数低于阈值 (0.4)，自动降级处理，避免返回错误答案。

### 3. 缓存体系与成本控制
- **Redis 高频缓存**：缓存 QA 对 (TTL=1h) 和路由决策 (TTL=30min)，命中率 60%，大幅降低 LLM 调用成本。
- **MySQL 持久化**：完整记录对话历史，支持会话隔离与数据回溯分析。

### 4. 工程化与接口
- **MCP 集成**：`mcp_server.py` 封装标准 Tool，支持 Claude Code 等 AI Agent 调用。
- **FastAPI 服务**：提供标准 `/ask` POST 接口。
- **配置化驱动**：所有 Prompt 和参数通过 YAML 管理，无需修改代码即可调整策略。

 ## 测试覆盖
覆盖率：43 个单元测试用例，覆盖配置加载、路由逻辑、缓存读写、向量检索等核心模块。
命令：pytest tests/ -v
## 后续规划
接入 LangGraph 实现多工具调用工作流。
增加 问答溯源，展示原始文档片段。
支持知识库 增量更新，无需全量重建。
 ## 个人总结
独立完成从需求分析到落地的全流程，深入解决了金融场景下 RAG 系统的检索质量与工程化问题。通过配置驱动、模块解耦、单元测试保障，实现了一个可维护、可迭代的垂直领域问答系统，具备大模型应用开发的完整工程能力。

##  项目结构
```text
.
├── config/                 # YAML 配置（模型参数、路由规则、提示词）
├── service/
│   ├── rag_service.py      # 核心 RAG 编排
│   ├── router_service.py   # 双层路由逻辑
│   └── vector_store.py     # 向量库管理
├── utils/
│   ├── mysql_client.py     # MySQL 连接池
│   ├── redis_client.py     # Redis 缓存操作
│   └── config_handler.py   # 配置加载
├── sql/init.sql            # 数据库初始化脚本
├── tests/                  # 单元测试（43 个用例）
├── api.py                  # FastAPI 入口
└── mcp_server.py           # MCP Server 封装



