# local-copilot

[![GitHub License](https://img.shields.io/github/license/cxkjn/local-copilot)](https://github.com/cxkjn/local-copilot/blob/main/LICENSE) [![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fcxkjn%2Flocal-copilot%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/cxkjn/local-copilot/blob/main/pyproject.toml)

local-copilot 是一个基于 [LangGraph](https://langchain-ai.github.io/langgraph/)、FastAPI 与 Streamlit 的本地代码助手服务。它把当前工作目录当作自己的代码仓库，可以读取、搜索、修改文件并运行命令，同时内置安全网关、人工确认（HITL）、路径越界拦截、记忆系统与长会话压缩，适合作为本地编程助手的可落地底座。

项目自带一个可交互的聊天界面（Streamlit），并提供流式/非流式 HTTP API，方便接入你自己的前端或其他工具。

## 功能特性

- **内置代码工具**：读文件、写文件、编辑、移动/复制/删除、列目录、内容搜索、运行 shell 命令，另有网页搜索（DuckDuckGo）与计算器；
- **安全网关**：工具调用前统一做用户鉴权、内容黑名单、工具白名单检查；高危工具（写文件、删除、执行命令等）触发 `interrupt()` 人工确认；文件路径强制收敛在 `PROJECT_ROOT` 内，越界直接拒绝；
- **记忆与持久规则**：偏好 / 反馈 / 知识 / 参考四类记忆，异步后台抽取；用户表达的“不要做 X”会记录为持久拒绝规则（Block Rule）并在后续请求中拦截；默认内存存储，配置 `REDIS_URL` 后可持久化；
- **上下文压缩**：会话超过 `MAX_TOKEN_LIMIT` 时自动把早期历史压缩为摘要，保留最近若干轮原文；
- **Plan-mode 任务拆解**：复杂请求自动拆分为相互独立的子任务并行执行，再汇总结果；
- **MCP 工具扩展**：通过 `MCP_SERVERS_JSON` 接入外部 MCP Server，为 agent 动态扩展工具；
- **完整服务端**：`/invoke`、`/stream`、`/history`、`/threads`、`/feedback`、`/info`、`/health` 与 AG-UI 协议端点，支持 Bearer Token 鉴权、Postgres / MongoDB / SQLite 多种检查点后端，以及 LangSmith / LangFuse 追踪；
- **聊天界面与语音**：Streamlit 网页聊天，可选 OpenAI 语音输入/输出（客户端侧配置）；
- **多种部署方式**：支持 Docker Compose 与本地 uv 虚拟环境两种运行方式。

## 快速开始

### 方式一：本地运行（uv）

至少需要配置一个 LLM API Key（如 `OPENAI_API_KEY`）：

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# 安装依赖（uv sync 会自动创建 .venv）
uv sync --frozen
source .venv/bin/activate

# 启动 agent 服务（默认 http://localhost:8080）
python src/run_service.py
```

另开一个终端启动聊天界面：

```sh
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

浏览器访问 `http://localhost:8501` 即可与 `code-assistant` 对话。

### 方式二：Docker

需要 Docker 与 Docker Compose（>= [v2.23.0](https://docs.docker.com/compose/release-notes/#2230)）：

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

`docker compose watch` 会启动 Postgres、agent 服务与 Streamlit 应用，并在代码变更时自动热更新。API 文档见 `http://localhost:8080/redoc`。

## HTTP API

统一使用 agent 标识 `code-assistant`（也是默认 agent），例如：

```sh
# 流式
curl -N -X POST http://localhost:8080/code-assistant/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"读一下 README.md 并总结","thread_id":"t1","user_id":"u1"}'

# 非流式
curl -X POST http://localhost:8080/code-assistant/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"列出 src 目录结构","thread_id":"t1","user_id":"u1"}'

# 历史 / 会话列表
curl -X POST http://localhost:8080/code-assistant/history \
  -H 'Content-Type: application/json' -d '{"thread_id":"t1","user_id":"u1"}'
curl "http://localhost:8080/code-assistant/threads?user_id=u1"
```

查看可用 agent 与信息：

```sh
curl http://localhost:8080/info
```

如果设置了 `AUTH_SECRET`，所有请求需携带请求头 `Authorization: Bearer <AUTH_SECRET>`。生产环境务必设置。

也可用仓库自带的 Python 客户端（`src/client/client.py`）：

```python
from client import AgentClient

client = AgentClient()
response = client.invoke("请总结 src 目录下有哪些模块")
print(response)
```

完整用法见 `src/run_client.py`。

## Agent 配置

code-assistant 的主要配置项（通过 `.env` 设置，均可选）：

| 配置项 | 说明 |
| --- | --- |
| `PROJECT_ROOT` | 代码助手可读写的工作目录，所有文件访问被限制在该路径内（默认 `.`） |
| `MAX_TOKEN_LIMIT` | 触发上下文压缩的 token 阈值（默认 8000） |
| `CONTEXT_KEEP_RECENT_ROUNDS` | 压缩时保留的最近完整轮数（默认 6） |
| `REDIS_URL` | 设置后记忆与 Block Rule 持久化到 Redis；留空使用内存态 |
| `HIGH_RISK_TOOLS` | 需要人工确认的高危工具 JSON 数组 |
| `TOOL_WHITELIST` | 允许使用的工具 JSON 数组；为空表示除高危工具外全部放行 |
| `CONTENT_BLOCKLIST` | 工具参数中命中的关键词即拒绝（如 `sudo`、`rm -rf`） |
| `BLOCK_RULE_SEMANTIC_CHECK` | 是否用 LLM 语义判断 Block Rule 命中（默认 true） |
| `PLAN_MODE_MAX_SUBTASKS` / `PLAN_MODE_MAX_SUB_STEPS` / `PLAN_MODE_COMPLEXITY_THRESHOLD` | plan-mode 拆解与并行执行上限 |
| `MCP_SERVERS_JSON` | MCP Server 配置（JSON），用于动态扩展工具 |

详细设计、工具清单与模块职责见 [docs/Code_Assistant.md](docs/Code_Assistant.md)，完整环境变量见 [.env.example](.env.example)。

支持多种模型提供商：OpenAI、Azure OpenAI、DeepSeek、Anthropic、Google Gemini、Groq、OpenRouter、AWS Bedrock、Vertex AI、Ollama 以及任意 OpenAI 兼容接口。设置 `DEFAULT_MODEL` 可指定默认模型；未设置时按已配置的提供商自动选择。

## LLM 之外的配置

- 历史会话与记忆的数据库后端：`DATABASE_TYPE=sqlite|postgres|mongo`（默认 SQLite），相关连接参数见 `.env.example`；
- 语音输入/输出：在 Streamlit 客户端配置 `VOICE_STT_PROVIDER=openai` 与 `VOICE_TTS_PROVIDER=openai`（需 `OPENAI_API_KEY`）；
- AG-UI 协议说明见 [docs/AGUI.md](docs/AGUI.md)；
- 本地模型（Ollama）与 Vertex AI 配置见 [docs/Ollama.md](docs/Ollama.md) 与 [docs/VertexAI.md](docs/VertexAI.md)；
- 文件型私有凭证的使用建议见 [docs/File_Based_Credentials.md](docs/File_Based_Credentials.md)（`privatecredentials/` 目录内容默认被 Git 与 Docker 忽略）。

## 目录结构

```text
local-copilot/
├── src/
│   ├── agents/code_assistant/   # code-assistant agent（主图、代码工具、安全网关、记忆、规划）
│   ├── agents/agents.py         # agent 注册表（默认 code-assistant）
│   ├── core/                    # 设置与 LLM 工厂
│   ├── schema/                  # 协议数据模型
│   ├── service/                 # FastAPI 服务（含 AG-UI、线程管理）
│   ├── client/                  # AgentClient 客户端
│   ├── streamlit_app.py         # Streamlit 聊天界面
│   └── run_service.py           # 服务启动入口
├── tests/                       # 单元与集成测试
├── docker/ + compose.yaml       # Docker 部署
├── docs/                        # 文档
└── pyproject.toml
```

## 测试

```sh
uv sync --frozen
pre-commit install
pytest
```

代码助手专项测试：

```sh
uv run pytest tests/agents/test_code_assistant.py
```

## License

本项目基于 MIT License 开源，详见 [LICENSE](LICENSE)。
