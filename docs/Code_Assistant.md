# Code Assistant（local-copilot）

**Agent 标识**：`code-assistant`（默认 agent）
**代码位置**：`src/agents/code_assistant/`

local-copilot 是一个本地代码助手：它把当前工作目录当作自己的代码仓库，可以直接读取、搜索、修改文件并运行命令，同时带有一层安全网关（写/删/命令需人工确认）和路径越界拦截，适合作为可实际落地的本地编程助手底座。

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────┐
│  LangGraph 编排层  src/agents/code_assistant/graph.py  │
│  block_rule_check → context_compress → plan_split      │
│    → (subtask_exec) → llm_infer → tool_exec → ...      │
└──────────────────────────────────────────────────────┘
                │
   ┌────────────┼─────────────┐
   ▼            ▼             ▼
 代码工具     安全网关       记忆/压缩
 code_tools   security.py    memory.py / context_compress.py
```

| 模块 | 职责 |
|------|------|
| `graph.py` | 主图编排：阻断规则 → 上下文压缩 → 任务拆分 → LLM 推理 → 工具执行 |
| `code_tools.py` | 内置文件/命令工具，路径统一限定在 `PROJECT_ROOT` 内 |
| `executor.py` | 工具执行包装：路径参数二次收敛 + 异常归一化为工具友好字符串 |
| `sandbox.py` | 工作区路径解析：`resolve_workspace_path()`，越界抛 `PathEscapesWorkspace` |
| `security.py` | 每次工具调用前的安全网关 |
| `plan.py` | plan-mode 任务拆解与并行子 Agent 执行 |
| `memory.py` | 四类记忆（偏好/反馈/知识/参考）+ 持久拒绝规则 |
| `context_compress.py` | 上下文超限时压缩历史 |

---

## 2. 内置工具

| 工具 | 说明 | 高危(需确认) |
|------|------|:---:|
| `ReadFile` | 读取文件 | |
| `WriteFile` | 覆盖写入文件 | ✅ |
| `EditFile` | 精确替换文件片段 | ✅ |
| `ListDirectory` | 列出目录 | |
| `SearchFiles` | 搜索文件内容 | |
| `RunCommand` | 运行 shell 命令 | ✅ |
| `MoveFile` / `CopyFile` / `DeleteFile` | 移动/复制/删除文件 | 删除 ✅ |
| `WebSearch` | DuckDuckGo 网页搜索 | |
| `Calculator` | 计算器 | |

文件类工具都会把路径强制收敛到 `PROJECT_ROOT` 内，`../` 越界直接拒绝。`HIGH_RISK_TOOLS` 默认包含 `WriteFile / DeleteFile / ShellExec / RunCommand`，这些工具在每次调用前会触发 LangGraph `interrupt()` 人工确认。

---

## 3. 安全网关（`security.py`）

每次工具调用前短路式检查，命中即拒绝：

1. **用户权限** — `permission_policy(user_id, tool_name)` 可注入自定义鉴权；
2. **内容安全** — 工具参数命中 `CONTENT_BLOCKLIST` 关键词即拒绝；
3. **工具白名单** — 不在 `TOOL_WHITELIST` 内的工具拒绝（为空 = 只拦高危）；
4. **高危确认** — 高危工具需 HITL 确认；
5. **路径收敛** — 由代码工具与 executor 强制，越界即 `PathEscapesWorkspace`。

---

## 4. 记忆与持久规则（`memory.py`）

- 四类记忆：`preference`（偏好）、`feedback`（反馈）、`knowledge`（知识）、`reference`（参考）。
- 后台异步抽取（`background_extract_memory`），不阻塞主链路。
- 持久拒绝规则：用户表达"不要做 X" 时记录为 `BlockRule`，后续请求命中即拦截。
- 存储默认内存态，设置 `REDIS_URL` 后持久化（TTL 由 `MEMORY_TTL_SECONDS` 控制）。

---

## 5. 上下文压缩（`context_compress.py`）

会话 token 超过 `MAX_TOKEN_LIMIT` 时，把早期历史总结为摘要并保留最近 `CONTEXT_KEEP_RECENT_ROUNDS` 轮，避免长会话撑爆上下文窗口。

---

## 6. Plan-mode 任务拆解（`plan.py`）

- 触发：请求长度 ≥ `PLAN_MODE_COMPLEXITY_THRESHOLD`，或 `agent_config.plan_mode=true`。
- 拆解为最多 `PLAN_MODE_MAX_SUBTASKS` 个相互独立的子任务，并行由子 Agent 执行（每步最多 `PLAN_MODE_MAX_SUB_STEPS`），结果汇总后由主 Agent 作答。

---

## 7. 配置（`.env`）

```bash
# 代码助手可读写的工作目录，文件访问被限制在该路径内
PROJECT_ROOT=.

# 上下文压缩
MAX_TOKEN_LIMIT=8000
CONTEXT_KEEP_RECENT_ROUNDS=6

# 记忆（留空 = 内存态）
REDIS_URL=

# 高危工具需人工确认（JSON 数组）
# HIGH_RISK_TOOLS=["WriteFile","DeleteFile","ShellExec","RunCommand"]

# 阻断规则语义判断
BLOCK_RULE_SEMANTIC_CHECK=true

# plan-mode
# PLAN_MODE_MAX_SUBTASKS=5
# PLAN_MODE_MAX_SUB_STEPS=6
# PLAN_MODE_COMPLEXITY_THRESHOLD=500
```

---

## 8. API 使用

统一走服务端 `agent_id=code-assistant`（也是 `DEFAULT_AGENT`）：

```bash
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

---

## 9. 测试

```bash
uv run pytest tests/agents/test_code_assistant.py
```
