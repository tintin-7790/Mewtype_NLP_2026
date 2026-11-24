# D + Graph Retrieval + Agent 集成验收报告

版本：`0.4.5-graph-agent-validation`

## 1. 代码链路结论

本轮逐段核对了 `clinical_graph.py`、`graph_retrieval.py`、`agent_orchestrator.py`、`pipeline.py`、`retrieval.py`、`model_client.py`、检索配置和离线图工具。结论如下。

1. `--graph-retrieval ppr` 会在 Clinical Planner 成功后构建当前病例图，并实例化 `GraphGuidedRetriever(strategy="ppr")`；不是仅记录 CLI 开关。
2. PPR 的种子来自 `DenseEvidenceRetriever.retrieve()` 的前 `seed_k` 个候选。真实 Dense 质量依赖显式配置的 EmbeddingClient；Graph 本身不生成语义种子。
3. PPR 使用带关系权重和边置信度的加权传播，默认 `alpha=0.2`、20 次迭代。归一化后的 source 图分数以 25% 权重进入最终 `ranking_score`；Dense、显著性、字段优先级和兼容性惩罚同时保留。
4. Clinical Plan 中的 supporting/contradicting Fact 会生成 `supports/contradicts` 边；鉴别、缺失证据、计划目标及共同来源还会形成 `considers/discriminates/motivates_plan/co_evidence`。只有 Source–Fact 溯源边时不再宣称临床图有效，而是显式回退 Dense。
5. 重排后的 candidates 由 `retrieve_problem_evidence()` 形成 Evidence Bundles，并进入 Writer 的 `problem_evidence_context.retrieval_evidence`。
6. 修复前 Counter-Evidence Agent 先于检索运行，看不到候选。现在先执行检索，再把同一组候选的 source、Fact ID、兼容性、Dense 分数和 Graph/PPR 分数以紧凑形式交给 Agent；提示词明确这些分数不是事实证明。
7. D 后处理保持不变：Writer 后仍经过 Evidence Policy、数值/单位修复、结构与来源校验、Compact Verifier、必要时一次 Local Reviser 和最终 Evidence Validation。

## 2. 检索配置与当前可用性

`config/retrieval.local.example.json` 的实际字段为：

| 字段 | 含义 |
|---|---|
| `provider` | `mock`、`local_bge` 或尚未实现协议的 `official` |
| `model` | 已存在的本地 BGE-M3 路径；`local_bge` 必填 |
| `allow_download` | 默认 false，禁止隐式下载 |
| `use_fp16` | 本地推理精度选项 |
| `batch_size` | 本地编码批量 |
| `max_length` | BGE 输入上限 |
| `top_k` | 每条问题查询进入下游的候选数 |
| `min_similarity` | Dense 候选最低余弦相似度 |
| `max_chunk_chars` | 病例来源切片上限 |

当前开发机检查结果：

- `config/retrieval.local.json` 不存在；
- `FlagEmbedding` 未安装；
- 项目目录、Hugging Face 默认缓存和 ModelScope 默认缓存均未发现 BAAI/bge-m3 权重；
- 当前 Codex 进程未继承 `DASHSCOPE_API_KEY/QWEN_BASE_URL/QWEN_MODEL`。

因此本轮没有运行真实 BGE+Qwen，也没有下载模型或安装依赖。`official` Embedding 仍会以 `OFFICIAL_EMBEDDING_PROTOCOL_PENDING` 明确失败，不能用于正式比赛。

如本机之后已有获准、已下载的 BGE-M3 与开发依赖，可复制示例（该文件被 `.gitignore` 排除）并只修改现有路径：

```bash
cp config/retrieval.local.example.json config/retrieval.local.json
# 将 model 改为本机已经存在的 BAAI/bge-m3 目录；保持 allow_download=false
```

程序是进程内 `FlagEmbedding` 适配器，不需要另起 HTTP 服务。没有合法权重或依赖时应停止，不得自动下载、切换公网 API 或把 Mock 结果标成 Dense+BGE。

## 3. 离线 PRACTICE03 PPR 验收

运行：

```bash
python3.11 tools/smoke_graph_retrieval.py \
  --output information/results/03/graph_ppr_smoke.json
```

工具直接读取 PRACTICE03 原始嵌套 JSON，经 Automatic Input Adapter 进入正式 Case/Facts/Graph/Retrieval 代码；使用固定 Mock Clinical Plan 和确定性哈希向量，不调用 Qwen、BGE 或网络。

验收结果：

| 指标 | 结果 |
|---|---:|
| Source Count | 52 |
| Fact Count | 59 |
| Clinical Problem Count | 3 |
| Node Count | 132 |
| Edge Count | 120 |
| Clinical Relation Edge Count | 54 |
| Dense Seed Count（三条查询合计） | 15 |
| Graph Candidate Count（三条查询合计） | 312 |
| PPR Iterations | 20 |
| Query Count | 3 |
| Ranking Changed | true |
| Fallback | false |

边类型为：`derived_from=59`、`observed_at=7`、`supports=25`、`considers=6`、`discriminates=6`、`motivates_plan=3`、`co_evidence=14`。这证明图不只有 Source–Fact 溯源边。

选中来源为：`P0015、P0043、P0039、P0014、P0016、P0042、P0047、P0049、P0046`。重排 Fact 包括：`F014、F015、F046、F050、F016、F017、F018、F019、F020、F049、F054、F053、F056`。完整分查询 PPR 分数和排序审计保存在结果 JSON。

该结果只能证明 Mock-Dense → Clinical Graph → PPR → 重排控制流真实执行，不证明 BGE 的语义召回质量或医学效果。

## 4. 运行日志验收字段

每例 `clinical.graph_retrieval` 现在记录：

- `graph_requested`、`graph_executed`、`graph_mode`；
- `graph_fallback_reason`、`fallback_status`；
- `embedding_provider`；
- `seed_fact_ids`、`candidate_fact_ids`、`reranked_fact_ids`；
- `graph_edge_type_counts`、`selected_source_ids`；
- `graph_elapsed_ms`、`ranking_changed`；
- 每条问题查询的 seed source、PPR source 分数和排序变化。

以下情况不会冒充 Graph 成功：

- CLI 请求 Graph 但未提供 `--retrieval-config`：任务级明确失败；
- 本地 BGE 路径或依赖不存在：配置阶段明确失败；
- 图没有 Clinical Problem 关系边：保留 Dense 结果，记录 `no_clinical_problem_fact_relations`；
- Graph 传播发生异常：保留 Dense 结果，记录 `graph_scoring_failed:<类型>`；
- Planner 降级到 Baseline：记录 `planner_baseline_fallback_bypassed_graph`；
- Dense 自身不可用：不伪造 Graph 结果，运行明确失败。

## 5. 真实 A/B 实验命令

前提：同一个终端已配置同一组百炼变量，且 `config/retrieval.local.json` 指向获准、真实可用的本地 BGE。两条命令除 Graph 开关和输出文件外完全一致。

### A：D + Agent + Dense，Graph off

```bash
python3.11 main.py \
  --input "information/临床智录_练习病历数据集JSON/03_老年病科_多病共存TIA待排.json" \
  --output information/results/03/d_agent_dense_graph_off.json \
  --log information/results/03/d_agent_dense_graph_off_log.json \
  --mode dev_api --pipeline clinical_pipeline \
  --clinical-prompt-profile contrastive \
  --clinical-verification on --clinical-revision on \
  --clinical-agents always \
  --retrieval-config config/retrieval.local.json \
  --graph-retrieval off \
  --clinical-response-format json_schema
```

### B：D + Agent + Dense + PPR

```bash
python3.11 main.py \
  --input "information/临床智录_练习病历数据集JSON/03_老年病科_多病共存TIA待排.json" \
  --output information/results/03/d_agent_dense_ppr.json \
  --log information/results/03/d_agent_dense_ppr_log.json \
  --mode dev_api --pipeline clinical_pipeline \
  --clinical-prompt-profile contrastive \
  --clinical-verification on --clinical-revision on \
  --clinical-agents always \
  --retrieval-config config/retrieval.local.json \
  --graph-retrieval ppr \
  --clinical-response-format json_schema
```

先运行 A，再只运行一次 B。不得把 Mock 配置替换进这两条命令后仍称为真实 Dense 实验。

## 6. A/B 验收方法

自动比较：

- output 的结构状态、evidence source_ids 和 review_items；
- log 的总调用数、Prompt/Completion/Total Token 与总耗时；
- `adaptive_agent.completed`、issue 数和覆盖问题数；
- B 的 `graph_requested=true`、`graph_executed=true`、`graph_mode=ppr`；
- B 的 PPR 分数、重排 Fact/Source、`ranking_changed` 和无 fallback；
- Writer、Verifier、Reviser 的阶段调用与最终 Evidence Validation。

人工比较：

- TIA/脑梗死待排是否保持不确定性；
- 发作特点、神经系统检查、多病共存和用药事实是否遗漏；
- 选中来源是否真实支持正文；
- 新增建议是否仍标为建议而非已执行；
- Graph 是否改善重要来源覆盖，而非只增加 Token 或引入无关事实。

单例 A/B 只能做集成验收，不能证明质量提升。若模型采样或服务端实现无法保证逐字确定，应记录该限制，并在后续固定病例集和人工标注上重复比较。
