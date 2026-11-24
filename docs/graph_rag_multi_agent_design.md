# Mewtype Graph-Guided RAG 与自适应反证代理设计

版本：v0.4.4-research（2026-11-24）
边界：教学比赛与获授权虚构病例；不用于真实诊疗。本文把结论标为 **[论文支持]**、**[工程设计]** 或 **[研究假设]**。

## 1. 最新相关论文与真实性核查

- **LightRAG，Findings of EMNLP 2025。[论文支持]** 论文提出图与向量结合、低层实体检索和高层主题检索，并支持增量更新。论文确实存在且有开源实现，但其实验并非首次病程生成。[论文页](https://aclanthology.org/2025.findings-emnlp.568/)
- **HippoRAG 2，ICML 2025。[论文支持]** 图含短语/实体与 passage，查询相关短语和 passage 形成 Personalized PageRank 个性化分布，最终按 passage 节点 PageRank 排序；论文报告关联记忆任务相对强 embedding 基线约 7% 提升。该结论不能直接外推到单病例事实忠实性。[论文页](https://proceedings.mlr.press/v267/gutierrez25a.html)；[代码](https://github.com/OSU-NLP-Group/HippoRAG)
- **MedGraphRAG，ACL 2025。[论文支持]** Triple Graph Construction 将用户文档连接到可信医学来源，U-Retrieval 组合 top-down precise retrieval 与 bottom-up response refinement；在 9 个医学问答、2 个事实核查及长文本集合上评估。任务仍不同于带 `source_id` 的首次病程生成。[论文页](https://aclanthology.org/2025.acl-long.1381/)
- **Microsoft GraphRAG。[论文/官方实现支持]** 标准索引会抽取 entity、relationship、claim，做社区发现和分层 community summary，并提供 Local、Global、DRIFT 搜索。官方明确提示索引昂贵，当前仓库以维护为主，不适合在单病例、隔离 CPU 的提交包内直接引入。[官方文档](https://microsoft.github.io/graphrag/)；[代码](https://github.com/microsoft/graphrag)
- **PathRAG。[预印本支持]** 以关系路径组织上下文，用 flow-based pruning 降低冗余，再以 path-based prompt 生成；其优势在多跳语料，论文最初以 arXiv 发布，已有公开代码，但不应把论文基准结论直接当成本项目结果。[论文页](https://arxiv.org/abs/2502.14902)；[代码](https://github.com/BUPT-GAMMA/PathRAG)
- **MEDDxAgent，ACL 2025。[论文支持]** 面向信息逐步获得的交互式鉴别诊断，使用 orchestrator、病史模拟器、知识检索与诊断策略模块，论文报告交互 DDx 准确率提升超过 10%。Mewtype 输入是一次性完整资料，不能照搬其交互收益。[论文页](https://aclanthology.org/2025.acl-long.677/)
- **Layered Debating，NAACL 2025 short。[论文支持]** 对相似疾病进行重复辩论与重评估，在四个诊断数据集上报告收益；它同时引入多轮调用与更大延迟，不适合未知调用限额下的当前提交。[论文页](https://aclanthology.org/2025.naacl-short.46/)
- **MAP，npj Health Systems 2026。[论文支持]** 用 triage、diagnosis、treatment 与 chief agent 模拟住院路径，在 IPDS 51,274 例上评估。其目标是路径决策和标准类别，而非首次病程正文与逐项来源引用；其训练/检索栈也重于本赛题。[论文页](https://www.nature.com/articles/s44401-026-00085-0)
- **Mixed-vendor clinical agents，HeaLing 2026。[论文支持]** 论文指出同一厂商模型可能产生相关错误，混合厂商团队在其诊断基准上更好。比赛固定同一 Qwen，因此新增同模型角色不等价于获得模型多样性。[论文页](https://aclanthology.org/2026.healing-1.1/)

## 2. Dense Top-K 与图方法适用性比较

| 方法 | 节点/边与建图 | 查询、路径与排序 | 真实优势 | 对 Mewtype 的局限 |
|---|---|---|---|---|
| Dense Top-K | 文本片段是独立单元，无显式边 | 查询 embedding 与片段余弦相似度排序 | 简单、低延迟、容易复现 | 容易漏掉与种子证据语义不近但在同一问题链上的时间、更正、反证 |
| LightRAG | LLM 从 chunk 抽 entity/relation，并保存来源片段；低层实体与高层关系分别建向量键 | 查询拆成具体/抽象关键词，检索 entity/relation 后汇集关联 chunk | 跨文档关系与全局主题更完整，支持增量更新 | 单病例只有几十 Facts；再次让 LLM 建开放域关系会增加成本和错误，且临床否定/时态需额外 Schema |
| HippoRAG 2 | OpenIE phrase/triple、passage 节点及连接 | embedding 产生 phrase/passage seeds，PPR 传播，按 passage 节点分数返回 | 适合关联与多跳记忆，PPR 路径可审计 | 单病例图很小且高度连通，PPR 可能趋同；OpenIE 与在线 LLM 会增加调用，论文任务不是医学文书忠实性 |
| MedGraphRAG | 用户文档、医学实体/知识与词典/来源形成三图 | top-down 精确检索，再 bottom-up 补充与精炼 | 明确强调可信医学来源，医学问答与事实核查结果有支持 | 建设、许可和版本治理成本高；“医学知识相关”不能证明“患者事实存在” |
| MS GraphRAG | entity/relation/claim、TextUnit、Leiden 社区和社区报告 | Local 组合实体图与原文，Global 聚合社区报告，DRIFT 在局部与全局间探索 | 大语料的全局归纳与关系检索能力强 | 索引涉及多次 LLM 抽取/总结与较重依赖；对单例提交明显过度 |
| PathRAG | 文档图上的实体/关系与可读关系路径 | query seed → 路径搜索 → flow pruning → 路径排序/提示 | 可减少重复片段并展示关系链 | 若路径边由错误推断生成，会把错误包装得更“可解释”；当前病例图路径短，收益待验证 |

结论：**[工程设计]** 当前不复制完整 GraphRAG 框架，而是在现有 Dense 检索上增加小型 typed graph 重排。**[研究假设]** 它可能提高反证、更正及时序证据的 Recall@K，但必须用 A/B/C 同条件实验验证。

## 3. Mewtype Typed Clinical Evidence Graph

### 3.1 图边界

**[工程设计]** Patient Evidence 与 Clinical Problem 放在同一个“病例内异构图”，因为二者共享 `Fact ID`，且生命周期只有一次病例运行。Medical Knowledge 使用独立、只读、带版本的图，通过显式 bridge edge 连接。理由是：

1. 患者图每例重建，必须严格 case-scoped；知识图跨病例复用。
2. `source_id` 表示患者资料，`knowledge_id` 表示一般知识，分图能减少身份混淆。
3. 正式知识库尚未获准/登记，所以当前代码只实现前两层；不存在时不会假装拥有医学知识。

### 3.2 Python 数据结构

已实现于 `src/clinical_graph.py`：

```python
@dataclass(frozen=True)
class ClinicalNode:
    node_id: str
    node_type: str
    label: str
    attributes: dict

@dataclass(frozen=True)
class ClinicalEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    provenance: str       # deterministic | planner_inference | knowledge_bridge
    evidence_ids: tuple[str, ...]
    confidence: float
```

患者层含 Source、Fact/Symptom/Examination/LaboratoryResult/Exposure、TimePoint；问题层含 ClinicalProblem、DifferentialDiagnosis、MissingEvidence、PlanGoal。当前边含 `derived_from`、`observed_at`、`corrects_record`、`supports`、`contradicts`、`considers`、`discriminates`、`motivates_plan` 与用于检索扩展的 `co_evidence`。

### 3.3 否定、时间、更正与状态

- **[工程设计]** 否定、未知、肯定/不确定是 Fact 的 `assertion` 维度；完成、进行中、计划与未知是独立 `status` 维度，不能压成一个布尔值。
- **[工程设计]** 时间保留原文字段和 TimePoint 边，不自动把记录时间推断为截止点。
- **[工程设计]** 更正关系保存在 Fact 的 `correction` 和 `corrects_record` 边；检索仍保留被更正文本，但兼容性检查和重排对更正优先。
- **[工程设计]** 同一鉴别下，若同一 Fact 同时出现于 supporting/contradicting，两条方向边都不建，并在 `ambiguities` 记录“需复核”；不会由程序擅自选边。

### 3.4 确定性关系与 Qwen 推断边

- 确定性：Source→Fact 来源、字段路径、解析到的时间/数值/单位/否定/状态/更正。
- Qwen 辅助：Fact→Problem 的支持/限制、Problem→Differential、Problem→PlanGoal。
- **[工程设计]** Qwen 关系统一标为 `planner_inference`，可信度只表示产生方式与排序启发，不表示医学正确概率。
- **[研究假设]** 几十 Facts 的病例不值得建设持久化完整知识图；内存邻接表足够。只有固定医学卡片跨病例复用且许可明确时，才值得单独索引知识图。

## 4. Hybrid Graph Retrieval 核心算法

### 4.1 四个实验臂

- A：现有 Dense Top-K。
- B：Dense seeds + 1-hop source expansion。
- C：Dense seeds + weighted Personalized PageRank。
- D：患者图 + 固定医学知识图；本轮只设计，不实现，防止在无授权资料时形成伪知识库。

### 4.2 可解释固定公式

**[工程设计，未训练]** 当前候选分数为：

`score = 0.50*dense + 0.25*graph + 0.15*salience + 0.10*source_priority - 0.20*compatibility_warning_count`

其中余弦从 `[-1,1]` 线性映射到 `[0,1]`；graph 是 1-hop 最大传播分或 PPR 分；salience 来自已有临床字段、异常、数值、否定、时序和更正规则；source priority 只作小权重先验。贪心选择时，同一 `source_id` 后续候选减 0.08 以促进证据多样性。

约束：

1. 否定、时间、状态与更正不靠向量判断；沿用 `retrieval.py` 兼容性警告。
2. 相似度和图边都不能自动写入最终 `source_ids`。
3. 病例间不共享图或向量缓存。
4. planner inference edge 不等于证据，只影响候选定位。

### 4.3 伪代码和接口

```python
class GraphGuidedRetriever:
    def retrieve(self, query: str, top_k: int):
        dense_all = dense.retrieve(query, top_k=len(units))
        seeds = dense_all[:seed_k]
        graph_score = one_hop(seeds) if mode == "one_hop" else ppr(seeds)
        ranked = fixed_weight_score(dense_all, graph_score, metadata)
        return diversity_aware_select(ranked, top_k)
```

`one_hop` 复杂度约为 `O(K·degree + N log N)`；PPR 为 `O(iterations·(V+E) + N log N)`。本地 01 病例离线图为 111 节点、63 边，标准库邻接表足够。真正成本仍主要是 BGE 编码和 Qwen，而非图算法。

超参数初始网格：`seed_k∈{3,5,8}`、`top_k∈{3,5}`、PPR `alpha∈{0.15,0.2,0.3}`、迭代 `{10,20}`。只在开发集选择一次，保留测试集不得调参；报告所有臂的 Recall@K 与事实指标，不只报告最好一次。

## 5. 多 Agent 角色与通信 Schema

| 角色 | 输入 | 输出 | 是否新增调用 |
|---|---|---|---|
| Planner | sources、编号 Facts | 现有 Clinical Plan Schema | 是，已有 |
| Counter-Evidence Reviewer | source目录、原文 Fact 索引、精简 problem 视图 | `reviewed_problem_ids + issues[]` | 仅触发时 1 次 |
| Writer | sources、plan、Problem–Evidence Context、可选反证审查 | 现有 Medical Schema | 是，已有 |
| Verifier | 原子 claims、sources、候选证据 | 现有 Verifier Schema | 可选，已有 |

新增内部 Schema：

```json
{
  "reviewed_problem_ids": ["PR001"],
  "issues": [{
    "problem_id": "PR001",
    "code": "COUNTER_EVIDENCE",
    "fact_ids": ["F012"],
    "note": "该事实限制当前鉴别方向"
  }]
}
```

**[工程设计]** Agent 只传 Fact ID 与结构化意见；完整 problem 覆盖由 Python 强制校验，正确问题不重复输出解释。Writer 仍回到 `fact_catalog` 和原始 sources。反证 Agent 不能修改 plan，冲突也不由“投票”决定。非法 problem/fact 引用由 Python 拒绝；代理失败被记录后继续原 Writer，避免次要增强阻断主流程。

## 6. 自适应调用策略

当前透明评分：问题数≥3 加2；鉴别总数≥4 加2；缺失证据≥3、未知事实≥3、存在更正、至少两个鉴别无反证各加1。总分≥3 才在 `adaptive` 模式触发。

- `off`：0 次附加调用，是默认与比赛冻结候选。
- `adaptive`：简单病例0次，复杂病例最多1次。
- `always`：每例恰好尝试1次，用于对照实验，不建议默认提交。
- 终止条件：一次结构化返回后立即终止；无辩论循环，无自动重试循环。
- 失败回退：记录 stage、格式、解析结果、Token、耗时和错误；忽略该审查并继续 Writer。若总预算不足，由已有 Budget 明确失败，不能静默增加调用。

**[研究假设]** 反证角色可能降低诊断过度确定与遗漏反证。要分辨“角色收益”与“多 Token 收益”，需加一个等 Token 的单 Planner 对照：保持总调用/Token 上限一致，仅改变提示角色和信息传递。

## 7. GraphRAG + Multi-Agent 整合架构

```text
adapter -> Case/sources -> deterministic Facts -> Planner
                               |                 |
                               +--> typed patient/problem graph
                                                   |
Dense seeds --------------------------------> 1-hop/PPR rerank
                                                   |
                      adaptive gate -> optional Counter-Evidence review
                                                   |
                      Problem–Evidence Context -> Writer -> Verifier -> Reviser
```

**[工程设计]** 图检索与 Agent 独立开关；Baseline 完全不经过这些模块。Evidence/Writer 只把图结果当候选，最终来源仍由原始 `source_id`、现有证据策略和数值/否定/时序校验约束。

## 8. 当前代码修改与适配器建议

已实现：

- `src/clinical_graph.py`：病例内异构图、溯源、推断边与歧义记录。
- `src/graph_retrieval.py`：1-hop/PPR 和可解释固定评分。
- `src/agent_orchestrator.py`：复杂度门控、反证 Schema、一次调用与严格引用校验。
- `prompts/counter_evidence_system.txt`：保守反证提示词。
- `tools/inspect_clinical_graph.py`：无需模型的 input→Facts→Graph 审计。
- `src/pipeline.py` / `src/app.py`：默认关闭的整合开关与日志。

转换器建议：

1. 保持 `tools/adapt_practice.py` 的显式 `case_id`/`record_cutoff`，禁止从文件名或记录时间猜截止点。
2. 批量转换使用人工审核 manifest（原路径、case_id、cutoff、输出路径），不做宽松协议猜测。
3. `field_path`、原文、确定性派生 `source_id` 是建图主键；转换后做原文条数、路径和哈希对账。
4. 01 病例作为固定调试例；02–06 在明确 cutoff 后形成开发/保留测试分层；不要用同模板变体同时调参和报最终结果。

## 9. CPU、Token、运行时与部署成本

- 图构建与 PPR：标准库、病例内百级节点，CPU/内存成本可忽略。
- Dense：首次索引编码 `N` 个片段，之后病例内缓存；每个新 query 只编码一次。官方 1024 维 BGE API 参数/配额未知，不能写死并发。
- Agent：触发时增加一次 Qwen 请求；输入通过“sources 一份 + 不重复 fact_text 的 Fact 索引”降低重复。日志分阶段记录 input chars、Token 与耗时。
- 完整 LightRAG/MS GraphRAG/MedGraphRAG：需要 LLM 关系抽取、社区/知识索引和额外资源治理，不适合 11月25日前成为强制依赖。
- 本地 BGE、Neo4j、NetworkX 均不是正式依赖；提交仍为 Python 3.11 标准库加赛方 API。

## 10. 2026-11-25 前的最小可运行版本

建议冻结候选不是“全部开启”，而是：

1. 保持 Baseline 和 v0.4.3 Clinical Pipeline 原语义。
2. 提交包可包含轻量图代码，但默认 `--graph-retrieval off --clinical-agents off`。
3. 如果 11月24 日内完成同条件测试且 C 明显改善、无失败率/延迟恶化，再考虑图 `one_hop`；PPR 与 Agent 仍作为实验开关。
4. 上传前进行一次独立环境复现，核对不含 `information/`、真实响应、凭据、本地权重和缓存。

## 11. 提交后研究版本

- 在许可明确后建立带版本、许可、发布日期和适用范围的医学知识卡片图；任何 `knowledge_id` 不得转写成患者 `source_id`。
- 做 A Dense、B Dense+1-hop、C Dense+PPR、D Patient+Knowledge 的固定测试集消融。
- 做单 Planner 等 Token、Planner+Counter、adaptive Counter 三臂，区分角色与额外计算量。
- 指标：JSON 合规率、关键事实完整率、无依据生成率、否定/时序/计划状态错误率、Recall@K、证据真实支持比例、Qwen/Embedding 调用和总耗时；医学合理性必须由人工/专业审核，不以模型自评代替。
- **[尚未验证]** 只有当图检索对反证/更正证据 Recall@K 和最终证据支持比例均稳定改善，才称为有效；当前实现本身不构成质量提升结论。
