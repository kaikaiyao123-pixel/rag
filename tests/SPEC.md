# 测试规格说明书（Test Specification）

> 被测对象：零依赖 RAG 检索引擎（`ragcore/`）及其 CLI（`rag.py`）与 Web 层（`rag_web.py`）
> 文档版本：1.0　|　适用代码版本：`ragcore.__version__`
> 入口：`python tests/run_tests.py`

---

## 1. 目的与范围

本文件定义该 RAG 引擎的**测试需求**与**测试设计依据**。所有自动化用例必须通过
`req_id` 字段追溯到下表中的某一条需求，`tests/run_tests.py` 会据此生成追溯矩阵，
并统计需求覆盖率。

### 1.1 被测范围（In Scope）

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 分词器 | `ragcore/tokenizer.py` | 中文 bigram、英文归一、停用词过滤 |
| 分块器 | `ragcore/chunker.py` | Markdown 标题感知分块、重叠窗口 |
| 索引 | `ragcore/index.py` | BM25 打分、倒排表、JSON 持久化 |
| 检索编排 | `ragcore/retriever.py` | 结果整形与去重 |
| 拒答判定 | `ragcore/abstain.py` | 查询内归一化特征与阈值判定 |
| 生成层 | `ragcore/generator.py` | 上下文拼装、离线抽取式回答、LLM 客户端 |
| 文档加载 | `ragcore/loader.py` | 文档发现、多编码读取 |
| 配置 | `ragcore/cli.py` | 默认配置、深合并、环境变量覆盖 |
| Web 层 | `ragcore/web.py` | HTTP 路由、请求校验、密钥脱敏 |

### 1.2 不测范围（Out of Scope）

| 项目 | 理由 |
| --- | --- |
| 检索**效果**（召回率、MRR 等） | 由 `benchmark.py` 的 143 条标注集负责，属评测而非功能测试 |
| 真实大模型返回质量 | 依赖外部服务，不可控；仅测客户端协议层与失败回退 |
| 浏览器端渲染效果 | 无稳定无头浏览器环境；仅测页面自包含约束（`check_html`）与接口契约 |
| `code/C8/` 子项目 | 独立技术路线，另有 `code/C8/selftest.py` |
| 并发性能极限 | 仅做冒烟级并发正确性，不做压测容量结论 |

---

## 2. 测试策略

### 2.1 分层

| 层级 | 目标 | 隔离方式 |
| --- | --- | --- |
| 单元 | 单个函数的输入输出契约 | 纯内存，无 I/O、无网络 |
| 集成 | 模块间协作与外部契约 | 临时目录 + 本机回环 HTTP |
| 系统 | CLI 端到端（建库→检索→回答） | 子进程调用真实入口 |
| 非功能 | 可复现性、性能、安全约束 | 子进程 + 计时 |

### 2.2 测试设计方法

| 方法 | 应用要点 |
| --- | --- |
| 等价类划分 | 按输入域划有效类 / 无效类，每类至少一个用例（如扩展名带点与不带点、本地与远程 api_base） |
| 边界值分析 | 取边界本身、边界 ±1（如 `max_chars = 0 / 1`、`top_k = 0 / 1 / 20 / 21`、指纹长度 60 的两侧） |
| 判定表 | 多条件组合的判定逻辑（拒答三阈值、表格行类型、`available` 规则） |
| 错误推测 | 基于经验构造易错输入（损坏索引、非 UTF-8、控制字符、超长请求体、二进制文件） |
| 回归 | 对已修复缺陷锁定用例，防止复发（每条注明历史缺陷） |
| 性能 | 给宽松上限，只用于发现数量级退化，不作为验收门槛 |

### 2.3 通过准则（Exit Criteria）

1. 全部用例 `PASS`，无 `FAIL`、无 `ERROR`；
2. 需求覆盖率 100%（`SPEC.md` 中每条需求至少被一个用例覆盖）；
3. 框架自验证（`python tests/run_tests.py --verify-harness`）全部通过；
4. `selftest.py` 与 `rag_web.py --selftest` 仍全部通过（不得因新增测试而破坏既有自检）。

---

## 3. 测试需求

> 下表是**需求的唯一来源**。用例通过 `req_id` 引用编号；`tests/run_tests.py` 解析
> 本表生成追溯矩阵，若某条需求没有对应用例，报告会标为「无用例」。

### 3.1 分词器（tokenizer）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-TOK-01 | tokenizer | 连续汉字串切分为 bigram；长度为 1 时退化为 unigram（停用字除外） | 4 字串产生 3 个 bigram；单字产生 1 个 unigram |
| REQ-TOK-02 | tokenizer | 英文小写化并做保守复数归一：`-ies`(长度>4)→`-y`，词尾单个 `-s` 剥离，`-ss` 与长度≤3 不剥离 | `databases`→`database`；`class` 保持 `class` |
| REQ-TOK-03 | tokenizer | 英文停用词不得进入词元序列 | `the` 不出现，同句实词保留 |
| REQ-TOK-04 | tokenizer | 数字（整数、小数、多段版本号）保留为单一词元 | `3.14` 不被拆成 `3` 与 `14` |
| REQ-TOK-05 | tokenizer | 两字皆为中文停用字的 bigram 丢弃，其余保留 | `是在` 产生空列表；`数据的` 两个 bigram 均保留 |
| REQ-TOK-06 | tokenizer | 空串、纯空白、纯标点输入返回空列表 | 均返回 `[]`，不抛异常 |
| REQ-TOK-07 | tokenizer | `tokenize_unique` 去重且保持首次出现顺序 | 重复词元只保留首次位置 |

### 3.2 分块器（chunker）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-CHK-01 | chunker | ATX 标题按层级维护标题栈，路径以 ` › ` 连接 | `# A` + `## B` 产生路径 `A › B` |
| REQ-CHK-02 | chunker | 围栏代码块（```/~~~）内的 `#` 行不得被解析为标题 | 代码块内注释不进标题路径；未闭合围栏吞掉后续标题 |
| REQ-CHK-03 | chunker | 按空行切段；未超长的段落整体进入同一块 | 短文档产生单块且段落不被拆开 |
| REQ-CHK-04 | chunker | 块长受 `max_chars` 约束；超长段落先按句末标点切分，必要时硬切 | 开启重叠时单块长度 ≤ `max_chars + overlap + 2` |
| REQ-CHK-05 | chunker | `overlap` 携带上一块尾部，且被 clamp 到 `max_chars // 2` | 传入 1000 与传入 `max_chars//2` 产生相同的重叠前缀 |
| REQ-CHK-06 | chunker | `max_chars <= 0` 抛 `ValueError` | 0 与 -1 均抛；`max_chars = 1` 仍可工作 |
| REQ-CHK-07 | chunker | 尾部短于 `min_chars` 的块，在同 doc 同 heading 时并入前一块 | 合并后块数减 1，内容完整保留 |
| REQ-CHK-08 | chunker | `start_line` 记录正文起始行，同一文档内单调不减 | 逐块非递减，且首块指向标题下一行 |
| REQ-CHK-09 | chunker | 空文档或纯空白文档返回空列表 | 返回 `[]`，不抛异常 |

### 3.3 索引（index）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-IDX-01 | index | `build` 为片段分配连续 id，并生成 postings / df / doc_len / avgdl / idf | id 为 0..N-1；`df` 与 postings 键一致 |
| REQ-IDX-02 | index | IDF 采用 `ln(1+(N-df+0.5)/(df+0.5))`，且恒为正 | 与手算值误差 < 1e-9；df=N 时仍 > 0 |
| REQ-IDX-03 | index | 索引文本 = 标题路径 + 正文，仅标题命中也能获得基础分 | 正文不含查询词但标题含时仍被召回 |
| REQ-IDX-04 | index | 排序键为 `(-score, chunk_id)`，同分时按 chunk_id 升序 | 内容完全相同的两片段按 id 升序返回 |
| REQ-IDX-05 | index | 空查询或空索引返回空列表 | 均返回 `[]` |
| REQ-IDX-06 | index | `candidate_k` 限制候选集；`<= 0` 表示不截断 | `candidate_k=1` 时结果数 ≤ 1 |
| REQ-IDX-07 | index | `max_per_doc` 限制单文档命中数；不足 `top_k` 时用溢出结果回填 | 限流后各文档计数 ≤ 上限；总数可达 `top_k` |
| REQ-IDX-08 | index | `min_rel_score` 按相对最高分过滤尾部结果 | 过滤后所有分数 ≥ `最高分 × min_rel_score` |
| REQ-IDX-09 | index | `save` / `load` 往返后 chunks、idf、检索结果一致 | 逐字段比对一致 |
| REQ-IDX-10 | index | 索引文件缺失、损坏或版本不符时 `load` 返回 `None` | 三种情形均返回 `None`，不抛异常 |
| REQ-IDX-11 | index | 空索引的 `avgdl` 为 0 且不触发除零 | `build([])` 后 `avgdl == 0.0` |
| REQ-IDX-12 | index | 检索结果与进程哈希种子无关（排序键完整，不依赖 `set()` 迭代顺序） | `PYTHONHASHSEED` 取 0 / 12345 / 999 时 chunk_id 序列逐项相同 |

### 3.4 检索编排（retriever）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-RET-01 | retriever | `dedupe` 以正文前 60 个非空白字符为指纹去重，保留高分者 | 前 60 字相同则去重；第 61 字起不同则保留 |
| REQ-RET-02 | retriever | `retrieve` 的 `top_k` 参数可覆盖构造时的默认值 | 默认 2 时传入 4 返回 4 条 |
| REQ-RET-03 | retriever | `idf` 属性透传索引的 IDF 表 | 与 `index.idf` 为同一对象 |

### 3.5 拒答判定（abstain）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-ABS-01 | abstain | 查询无有效词项时直接拒答 | 空串与纯标点均返回 `True` |
| REQ-ABS-02 | abstain | OOV 词项按语料最大 IDF 计权 | 词表内 1 项 + OOV 1 项时 `total_idf` 等于两权重之和 |
| REQ-ABS-03 | abstain | 未提供 `postings` 时 `idf_cov` 退化为 1.0（不参与否决） | 返回 1.0 且不因该特征拒答 |
| REQ-ABS-04 | abstain | 三项阈值任一不满足即拒答 | 判定表四行全部符合预期 |
| REQ-ABS-05 | abstain | `thresholds` 参数可覆盖模块默认阈值 | 传入更高阈值可把「不拒答」翻转为「拒答」 |
| REQ-ABS-06 | abstain | `vocab_cov` / `idf_cov` / `norm_score` 的计算口径 | 给定构造数据时三个特征取到精确值 |

### 3.6 生成层（generator）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-GEN-01 | generator | `format_context` 输出带编号、来源、标题路径、行号 | 首行形如 `[1] 来源：doc › heading（约第 N 行）` |
| REQ-GEN-02 | generator | `format_context` 超长时截断并标注「已截断」；余量不足 200 字时直接停止 | 有标注 / 无标注两种分支均可复现 |
| REQ-GEN-03 | generator | 图片语法 `![]()` 必须在切句前移除 | 答案中不出现图片路径或 alt 残片 |
| REQ-GEN-04 | generator | 表格行转逗号文本；分隔行与短表头行丢弃 | 四类行各有明确去留 |
| REQ-GEN-05 | generator | Markdown 链接、双向链接、加粗、行内代码、列表符号被清洗 | 清洗后只剩可朗读文本 |
| REQ-GEN-06 | generator | 无命中 / 无词项 / 无重叠时返回统一拒答文案 | 三者文案完全一致 |
| REQ-GEN-07 | generator | `max_sentences` 与 `max_chars` 约束抽取结果规模 | 句子数 ≤ `max_sentences`，总长 ≤ `max_chars`（单句超长除外） |
| REQ-GEN-08 | generator | `available` 规则：远程需密钥，本机/内网地址可免密钥 | 四类组合判定正确 |
| REQ-GEN-09 | generator | `_endpoint` 按 `api_base` 形态补全 `/chat/completions` | 三种形态均得到完整 URL |
| REQ-GEN-10 | generator | 非 JSON、结构不符、HTTP 错误、网络不可达均抛 `LLMError` | 异常类型正确且信息可读 |
| REQ-GEN-11 | generator | ASCII 句点出现在文件名 / 版本号 / 小数中时不得被当作句末标点 | `.txt`、`.md`、`3.14`、`v2.3.0` 所在句子保持完整，不被切成残片 |

### 3.7 文档加载（loader）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-LOD-01 | loader | 扩展名过滤大小写不敏感，带点与不带点均可 | `.MD` 被识别；`["md"]` 与 `[".md"]` 等价 |
| REQ-LOD-02 | loader | `exclude` 同时支持子串匹配与文件名精确匹配 | `template` 命中 `template.md` |
| REQ-LOD-03 | loader | 跳过隐藏目录与 `SKIP_DIRS` 中的目录 | `.git`、`node_modules` 内的文件不被发现 |
| REQ-LOD-04 | loader | 多编码回退读取，换行统一为 `\n` | GB18030 文件正确解码；CRLF 被归一 |

### 3.8 配置（cli）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-CLI-01 | cli | `deep_merge` 递归合并嵌套字典，标量覆盖 | 嵌套键保留未覆盖项；入参不被修改 |
| REQ-CLI-02 | cli | 环境变量覆盖 `llm` 配置；数值转换失败则跳过该变量 | 合法值生效且类型正确；非法值回退默认 |
| REQ-CLI-03 | cli | CLI 端到端链路：`ingest` 建库 → `search` 检索 → `ask` 回答 → `stats` 统计 | 各子命令退出码 0，输出含预期内容 |
| REQ-CLI-04 | cli | 增量 `ingest` 复用未变更文档的缓存片段，文档删除后索引同步移除 | 第二次运行输出「复用 N 篇」；删除文档后片段数下降 |
| REQ-CLI-05 | cli | 索引缺失或未发现任何文档时返回退出码 2，并给出可操作提示 | 退出码 2，提示含「请先执行 ingest」或「未发现任何文档」 |
| REQ-CLI-06 | cli | 配置文件损坏时回退默认配置并输出警告，不中断执行 | 返回默认值，stderr 含「警告」 |

### 3.9 Web 层（web）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-WEB-01 | web | 路由与状态码契约：`/` 200、`/favicon.ico` 204、未知路径 404、`GET /api/ask` 405 | 逐路径断言 |
| REQ-WEB-02 | web | `/api/ask` 请求校验：JSON 对象、`question` 非空且不超长、`top_k` 范围、`no_llm` 布尔 | 六类非法输入均 400 |
| REQ-WEB-03 | web | 密钥只以掩码形式出现，响应体不得包含明文 | 全量响应文本中不出现原始密钥 |
| REQ-WEB-04 | web | 含控制字符的 `api_base` / `model` / `api_key` 一律 400 | 换行注入被拒 |
| REQ-WEB-05 | web | `/api/ask` 与 CLI 的回答必须一致（不得分叉） | 同一问题两边答案字符串完全相同 |
| REQ-WEB-06 | web | 请求体非 JSON 或超过 `MAX_BODY_BYTES` 时返回 400 | 两种情形均 400 |
| REQ-WEB-07 | web | 单个请求异常不得打挂服务 | 异常请求后服务仍能正常响应 |
| REQ-WEB-08 | web | `/api/llm` 更新配置后可读回，且 `key_set` 状态正确 | 保存后 `GET /api/llm` 反映新值 |
| REQ-WEB-09 | web | 页面自包含：`check_html` 拦截外部资源与 `innerHTML` | 真实页面全部检查项通过 |

### 3.10 跨模块（meta）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-META-01 | 测试框架 | 执行器对 PASS / FAIL / ERROR 的判定正确，断言助手不会"永远通过" | 注入式反向验证全部符合预期 |
| REQ-META-02 | 测试框架 | 全部需求至少被一个用例覆盖，用例引用的需求编号必须存在 | 覆盖率 100%，无悬空引用 |
| REQ-META-03 | 既有自检 | 既有自检脚本仍全部通过，新增测试不得破坏既有自检 | `selftest.py` 与 `rag_web.py --selftest` 退出码均为 0 |

### 3.11 文档一致性（doc）

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-DOC-01 | 文档 | `README.md` 的指标表与 `benchmark_report.json` 逐项一致 | 四个方法行 × 四个指标全部匹配，误差 < 5e-4 |

### 3.12 性能（perf）

> 性能用例的阈值刻意放宽。它们**不是**验收门槛，而是"数量级退化探测器"：
> 一旦有人把倒排索引改成全表扫描、或把排序写成 O(n²)，这里会立刻超时。

| 需求 ID | 模块 | 需求描述 | 验收准则 |
| --- | --- | --- | --- |
| REQ-PERF-01 | 性能 | 内存建库吞吐：2000 个片段在 10 秒内完成 | 耗时 < 10 s |
| REQ-PERF-02 | 性能 | 单次检索平均延迟在小语料上低于 5 ms（1000 次查询） | 平均值 < 5 ms |
| REQ-PERF-03 | 性能 | 索引 JSON 序列化 + 反序列化 5000 片段在 15 秒内完成 | 往返耗时 < 15 s |
| REQ-PERF-04 | 性能 | 检索耗时不随语料规模线性放大（倒排索引应只扫命中词项的倒排表） | 5000 片段上的耗时 < 50 片段上的 50 倍 |

---

## 4. 缺陷严重度定义

| 级别 | 判定 | 处理 |
| --- | --- | --- |
| S1 致命 | 数据损坏、崩溃、安全漏洞（密钥泄漏 / 头注入） | 必须修复后才能发布 |
| S2 严重 | 核心功能结果错误（检索错文档、拒答方向反了） | 必须修复 |
| S3 一般 | 边界输入行为不符预期（不崩溃但结果偏差） | 排期修复 |
| S4 轻微 | 文案、格式、非关键路径 | 可接受 |

用例失败时，报告中「详情」一栏给出实际值与预期值，可直接用于提单。

---

## 5. 环境与依赖

- Python 3.9+（开发环境 3.13），**仅标准库**；
- 无需网络：LLM 相关用例使用本机回环上的假服务，或直接断言客户端侧规则；
- 临时目录固定为 `tests/.tmp/`（ASCII 路径），用后即删；
- 运行：`python tests/run_tests.py`（可选 `--verify-harness`、`--junit`、`--filter 关键字`）。
