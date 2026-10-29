# RAG 引擎测试报告

> 生成时间：2026-10-29 15:39:08　|　总耗时：72.34 s

- 被测版本：ragcore 1.0.0
- Python：3.13.14（CPython）
- 操作系统：Windows 11 AMD64
- 工作目录：`F:\work-buddyAi\rag`

## 1. 结果总览

| 指标 | 数值 |
| --- | --- |
| 用例总数 | 97 |
| 通过 | 97 |
| 失败 | 0 |
| 错误（非断言异常） | 0 |
| 通过率 | 100.0% |

## 2. 按模块统计

| 模块 | 用例数 | 通过 | 失败 | 错误 | 通过率 |
| --- | --- | --- | --- | --- | --- |
| abstain | 8 | 8 | 0 | 0 | 100.0% |
| chunker | 13 | 13 | 0 | 0 | 100.0% |
| cli | 10 | 10 | 0 | 0 | 100.0% |
| generator | 16 | 16 | 0 | 0 | 100.0% |
| index | 12 | 12 | 0 | 0 | 100.0% |
| loader | 6 | 6 | 0 | 0 | 100.0% |
| retriever | 3 | 3 | 0 | 0 | 100.0% |
| tokenizer | 9 | 9 | 0 | 0 | 100.0% |
| web | 12 | 12 | 0 | 0 | 100.0% |
| 性能 | 4 | 4 | 0 | 0 | 100.0% |
| 文档 | 1 | 1 | 0 | 0 | 100.0% |
| 既有自检 | 1 | 1 | 0 | 0 | 100.0% |
| 测试框架 | 2 | 2 | 0 | 0 | 100.0% |

## 3. 按测试层级

| 层级 | 用例数 | 通过 | 失败 | 错误 | 通过率 |
| --- | --- | --- | --- | --- | --- |
| 单元 | 67 | 67 | 0 | 0 | 100.0% |
| 集成 | 18 | 18 | 0 | 0 | 100.0% |
| 系统 | 7 | 7 | 0 | 0 | 100.0% |
| 非功能 | 5 | 5 | 0 | 0 | 100.0% |

## 4. 按测试设计方法

| 设计方法 | 用例数 | 通过 | 失败 | 错误 |
| --- | --- | --- | --- | --- |
| 等价类划分 | 25 | 25 | 0 | 0 |
| 边界值分析 | 28 | 28 | 0 | 0 |
| 判定表 | 13 | 13 | 0 | 0 |
| 错误推测 | 18 | 18 | 0 | 0 |
| 回归 | 9 | 9 | 0 | 0 |
| 性能 | 4 | 4 | 0 | 0 |

## 5. 需求追溯矩阵

| 需求 ID | 模块 | 需求描述 | 用例 | 结果 |
| --- | --- | --- | --- | --- |
| REQ-TOK-01 | tokenizer | 连续汉字串切分为 bigram；长度为 1 时退化为 unigram（停用字除外） | TC-TOK-001、TC-TOK-002 | 通过 |
| REQ-TOK-02 | tokenizer | 英文小写化并做保守复数归一：`-ies`(长度>4)→`-y`，词尾单个 `-s` 剥离，`-ss` 与长度≤3 不剥离 | TC-TOK-003、TC-TOK-004 | 通过 |
| REQ-TOK-03 | tokenizer | 英文停用词不得进入词元序列 | TC-TOK-005 | 通过 |
| REQ-TOK-04 | tokenizer | 数字（整数、小数、多段版本号）保留为单一词元 | TC-TOK-006 | 通过 |
| REQ-TOK-05 | tokenizer | 两字皆为中文停用字的 bigram 丢弃，其余保留 | TC-TOK-009 | 通过 |
| REQ-TOK-06 | tokenizer | 空串、纯空白、纯标点输入返回空列表 | TC-TOK-007 | 通过 |
| REQ-TOK-07 | tokenizer | `tokenize_unique` 去重且保持首次出现顺序 | TC-TOK-008 | 通过 |
| REQ-CHK-01 | chunker | ATX 标题按层级维护标题栈，路径以 ` › ` 连接 | TC-CHK-001、TC-CHK-002 | 通过 |
| REQ-CHK-02 | chunker | 围栏代码块（```/~~~）内的 `#` 行不得被解析为标题 | TC-CHK-003、TC-CHK-004 | 通过 |
| REQ-CHK-03 | chunker | 按空行切段；未超长的段落整体进入同一块 | TC-CHK-005 | 通过 |
| REQ-CHK-04 | chunker | 块长受 `max_chars` 约束；超长段落先按句末标点切分，必要时硬切 | TC-CHK-006、TC-CHK-007 | 通过 |
| REQ-CHK-05 | chunker | `overlap` 携带上一块尾部，且被 clamp 到 `max_chars // 2` | TC-CHK-008、TC-CHK-009 | 通过 |
| REQ-CHK-06 | chunker | `max_chars <= 0` 抛 `ValueError` | TC-CHK-010 | 通过 |
| REQ-CHK-07 | chunker | 尾部短于 `min_chars` 的块，在同 doc 同 heading 时并入前一块 | TC-CHK-011 | 通过 |
| REQ-CHK-08 | chunker | `start_line` 记录正文起始行，同一文档内单调不减 | TC-CHK-012 | 通过 |
| REQ-CHK-09 | chunker | 空文档或纯空白文档返回空列表 | TC-CHK-013 | 通过 |
| REQ-IDX-01 | index | `build` 为片段分配连续 id，并生成 postings / df / doc_len / avgdl / idf | TC-IDX-001 | 通过 |
| REQ-IDX-02 | index | IDF 采用 `ln(1+(N-df+0.5)/(df+0.5))`，且恒为正 | TC-IDX-002 | 通过 |
| REQ-IDX-03 | index | 索引文本 = 标题路径 + 正文，仅标题命中也能获得基础分 | TC-IDX-003 | 通过 |
| REQ-IDX-04 | index | 排序键为 `(-score, chunk_id)`，同分时按 chunk_id 升序 | TC-IDX-004 | 通过 |
| REQ-IDX-05 | index | 空查询或空索引返回空列表 | TC-IDX-005 | 通过 |
| REQ-IDX-06 | index | `candidate_k` 限制候选集；`<= 0` 表示不截断 | TC-IDX-006 | 通过 |
| REQ-IDX-07 | index | `max_per_doc` 限制单文档命中数；不足 `top_k` 时用溢出结果回填 | TC-IDX-007 | 通过 |
| REQ-IDX-08 | index | `min_rel_score` 按相对最高分过滤尾部结果 | TC-IDX-008 | 通过 |
| REQ-IDX-09 | index | `save` / `load` 往返后 chunks、idf、检索结果一致 | TC-IDX-009 | 通过 |
| REQ-IDX-10 | index | 索引文件缺失、损坏或版本不符时 `load` 返回 `None` | TC-IDX-010 | 通过 |
| REQ-IDX-11 | index | 空索引的 `avgdl` 为 0 且不触发除零 | TC-IDX-011 | 通过 |
| REQ-IDX-12 | index | 检索结果与进程哈希种子无关（排序键完整，不依赖 `set()` 迭代顺序） | TC-REG-001 | 通过 |
| REQ-RET-01 | retriever | `dedupe` 以正文前 60 个非空白字符为指纹去重，保留高分者 | TC-RET-001 | 通过 |
| REQ-RET-02 | retriever | `retrieve` 的 `top_k` 参数可覆盖构造时的默认值 | TC-RET-002 | 通过 |
| REQ-RET-03 | retriever | `idf` 属性透传索引的 IDF 表 | TC-RET-003 | 通过 |
| REQ-ABS-01 | abstain | 查询无有效词项时直接拒答 | TC-ABS-001 | 通过 |
| REQ-ABS-02 | abstain | OOV 词项按语料最大 IDF 计权 | TC-ABS-002 | 通过 |
| REQ-ABS-03 | abstain | 未提供 `postings` 时 `idf_cov` 退化为 1.0（不参与否决） | TC-ABS-003 | 通过 |
| REQ-ABS-04 | abstain | 三项阈值任一不满足即拒答 | TC-ABS-004、TC-REG-002 | 通过 |
| REQ-ABS-05 | abstain | `thresholds` 参数可覆盖模块默认阈值 | TC-ABS-005、TC-REG-004 | 通过 |
| REQ-ABS-06 | abstain | `vocab_cov` / `idf_cov` / `norm_score` 的计算口径 | TC-ABS-006 | 通过 |
| REQ-GEN-01 | generator | `format_context` 输出带编号、来源、标题路径、行号 | TC-GEN-001 | 通过 |
| REQ-GEN-02 | generator | `format_context` 超长时截断并标注「已截断」；余量不足 200 字时直接停止 | TC-GEN-002 | 通过 |
| REQ-GEN-03 | generator | 图片语法 `![]()` 必须在切句前移除 | TC-GEN-003 | 通过 |
| REQ-GEN-04 | generator | 表格行转逗号文本；分隔行与短表头行丢弃 | TC-GEN-004 | 通过 |
| REQ-GEN-05 | generator | Markdown 链接、双向链接、加粗、行内代码、列表符号被清洗 | TC-GEN-005 | 通过 |
| REQ-GEN-06 | generator | 无命中 / 无词项 / 无重叠时返回统一拒答文案 | TC-GEN-006 | 通过 |
| REQ-GEN-07 | generator | `max_sentences` 与 `max_chars` 约束抽取结果规模 | TC-GEN-007 | 通过 |
| REQ-GEN-08 | generator | `available` 规则：远程需密钥，本机/内网地址可免密钥 | TC-GEN-008 | 通过 |
| REQ-GEN-09 | generator | `_endpoint` 按 `api_base` 形态补全 `/chat/completions` | TC-GEN-009 | 通过 |
| REQ-GEN-10 | generator | 非 JSON、结构不符、HTTP 错误、网络不可达均抛 `LLMError` | TC-GEN-010、TC-ROB-004、TC-ROB-005、TC-ROB-006、TC-ROB-007、TC-ROB-008 | 通过 |
| REQ-GEN-11 | generator | ASCII 句点出现在文件名 / 版本号 / 小数中时不得被当作句末标点 | TC-GEN-011 | 通过 |
| REQ-LOD-01 | loader | 扩展名过滤大小写不敏感，带点与不带点均可 | TC-LOD-001、TC-ROB-010 | 通过 |
| REQ-LOD-02 | loader | `exclude` 同时支持子串匹配与文件名精确匹配 | TC-LOD-002 | 通过 |
| REQ-LOD-03 | loader | 跳过隐藏目录与 `SKIP_DIRS` 中的目录 | TC-LOD-003 | 通过 |
| REQ-LOD-04 | loader | 多编码回退读取，换行统一为 `\n` | TC-LOD-004、TC-ROB-009 | 通过 |
| REQ-CLI-01 | cli | `deep_merge` 递归合并嵌套字典，标量覆盖 | TC-CLI-001、TC-CLI-002、TC-CLI-003 | 通过 |
| REQ-CLI-02 | cli | 环境变量覆盖 `llm` 配置；数值转换失败则跳过该变量 | TC-CLI-004、TC-CLI-005 | 通过 |
| REQ-CLI-03 | cli | CLI 端到端链路：`ingest` 建库 → `search` 检索 → `ask` 回答 → `stats` 统计 | TC-INT-001 | 通过 |
| REQ-CLI-04 | cli | 增量 `ingest` 复用未变更文档的缓存片段，文档删除后索引同步移除 | TC-INT-002 | 通过 |
| REQ-CLI-05 | cli | 索引缺失或未发现任何文档时返回退出码 2，并给出可操作提示 | TC-ROB-001、TC-ROB-002 | 通过 |
| REQ-CLI-06 | cli | 配置文件损坏时回退默认配置并输出警告，不中断执行 | TC-ROB-003 | 通过 |
| REQ-WEB-01 | web | 路由与状态码契约：`/` 200、`/favicon.ico` 204、未知路径 404、`GET /api/ask` 405 | TC-INT-003、TC-INT-004、TC-ROB-012 | 通过 |
| REQ-WEB-02 | web | `/api/ask` 请求校验：JSON 对象、`question` 非空且不超长、`top_k` 范围、`no_llm` 布尔 | TC-INT-005 | 通过 |
| REQ-WEB-03 | web | 密钥只以掩码形式出现，响应体不得包含明文 | TC-INT-007 | 通过 |
| REQ-WEB-04 | web | 含控制字符的 `api_base` / `model` / `api_key` 一律 400 | TC-INT-008 | 通过 |
| REQ-WEB-05 | web | `/api/ask` 与 CLI 的回答必须一致（不得分叉） | TC-INT-006 | 通过 |
| REQ-WEB-06 | web | 请求体非 JSON 或超过 `MAX_BODY_BYTES` 时返回 400 | TC-INT-009 | 通过 |
| REQ-WEB-07 | web | 单个请求异常不得打挂服务 | TC-ROB-011、TC-ROB-013 | 通过 |
| REQ-WEB-08 | web | `/api/llm` 更新配置后可读回，且 `key_set` 状态正确 | TC-INT-010 | 通过 |
| REQ-WEB-09 | web | 页面自包含：`check_html` 拦截外部资源与 `innerHTML` | TC-INT-011 | 通过 |
| REQ-META-01 | 测试框架 | 执行器对 PASS / FAIL / ERROR 的判定正确，断言助手不会"永远通过" | TC-META-001 | 通过 |
| REQ-META-02 | 测试框架 | 全部需求至少被一个用例覆盖，用例引用的需求编号必须存在 | TC-META-002 | 通过 |
| REQ-META-03 | 既有自检 | 既有自检脚本仍全部通过，新增测试不得破坏既有自检 | TC-META-003 | 通过 |
| REQ-DOC-01 | 文档 | `README.md` 的指标表与 `benchmark_report.json` 逐项一致 | TC-REG-003 | 通过 |
| REQ-PERF-01 | 性能 | 内存建库吞吐：2000 个片段在 10 秒内完成 | TC-PERF-001 | 通过 |
| REQ-PERF-02 | 性能 | 单次检索平均延迟在小语料上低于 5 ms（1000 次查询） | TC-PERF-002 | 通过 |
| REQ-PERF-03 | 性能 | 索引 JSON 序列化 + 反序列化 5000 片段在 15 秒内完成 | TC-PERF-003 | 通过 |
| REQ-PERF-04 | 性能 | 检索耗时不随语料规模线性放大（倒排索引应只扫命中词项的倒排表） | TC-PERF-004 | 通过 |

需求覆盖率：**75/75 = 100.0%**

## 6. 失败与错误明细

无。全部用例通过。

## 7. 用例清单

| 用例 ID | 需求 | 标题 | 层级 | 设计方法 | 预期 | 结果 | 耗时(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TC-TOK-001 | REQ-TOK-01 | 四字汉字串切分为 3 个 bigram | 单元 | 等价类划分 | tokenize("云雀笔记") == ["云雀", "雀笔", "笔记"] | PASS | 0.0 |
| TC-TOK-002 | REQ-TOK-01 | 单字汉字退化为 unigram，停用字除外 | 单元 | 边界值分析 | tokenize("税") == ["税"]；tokenize("的") == [] | PASS | 0.0 |
| TC-TOK-003 | REQ-TOK-02 | 英文标识符统一小写 | 单元 | 等价类划分 | tokenize("Markdown") == ["markdown"] | PASS | 0.0 |
| TC-TOK-004 | REQ-TOK-02 | 复数归一覆盖 -ies / -s / -ss / 短词四个边界 | 单元 | 边界值分析 | databases→database；class 不变；studies→study；ties→tie | PASS | 0.0 |
| TC-TOK-005 | REQ-TOK-03 | 英文停用词被过滤，同句实词保留 | 单元 | 判定表 | tokenize("the cat is on the mat") 只含 cat / mat | PASS | 0.0 |
| TC-TOK-006 | REQ-TOK-04 | 小数与多段版本号保留为单一词元 | 单元 | 边界值分析 | tokenize("3.14") == ["3.14"]；"2024.10.26" 不被拆分 | PASS | 0.0 |
| TC-TOK-009 | REQ-TOK-05 | 双停用字 bigram 丢弃，其余保留 | 单元 | 判定表 | tokenize("是在") == []；tokenize("数据的") 保留两个 bigram | PASS | 0.0 |
| TC-TOK-007 | REQ-TOK-06 | 空串 / 纯空白 / 纯标点均返回空列表 | 单元 | 等价类划分 | 三种输入都返回 []，不抛异常 | PASS | 0.0 |
| TC-TOK-008 | REQ-TOK-07 | tokenize_unique 去重且保持首次出现顺序 | 单元 | 等价类划分 | tokenize_unique("数据 数据 分析") == ["数据", "分析"] | PASS | 0.0 |
| TC-CHK-001 | REQ-CHK-01 | 标题栈生成 A › B 形式的层级路径 | 单元 | 等价类划分 | 两级标题产生 heading='A › B' | PASS | 0.1 |
| TC-CHK-002 | REQ-CHK-01 | 同级标题不叠加，栈正确回退 | 单元 | 边界值分析 | ## B 之后的 ## C 路径为 'A › C'，而非 'A › B › C' | PASS | 0.0 |
| TC-CHK-003 | REQ-CHK-02 | 围栏代码块内的 # 行不被解析为标题 | 单元 | 判定表 | ``` 与 ~~~ 内的注释都不进入 heading | PASS | 0.1 |
| TC-CHK-004 | REQ-CHK-02 | 未闭合围栏吞掉后续标题（状态机不退出） | 单元 | 边界值分析 | ``` 未闭合时，其后的 # B 不被当作标题 | PASS | 0.0 |
| TC-CHK-005 | REQ-CHK-03 | 短文档产生单块且两段均完整保留 | 单元 | 等价类划分 | 块数 == 1，且同时包含两段正文 | PASS | 0.0 |
| TC-CHK-006 | REQ-CHK-04 | 块长不超过 max_chars（含重叠时的额外开销） | 单元 | 边界值分析 | overlap=0 时每块 ≤ 200；overlap=100 时每块 ≤ 302 | PASS | 0.4 |
| TC-CHK-007 | REQ-CHK-04 | 无标点超长段落被硬切且不丢字符 | 单元 | 边界值分析 | 切分后各片段长度 ≤ max_chars，拼接后字符总数不变 | PASS | 0.1 |
| TC-CHK-008 | REQ-CHK-05 | overlap 超过 max_chars//2 时被 clamp | 单元 | 边界值分析 | overlap=1000 与 overlap=max_chars//2 产生完全相同的分块 | PASS | 0.3 |
| TC-CHK-009 | REQ-CHK-05 | overlap=0 时不携带上一块尾部 | 单元 | 边界值分析 | 第二块文本中不含双换行拼接的残留前缀 | PASS | 0.1 |
| TC-CHK-010 | REQ-CHK-06 | max_chars<=0 抛 ValueError，=1 仍可工作 | 单元 | 边界值分析 | 0 与 -1 抛 ValueError；1 正常返回非空块 | PASS | 0.0 |
| TC-CHK-011 | REQ-CHK-07 | 尾部短块在同标题下并入前一块 | 单元 | 边界值分析 | 两块的文档合并为一块，且内容完整保留 | PASS | 0.0 |
| TC-CHK-012 | REQ-CHK-08 | start_line 指向正文起始行且单调不减 | 单元 | 等价类划分 | 首块 start_line == 3（标题下一行的正文行）；逐块非递减 | PASS | 0.1 |
| TC-CHK-013 | REQ-CHK-09 | 空文档与纯空白文档返回空列表 | 单元 | 边界值分析 | chunk_document 返回 []，不抛异常 | PASS | 0.0 |
| TC-LOD-001 | REQ-LOD-01 | 扩展名过滤大小写不敏感且点号可省略 | 单元 | 等价类划分 | [".md",".txt"] 与 ["md","txt"] 结果一致，且 B.MD 被识别 | PASS | 2825.4 |
| TC-LOD-002 | REQ-LOD-02 | exclude 支持子串匹配与文件名精确匹配 | 单元 | 判定表 | exclude=['template'] 排除 template.md，保留 note.md | PASS | 2200.5 |
| TC-LOD-003 | REQ-LOD-03 | 隐藏目录与依赖目录中的文档不被发现 | 单元 | 判定表 | .hidden/、node_modules/、.git/ 内文件被跳过，仅 ok.md 被发现 | PASS | 2744.8 |
| TC-LOD-004 | REQ-LOD-04 | GB18030 文件可解码且 CRLF 被归一为 LF | 单元 | 错误推测 | 中文内容正确还原，换行全部为 \n | PASS | 1765.7 |
| TC-IDX-001 | REQ-IDX-01 | build 分配连续 id 并生成倒排统计 | 单元 | 等价类划分 | id 为 0..2；df 键与 postings 键一致；doc_len 长度 == 片段数 | PASS | 0.3 |
| TC-IDX-002 | REQ-IDX-02 | IDF 符合 BM25+ 形式且 df=N 时仍为正 | 单元 | 边界值分析 | idf(alpha)=ln(1.142857)≈0.1335；idf(beta)=ln(2.666667)≈0.9808；全部 > 0 | PASS | 0.1 |
| TC-IDX-003 | REQ-IDX-03 | 正文不含查询词、仅标题命中时仍被召回 | 单元 | 判定表 | 查询「退款」只召回标题为「退款政策」的片段 | PASS | 0.1 |
| TC-IDX-004 | REQ-IDX-04 | 得分相同时按 chunk_id 升序，结果确定 | 单元 | 边界值分析 | 两个内容完全相同的片段按 0、1 顺序返回 | PASS | 0.1 |
| TC-IDX-005 | REQ-IDX-05 | 空查询与空索引都返回空列表 | 单元 | 边界值分析 | search('')、search('。')、空索引 search('alpha') 均为 [] | PASS | 0.1 |
| TC-IDX-006 | REQ-IDX-06 | candidate_k 限制候选集，<=0 表示不截断 | 单元 | 边界值分析 | candidate_k=1 时只返回 1 条；candidate_k=0 时返回全部 5 条 | PASS | 0.3 |
| TC-IDX-007 | REQ-IDX-07 | max_per_doc 限流，不足 top_k 时用溢出结果回填 | 单元 | 判定表 | top_k=2 时两文档各 1 条；top_k=4 时回填到 4 条 | PASS | 0.4 |
| TC-IDX-008 | REQ-IDX-08 | min_rel_score 按相对最高分过滤尾部结果 | 单元 | 边界值分析 | 阈值 0.9 时低分片段被剔除，剩余分数均 >= 最高分×0.9 | PASS | 0.8 |
| TC-IDX-009 | REQ-IDX-09 | save/load 往返后片段、IDF 与检索结果一致 | 单元 | 等价类划分 | chunk_id 序列一致，IDF 逐项误差 < 1e-9 | PASS | 1942.5 |
| TC-IDX-010 | REQ-IDX-10 | 索引缺失、损坏、版本不符时 load 返回 None | 单元 | 错误推测 | 三种情形都返回 None 且不抛异常 | PASS | 3211.6 |
| TC-IDX-011 | REQ-IDX-11 | 空索引的 avgdl 为 0 且不触发除零 | 单元 | 边界值分析 | build([]) 后 avgdl == 0.0，stats 各项为 0 | PASS | 0.0 |
| TC-RET-001 | REQ-RET-01 | 去重指纹取前 60 个非空白字符 | 单元 | 边界值分析 | 前 60 字相同则去重；第 60 字起不同则两条都保留 | PASS | 0.1 |
| TC-RET-002 | REQ-RET-02 | retrieve 的 top_k 参数覆盖构造默认值 | 单元 | 边界值分析 | 默认 top_k=2，调用时传 4 返回 4 条 | PASS | 0.2 |
| TC-RET-003 | REQ-RET-03 | Retriever.idf 与索引 IDF 表为同一对象 | 单元 | 等价类划分 | r.idf is idx.idf | PASS | 0.1 |
| TC-ABS-001 | REQ-ABS-01 | 查询无有效词项时直接拒答 | 单元 | 边界值分析 | 空串与纯标点都返回 True | PASS | 0.0 |
| TC-ABS-002 | REQ-ABS-02 | OOV 词项按语料最大 IDF 计权 | 单元 | 等价类划分 | 词表内甲=1.0、OOV 丙=3.0 → total_idf=4.0，vocab_cov=0.5 | PASS | 0.0 |
| TC-ABS-003 | REQ-ABS-03 | 未提供 postings 时 idf_cov 退化为 1.0 且不否决 | 单元 | 判定表 | postings=None → idf_cov=1.0、不拒答；postings={} → idf_cov=0.0、拒答 | PASS | 0.1 |
| TC-ABS-004 | REQ-ABS-04 | 三项阈值任一不满足即拒答（判定表） | 单元 | 判定表 | vocab_cov 不足→拒答；idf_cov 不足→拒答；两者充足→不拒答；norm_score 不足→拒答 | PASS | 0.1 |
| TC-ABS-005 | REQ-ABS-05 | thresholds 参数可覆盖默认阈值并翻转判定 | 单元 | 边界值分析 | vocab_cov 阈值抬到 1.01 后，同一输入由「不拒答」变为「拒答」 | PASS | 0.0 |
| TC-ABS-006 | REQ-ABS-06 | 三个特征在构造数据上取到精确值 | 单元 | 等价类划分 | vocab_cov=1.0、idf_cov=1.0、norm_score=0.5、total_idf=1.0 | PASS | 0.0 |
| TC-GEN-001 | REQ-GEN-01 | format_context 输出编号、来源、标题与行号 | 单元 | 等价类划分 | 首行 == "[1] 来源：a.md › H（约第 10 行）" | PASS | 0.0 |
| TC-GEN-002 | REQ-GEN-02 | 超长时截断并标注，余量不足 200 字则直接停止 | 单元 | 边界值分析 | 余量 628 字时输出「已截断」；余量 178 字时静默停止 | PASS | 0.0 |
| TC-GEN-003 | REQ-GEN-03 | 图片语法被整体移除，不残留路径与残片 | 单元 | 回归 | 答案含「成品色泽红亮」，且不含 jpg / 000 / ![ / 孤立感叹号 | PASS | 0.5 |
| TC-GEN-004 | REQ-GEN-04 | 表格四类行的去留（判定表） | 单元 | 判定表 | 数据行转逗号文本；分隔行丢弃；短表头行丢弃；两列行保留 | PASS | 0.2 |
| TC-GEN-005 | REQ-GEN-05 | 链接 / 双链 / 加粗 / 代码 / 列表符号被清洗 | 单元 | 等价类划分 | 清洗后只剩可朗读文本 | PASS | 0.1 |
| TC-GEN-006 | REQ-GEN-06 | 无命中 / 无词项 / 无重叠返回统一拒答文案 | 单元 | 等价类划分 | 三者都等于「根据现有资料无法回答该问题。」 | PASS | 0.1 |
| TC-GEN-007 | REQ-GEN-07 | max_sentences 与 max_chars 约束抽取规模 | 单元 | 边界值分析 | max_sentences=3 → 3 行；max_chars=20 → 1 行 | PASS | 0.6 |
| TC-GEN-008 | REQ-GEN-08 | 远程需密钥、本机与内网可免密钥（判定表） | 单元 | 判定表 | 远程无密钥→False；远程有密钥→True；127.0.0.1 与 192.168.x 免密钥→True | PASS | 0.3 |
| TC-GEN-009 | REQ-GEN-09 | api_base 三种形态都补全为 chat/completions | 单元 | 等价类划分 | /v1 结尾补 /chat/completions；裸域名补 /v1/chat/completions；已完整则不变 | PASS | 0.0 |
| TC-GEN-010 | REQ-GEN-10 | 未配置时 chat 抛 LLMError 而非发起请求 | 单元 | 错误推测 | LLMError 信息包含「未配置」 | PASS | 0.1 |
| TC-GEN-011 | REQ-GEN-11 | ASCII 句点只在后接空白或串尾时断句 | 单元 | 回归 | `.txt` / `3.14` / `v2.3.0` 所在句子完整；答案不再只剩「纯文本批量导入：.」 | PASS | 0.2 |
| TC-CLI-001 | REQ-CLI-01 | deep_merge 递归合并嵌套字典，标量覆盖 | 单元 | 等价类划分 | 嵌套键保留未覆盖项，同名标量被覆盖，新增键被加入 | PASS | 0.0 |
| TC-CLI-002 | REQ-CLI-01 | deep_merge 不修改入参 | 单元 | 边界值分析 | 合并后 base 与 override 保持原值 | PASS | 0.0 |
| TC-CLI-003 | REQ-CLI-01 | 配置文件只覆盖指定键，其余取默认值 | 单元 | 等价类划分 | 仅 chunk.max_chars 被改，retrieval 与 abstain 保持默认 | PASS | 1644.3 |
| TC-CLI-004 | REQ-CLI-02 | 环境变量覆盖 llm 配置并完成类型转换 | 单元 | 等价类划分 | RAG_TIMEOUT='30' → int 30；RAG_TEMPERATURE='0.7' → float 0.7 | PASS | 0.4 |
| TC-CLI-005 | REQ-CLI-02 | 非法数值与空值的环境变量被忽略 | 单元 | 边界值分析 | RAG_TIMEOUT='abc' 与空字符串都回退默认值，不抛异常 | PASS | 0.2 |
| TC-INT-001 | REQ-CLI-03 | CLI 全链路：ingest → search → ask → stats | 系统 | 等价类划分 | 四个子命令退出码均为 0，检索命中正确文档，统计输出可读 | PASS | 6134.8 |
| TC-INT-002 | REQ-CLI-04 | 增量 ingest 复用缓存并在文档删除后移除片段 | 系统 | 边界值分析 | 第二次运行复用 4 篇；删除 1 篇后提示移除 1 篇、文档数降为 3 | PASS | 6397.6 |
| TC-INT-003 | REQ-WEB-01 | 路由与状态码契约 | 集成 | 判定表 | / →200；/favicon.ico →204；未知路径 →404；GET /api/ask →405 | PASS | 787.3 |
| TC-INT-004 | REQ-WEB-01 | /api/meta 返回索引统计与生效参数 | 集成 | 等价类划分 | 含 index / documents / llm / retrieval / abstain / limits 六组字段 | PASS | 517.7 |
| TC-INT-005 | REQ-WEB-02 | /api/ask 请求校验（判定表六行） | 集成 | 判定表 | 非对象 / 空问题 / 超长问题 / top_k 越界 / 非整数 / 非布尔 一律 400 | PASS | 2603.1 |
| TC-INT-006 | REQ-WEB-05 | /api/ask 与 CLI 的回答完全一致（不得分叉） | 集成 | 回归 | 同一问题、同一 top_k 下两侧 answer 字符串逐字相同 | PASS | 568.0 |
| TC-INT-007 | REQ-WEB-03 | 密钥只以掩码形式出现，响应体不含明文 | 集成 | 错误推测 | POST 后 GET /api/llm 与 /api/meta 的原始响应文本中找不到明文密钥 | PASS | 565.0 |
| TC-INT-008 | REQ-WEB-04 | 含控制字符的配置项一律 400（头注入防护） | 集成 | 错误推测 | api_key / api_base / model 含换行、制表符时均被拒 | PASS | 554.8 |
| TC-INT-009 | REQ-WEB-06 | 请求体非 JSON 或超过 64KB 时返回 400 | 集成 | 边界值分析 | 垃圾字节与 70KB 请求体都返回 400，且服务不崩溃 | PASS | 2577.2 |
| TC-INT-010 | REQ-WEB-08 | /api/llm 更新后可读回，本地地址判定正确 | 集成 | 等价类划分 | 保存 api_base/model 后 GET 反映新值，is_local=True | PASS | 582.9 |
| TC-INT-011 | REQ-WEB-09 | 前端页面满足自包含约束 | 集成 | 回归 | check_html 全部检查项通过：无外链、无 innerHTML、关键控件齐备 | PASS | 10.5 |
| TC-ROB-001 | REQ-CLI-05 | 未发现任何文档时 ingest 返回退出码 2 | 系统 | 错误推测 | 退出码 2，提示「未在 ... 下发现任何文档」 | PASS | 2758.8 |
| TC-ROB-002 | REQ-CLI-05 | 索引缺失时 ask / search 返回退出码 2 | 系统 | 错误推测 | 退出码 2，提示「请先执行：python rag.py ingest」 | PASS | 5413.3 |
| TC-ROB-003 | REQ-CLI-06 | 配置文件损坏时回退默认配置并给出警告 | 系统 | 错误推测 | 退出码 0，stderr 含「警告」，生效配置为默认值 | PASS | 2613.6 |
| TC-ROB-004 | REQ-GEN-10 | 服务端返回 5xx 时抛 LLMError 且带上状态码 | 集成 | 错误推测 | LLMError 信息含「HTTP 500」 | PASS | 586.2 |
| TC-ROB-005 | REQ-GEN-10 | 响应不是 JSON 时抛 LLMError | 集成 | 错误推测 | LLMError 信息含「响应解析失败」 | PASS | 512.6 |
| TC-ROB-006 | REQ-GEN-10 | JSON 结构不符（choices 为空）时抛 LLMError | 集成 | 错误推测 | LLMError 信息含「响应解析失败」 | PASS | 517.5 |
| TC-ROB-007 | REQ-GEN-10 | 端口不可达时抛 LLMError 而非阻塞或泄漏原始异常 | 集成 | 错误推测 | 直连不可达端口 → LLMError，信息含「网络错误」 | PASS | 2951.9 |
| TC-ROB-008 | REQ-GEN-10 | 正常响应能取到 content | 集成 | 等价类划分 | chat 返回 choices[0].message.content 的去空白结果 | PASS | 580.1 |
| TC-ROB-009 | REQ-LOD-04 | 二进制文件不会让读取崩溃 | 单元 | 错误推测 | read_text 返回 str（latin-1 兜底），不抛异常 | PASS | 1835.9 |
| TC-ROB-010 | REQ-LOD-01 | 含空格与中文的路径可正常发现与读取 | 单元 | 错误推测 | 文件名含中文、目录名含空格时仍能发现并读到内容 | PASS | 2002.7 |
| TC-ROB-011 | REQ-WEB-07 | 并发 20 个提问请求全部成功且服务不崩溃 | 集成 | 错误推测 | 20 个并发请求均返回 200，且响应内容非空 | PASS | 3125.7 |
| TC-ROB-012 | REQ-WEB-01 | 未实现的 HTTP 方法返回 501 而不是崩溃 | 集成 | 错误推测 | PUT /api/ask → 501，随后 GET / 仍为 200 | PASS | 519.3 |
| TC-ROB-013 | REQ-WEB-07 | 答案生成抛异常时返回 500 且服务存活 | 集成 | 错误推测 | 注入故障后请求返回 500，恢复正常后请求返回 200 | PASS | 2620.5 |
| TC-REG-001 | REQ-IDX-12 | 检索排序在三个哈希种子下完全一致 | 非功能 | 回归 | PYTHONHASHSEED=0/12345/999 三次子进程输出的 chunk_id 序列逐项相同 | PASS | 1620.4 |
| TC-REG-002 | REQ-ABS-04 | 领域内问题不拒答、领域外问题拒答（方向回归） | 集成 | 回归 | 「如何导入已有笔记」不拒答；「量子退相干的持续时间」拒答 | PASS | 0.2 |
| TC-REG-003 | REQ-DOC-01 | README 指标表与 benchmark_report.json 一致 | 系统 | 回归 | 本方案 / 纯 Okapi BM25 / TF-IDF 余弦 / 关键词命中计数 四行的四个指标全部匹配 | PASS | 12.0 |
| TC-REG-004 | REQ-ABS-05 | 生产阈值与标定最优阈值保持分离 | 单元 | 回归 | DEFAULT_CONFIG.abstain == {vocab_cov:0.32, idf_cov:0.18, norm_score:0.0} | PASS | 0.0 |
| TC-PERF-001 | REQ-PERF-01 | 2000 个片段的建库在 10 秒内完成 | 非功能 | 性能 | build 耗时 < 10 s | PASS | 51.9 |
| TC-PERF-002 | REQ-PERF-02 | 小语料上 1000 次查询平均延迟低于 5 ms | 非功能 | 性能 | 平均单次检索 < 5 ms | PASS | 20.7 |
| TC-PERF-003 | REQ-PERF-03 | 5000 片段的索引保存与加载在 15 秒内完成 | 非功能 | 性能 | save + load 往返 < 15 s，且片段数一致 | PASS | 1179.6 |
| TC-PERF-004 | REQ-PERF-04 | 检索耗时不随语料规模线性放大 | 非功能 | 性能 | 5000 片段上的单次检索耗时 < 50 片段上的 50 倍 | PASS | 137.3 |
| TC-META-001 | REQ-META-01 | 执行器与断言助手通过反向注入验证 | 单元 | 错误推测 | 注入的 PASS/FAIL/ERROR 三种探针都被正确判定，断言助手能识别失效 | PASS | 3.4 |
| TC-META-002 | REQ-META-02 | 需求覆盖率 100% 且无悬空需求引用 | 单元 | 等价类划分 | SPEC.md 中每条需求至少被一个用例覆盖，所有 req_id 均可在 SPEC.md 中找到 | PASS | 0.7 |
| TC-META-003 | REQ-META-03 | 既有 selftest 与 web selftest 仍然全部通过 | 系统 | 回归 | selftest.py 退出码 0；rag_web.py --selftest 退出码 0 | PASS | 5638.6 |

---

> 本报告由 `tests/run_tests.py` 自动生成，请勿手工编辑。
