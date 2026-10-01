# Chapter 8 · 食谱 RAG 系统（复刻实现）

对 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) 第 8 章的
完整复刻。教程原意是做一个「今天吃什么」的菜谱问答助手：以
[HowToCook](https://github.com/Anduin2017/HowToCook) 的菜谱为语料，
用「向量检索 + BM25 关键词检索」混合召回，再交给大模型生成回答。

本目录 = **教程代码（逐字节一致）** + **本项目的验证与量化脚本**。

---

## 1. 目录结构

```
code/C8/
├── config.py                      # 教程原文：RAGConfig 数据类
├── main.py                        # 教程原文：交互式问答主程序
├── requirements.txt               # 教程原文：依赖清单
├── rag_modules/                   # 教程原文：四个核心模块
│   ├── __init__.py
│   ├── data_preparation.py        #   数据加载 / 元数据增强 / Markdown 结构分块
│   ├── index_construction.py      #   BGE 嵌入 + FAISS 向量库
│   ├── retrieval_optimization.py  #   向量 + BM25 双路检索 + RRF 融合
│   └── generation_integration.py  #   查询路由 / 重写 / 三种生成模式（LCEL）
│
├── selftest.py                    # 新增：离线自检（35 条断言，无需 API Key）
├── verify_generation.py           # 新增：生成链路端到端验证（本地 mock，无需 Key）
├── demo.py                        # 新增：非交互式演示（可只跑检索链路）
├── benchmark_retrieval.py         # 新增：BM25 中文有效性量化基准
├── zh_tokenizer.py                # 新增：中文友好分词（修复 BM25 缺陷）
├── .env.example                   # 新增：环境变量模板
└── .gitignore                     # 新增
```

### 教程代码一致性

下列 8 个文件与上游仓库**逐字节一致**，未做任何修改（可用文件大小核对）：

| 文件 | 字节数 |
| --- | --- |
| `config.py` | 814 |
| `main.py` | 11440 |
| `requirements.txt` | 297 |
| `rag_modules/__init__.py` | 404 |
| `rag_modules/data_preparation.py` | 12690 |
| `rag_modules/generation_integration.py` | 12275 |
| `rag_modules/index_construction.py` | 4156 |
| `rag_modules/retrieval_optimization.py` | 5276 |

新增脚本全部是**独立文件**，不侵入教程代码路径。教程原版的缺陷（见第 5 节）
以「可复现的对照实验 + 可选修复」的方式处理，而不是直接改掉教程源码。

---

## 2. 技术栈与版本

教程 `requirements.txt` 只对 langchain 系列做了精确 pin，其余用 `>=`。
实际解析结果如下（Python 3.13.12 / Windows）：

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| langchain | 0.3.26 | 教程 pin |
| langchain-core | 0.3.86 | 传递依赖 |
| langchain-community | 0.3.27 | 教程 pin |
| langchain-huggingface | 0.3.1 | 教程 pin |
| langchain-text-splitters | 0.3.8 | 教程 pin |
| faiss-cpu | 1.15.1 | 向量索引 |
| torch | 2.14.0+cpu | CPU 版 |
| sentence-transformers | 6.1.0 | 传递依赖，大版本已漂移 |
| transformers | 5.17.0 | 传递依赖，大版本已漂移 |
| huggingface-hub | 1.32.0 | 传递依赖，大版本已漂移 |
| numpy | 2.5.3 | |
| rank_bm25 | 0.2.2 | BM25 底层实现 |
| openai | 1.109.1 | |
| Markdown | 3.8.2 | |
| python-dotenv | 1.2.3 | |

> **注意**：`sentence-transformers` / `transformers` / `huggingface-hub` 都没有上界约束，
> 教程写作时是 transformers 4.x，现在已解析到 **5.x / 6.x / 1.x**。
> 本目录已实测这套组合可用（512 维向量、模长归一化正确），但若上游继续破坏性变更，
> 建议按需加 pin。

---

## 3. 环境搭建

### 3.1 创建虚拟环境

```bash
cd code/C8
python -m venv .venv
```

### 3.2 安装依赖（**不要用清华源**）

```bash
.venv/Scripts/python.exe -m pip install \
  -i https://mirrors.aliyun.com/pypi/simple \
  --timeout 60 --retries 3 --prefer-binary \
  torch "sentence-transformers>=3.0.0"

.venv/Scripts/python.exe -m pip install \
  -i https://mirrors.aliyun.com/pypi/simple \
  "langchain==0.3.26" "langchain-huggingface==0.3.1" \
  "langchain-text-splitters==0.3.8" "langchain-community==0.3.27"

.venv/Scripts/python.exe -m pip install \
  -i https://mirrors.aliyun.com/pypi/simple \
  "faiss-cpu>=1.7.0" "rank_bm25==0.2.2" \
  "openai>=1.86.0,<2.0.0" "python-dotenv>=1.0.0" "Markdown==3.8.2"
```

> **实测坑**：`pypi.tuna.tsinghua.edu.cn`（清华源）对本机 pip 的请求返回
> **HTTP 403**，症状是 `Could not find a version that satisfies the requirement ... (from versions: none)`，
> 极易被误判为「包不存在」。同样的 URL 用 urllib 请求却是 200。
> 换用阿里云镜像后一切正常。若你遇到「明明有包却报 no matching distribution」，
> 先用 `pip download -vvv` 看真实状态码。

> `unstructured==0.18.11` 与 `langchain-unstructured==0.1.6` 虽然在
> `requirements.txt` 里，但**代码从未 import 过它们**（数据准备走的是原生文件读取），
> 因此上面没有安装。它们会引入 pdfminer、pypandoc 等一大串无关依赖。

### 3.3 配置环境变量

```bash
cp .env.example .env
```

`.env` 内容：

```ini
# 生成环节需要（Moonshot / Kimi 平台申请）
MOONSHOT_API_KEY=sk-xxxxxxxx

# 国内下载 BGE 模型走镜像（HuggingFace 官方域名在本机被阻断）
HF_ENDPOINT=https://hf-mirror.com
```

### 3.4 准备语料

```bash
cd data/C8
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Anduin2017/HowToCook.git cook
cd cook
git sparse-checkout set --no-cone '/dishes/**/*.md'
```

稀疏检出只取 Markdown，**不要用 cone 模式**——`dishes/` 下图片很多，
cone 模式会把整个目录拉下来，实测卡住超过 12 分钟。

预期结果：`dishes/` 下 10 个分类、**373 个 `.md`**（约 1.6 MB）。
注意有两种目录形态并存，`dish_name` 取自文件名，两种都能正确抽取：

```
dishes/meat_dish/宫保鸡丁/宫保鸡丁.md     # 带图片的菜谱（201 篇）
dishes/aquatic/咖喱炒蟹.md                # 单文件菜谱（172 篇）
```

---

## 4. 运行方式

### 4.1 离线自检（推荐先跑这个，无需 API Key）

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/Scripts/python.exe selftest.py
```

用 `FakeListChatModel` 替换真实 LLM：LCEL 链的组装、提示词渲染、输出解析、
路由分发**全部真实执行**，只有最后一次模型推理被替换为固定返回。
因此能在零成本、不联网的前提下验证整条链路。

覆盖 8 个板块：数据加载与元数据、Markdown 结构分块与父子关系、FAISS 构建与
持久化往返、RRF 数值正确性（与手算比对）、混合检索与元数据过滤、父子去重、
查询路由与生成链、统计信息。

上游缺陷（5.1 的 BM25 中文失效）以 `[INFO]` 打印而不计入失败，本项目的修复
（中文二元字组分词）则作为正式断言——这样自检能长期保持全绿，同时缺陷仍被
如实记录在案。

### 4.2 检索链路演示（无需 API Key）

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/Scripts/python.exe demo.py --retrieval-only
```

首次运行会构建向量索引（2062 个子块，CPU 约 40 秒），之后从
`./vector_index/` 加载。

### 4.3 完整链路演示（需要 `MOONSHOT_API_KEY`）

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/Scripts/python.exe demo.py
HF_ENDPOINT=https://hf-mirror.com .venv/Scripts/python.exe demo.py \
  --queries "宫保鸡丁怎么做" "推荐几个简单的素菜"
```

> **每次运行都要带 `HF_ENDPOINT`**，即使模型已经缓存。
> `HuggingFaceEmbeddings` 在构造时会向 HF 仓库查询一次元数据，
> 漏掉这个变量就会抛 `httpx.ProxyError: 502 Bad Gateway`
> （实测踩过：模型明明已在本地缓存，仍然报错）。
> 若确定要完全离线，可改设 `HF_HUB_OFFLINE=1` 跳过查询。

`demo.py` 是本项目新增的**非交互式**驱动：教程的 `main.py` 依赖 `input()`
逐条输入，无法在无人值守场景下验证，`demo.py` 把同一条链路跑成脚本化批量查询。

### 4.4 教程原版交互式问答

```bash
.venv/Scripts/python.exe main.py
```

> `main.py` 在构造 `RecipeRAGSystem` 时就会校验 `MOONSHOT_API_KEY`，
> **没有密钥会直接抛异常**，连检索链路都跑不起来。
> 只是想看检索效果的话，用 `demo.py --retrieval-only`。

### 4.5 BM25 有效性量化

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/Scripts/python.exe benchmark_retrieval.py --n 60
```

### 4.6 生成链路端到端验证（无需 API Key）

```bash
.venv/Scripts/python.exe verify_generation.py
```

在本地起一个最小的 OpenAI 兼容服务（`/v1/chat/completions`，支持流式与非流式），
把 `MoonshotChat` 的请求重定向过去，从而真正走通一次 HTTP：

```
查询路由 → 查询重写 → 分步/基础/列表三种生成 → 流式输出（SSE）
```

这一步补上了 `selftest.py` 的盲区——后者用 `FakeListChatModel` 替换了 `gen.llm`，
**从未真正发出过请求**。`MoonshotChat` 继承自 `ChatOpenAI`，其请求构造依赖
openai SDK 版本，只有真的发一次才能确认兼容。

---

## 5. 复刻过程中实测到的问题

### 5.1 BM25 分支对中文完全失效（已量化）

**现象**：查询 `宫保鸡丁`，BM25 分支返回的却是 `黄油煎虾`——与查询毫无关系。
`selftest.py` 会把这个行为作为 `[INFO]` 如实打印出来。

**根因**：教程构造检索器时没有传 `preprocess_func`：

```python
self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=5)
```

于是使用 langchain 默认的 `default_preprocessing_func = lambda text: text.split()`
——**按空白字符切分**。中文没有词间空格：

| | 分词结果 |
| --- | --- |
| 查询 `宫保鸡丁` | `['宫保鸡丁']` |
| 文档块内容 | `['#', '宫保鸡丁的做法', '这道川味名菜以糊辣荔枝味见长，……（整段成一个词元）', ...]` |

查询词元是 `宫保鸡丁`，文档词元是 `宫保鸡丁的做法`，**永远不可能相等**。

**量化**（`benchmark_retrieval.py`，60 条「`<菜名>怎么做`」查询，top_k=3）：

| 配置 | hit@1 | hit@3 | 上下文纯净度 | 无关父文档/条 |
| --- | --- | --- | --- | --- |
| A 仅向量分支 | 100.0% | 100.0% | 57.1% | 0.75 |
| B 教程原版混合（BM25 默认切分） | 98.3% | 100.0% | 43.5% | 1.30 |
| C 混合 + 中文友好分词 | 100.0% | 100.0% | 54.5% | 0.83 |
| D BM25 分支单独（默认切分） | **0.0%** | 0.0% | 0.0% | 1.00 |
| E BM25 分支单独（中文分词） | 91.7% | 98.3% | 43.7% | 1.27 |

关键读数：

* **D = 0%**：BM25 分支单独使用时，对中文查询的召回率是零，**完全失效**。
* **B vs A**：缺陷并非无害。hit@1 从 100% 降到 98.3%（−1.7pt）；
  更明显的代价在上下文——RRF 把两路结果交替合并

  ```
  vec1(1/61), bm1(1/61), vec2(1/62), bm2(1/62), vec3(1/63), ...
  ```

  而 `main.py` 会把 top-3 子块对应的**全部父文档**交给 LLM，
  于是 BM25 返回的无关文档占用了约 1/3 的上下文槽位：
  每条查询的无关父文档从 **0.75 篇增至 1.30 篇（+73%）**，纯净度 57.1% → 43.5%。
  既浪费 token，也可能误导生成。
* **E = 91.7%**：换成中文分词后，BM25 分支单独召回率接近满分。
* **C vs B**：修正分词后 hit@1 回到 100%，无关父文档降到 0.83 篇。

**为什么只测「答案有没有出现」会漏掉这个缺陷**：BGE 稠密分支在 373 篇这个规模上
已经很强，单靠它 hit@3 就是 100%。缺陷是**潜伏**的——它不改变「有没有命中」，
但实实在在地污染了送给 LLM 的上下文。要发现它，必须看**上下文纯净度**这类指标。

**端到端对照**（`demo.py --retrieval-only`，同一组问题，只切换 BM25 分词）：

| 问题 | `--bm25 tutorial`（教程原版） | `--bm25 zh`（修复后） |
| --- | --- | --- |
| 推荐几个简单的素菜 | 清炒花菜、炒青菜 | 清炒花菜、炒青菜 |
| 宫保鸡丁怎么做 | 宫保鸡丁、**黄油煎虾**、宫保鸡丁 | 宫保鸡丁 ×3 |
| 红烧肉需要什么食材 | 简易红烧肉、**黄油煎虾**、徽派红烧肉 | 简易红烧肉、湖南家常红烧肉、徽派红烧肉 |
| 有哪些中等难度的荤菜 | **0 块（完全检索不到）** | 芥末罗氏虾、农家一碗香 |

最后一行是**硬失败**，值得单独说明：`metadata_filtered_search` 先取
`top_k*3 = 9` 个候选再按 category/difficulty 过滤，而 BM25 返回的 5 个无关文档
占掉了候选槽位的一半，过滤后一无所获，用户拿到的是
"抱歉，没有找到相关的食谱信息"。
**这类失败在只看平均命中率时被完全掩盖**——它只在特定查询形态下暴露。

**修复方式**（本项目提供，但不改动教程源码）：

`zh_tokenizer.py` 用**二元字组**替代空格切分：

```
宫保鸡丁  ->  ['宫保', '保鸡', '鸡丁']
```

只要文档里出现同一串汉字就必然产生重叠词元，无需分词词典即可稳定召回。
代价是词表规模约 2 倍、精度略低于 jieba 这类专业分词器。

`demo.py --bm25 zh`（默认）启用修复，`--bm25 tutorial` 复现教程原版行为。

> 上游教程未讨论中文分词问题。如果你的语料是英文，本缺陷不存在。

### 5.2 FAISS 无法写入含非 ASCII 字符的路径（Windows）

**现象**：`save_index()` 抛

```
RuntimeError: Error in faiss::FileIOWriter::FileIOWriter(const char *)
  ... could not open ...\index.faiss for writing: No such file or directory
```

**根因**：FAISS 的 C++ `FileIOWriter` 用 `char*` 接收路径并交给 `fopen`，
Windows 按 ANSI 代码页解码，UTF-8 编码的中文路径字节无法解析。

**对照实验**（`selftest.py` 第 3 节会做信息性探测）：

| 路径 | 结果 |
| --- | --- |
| `F:\work-buddyAi\rag\code\C8\_selftest_index`（全 ASCII） | ✅ 成功 |
| `C:\Users\黄凯斌\AppData\Local\Temp\...`（父目录含中文） | ❌ 失败 |
| `...\code\C8\_faiss_中文`（路径本身含中文） | ❌ 失败 |

**影响**：路径中**任何一段**含非 ASCII 字符都会失败，不只是末级目录。
教程默认的 `index_save_path="./vector_index"` 是相对路径，只要**项目根目录
本身不含中文**就能正常工作；若把项目克隆到 `D:\我的项目\all-in-rag`
这类目录下，索引落盘会直接失败。

**规避**：让项目路径保持纯 ASCII；或把 `index_save_path` 指到纯 ASCII 位置。

### 5.3 镜像源与网络

| 目标 | 结果 | 处置 |
| --- | --- | --- |
| 清华 PyPI 镜像 | pip 请求返回 **403** | 改用阿里云镜像 |
| HuggingFace 官方 | TLS 被阻断（`UNEXPECTED_EOF_WHILE_READING`） | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| hf-mirror.com | 200 正常 | 用它下载 BGE 模型 |

BGE 模型 `BAAI/bge-small-zh-v1.5` 约 95 MB，首次运行自动下载到 HF 缓存。

> **注意**：模型缓存**不能**让你免于设置 `HF_ENDPOINT`。
> `HuggingFaceEmbeddings` 构造时会向 HF 做一次仓库查询，漏设变量就报
> `httpx.ProxyError: 502 Bad Gateway`——即使模型已在本地。
> 这一点很容易在"首次成功后第二次忘了带变量"时踩到。

### 5.4 生成模型名可能已过期

`config.py` 里 `llm_model = "kimi-k2-0711-preview"`。Moonshot 的模型列表会更新，
若调用报模型不存在，改成本地可用的型号即可（如 `moonshot-v1-8k`）。

### 5.5 `MoonshotChat` 忽略所有端点覆盖参数

**现象**：想让 `MoonshotChat` 指向自建或第三方的 OpenAI 兼容端点时，
`base_url=`、`openai_api_base=`、`api_base=` **三个参数全部无效**，
请求仍然发往 `https://api.moonshot.cn/v1`。

**实测**：

```
传 base_url=        -> llm.base_url = https://api.moonshot.cn/v1
传 openai_api_base= -> llm.base_url = https://api.moonshot.cn/v1
```

**根因**：`MoonshotChat(MoonshotCommon, ChatOpenAI)` 的 MRO 中，`MoonshotCommon`
声明了真实字段 `base_url`，而父类 `ChatOpenAI` 的 `openai_api_base` 的**别名恰好
也是 `base_url`**。两个字段发生别名冲突，用户传入的值没有落到
`MoonshotCommon.base_url` 上，`pre_init` 校验器读到的仍是默认值。
其文档字符串却宣称支持 `api_base`，属上游文档与实现不一致。

**影响**：只能对 Moonshot 官方端点使用；想换供应商不能靠改配置。

**可行的重定向方式**：注入预构建的 `openai` client——

```python
import openai
MoonshotChat(
    model=..., moonshot_api_key=key,
    client=openai.OpenAI(api_key=key, base_url=my_url).chat.completions,
    async_client=openai.AsyncOpenAI(api_key=key, base_url=my_url).chat.completions,
)
```

`verify_generation.py` 正是用这个方法把请求接到本地 mock 服务，
从而在**不需要真实 API Key** 的前提下验证了完整的生成链路。

---

## 6. 复刻的架构（教程设计）

```
dishes/**/*.md  (373 篇)
      │
      ▼  ① 数据准备  DataPreparationModule
   load_documents()        rglob 读入，按相对路径 md5 生成确定性 parent_id
   _enhance_metadata()     从路径推分类、从文件名取菜名、正则 ★+ 解析难度
   chunk_documents()       MarkdownHeaderTextSplitter
                           headers=(#, ##, ###)，strip_headers=False
      │                    → 2062 个子块，各自带 parent_id / chunk_id
      ▼  ② 索引构建  IndexConstructionModule
   HuggingFaceEmbeddings("BAAI/bge-small-zh-v1.5")   512 维，归一化
   FAISS.from_documents(chunks)  →  save_local("./vector_index")
      │
      ▼  ③ 检索优化  RetrievalOptimizationModule
   向量检索 top-5  ─┐
                    ├─ RRF 融合  score = Σ 1/(60 + rank + 1)  →  top-k
   BM25 检索 top-5 ─┘
   （可选）metadata_filtered_search：先混合检索 top_k*3，再按 category /
   difficulty 过滤）
      │
      ▼  ④ 生成集成  GenerationIntegrationModule
   query_router()   LLM 分类 → list / detail / general
   query_rewrite()  非 list 类先重写查询
   get_parent_documents()  子块 → 父文档，按命中块数排序去重
   三种模式（均用 LCEL 链）：
     list    → 纯规则拼接菜名（**不调用 LLM**）
     detail  → 分步骤指导
     general → 基础回答
   MoonshotChat(kimi-k2-0711-preview, temperature=0.1, max_tokens=2048)
```

关键设计点：

* **父文档 / 子块分离**：小块用于精准检索，大块（完整菜谱）用于生成，
  兼顾检索精度与上下文完整性。
* **RRF 而非加权求和**：只用排名不用分数，避免向量余弦相似度与 BM25 分数
  量纲不可比的问题。
* **确定性 ID**：父文档 ID 由相对路径 md5 得出（重建索引后仍稳定），
  子块 ID 用 uuid4（每次重建都变）。
* **`list` 路由不调 LLM**：直接从元数据取菜名，省一次调用且结果稳定。

---

## 7. 实测数据速查

| 项 | 值 |
| --- | --- |
| 语料 | 373 篇菜谱 / 10 个分类 |
| 子块数 | 2062 |
| 嵌入维度 | 512 |
| 索引构建耗时 | 约 30–40 秒（CPU） |
| 离线自检 | 35 项断言，**35 通过 / 0 失败** |
| 生成链路验证 | 15 项断言，**15 通过 / 0 失败**（本地 mock，走真实 HTTP） |
| BM25 中文召回（默认切分） | 0.0% |
| BM25 中文召回（二元字组） | 91.7% (hit@1) / 98.3% (hit@3) |

---

## 8. 引用

* 教程：[datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) 第 8 章
* 语料：[Anduin2017/HowToCook](https://github.com/Anduin2017/HowToCook)（MIT）
* 嵌入模型：[BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
