#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BM25 分支在中文语料上的有效性量化。

背景
----
教程 `rag_modules/retrieval_optimization.py` 用

    BM25Retriever.from_documents(self.chunks, k=5)

构造稀疏检索分支，没有传 `preprocess_func`，于是使用 langchain 的默认实现
`default_preprocessing_func = lambda text: text.split()`——**按空白字符切分**。

中文没有词间空格，该切分方式会导致两个后果：

1. 文档里一整段中文会被切成**一个超长词元**，例如
   `'这道川味名菜以糊辣荔枝味见长，小酸甜中透着干辣椒与花椒的香气……'`；
2. 查询 `'宫保鸡丁'` 本身也只切出一个词元 `'宫保鸡丁'`，而文档中的词元是
   `'宫保鸡丁的做法'`，两者**不相等**，因此完全无法命中。

为什么端到端指标看不出来
------------------------
本脚本实测：BM25 分支单独使用时中文命中率为 **0%**，但混合检索的 top-3 命中率
仍是 100%——因为 BGE 稠密分支在这个规模（373 篇）上已经足够强，单靠它就能全部
命中。缺陷因此是**潜伏**的，只测「答案有没有出现」会完全漏掉它。

真正被影响的是**上下文纯净度**：RRF 把两路结果交替合并

    vec1(1/61), bm1(1/61), vec2(1/62), bm2(1/62), vec3(1/63), ...

而 `main.py` 会把 top-3 子块对应的**全部父文档**交给 LLM，于是当 bm 分支返回
无关文档时，约 1/3 的上下文槽位被无关菜谱占用：既浪费 token，也可能误导生成。

本脚本同时给出「命中率」和「上下文纯净度」两组指标，并对比三种配置：

    A. 仅向量分支
    B. 教程原版混合检索（BM25 默认按空格切分）
    C. 混合检索 + 中文友好分词（二元字组 + 英文/数字按词）

用法
----
    python benchmark_retrieval.py            # 默认 60 条查询
    python benchmark_retrieval.py --n 120
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.retrievers import BM25Retriever  # noqa: E402

from rag_modules import (  # noqa: E402
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)
from zh_tokenizer import zh_preprocess  # noqa: E402


def build_queries(documents, n: int) -> list[tuple[str, str]]:
    """按菜名构造 (query, 期望菜名) 列表，采样保持确定性。"""
    names = sorted({d.metadata["dish_name"] for d in documents})
    if n >= len(names):
        picked = names
    else:
        step = len(names) / n
        picked = [names[int(i * step)] for i in range(n)]
    return [(f"{name}怎么做", name) for name in picked]


def evaluate(docs_per_query, parents_per_query, queries, top_k):
    """返回 (hit@1, hit@top_k, 上下文纯净度, 失败样例)"""
    hit1 = hittop = 0
    total_parents = irrelevant = 0
    misses = []
    for (query, expected), docs, parents in zip(queries, docs_per_query, parents_per_query):
        got = [d.metadata.get("dish_name") for d in docs]
        if got and got[0] == expected:
            hit1 += 1
        if expected in got:
            hittop += 1
        else:
            misses.append(f"{expected} -> {got[:top_k]}")
        total_parents += len(parents)
        irrelevant += sum(1 for p in parents
                          if p.metadata.get("dish_name") != expected)
    n = len(queries)
    purity = 1 - irrelevant / total_parents if total_parents else 0.0
    return hit1 / n, hittop / n, purity, irrelevant / n, misses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="查询条数")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    data_path = str(Path(__file__).parent / "../../data/C8/cook/dishes")
    if not Path(data_path).is_dir():
        print(f"数据目录不存在: {data_path}")
        return 2

    print("=" * 74)
    print("BM25 中文有效性量化")
    print("=" * 74)

    data = DataPreparationModule(data_path)
    documents = data.load_documents()
    chunks = data.chunk_documents()
    print(f"语料: {len(documents)} 篇菜谱 / {len(chunks)} 个子块")

    queries = build_queries(documents, args.n)
    print(f"查询: {len(queries)} 条（形如「<菜名>怎么做」），top_k={args.top_k}")
    print("判定: 命中=结果中含期望菜名；纯净度=父文档中期望菜谱的占比\n")

    index_mod = IndexConstructionModule(
        model_name="BAAI/bge-small-zh-v1.5",
        index_save_path=str(Path(__file__).parent / "_bench_index"),
    )
    t0 = time.time()
    vectorstore = index_mod.build_vector_index(chunks)
    print(f"向量索引构建完成（{time.time() - t0:.1f}s）\n")

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5})
    bm25_default = BM25Retriever.from_documents(chunks, k=5)
    bm25_zh = BM25Retriever.from_documents(chunks, k=5, preprocess_func=zh_preprocess)

    retr_default = RetrievalOptimizationModule(vectorstore, chunks)
    retr_default.bm25_retriever = bm25_default
    retr_zh = RetrievalOptimizationModule(vectorstore, chunks)
    retr_zh.bm25_retriever = bm25_zh

    print(f"默认分词示例: {bm25_default.preprocess_func('宫保鸡丁')}")
    print(f"中文分词示例: {zh_preprocess('宫保鸡丁')}\n")

    configs = [
        ("A 仅向量分支", lambda q: vector_retriever.invoke(q)[:args.top_k]),
        ("B 教程原版混合(BM25 默认切分)", lambda q: retr_default.hybrid_search(q, args.top_k)),
        ("C 混合 + 中文友好分词", lambda q: retr_zh.hybrid_search(q, args.top_k)),
        ("D BM25 分支单独(默认切分)", lambda q: bm25_default.invoke(q)[:args.top_k]),
        ("E BM25 分支单独(中文分词)", lambda q: bm25_zh.invoke(q)[:args.top_k]),
    ]

    rows = []
    for label, fn in configs:
        t0 = time.time()
        per_query = [fn(q) for q, _ in queries]
        parents = [data.get_parent_documents(docs) for docs in per_query]
        h1, hk, purity, irr, misses = evaluate(per_query, parents, queries, args.top_k)
        rows.append((label, h1, hk, purity, irr, misses, time.time() - t0))

    print("-" * 74)
    print(f"{'配置':<30}{'hit@1':>8}{'hit@3':>8}{'上下文纯净度':>14}{'无关父文档/条':>15}")
    print("-" * 74)
    for label, h1, hk, purity, irr, _, _ in rows:
        print(f"{label:<30}{h1:>7.1%}{hk:>8.1%}{purity:>13.1%}{irr:>14.2f}")
    print("-" * 74)

    for label, _, _, _, _, misses, _ in rows:
        if misses:
            print(f"\n失败样例 · {label}（共 {len(misses)} 条，列出前 3）")
            for m in misses[:3]:
                print(f"    {m}")

    by = {label: (h1, hk, purity, irr) for label, h1, hk, purity, irr, _, _ in rows}
    a = by["A 仅向量分支"]
    b = by["B 教程原版混合(BM25 默认切分)"]
    c = by["C 混合 + 中文友好分词"]
    d = by["D BM25 分支单独(默认切分)"]
    e = by["E BM25 分支单独(中文分词)"]

    print("\n结论（数值均由本次实测得出）")
    print(f"  · BM25 分支单独使用：默认切分 hit@1 = {d[0]:.1%}，即对中文查询完全失效；"
          f"换成中文分词后回升到 {e[0]:.1%}。")
    print(f"  · 缺陷并非无害：端到端 hit@1 从 {a[0]:.1%} 降到 {b[0]:.1%}"
          f"（{(b[0] - a[0]) * 100:+.1f}pt）。")
    print(f"  · 更主要的代价在上下文：每条查询的无关父文档由 {a[3]:.2f} 篇增至 "
          f"{b[3]:.2f} 篇（{(b[3] / a[3] - 1) * 100:+.0f}%），"
          f"纯净度 {a[2]:.1%} → {b[2]:.1%}。")
    print(f"  · 修正分词后 hit@1 回到 {c[0]:.1%}，无关父文档降至 {c[3]:.2f} 篇，"
          f"纯净度回到 {c[2]:.1%}。")

    shutil.rmtree(Path(__file__).parent / "_bench_index", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
