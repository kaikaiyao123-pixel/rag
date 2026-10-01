"""
Chapter 8 RAG 系统的非交互式演示脚本。

说明：本文件是复刻教程之外**新增**的辅助脚本。教程原文只提供 main.py 的
交互式问答（依赖 input() 逐条输入，且启动即强校验 MOONSHOT_API_KEY），
在无人值守场景下无法自动验证。本脚本把同一条链路跑成脚本化批量查询：

    查询路由 → 查询重写 → 元数据过滤 → 混合检索(向量+BM25, RRF 重排)
    → 父文档回溯去重 → 生成

用法：
    # 完整链路（需要 MOONSHOT_API_KEY）
    python demo.py

    # 仅检索链路（无需 API Key，跳过全部 LLM 调用）
    python demo.py --retrieval-only

    # 自定义问题
    python demo.py --queries "宫保鸡丁怎么做" "推荐几个简单的素菜"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG  # noqa: E402
from rag_modules import (  # noqa: E402
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)
from zh_tokenizer import zh_preprocess  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DEFAULT_QUERIES = [
    "推荐几个简单的素菜",
    "宫保鸡丁怎么做",
    "红烧肉需要什么食材",
    "有哪些中等难度的荤菜",
]

SEP = "=" * 68


def apply_bm25_mode(retrieval_module, chunks, mode: str):
    """按模式设置 BM25 分支的分词函数。

    教程原文构造 BM25Retriever 时未传 ``preprocess_func``，langchain 默认按空白
    切分，中文查询无法命中任何文档（详见 zh_tokenizer.py）。这里提供两种模式：

      - ``zh``       : 中文友好（二元字组），默认，检索质量正常
      - ``tutorial`` : 教程原版行为，用于复现问题与对照实验
    """
    if mode == "tutorial":
        return retrieval_module
    from langchain_community.retrievers import BM25Retriever

    retrieval_module.bm25_retriever = BM25Retriever.from_documents(
        chunks, k=5, preprocess_func=zh_preprocess)
    return retrieval_module


def build_knowledge_base(config, rebuild: bool = False, bm25_mode: str = "zh"):
    """构建/加载知识库，返回 (data_module, retrieval_module, vectorstore, chunks)"""
    data_module = DataPreparationModule(config.data_path)
    index_module = IndexConstructionModule(
        model_name=config.embedding_model,
        index_save_path=config.index_save_path,
    )

    vectorstore = None if rebuild else index_module.load_index()
    if vectorstore is not None:
        print("[索引] 已加载本地向量索引")
        data_module.load_documents()
        chunks = data_module.chunk_documents()
    else:
        print("[索引] 未找到本地索引，开始构建（首次需下载嵌入模型）...")
        data_module.load_documents()
        t0 = time.time()
        chunks = data_module.chunk_documents()
        print(f"[索引] 分块完成: {len(chunks)} 块，耗时 {time.time() - t0:.1f}s")
        t0 = time.time()
        vectorstore = index_module.build_vector_index(chunks)
        print(f"[索引] 向量化完成，耗时 {time.time() - t0:.1f}s")
        index_module.save_index()
        print(f"[索引] 已保存至 {config.index_save_path}")

    retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
    retrieval_module = apply_bm25_mode(retrieval_module, chunks, bm25_mode)
    return data_module, retrieval_module, vectorstore, chunks


def describe_hits(docs):
    """把检索结果压成一行摘要"""
    parts = []
    for doc in docs:
        name = doc.metadata.get("dish_name", "未知")
        score = doc.metadata.get("rrf_score")
        parts.append(f"{name}({score:.5f})" if isinstance(score, float) else name)
    return ", ".join(parts) if parts else "（无）"


def run_retrieval_only(queries, config, rebuild: bool = False, bm25_mode: str = "zh"):
    """仅跑检索链路，不触碰任何 LLM"""
    data_module, retrieval_module, _, chunks = build_knowledge_base(
        config, rebuild=rebuild, bm25_mode=bm25_mode)

    print(f"\n知识库规模: {len(data_module.documents)} 篇菜谱 / {len(chunks)} 个文本块")
    print(f"模式: 仅检索（不调用 LLM） / BM25 分词: {bm25_mode}\n")

    for query in queries:
        print(SEP)
        print(f"问题: {query}")

        filters = _extract_filters(data_module, query)
        if filters:
            print(f"元数据过滤: {filters}")
            hits = retrieval_module.metadata_filtered_search(query, filters, top_k=config.top_k)
        else:
            hits = retrieval_module.hybrid_search(query, top_k=config.top_k)

        print(f"混合检索命中 {len(hits)} 块: {describe_hits(hits)}")

        parents = data_module.get_parent_documents(hits)
        names = [d.metadata.get("dish_name", "未知") for d in parents]
        print(f"回溯父文档 {len(parents)} 篇: {', '.join(names)}")

        # 列表类回答不依赖 LLM，可在此模式下直接演示
        if any(k in query for k in ("推荐", "有哪些", "几个", "给我")):
            print("列表式回答（纯规则拼接，无需 LLM）:")
            print("  " + "\n  ".join(
                f"{i}. {n}" for i, n in enumerate(names[:3], 1)
            ))
        print()


def _extract_filters(data_module, query: str) -> dict:
    """复刻 main.py 的 _extract_filters_from_query 逻辑"""
    filters = {}
    for cat in data_module.get_supported_categories():
        if cat in query:
            filters["category"] = cat
            break
    for diff in sorted(data_module.get_supported_difficulties(), key=len, reverse=True):
        if diff in query:
            filters["difficulty"] = diff
            break
    return filters


def run_full(queries, config, rebuild: bool = False, bm25_mode: str = "zh"):
    """完整链路：检索 + 真实 LLM 生成"""
    generation_module = GenerationIntegrationModule(
        model_name=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    data_module, retrieval_module, _, chunks = build_knowledge_base(
        config, rebuild=rebuild, bm25_mode=bm25_mode)

    print(f"\n知识库规模: {len(data_module.documents)} 篇菜谱 / {len(chunks)} 个文本块")
    print(f"LLM: {config.llm_model} / BM25 分词: {bm25_mode}\n")

    for query in queries:
        print(SEP)
        print(f"问题: {query}")

        route = generation_module.query_router(query)
        print(f"查询路由: {route}")

        if route == "list":
            rewritten = query
        else:
            rewritten = generation_module.query_rewrite(query)
            if rewritten != query:
                print(f"查询重写: {query} -> {rewritten}")

        filters = _extract_filters(data_module, query)
        if filters:
            print(f"元数据过滤: {filters}")
            hits = retrieval_module.metadata_filtered_search(rewritten, filters, top_k=config.top_k)
        else:
            hits = retrieval_module.hybrid_search(rewritten, top_k=config.top_k)
        print(f"检索命中 {len(hits)} 块: {describe_hits(hits)}")

        if not hits:
            print("回答: 抱歉，没有找到相关的食谱信息。")
            print()
            continue

        parents = data_module.get_parent_documents(hits)
        print(f"父文档 {len(parents)} 篇: " + ", ".join(d.metadata.get("dish_name", "未知") for d in parents))

        t0 = time.time()
        if route == "list":
            answer = generation_module.generate_list_answer(query, parents)
        elif route == "detail":
            answer = generation_module.generate_step_by_step_answer(query, parents)
        else:
            answer = generation_module.generate_basic_answer(query, parents)
        print(f"生成耗时: {time.time() - t0:.2f}s")
        print("回答:")
        print(answer)
        print()


def main():
    parser = argparse.ArgumentParser(description="Chapter 8 RAG 非交互式演示")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="只跑检索链路，跳过所有 LLM 调用（无需 API Key）")
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量索引")
    parser.add_argument("--queries", nargs="*", default=None, help="自定义问题列表")
    parser.add_argument("--bm25", choices=("zh", "tutorial"), default="zh",
                        help="BM25 分词方式：zh=中文二元字组（默认）；tutorial=教程原版按空格切分")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    config = DEFAULT_CONFIG

    if not Path(config.data_path).exists():
        print(f"数据路径不存在: {config.data_path}")
        print("请先克隆 HowToCook 语料到该路径，详见 README。")
        return 2

    if args.retrieval_only:
        run_retrieval_only(queries, config, rebuild=args.rebuild, bm25_mode=args.bm25)
    else:
        run_full(queries, config, rebuild=args.rebuild, bm25_mode=args.bm25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
