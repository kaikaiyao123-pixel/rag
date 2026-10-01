# -*- coding: utf-8 -*-
"""命令行入口。

    python rag.py ingest [路径...]   建立 / 增量更新索引
    python rag.py ask "问题"         检索 + 生成回答
    python rag.py search "查询"      仅检索，展示片段与得分
    python rag.py chat               交互式问答
    python rag.py stats              索引统计
    python rag.py docs               列出已索引文档
    python rag.py config             显示生效配置
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .abstain import features as abstain_features
from .abstain import should_abstain
from .chunker import chunk_document
from .generator import (LLMClient, LLMError, build_messages, extractive_answer,
                        format_context)
from .index import BM25Index
from .loader import DEFAULT_EXTENSIONS, discover_documents, read_text, sha1_of
from .retriever import Retriever
from .tokenizer import tokenize

DEFAULT_CONFIG: Dict[str, Any] = {
    "docs": ["docs"],
    "index_dir": ".rag_index",
    "extensions": list(DEFAULT_EXTENSIONS),
    "exclude": [],
    "chunk": {"max_chars": 600, "min_chars": 120, "overlap": 100},
    "retrieval": {
        "top_k": 5,
        "candidate_k": 30,
        "max_per_doc": 3,
        "min_rel_score": 0.0,
        "heading_boost": 1.8,
        "coverage_weight": 0.4,
    },
    "abstain": {
        "vocab_cov": 0.32,
        "idf_cov": 0.18,
        "norm_score": 0.0,
    },
    "llm": {
        "api_base": "https://api.deepseek.com/v1",
        "api_key": "",
        "model": "deepseek-chat",
        "temperature": 0.2,
        "timeout": 60,
        "max_context_chars": 6000,
    },
}

# 环境变量覆盖映射：RAG_<SECTION>_<KEY> 或全局 RAG_<KEY>
ENV_MAP = {
    "api_base": "RAG_API_BASE",
    "api_key": "RAG_API_KEY",
    "model": "RAG_MODEL",
    "timeout": "RAG_TIMEOUT",
    "temperature": "RAG_TEMPERATURE",
}


# --------------------------------------------------------------------- 工具
def deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = deep_merge(DEFAULT_CONFIG, {})
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = deep_merge(cfg, json.load(f))
        except (OSError, ValueError) as e:
            print(f"警告：配置文件读取失败（{path}）：{e}", file=sys.stderr)

    llm = cfg.setdefault("llm", {})
    for key, env in ENV_MAP.items():
        val = os.environ.get(env)
        if val:
            if key in ("timeout",):
                try:
                    val = int(val)
                except ValueError:
                    continue
            elif key in ("temperature",):
                try:
                    val = float(val)
                except ValueError:
                    continue
            llm[key] = val
    return cfg


def open_index(index_dir: str) -> Optional[BM25Index]:
    return BM25Index.load(index_dir)


def require_index(cfg: Dict) -> Optional[BM25Index]:
    idx = open_index(cfg["index_dir"])
    if idx is None or not idx.chunks:
        print(f"索引不存在或为空：{cfg['index_dir']}\n"
              f"请先执行：python rag.py ingest", file=sys.stderr)
        return None
    return idx


def make_retriever(idx: BM25Index, cfg: Dict, top_k: Optional[int] = None,
                   max_per_doc: Optional[int] = None) -> Retriever:
    r = cfg["retrieval"]
    return Retriever(
        idx,
        top_k=top_k if top_k is not None else r["top_k"],
        candidate_k=r["candidate_k"],
        max_per_doc=max_per_doc if max_per_doc is not None else r["max_per_doc"],
        min_rel_score=r.get("min_rel_score", 0.0),
    )


def print_hits(hits: List[Dict], show_terms: bool = True, snippet: int = 220) -> None:
    if not hits:
        print("（无命中）")
        return
    for i, h in enumerate(hits, 1):
        loc = h.get("doc", "")
        if h.get("heading"):
            loc += f" › {h['heading']}"
        if h.get("start_line"):
            loc += f"（第 {h['start_line']} 行起）"
        print(f"[{i}] 得分 {h['score']:.4f}  {loc}")
        body = " ".join(h.get("text", "").split())
        if len(body) > snippet:
            body = body[:snippet] + " …"
        print(f"    {body}")
        if show_terms and h.get("matched_terms"):
            print(f"    命中词项：{'、'.join(h['matched_terms'])}")
        print()


def build_llm(cfg: Dict) -> LLMClient:
    l = cfg["llm"]
    return LLMClient(
        api_base=l.get("api_base", ""),
        api_key=l.get("api_key", ""),
        model=l.get("model", ""),
        temperature=l.get("temperature", 0.2),
        timeout=l.get("timeout", 60),
    )


# --------------------------------------------------------------------- ingest
def cmd_ingest(args, cfg: Dict) -> int:
    roots = args.paths or cfg["docs"]
    index_dir = args.index_dir or cfg["index_dir"]
    ck = cfg["chunk"]

    docs = discover_documents(roots, cfg["extensions"], cfg.get("exclude", []))
    if not docs:
        print(f"未在 {roots} 下发现任何文档（支持后缀：{', '.join(cfg['extensions'])}）",
              file=sys.stderr)
        return 2

    old: Optional[BM25Index] = None if args.rebuild else open_index(index_dir)
    cached: Dict[str, List[Dict]] = {}
    if old:
        for ch in old.chunks:
            cached.setdefault(ch.get("doc", ""), []).append(ch)

    all_chunks: List[Dict] = []
    files_meta: Dict[str, Dict] = {}
    n_new = n_reuse = 0
    total_chars = 0

    for rel in docs:
        try:
            text = read_text(rel)
        except OSError as e:
            print(f"  跳过 {rel}：{e}", file=sys.stderr)
            continue
        sha = sha1_of(text)
        total_chars += len(text)
        files_meta[rel] = {"sha1": sha, "chars": len(text)}

        if old and old.files.get(rel, {}).get("sha1") == sha and rel in cached:
            all_chunks.extend(cached[rel])
            n_reuse += 1
            if args.verbose:
                print(f"  复用 {rel}（未变更，{len(cached[rel])} 块）")
        else:
            chunks = chunk_document(rel, text,
                                    max_chars=ck["max_chars"],
                                    min_chars=ck["min_chars"],
                                    overlap=ck["overlap"])
            all_chunks.extend(chunks)
            n_new += 1
            if args.verbose or not args.quiet:
                print(f"  解析 {rel}：{len(text)} 字符 → {len(chunks)} 块")

    # 稳定排序，保证 chunk id 与内容一一对应、可复现
    all_chunks.sort(key=lambda c: (c.get("doc", ""), c.get("start_line", 0)))

    if args.dry_run:
        print(f"\n[dry-run] 文档 {len(files_meta)} 篇，将写入 {len(all_chunks)} 块，"
              f"未改动任何文件。")
        return 0

    idx = BM25Index(
        heading_boost=cfg["retrieval"].get("heading_boost", 1.8),
        coverage_weight=cfg["retrieval"].get("coverage_weight", 0.4),
    ).build(all_chunks, files_meta)

    path = idx.save(index_dir)
    removed = set(cached) - set(files_meta) if old else set()

    print(f"\n索引完成：{len(files_meta)} 篇文档 / {len(all_chunks)} 个片段 / "
          f"{len(idx.df)} 个词项")
    print(f"  新增或更新：{n_new} 篇，复用缓存：{n_reuse} 篇"
          + (f"，移除：{len(removed)} 篇" if removed else ""))
    print(f"  平均片段长度：{idx.avgdl:.1f} 词元，原文合计 {total_chars} 字符")
    print(f"  索引文件：{path}")
    return 0


# ------------------------------------------------------------------------ ask
def _answer_once(question: str, retriever: Retriever, cfg: Dict,
                 no_llm: bool, top_k: Optional[int]) -> Dict:
    hits = retriever.retrieve(question, top_k=top_k)
    l = cfg["llm"]
    llm = build_llm(cfg)
    used_llm = False
    note = ""

    # 拒答判定先于生成：资料不足时直接拒答，避免大模型在无关片段上编造答案
    abstained = should_abstain(question, hits, retriever.idf,
                               retriever.index.postings,
                               cfg.get("abstain"))
    if abstained:
        answer = "根据现有资料无法回答该问题。"
        if not hits:
            note = "（知识库中没有任何片段与问题相关。）"
        else:
            f = abstain_features(question, hits, retriever.idf,
                                 retriever.index.postings)
            note = (f"（拒答判定：资料不足。vocab_cov={f['vocab_cov']:.2f}，"
                    f"idf_cov={f['idf_cov']:.2f}。问题词项在知识库中覆盖过低，"
                    "强行作答极易产生幻觉，故不予回答。）")
    elif no_llm or not llm.available:
        answer = extractive_answer(question, hits, retriever.idf)
        note = ("（离线抽取式回答：未调用大模型，仅摘录原文最相关的句子。"
                "该模式不做推理与汇总，摘录内容可能来自相邻条目，"
                "请以引用来源原文为准；配置大模型可显著提升回答质量。）")
    else:
        try:
            answer = llm.chat(build_messages(question, hits,
                                             l.get("max_context_chars", 6000)))
            used_llm = True
        except LLMError as e:
            answer = extractive_answer(question, hits, retriever.idf)
            note = f"（大模型调用失败，已回退离线抽取：{e}）"

    return {"question": question, "answer": answer, "hits": hits,
            "abstained": abstained, "used_llm": used_llm, "note": note}


def cmd_ask(args, cfg: Dict) -> int:
    idx = require_index(cfg)
    if idx is None:
        return 2
    retriever = make_retriever(idx, cfg, top_k=args.top_k)
    result = _answer_once(args.question, retriever, cfg, args.no_llm, args.top_k)

    if args.json:
        print(json.dumps({
            "question": result["question"],
            "answer": result["answer"],
            "used_llm": result["used_llm"],
            "sources": [
                {"n": i, "doc": h["doc"], "heading": h["heading"],
                 "line": h["start_line"], "score": round(h["score"], 4)}
                for i, h in enumerate(result["hits"], 1)
            ],
        }, ensure_ascii=False, indent=2))
        return 0

    print(result["answer"])
    if result["note"]:
        print(f"\n{result['note']}")
    if result["hits"]:
        print("\n引用来源：")
        for i, h in enumerate(result["hits"], 1):
            loc = h["doc"] + (f" › {h['heading']}" if h["heading"] else "")
            print(f"  [{i}] {loc}（第 {h['start_line']} 行起，得分 {h['score']:.4f}）")
    if args.show_context:
        print("\n" + "=" * 60 + "\n召回片段：\n")
        print_hits(result["hits"])
    return 0


# --------------------------------------------------------------------- search
def cmd_search(args, cfg: Dict) -> int:
    idx = require_index(cfg)
    if idx is None:
        return 2
    retriever = make_retriever(idx, cfg, top_k=args.top_k)
    hits = retriever.retrieve(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0
    print(f"查询：{args.query}\n命中 {len(hits)} 条\n")
    print_hits(hits, show_terms=not args.no_terms)
    return 0


# ----------------------------------------------------------------------- chat
def cmd_chat(args, cfg: Dict) -> int:
    idx = require_index(cfg)
    if idx is None:
        return 2
    retriever = make_retriever(idx, cfg, top_k=args.top_k)
    llm = build_llm(cfg)
    mode = "离线抽取式" if (args.no_llm or not llm.available) else f"大模型（{llm.model}）"
    print(f"交互式问答已启动｜检索片段数 {retriever.top_k}｜回答模式 {mode}")
    print("输入问题后回车；输入 :exit 或 :quit 退出；输入 :k N 调整片段数。\n")

    note_shown = False
    while True:
        try:
            line = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":exit", ":quit", "exit", "quit", "退出"):
            break
        if line.startswith(":k "):
            try:
                retriever.top_k = max(1, int(line[3:]))
                print(f"检索片段数已设为 {retriever.top_k}\n")
            except ValueError:
                print("用法：:k 5\n")
            continue

        result = _answer_once(line, retriever, cfg, args.no_llm, args.top_k)
        print(f"\n{result['answer']}")
        if result["note"] and not note_shown:
            print(result["note"])
            note_shown = True
        elif result["note"]:
            print("（离线抽取式回答，说明见首次提示）")
        if result["hits"]:
            print("来源：" + "；".join(
                f"[{i}] {h['doc']}" + (f" › {h['heading']}" if h["heading"] else "")
                for i, h in enumerate(result["hits"], 1)))
        print()
    return 0


# ---------------------------------------------------------------------- stats
def cmd_stats(args, cfg: Dict) -> int:
    idx = require_index(cfg)
    if idx is None:
        return 2
    s = idx.stats()
    print(f"索引目录   : {cfg['index_dir']}")
    print(f"构建时间   : {s['built_at']}")
    print(f"文档数     : {s['documents']}")
    print(f"片段数     : {s['chunks']}")
    print(f"词项数     : {s['terms']}")
    print(f"平均片段长度: {s['avgdl']} 词元")
    if args.verbose:
        print("\n各文档片段数：")
        for doc, n in sorted(s["per_doc"].items(), key=lambda x: -x[1]):
            print(f"  {n:>5}  {doc}")
        top = sorted(idx.df.items(), key=lambda x: -x[1])[:20]
        print("\n高频词项（Top 20）：")
        for t, df in top:
            print(f"  df={df:<5} {t}")
    return 0


def cmd_docs(args, cfg: Dict) -> int:
    idx = require_index(cfg)
    if idx is None:
        return 2
    per_doc = idx.stats()["per_doc"]
    for doc, n in sorted(per_doc.items()):
        print(f"{n:>5} 块  {doc}")
    print(f"\n合计 {len(per_doc)} 篇")
    return 0


def cmd_config(args, cfg: Dict) -> int:
    shown = deep_merge(cfg, {})
    key = shown.get("llm", {}).get("api_key", "")
    if key:
        shown["llm"]["api_key"] = key[:6] + "…" + key[-4:] if len(key) > 12 else "***"
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag",
        description="零依赖小型 RAG：BM25 检索 + 可选大模型生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python rag.py ingest\n"
               "  python rag.py ask \"如何导入笔记？\" --show-context\n"
               "  python rag.py search \"导出格式\" -k 8\n"
               "  python rag.py chat --no-llm",
    )
    p.add_argument("--version", action="version", version=f"rag {__version__}")
    p.add_argument("--config", default=None, help="配置文件路径（默认 ./config.json）")
    p.add_argument("--index-dir", default=None, help="索引目录（默认取配置 index_dir）")
    sub = p.add_subparsers(dest="command")

    pi = sub.add_parser("ingest", help="建立或增量更新索引")
    pi.add_argument("paths", nargs="*", help="文档文件或目录（默认取配置 docs）")
    pi.add_argument("--rebuild", action="store_true", help="忽略缓存，全量重建")
    pi.add_argument("--dry-run", action="store_true", help="只扫描不写索引")
    pi.add_argument("-v", "--verbose", action="store_true", help="显示每个文件的处理详情")
    pi.add_argument("-q", "--quiet", action="store_true", help="仅输出汇总")
    pi.set_defaults(func=cmd_ingest)

    pa = sub.add_parser("ask", help="检索并生成回答")
    pa.add_argument("question", help="问题文本")
    pa.add_argument("-k", "--top-k", type=int, default=None, help="召回片段数")
    pa.add_argument("--no-llm", action="store_true", help="不调用大模型，使用离线抽取式回答")
    pa.add_argument("--show-context", action="store_true", help="额外打印召回片段")
    pa.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本调用")
    pa.set_defaults(func=cmd_ask)

    ps = sub.add_parser("search", help="仅检索，不生成回答")
    ps.add_argument("query", help="查询文本")
    ps.add_argument("-k", "--top-k", type=int, default=None, help="返回条数")
    ps.add_argument("--no-terms", action="store_true", help="不显示命中的词项")
    ps.add_argument("--json", action="store_true", help="以 JSON 输出")
    ps.set_defaults(func=cmd_search)

    pc = sub.add_parser("chat", help="交互式问答")
    pc.add_argument("-k", "--top-k", type=int, default=None, help="召回片段数")
    pc.add_argument("--no-llm", action="store_true", help="使用离线抽取式回答")
    pc.set_defaults(func=cmd_chat)

    pst = sub.add_parser("stats", help="索引统计")
    pst.add_argument("-v", "--verbose", action="store_true", help="附加每文档与高频词项")
    pst.set_defaults(func=cmd_stats)

    pd = sub.add_parser("docs", help="列出已索引文档")
    pd.set_defaults(func=cmd_docs)

    pcf = sub.add_parser("config", help="显示生效配置（API Key 已脱敏）")
    pcf.set_defaults(func=cmd_config)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    # Windows 控制台默认 GBK，强制 UTF-8 以避免中文输出乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    cfg_path = args.config or ("config.json" if os.path.isfile("config.json") else None)
    cfg = load_config(cfg_path)
    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0
