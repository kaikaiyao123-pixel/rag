#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单元测试：BM25 索引与检索编排。

覆盖需求：REQ-IDX-01..11、REQ-RET-01..03
"""
from __future__ import annotations

import json
import math
import os

from fixtures import build_index, chunk, tempdir, write_file

from ragcore.index import INDEX_VERSION, BM25Index
from ragcore.retriever import Retriever, dedupe

from harness import (assert_close, assert_eq, assert_ge, assert_in, assert_le,
                     assert_true, case)


def _three_chunks() -> BM25Index:
    """3 个片段，alpha 出现在全部片段（df=3），beta/gamma/delta 各 1 次。"""
    return build_index([
        chunk("a.md", "H", "alpha beta"),
        chunk("b.md", "H", "alpha gamma"),
        chunk("c.md", "H", "alpha delta"),
    ])


# ================================================================ 索引 IDX
# build 的结构不变量（REQ-IDX-01）
@case("TC-IDX-001", "REQ-IDX-01", "build 分配连续 id 并生成倒排统计",
      technique="等价类划分", level="单元",
      expect="id 为 0..2；df 键与 postings 键一致；doc_len 长度 == 片段数")
def tc_idx_001() -> None:
    idx = _three_chunks()
    assert_eq([c["id"] for c in idx.chunks], [0, 1, 2])
    assert_eq(len(idx.doc_len), 3)
    assert_eq(set(idx.df), set(idx.postings), "df 与 postings 的词项集合必须一致")
    assert_ge(idx.avgdl, 1.0)
    assert_eq(idx.stats()["chunks"], 3)
    assert_eq(idx.stats()["documents"], 3)


# IDF 公式与正定性（REQ-IDX-02 边界：df == N 时经典 BM25 会出负分，此处必须为正）
@case("TC-IDX-002", "REQ-IDX-02", "IDF 符合 BM25+ 形式且 df=N 时仍为正",
      technique="边界值分析", level="单元",
      expect="idf(alpha)=ln(1.142857)≈0.1335；idf(beta)=ln(2.666667)≈0.9808；全部 > 0")
def tc_idx_002() -> None:
    idx = _three_chunks()
    n = 3
    for term, df in (("alpha", 3), ("beta", 1)):
        want = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        assert_close(idx.idf[term], want, 1e-9, f"idf({term})")
    assert_true(all(v > 0 for v in idx.idf.values()), "所有 IDF 必须为正")
    assert_close(idx.idf["alpha"], math.log(1 + 0.5 / 3.5), 1e-9)


# 标题参与索引（REQ-IDX-03）
@case("TC-IDX-003", "REQ-IDX-03", "正文不含查询词、仅标题命中时仍被召回",
      technique="判定表", level="单元",
      expect="查询「退款」只召回标题为「退款政策」的片段")
def tc_idx_003() -> None:
    idx = build_index([
        chunk("a.md", "退款政策", "七天无理由退货。"),
        chunk("b.md", "其他", "七天无理由退货。"),
    ])
    hits = idx.search("退款", top_k=5)
    assert_eq(len(hits), 1, "只有标题含「退款」的片段应命中")
    assert_eq(hits[0]["chunk_id"], 0)
    assert_in("退款", hits[0]["heading"])


# 同分时的二级排序键（REQ-IDX-04）
@case("TC-IDX-004", "REQ-IDX-04", "得分相同时按 chunk_id 升序，结果确定",
      technique="边界值分析", level="单元",
      expect="两个内容完全相同的片段按 0、1 顺序返回")
def tc_idx_004() -> None:
    idx = build_index([
        chunk("a.md", "H", "alpha beta"),
        chunk("b.md", "H", "alpha beta"),
    ])
    hits = idx.search("alpha", top_k=5)
    assert_eq([h["chunk_id"] for h in hits], [0, 1])
    assert_close(hits[0]["score"], hits[1]["score"], 1e-12, "两片段应同分")


# 空查询与空索引（REQ-IDX-05）
@case("TC-IDX-005", "REQ-IDX-05", "空查询与空索引都返回空列表",
      technique="边界值分析", level="单元",
      expect="search('')、search('。')、空索引 search('alpha') 均为 []")
def tc_idx_005() -> None:
    idx = _three_chunks()
    assert_eq(idx.search(""), [])
    assert_eq(idx.search("。！"), [])
    empty = BM25Index().build([])
    assert_eq(empty.search("alpha"), [])


# candidate_k 截断（REQ-IDX-06 边界：1 与 0）
@case("TC-IDX-006", "REQ-IDX-06", "candidate_k 限制候选集，<=0 表示不截断",
      technique="边界值分析", level="单元",
      expect="candidate_k=1 时只返回 1 条；candidate_k=0 时返回全部 5 条")
def tc_idx_006() -> None:
    idx = build_index([chunk(f"{i}.md", "H", "alpha beta") for i in range(5)])
    assert_eq(len(idx.search("alpha", top_k=5, candidate_k=1)), 1)
    assert_eq(len(idx.search("alpha", top_k=5, candidate_k=0)), 5)


# 单文档限流与溢出回填（REQ-IDX-07 判定表两行）
@case("TC-IDX-007", "REQ-IDX-07", "max_per_doc 限流，不足 top_k 时用溢出结果回填",
      technique="判定表", level="单元",
      expect="top_k=2 时两文档各 1 条；top_k=4 时回填到 4 条")
def tc_idx_007() -> None:
    idx = build_index([
        chunk("a.md", "H", "alpha beta"),
        chunk("a.md", "H", "alpha beta"),
        chunk("a.md", "H", "alpha beta"),
        chunk("b.md", "H", "alpha beta " + "填充" * 200),
    ])
    limited = idx.search("alpha", top_k=2, max_per_doc=1)
    assert_eq(len(limited), 2)
    assert_eq(len({h["doc"] for h in limited}), 2, "限流后两文档各取一条")

    backfilled = idx.search("alpha", top_k=4, max_per_doc=1)
    assert_eq(len(backfilled), 4, "候选不足 top_k 时应回填溢出结果")


# 相对分数过滤（REQ-IDX-08）
@case("TC-IDX-008", "REQ-IDX-08", "min_rel_score 按相对最高分过滤尾部结果",
      technique="边界值分析", level="单元",
      expect="阈值 0.9 时低分片段被剔除，剩余分数均 >= 最高分×0.9")
def tc_idx_008() -> None:
    idx = build_index([
        chunk("a.md", "H", "alpha alpha alpha alpha alpha alpha"),
        chunk("b.md", "H", "alpha " + "填充内容" * 300),
    ])
    full = idx.search("alpha", top_k=5)
    assert_eq(len(full), 2, "未过滤时应两条都在")

    filtered = idx.search("alpha", top_k=5, min_rel_score=0.9)
    floor = full[0]["score"] * 0.9
    assert_le(len(filtered), len(full))
    for h in filtered:
        assert_ge(h["score"], floor - 1e-12, "被保留的分数不得低于阈值")


# 持久化往返（REQ-IDX-09）
@case("TC-IDX-009", "REQ-IDX-09", "save/load 往返后片段、IDF 与检索结果一致",
      technique="等价类划分", level="单元",
      expect="chunk_id 序列一致，IDF 逐项误差 < 1e-9")
def tc_idx_009() -> None:
    idx = _three_chunks()
    with tempdir("TC-IDX-009") as d:
        path = idx.save(d)
        assert_true(os.path.isfile(path))
        loaded = BM25Index.load(d)
        assert_true(loaded is not None, "应能重新加载")
        assert_eq([c["id"] for c in loaded.chunks], [c["id"] for c in idx.chunks])
        for term, val in idx.idf.items():
            assert_close(loaded.idf[term], val, 1e-9, f"idf({term})")
        a = [h["chunk_id"] for h in idx.search("alpha beta", top_k=3)]
        b = [h["chunk_id"] for h in loaded.search("alpha beta", top_k=3)]
        assert_eq(b, a)


# 索引损坏 / 版本不符（REQ-IDX-10 错误推测：三种失效路径）
@case("TC-IDX-010", "REQ-IDX-10", "索引缺失、损坏、版本不符时 load 返回 None",
      technique="错误推测", level="单元",
      expect="三种情形都返回 None 且不抛异常")
def tc_idx_010() -> None:
    with tempdir("TC-IDX-010") as d:
        assert_true(BM25Index.load(d) is None, "目录内无索引文件")

        write_file(os.path.join(d, "index.json"), "{ 这不是 JSON ")
        assert_true(BM25Index.load(d) is None, "JSON 损坏")

        write_file(os.path.join(d, "index.json"),
                   json.dumps({"version": INDEX_VERSION + 99, "chunks": []}))
        assert_true(BM25Index.load(d) is None, "版本不符应强制重建")


# 空索引的 avgdl（REQ-IDX-11 边界：不得除零）
@case("TC-IDX-011", "REQ-IDX-11", "空索引的 avgdl 为 0 且不触发除零",
      technique="边界值分析", level="单元",
      expect="build([]) 后 avgdl == 0.0，stats 各项为 0")
def tc_idx_011() -> None:
    idx = BM25Index().build([])
    assert_eq(idx.avgdl, 0.0)
    assert_eq(idx.chunks, [])
    s = idx.stats()
    assert_eq((s["chunks"], s["documents"], s["terms"]), (0, 0, 0))


# ============================================================ 检索 RET
# 指纹去重的边界（REQ-RET-01：第 60 字相同/不同）
@case("TC-RET-001", "REQ-RET-01", "去重指纹取前 60 个非空白字符",
      technique="边界值分析", level="单元",
      expect="前 60 字相同则去重；第 60 字起不同则两条都保留")
def tc_ret_001() -> None:
    same = [{"text": "A" * 60 + "X"}, {"text": "A" * 60 + "Y"}]
    assert_eq(len(dedupe(same)), 1, "第 60 字之后不同仍视为重复")

    differ = [{"text": "A" * 59 + "X"}, {"text": "A" * 59 + "Y"}]
    assert_eq(len(dedupe(differ)), 2, "第 60 字不同应视为不重复")

    ws = [{"text": "A B C"}, {"text": "ABC"}]
    assert_eq(len(dedupe(ws)), 1, "空白差异不构成新片段")

    # 空指纹（全空白正文）不参与去重：fp 为空时判定为「不是重复」，一律保留。
    # 这是既有行为，用例锁定它以免后续改动悄悄改变结果条数。
    blank = [{"text": ""}, {"text": "   "}, {"text": "real"}]
    assert_eq(len(dedupe(blank)), 3, "空指纹一律保留（不去重）")


# top_k 覆盖（REQ-RET-02）
@case("TC-RET-002", "REQ-RET-02", "retrieve 的 top_k 参数覆盖构造默认值",
      technique="边界值分析", level="单元",
      expect="默认 top_k=2，调用时传 4 返回 4 条")
def tc_ret_002() -> None:
    # 每个片段正文必须不同：dedupe 会按正文前 60 字去重，正文相同会被合并成一条
    idx = build_index([chunk(f"{i}.md", "H", f"alpha beta{i}") for i in range(5)])
    r = Retriever(idx, top_k=2, candidate_k=30, max_per_doc=0)
    assert_eq(len(r.retrieve("alpha")), 2, "默认 top_k")
    assert_eq(len(r.retrieve("alpha", top_k=4)), 4, "参数覆盖")


# idf 透传（REQ-RET-03）
@case("TC-RET-003", "REQ-RET-03", "Retriever.idf 与索引 IDF 表为同一对象",
      technique="等价类划分", level="单元",
      expect="r.idf is idx.idf")
def tc_ret_003() -> None:
    idx = _three_chunks()
    r = Retriever(idx)
    assert_true(r.idf is idx.idf, "应为同一对象而非副本")
