#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""非功能测试：性能与规模行为。

覆盖需求：REQ-PERF-01..04

阈值刻意放宽（10 秒、5 ms、15 秒、50 倍），目的是发现**数量级**退化而不是
卡毫秒数。这类用例如果在 CI 的慢机器上偶发超时，说明阈值定得太紧，应调整
阈值而不是删用例 —— 删掉等于放弃对算法复杂度的守门。
"""
from __future__ import annotations

import time

from fixtures import chunk, mini_index, tempdir

from ragcore.index import BM25Index

from harness import assert_ge, assert_le, assert_true, case


def _synthetic(n: int) -> list:
    """生成 n 个片段：每个片段含两个高频词和一个唯一词。"""
    return [chunk(f"doc{i:05d}.md", "H",
                  f"alpha beta{i % 50} unique{i} " + "填充内容" * 10)
            for i in range(n)]


def _median_ms(fn, repeat: int = 5) -> float:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


# 建库吞吐（REQ-PERF-01）
@case("TC-PERF-001", "REQ-PERF-01", "2000 个片段的建库在 10 秒内完成",
      technique="性能", level="非功能",
      expect="build 耗时 < 10 s")
def tc_perf_001() -> None:
    chunks = _synthetic(2000)
    t0 = time.perf_counter()
    idx = BM25Index().build(chunks)
    elapsed = time.perf_counter() - t0
    assert_le(elapsed, 10.0, f"建库耗时 {elapsed:.2f}s")
    assert_ge(len(idx.df), 100, "词项数异常，可能未正确建立倒排表")


# 单次检索延迟（REQ-PERF-002）
@case("TC-PERF-002", "REQ-PERF-02", "小语料上 1000 次查询平均延迟低于 5 ms",
      technique="性能", level="非功能",
      expect="平均单次检索 < 5 ms")
def tc_perf_002() -> None:
    idx = mini_index()
    queries = ["导入已有笔记", "团队版多少钱", "导出格式", "数据存储",
               "检索接口 参数", "免费版 设备"]
    t0 = time.perf_counter()
    for i in range(1000):
        idx.search(queries[i % len(queries)], top_k=5)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / 1000
    assert_le(avg_ms, 5.0, f"平均 {avg_ms:.3f} ms/次")


# 持久化往返（REQ-PERF-003）
@case("TC-PERF-003", "REQ-PERF-03", "5000 片段的索引保存与加载在 15 秒内完成",
      technique="性能", level="非功能",
      expect="save + load 往返 < 15 s，且片段数一致")
def tc_perf_003() -> None:
    idx = BM25Index().build(_synthetic(5000))
    with tempdir("TC-PERF-003") as d:
        t0 = time.perf_counter()
        idx.save(d)
        loaded = BM25Index.load(d)
        elapsed = time.perf_counter() - t0
        assert_true(loaded is not None, "加载失败")
        assert_le(elapsed, 15.0, f"往返耗时 {elapsed:.2f}s")
        assert_ge(len(loaded.chunks), 5000)


# 规模放大不显著（REQ-PERF-004）
@case("TC-PERF-004", "REQ-PERF-04", "检索耗时不随语料规模线性放大",
      technique="性能", level="非功能",
      expect="5000 片段上的单次检索耗时 < 50 片段上的 50 倍")
def tc_perf_004() -> None:
    small = BM25Index().build(_synthetic(50))
    large = BM25Index().build(_synthetic(5000))

    # 选一个倒排表长度为 1 的稀有词，排除"命中数本身变多"的干扰
    q_small = "unique49"
    q_large = "unique4999"
    assert_ge(len(small.search(q_small, top_k=5)), 1, "小索引未命中")
    assert_ge(len(large.search(q_large, top_k=5)), 1, "大索引未命中")

    def bench(idx, q):
        def run():
            for _ in range(200):
                idx.search(q, top_k=5)
        return run

    t_small = _median_ms(bench(small, q_small))
    t_large = _median_ms(bench(large, q_large))
    ratio = t_large / max(t_small, 1e-6)
    assert_le(ratio, 50.0,
              f"耗时放大 {ratio:.1f} 倍（小 {t_small:.2f}ms → 大 {t_large:.2f}ms），"
              "疑似退化为全表扫描")
