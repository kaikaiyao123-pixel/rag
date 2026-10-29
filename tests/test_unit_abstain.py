#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单元测试：拒答判定（abstention）。

覆盖需求：REQ-ABS-01..06

这一层的三个特征都是**查询内归一化**量，因此可以用极小的构造数据把每个特征
钉死在精确值上 —— 不需要真实语料，也就不会随语料变化而漂移。
"""
from __future__ import annotations

from ragcore.abstain import features, should_abstain

from harness import assert_close, assert_eq, assert_false, assert_true, case

IDF = {"甲": 1.0}
HIT = [{"chunk_id": 0, "score": 0.5}]
POSTINGS_FULL = {"甲": {0: 1}}
POSTINGS_EMPTY: dict = {}


# 无有效词项必须拒答（REQ-ABS-01 边界：空串与纯标点）
@case("TC-ABS-001", "REQ-ABS-01", "查询无有效词项时直接拒答",
      technique="边界值分析", level="单元",
      expect="空串与纯标点都返回 True")
def tc_abs_001() -> None:
    assert_true(should_abstain("", [], {}))
    assert_true(should_abstain("。！？", [], {}))
    assert_true(should_abstain("", HIT, IDF), "即使有召回也必须拒答")


# OOV 词项按最大 IDF 计权（REQ-ABS-02：这是历史缺陷的回归点）
@case("TC-ABS-002", "REQ-ABS-02", "OOV 词项按语料最大 IDF 计权",
      technique="等价类划分", level="单元",
      expect="词表内甲=1.0、OOV 丙=3.0 → total_idf=4.0，vocab_cov=0.5")
def tc_abs_002() -> None:
    f = features("甲 丙", [], {"甲": 1.0, "乙": 3.0})
    assert_close(f["total_idf"], 4.0, 1e-6, "OOV 应按 max(idf)=3.0 计权")
    assert_close(f["vocab_cov"], 0.5, 1e-6, "2 个词项中 1 个在词表内")
    assert_eq(f["terms"], 2.0)


# postings 缺省时 idf_cov 退化为 1.0（REQ-ABS-03）
@case("TC-ABS-003", "REQ-ABS-03", "未提供 postings 时 idf_cov 退化为 1.0 且不否决",
      technique="判定表", level="单元",
      expect="postings=None → idf_cov=1.0、不拒答；postings={} → idf_cov=0.0、拒答")
def tc_abs_003() -> None:
    f = features("甲", HIT, IDF, None)
    assert_close(f["idf_cov"], 1.0, 1e-9)
    assert_false(should_abstain("甲", HIT, IDF, None), "退化值不得触发否决")

    f2 = features("甲", HIT, IDF, POSTINGS_EMPTY)
    assert_close(f2["idf_cov"], 0.0, 1e-9)
    assert_true(should_abstain("甲", HIT, IDF, POSTINGS_EMPTY), "零覆盖应拒答")


# 三阈值判定表（REQ-ABS-04）
@case("TC-ABS-004", "REQ-ABS-04", "三项阈值任一不满足即拒答（判定表）",
      technique="判定表", level="单元",
      expect="vocab_cov 不足→拒答；idf_cov 不足→拒答；两者充足→不拒答；norm_score 不足→拒答")
def tc_abs_004() -> None:
    # 行 1：vocab_cov = 0.25 < 0.32 → 拒答
    assert_true(should_abstain("甲 丙 丁 戊", HIT, IDF, POSTINGS_FULL),
                "词表覆盖率过低应拒答")
    # 行 2：vocab_cov = 1.0 但 idf_cov = 0.0 → 拒答
    assert_true(should_abstain("甲", HIT, IDF, POSTINGS_EMPTY),
                "最高分片段未覆盖查询词项应拒答")
    # 行 3：两项均满足 → 不拒答
    assert_false(should_abstain("甲", HIT, IDF, POSTINGS_FULL))
    # 行 4：norm_score 阈值被抬高后不满足 → 拒答
    assert_true(should_abstain("甲", HIT, IDF, POSTINGS_FULL,
                               {"norm_score": 0.9}),
                "norm_score 0.5 < 0.9 应拒答")


# 阈值可覆盖（REQ-ABS-05）
@case("TC-ABS-005", "REQ-ABS-05", "thresholds 参数可覆盖默认阈值并翻转判定",
      technique="边界值分析", level="单元",
      expect="vocab_cov 阈值抬到 1.01 后，同一输入由「不拒答」变为「拒答」")
def tc_abs_005() -> None:
    assert_false(should_abstain("甲", HIT, IDF, POSTINGS_FULL))
    assert_true(should_abstain("甲", HIT, IDF, POSTINGS_FULL,
                               {"vocab_cov": 1.01}))
    assert_true(should_abstain("甲", HIT, IDF, POSTINGS_FULL,
                               {"idf_cov": 1.01}))


# 特征计算口径（REQ-ABS-06）
@case("TC-ABS-006", "REQ-ABS-06", "三个特征在构造数据上取到精确值",
      technique="等价类划分", level="单元",
      expect="vocab_cov=1.0、idf_cov=1.0、norm_score=0.5、total_idf=1.0")
def tc_abs_006() -> None:
    f = features("甲", HIT, IDF, POSTINGS_FULL)
    assert_close(f["vocab_cov"], 1.0, 1e-9, "vocab_cov")
    assert_close(f["idf_cov"], 1.0, 1e-9, "idf_cov")
    assert_close(f["norm_score"], 0.5, 1e-9, "norm_score = 0.5 / 1.0")
    assert_close(f["total_idf"], 1.0, 1e-6, "total_idf")
    assert_eq(f["terms"], 1.0)
