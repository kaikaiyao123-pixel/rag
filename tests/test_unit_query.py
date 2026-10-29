#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单元测试：查询改写（同义词扩展）。

覆盖需求：REQ-RET-04..06

设计要点
--------
这个功能的风险不在"加了别名能不能命中"，而在**默认关闭时会不会悄悄改变
既有行为**。因此第一组用例全部围绕"不扩展时逐字符原样返回"，并额外用
集成用例确认扩展串确实被检索与拒答判定共同使用（两处若用了不同的查询串，
会出现"按扩展词命中、按原词判低覆盖"的自相矛盾）。
"""
from __future__ import annotations

import copy

from fixtures import build_index, chunk

from ragcore.cli import DEFAULT_CONFIG, _answer_once
from ragcore.query import describe, expand
from ragcore.retriever import Retriever

from harness import assert_eq, assert_in, assert_true, case


# ================================================================ 原样返回
@case("TC-RET-004", "REQ-RET-04", "未配置同义词表时 expand 原样返回",
      technique="等价类划分", level="单元",
      expect="synonyms 为 None / 空字典时返回串与输入逐字符相等，且不追加空格")
def tc_ret_004() -> None:
    q = "番茄炒鸡蛋怎么做"
    assert_eq(expand(q, None), q, "None 必须原样返回")
    assert_eq(expand(q, {}), q, "空字典必须原样返回")
    assert_eq(expand("", {"番茄": ["西红柿"]}), "", "空问题原样返回")
    # 表非空但一个词都没命中，同样不能追加任何字符（否则会多出一个空格，
    # 影响下游 should_abstain 的词项集合与缓存键）
    assert_eq(expand(q, {"土豆": ["马铃薯"]}), q, "无命中不得追加空格")
    assert_eq(expand(q, {"番茄": []}), q, "空别名列表视同无命中")


# ================================================================ 追加别名
@case("TC-RET-005", "REQ-RET-05", "命中同义词时追加别名且保留原查询",
      technique="等价类划分", level="单元",
      expect="结果以原问题为前缀、以别名结尾；多个词命中时全部追加")
def tc_ret_005() -> None:
    q = "番茄炒鸡蛋怎么做"
    out = expand(q, {"番茄": ["西红柿"]})
    assert_true(out.startswith(q), f"原查询必须完整保留为前缀，实际 {out!r}")
    assert_eq(out, q + " 西红柿")
    assert_in("西红柿", out)

    multi = expand("番茄土豆汤", {"番茄": ["西红柿"], "土豆": ["马铃薯"]})
    assert_eq(multi, "番茄土豆汤 西红柿 马铃薯", "多个词命中时按表序追加")

    info = describe(q, {"番茄": ["西红柿"]})
    assert_eq(info["added"], ["西红柿"], "describe 应报告追加的别名")


# ================================================================ 非法配置
@case("TC-RET-006", "REQ-RET-06", "别名重复不追加，非法配置降级为原样返回",
      technique="错误推测", level="单元",
      expect="别名已在问题中或已追加过时不重复；synonyms 为 list / 值为 int / 值为 None 均不抛异常")
def tc_ret_006() -> None:
    q = "西红柿炒鸡蛋"
    # 问题里已经有别名，再追加等于把同一个词计两次权重
    assert_eq(expand(q, {"番茄": ["西红柿"]}), q, "别名已在问题中时不得重复追加")
    # 两个原词映射到同一个别名，只应追加一次
    out = expand("番茄洋柿子", {"番茄": ["西红柿"], "洋柿子": ["西红柿"]})
    assert_eq(out.count("西红柿"), 1, f"同一别名不得重复追加，实际 {out!r}")

    # 配置写错不能让服务起不来：一律降级为原样返回
    assert_eq(expand(q, ["番茄", "西红柿"]), q, "synonyms 为列表时降级")
    assert_eq(expand(q, {"番茄": 3}), q, "值为 int 时降级")
    assert_eq(expand(q, {"番茄": [None, 3]}), q, "别名列表含非字符串时跳过")
    assert_eq(expand(q, {3: ["西红柿"]}), q, "键非字符串时跳过")
    # 单值写成字符串应被当作单元素列表处理（配置文件里手写常见）
    assert_eq(expand("番茄", {"番茄": "西红柿"}), "番茄 西红柿")


# ================================================================ 集成
@case("TC-RET-007", "REQ-RET-05", "扩展后的查询串同时用于检索与拒答判定",
      technique="回归", level="集成",
      expect="菜谱语料里问「番茄炒鸡蛋」能命中「西红柿炒鸡蛋」，且 query_used 回传扩展串")
def tc_ret_007() -> None:
    idx = build_index([
        chunk("cook/x.md", "西红柿炒鸡蛋的做法",
              "西红柿炒鸡蛋的做法：西红柿洗净切块，鸡蛋打散，热锅下油，"
              "先炒鸡蛋再下西红柿翻炒，加盐出锅。"),
        chunk("cook/y.md", "菠菜炒鸡蛋的做法",
              "菠菜炒鸡蛋的做法：菠菜焯水切段，鸡蛋打散，热锅下油翻炒。"),
    ])
    retriever = Retriever(idx, top_k=5, candidate_k=30, max_per_doc=3)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["synonyms"] = {"番茄": ["西红柿"]}
    res = _answer_once("番茄炒鸡蛋怎么做", retriever, cfg, True, None)

    assert_eq(res["question"], "番茄炒鸡蛋怎么做", "回传给用户的 question 不得被改写")
    assert_eq(res["query_used"], "番茄炒鸡蛋怎么做 西红柿")
    assert_true(res["abstained"] is False, "扩展后不应拒答")
    assert_eq(res["hits"][0]["doc"], "cook/x.md",
              f"top1 应为西红柿炒鸡蛋，实际 {res['hits'][0]['doc']}")
