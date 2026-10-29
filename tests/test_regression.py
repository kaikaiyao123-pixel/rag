#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回归测试：锁定已修复缺陷与必须长期成立的全局不变量。

覆盖需求：REQ-IDX-12、REQ-ABS-04、REQ-DOC-01

每条用例都对应一次**真实发生过的失效**。回归用例的价值不在于"再测一遍功能"，
而在于一旦有人重新引入同样的错误，测试会立刻变红。因此这里的断言都写成
"能区分对错的最小判据"，而不是笼统的"结果看起来正常"。
"""
from __future__ import annotations

import json
import os

from fixtures import ROOT, mini_cfg, mini_index, run_python

from ragcore.abstain import should_abstain
from ragcore.cli import DEFAULT_CONFIG, make_retriever

from harness import assert_eq, assert_ge, assert_true, case

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- 跨进程可复现
# 历史缺陷：keyword 基线的排序漏了 (score, chunk_id) 二级键，稳定排序继承
# set() 的迭代顺序 → 受 PYTHONHASHSEED 影响，同一份代码换进程结果就变。
@case("TC-REG-001", "REQ-IDX-12", "检索排序在三个哈希种子下完全一致",
      technique="回归", level="非功能",
      expect="PYTHONHASHSEED=0/12345/999 三次子进程输出的 chunk_id 序列逐项相同")
def tc_reg_001() -> None:
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {TESTS_DIR!r})\n"
        "from fixtures import mini_index\n"
        "from ragcore.retriever import Retriever\n"
        "idx = mini_index()\n"
        "r = Retriever(idx, top_k=5, candidate_k=30, max_per_doc=3)\n"
        "qs = ['导入已有笔记', '团队版多少钱', '导出格式', '数据存储',\n"
        "      '检索接口 参数', '免费版 设备']\n"
        "print(json.dumps([[h['chunk_id'] for h in r.retrieve(q)] for q in qs]))\n"
    )
    outs = []
    for seed in ("0", "12345", "999"):
        cp = run_python(code, env_extra={"PYTHONHASHSEED": seed})
        assert_eq(cp.returncode, 0, f"seed={seed} 子进程失败：{cp.stderr}")
        outs.append(cp.stdout.strip())

    assert_eq(outs[0], outs[1], "seed=0 与 seed=12345 的结果不同")
    assert_eq(outs[0], outs[2], "seed=0 与 seed=999 的结果不同")
    assert_ge(len(json.loads(outs[0])), 4, "子进程输出结构异常")


# ---------------------------------------------------------------- 拒答方向
# 历史缺陷：OOV 词项未计入分母，导致领域外查询的归一化比值被放大到 1.0 以上，
# 特征方向完全反转 —— 越是不认识的问题越倾向于作答。
@case("TC-REG-002", "REQ-ABS-04", "领域内问题不拒答、领域外问题拒答（方向回归）",
      technique="回归", level="集成",
      expect="「如何导入已有笔记」不拒答；「量子退相干的持续时间」拒答")
def tc_reg_002() -> None:
    idx = mini_index()
    cfg = mini_cfg()
    r = make_retriever(idx, cfg, top_k=5)

    def judge(q: str) -> bool:
        hits = r.retrieve(q)
        return should_abstain(q, hits, r.idf, idx.postings, cfg["abstain"])

    assert_eq(judge("如何导入已有笔记"), False, "领域内问题不应拒答")
    assert_eq(judge("导出支持哪些格式"), False, "领域内问题不应拒答")
    assert_eq(judge("量子退相干的持续时间是多少"), True, "领域外问题必须拒答")
    assert_eq(judge("红烧肉需要炖多久"), True, "另一领域的查询必须拒答")


# ---------------------------------------------------------------- 文档一致性
# 历史缺陷：README 的指标表长期停留在更早一次运行的残值上，改评测后忘了同步。
@case("TC-REG-003", "REQ-DOC-01", "README 指标表与 benchmark_report.json 一致",
      technique="回归", level="系统",
      expect="本方案 / 纯 Okapi BM25 / TF-IDF 余弦 / 关键词命中计数 四行的四个指标全部匹配")
def tc_reg_003() -> None:
    rep_path = os.path.join(ROOT, "benchmark_report.json")
    readme_path = os.path.join(ROOT, "README.md")
    assert_true(os.path.isfile(rep_path), f"缺少 {rep_path}")
    assert_true(os.path.isfile(readme_path), f"缺少 {readme_path}")

    with open(rep_path, encoding="utf-8") as f:
        metrics = json.load(f)["metrics"]
    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()

    label2key = [("本方案", "full"), ("纯 Okapi BM25", "bm25_plain"),
                 ("TF-IDF 余弦", "tfidf"), ("关键词命中计数", "keyword")]
    found = set()
    for line in readme.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("*", "") for c in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        hit = next((lb for lb, _ in label2key if cells[0].startswith(lb)), None)
        if hit is None or hit in found:
            continue
        found.add(hit)
        m = metrics[dict(label2key)[hit]]
        want = [m["doc_recall"]["@1"], m["passage_hit"]["@1"],
                m["passage_hit"]["@5"], m["mrr@10"]]
        try:
            got = [float(c) for c in cells[1:]]
        except ValueError:
            raise AssertionError(f"README「{hit}」行含非数字：{cells[1:]}") from None
        for i, (w, g) in enumerate(zip(want, got)):
            assert_true(abs(round(w, 3) - g) < 5e-4,
                        f"README「{hit}」第 {i + 1} 个指标：README={g}，"
                        f"报告={round(w, 3)}")

    assert_eq(sorted(found), sorted(lb for lb, _ in label2key),
              "README 指标表缺少方法行")


# ------------------------------------------------------ 生产阈值不得被诊断值覆盖
# 标定最优（0.25/0.12/0.3）因 max-min 间隔仅 0.001 而未被采纳。若有人"顺手"
# 把网格最优写进 DEFAULT_CONFIG，误拒率会变化而文档仍宣称旧数字。
@case("TC-REG-004", "REQ-ABS-05", "生产阈值与标定最优阈值保持分离",
      technique="回归", level="单元",
      expect="DEFAULT_CONFIG.abstain == {vocab_cov:0.32, idf_cov:0.18, norm_score:0.0}")
def tc_reg_004() -> None:
    th = DEFAULT_CONFIG["abstain"]
    assert_eq(th["vocab_cov"], 0.32)
    assert_eq(th["idf_cov"], 0.18)
    assert_eq(th["norm_score"], 0.0, "norm_score 应保持禁用（冗余特征）")
