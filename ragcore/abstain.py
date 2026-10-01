# -*- coding: utf-8 -*-
"""拒答判定（abstention）。

问题背景
--------
离线抽取式回答最初的拒答条件是"召回片段中没有任何句子与问题共享词项"。
在中文 bigram 分词下这个条件几乎不可能满足：任意两个中文句子都会因为
"可以""一个""怎么"这类高频二元组产生重叠。基准测试实测拒答准确率仅 28.6%，
即系统倾向于对领域外问题强行编造摘录。

特征设计
--------
BM25 的绝对分数不可跨查询比较（取决于查询词项数量与词项稀有度），因此改用
三个**查询内归一化**的特征判断"知识库里到底有没有这条知识"。

关键细节：词表外（OOV）词项必须参与分母，且要赋予高权重。早期实现只把
"存在于词表中的查询词项"计入分母，结果领域外查询的分母趋近于 0，归一化
比值反而被放大到 1.0 以上 —— 特征方向完全反了。正确做法是把 OOV 词项按
"语料中最稀有词项的 IDF"计权：语料里根本不存在的词，恰恰说明语料答不了
这个问题。

1. ``vocab_cov`` —— 问题词项中能在索引词表里找到的比例。
   领域外问题的专有名词（量子、退相干、川菜）不在词表中，该值显著偏低。

2. ``idf_cov`` —— 最高分片段覆盖的查询词项权重占**全部**查询词项权重的比例
   （OOV 按最稀有词计权）。衡量"最相关的片段是否真的覆盖了问题的主要信息"，
   而不是只沾了一个词。

3. ``norm_score`` —— 最高分片段得分 ÷ 全部查询词项总权重。
   连续型综合指标。**标定结论：该特征本身有区分度（领域内中位 0.750，
   领域外中位 0.172），但在网格搜索中其最优阈值退化为 0，说明它提供的
   信息已被前两项覆盖，属于冗余特征。** 因此默认不参与判定（阈值 0 即
   禁用），仅保留计算用于诊断与换语料后重新标定。

三项阈值均可通过 ``thresholds`` 参数覆盖，默认值为 benchmark.py 在标注集上
网格搜索的结果，目标函数为平衡准确率（领域外拒答率与领域内不误拒率的
算术平均），并列解按 max-min 间隔准则二次筛选。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from .tokenizer import tokenize

# 经 benchmark.py 标定：vocab_cov 与 idf_cov 承担判定，norm_score 冗余故置 0
MIN_VOCAB_COV = 0.32
MIN_IDF_COV = 0.18
MIN_NORM_SCORE = 0.0


def features(query: str,
             hits: List[Dict],
             idf: Dict[str, float],
             postings: Optional[Dict[str, Dict[int, int]]] = None) -> Dict[str, float]:
    """计算拒答判定所用的归一化特征。

    postings 省略时无法计算 idf_cov，该特征退化为 1.0（即不参与否决），
    调用方应尽量传入索引的倒排表。
    """
    q_terms: Set[str] = set(tokenize(query))
    if not q_terms:
        return {"vocab_cov": 0.0, "idf_cov": 0.0, "norm_score": 0.0,
                "total_idf": 0.0, "terms": 0.0}

    # OOV 词项按语料中最稀有的词项计权：语料里没有的词，权重理应最高
    max_idf = max(idf.values()) if idf else 1.0

    def weight(t: str) -> float:
        return idf.get(t, max_idf)

    total_idf = sum(weight(t) for t in q_terms)
    vocab_cov = sum(1 for t in q_terms if t in idf) / len(q_terms)

    if not hits or total_idf <= 0:
        return {"vocab_cov": round(vocab_cov, 4), "idf_cov": 0.0,
                "norm_score": 0.0, "total_idf": round(total_idf, 3),
                "terms": float(len(q_terms))}

    top1 = hits[0]
    if postings is None:
        idf_cov = 1.0
    else:
        cid = top1.get("chunk_id", -1)
        matched = sum(weight(t) for t in q_terms if cid in postings.get(t, {}))
        idf_cov = matched / total_idf

    return {
        "vocab_cov": round(vocab_cov, 4),
        "idf_cov": round(idf_cov, 4),
        "norm_score": round(top1.get("score", 0.0) / total_idf, 4),
        "total_idf": round(total_idf, 3),
        "terms": float(len(q_terms)),
    }


def should_abstain(query: str,
                   hits: List[Dict],
                   idf: Dict[str, float],
                   postings: Optional[Dict[str, Dict[int, int]]] = None,
                   thresholds: Optional[Dict[str, float]] = None) -> bool:
    """判定是否应当拒答。"""
    th = thresholds or {}
    f = features(query, hits, idf, postings)
    if f["terms"] == 0:
        return True
    return (f["vocab_cov"] < th.get("vocab_cov", MIN_VOCAB_COV)
            or f["idf_cov"] < th.get("idf_cov", MIN_IDF_COV)
            or f["norm_score"] < th.get("norm_score", MIN_NORM_SCORE))
