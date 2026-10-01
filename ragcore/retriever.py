# -*- coding: utf-8 -*-
"""检索编排层：在 BM25 之上补充去重与结果整形。"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .index import BM25Index

_WS_RE = re.compile(r"\s+")


def _fingerprint(text: str, n: int = 60) -> str:
    """取正文前 n 个非空白字符作为近似重复指纹（忽略标题与换行差异）。"""
    return _WS_RE.sub("", text)[:n]


def dedupe(hits: List[Dict], max_chars: int = 60) -> List[Dict]:
    """去掉内容高度重合的片段，保留得分最高者。

    分块时的重叠窗口会让相邻块携带相同的尾部文本，若两块都被召回，会把
    宝贵的上下文配额浪费在重复内容上。
    """
    seen = set()
    out: List[Dict] = []
    for h in hits:
        fp = _fingerprint(h.get("text", ""), max_chars)
        if fp and fp in seen:
            continue
        seen.add(fp)
        out.append(h)
    return out


class Retriever:
    """封装索引 + 检索参数，供 CLI 与生成层共用。"""

    def __init__(self,
                 index: BM25Index,
                 top_k: int = 5,
                 candidate_k: int = 30,
                 max_per_doc: int = 3,
                 min_rel_score: float = 0.0):
        self.index = index
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.max_per_doc = max_per_doc
        self.min_rel_score = min_rel_score

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        hits = self.index.search(
            query,
            top_k=top_k or self.top_k,
            candidate_k=self.candidate_k,
            max_per_doc=self.max_per_doc,
            min_rel_score=self.min_rel_score,
        )
        return dedupe(hits)

    @property
    def idf(self) -> Dict[str, float]:
        return self.index.idf
