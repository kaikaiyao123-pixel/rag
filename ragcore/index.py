# -*- coding: utf-8 -*-
"""Okapi BM25 倒排索引，附 JSON 持久化与增量更新支持（纯标准库实现）。

评分公式
--------
对查询中的每个词项 t：

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    tf_norm = f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * dl(d) / avgdl))
    score += idf(t) * tf_norm * (1 + ln(qtf(t)))

采用 BM25+ 风格的 idf 形式（加 1 后取对数），保证高 df 词项的权重恒为正，
避免经典 BM25 在词项出现于过半文档时给出负分。

在此之上叠加两个启发式加权因子：

    final = score * (1 + heading_boost * 标题命中率 + coverage_weight * 词项覆盖率)

- **标题命中率**：命中的查询词项中，出现在标题路径里的部分按 IDF 归一后的
  比例。用于把"标题即答案"的片段（如 FAQ 中的问题标题）提前。
- **词项覆盖率**：命中的去重查询词项占全部查询词项的比例。缓解"单个高频
  词重复命中"压过"多词共同命中"的问题。

索引文本 = 标题路径 + 正文，因此标题命中同时也会产生基础分，加权因子只做
放大，不会把零分块提升上来。
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence

from .tokenizer import tokenize

INDEX_VERSION = 2
INDEX_FILENAME = "index.json"


class BM25Index:
    def __init__(self,
                 k1: float = 1.5,
                 b: float = 0.75,
                 heading_boost: float = 1.8,
                 coverage_weight: float = 0.4):
        self.k1 = float(k1)
        self.b = float(b)
        self.heading_boost = float(heading_boost)
        self.coverage_weight = float(coverage_weight)

        self.chunks: List[Dict] = []
        self.postings: Dict[str, Dict[int, int]] = {}
        self.heading_terms: List[set] = []
        self.doc_len: List[int] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.files: Dict[str, Dict] = {}
        self.built_at: str = ""

    # ------------------------------------------------------------------ 构建
    def build(self, chunks: Sequence[Dict], files: Optional[Dict[str, Dict]] = None) -> "BM25Index":
        self.chunks = [dict(c) for c in chunks]
        for i, ch in enumerate(self.chunks):
            ch["id"] = i

        self.postings = {}
        self.heading_terms = []
        self.doc_len = []
        self.df = {}

        for cid, ch in enumerate(self.chunks):
            heading = ch.get("heading", "") or ""
            text = f"{heading}\n{ch.get('text', '')}"
            toks = tokenize(text)
            self.doc_len.append(len(toks))

            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                self.postings.setdefault(t, {})[cid] = f
                self.df[t] = self.df.get(t, 0) + 1

            self.heading_terms.append(set(tokenize(heading)))

        n = len(self.chunks)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {
            t: math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }
        self.files = dict(files or {})
        self.built_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return self

    # ------------------------------------------------------------------ 检索
    def search(self,
               query: str,
               top_k: int = 5,
               candidate_k: int = 30,
               max_per_doc: int = 0,
               min_rel_score: float = 0.0) -> List[Dict]:
        """检索并返回排序后的命中片段。

        min_rel_score：相对最低分阈值（相对最高分），用于过滤明显无关的尾部
        结果；0 表示不过滤。
        """
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []

        q_tf: Dict[str, int] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        # 仅在词表内的查询词项参与归一，避免生僻词拉低所有结果的加权因子
        vocab_terms = {t: self.idf[t] for t in q_tf if t in self.idf}
        total_idf = sum(vocab_terms.values()) or 1.0

        scores: Dict[int, float] = {}
        hits_by_chunk: Dict[int, set] = {}
        avgdl = self.avgdl or 1.0

        for t, qtf in q_tf.items():
            posting = self.postings.get(t)
            if not posting:
                continue
            idf = self.idf[t]
            q_weight = 1.0 + math.log(qtf)
            for cid, f in posting.items():
                dl = self.doc_len[cid]
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                if denom <= 0:
                    continue
                scores[cid] = scores.get(cid, 0.0) + idf * f * (self.k1 + 1.0) / denom * q_weight
                hits_by_chunk.setdefault(cid, set()).add(t)

        if not scores:
            return []

        results: List[Dict] = []
        for cid, base in scores.items():
            matched = hits_by_chunk[cid]
            h_idf = sum(self.idf.get(t, 0.0) for t in matched if t in self.heading_terms[cid])
            heading_ratio = h_idf / total_idf
            coverage = len(matched) / len(q_tf)
            final = base * (1.0 + self.heading_boost * heading_ratio
                            + self.coverage_weight * coverage)
            ch = self.chunks[cid]
            results.append({
                "chunk_id": cid,
                "score": final,
                "doc": ch.get("doc", ""),
                "heading": ch.get("heading", ""),
                "start_line": ch.get("start_line", 0),
                "text": ch.get("text", ""),
                "matched_terms": sorted(matched, key=lambda t: -self.idf.get(t, 0.0))[:8],
            })

        results.sort(key=lambda r: (-r["score"], r["chunk_id"]))

        # 候选集截断 → 单文档限流 → 取 top_k
        candidates = results[:candidate_k] if candidate_k and candidate_k > 0 else results
        if max_per_doc and max_per_doc > 0:
            kept, per_doc = [], {}
            overflow = []
            for r in candidates:
                d = r["doc"]
                if per_doc.get(d, 0) < max_per_doc:
                    per_doc[d] = per_doc.get(d, 0) + 1
                    kept.append(r)
                else:
                    overflow.append(r)
            if len(kept) < top_k:
                kept.extend(overflow[: top_k - len(kept)])
            candidates = kept

        if min_rel_score > 0 and candidates:
            floor = candidates[0]["score"] * min_rel_score
            candidates = [r for r in candidates if r["score"] >= floor]

        return candidates[:top_k]

    # ------------------------------------------------------------ 统计信息
    def stats(self) -> Dict:
        docs = {}
        for ch in self.chunks:
            docs[ch.get("doc", "")] = docs.get(ch.get("doc", ""), 0) + 1
        return {
            "chunks": len(self.chunks),
            "documents": len(docs),
            "terms": len(self.df),
            "avgdl": round(self.avgdl, 1),
            "built_at": self.built_at,
            "per_doc": docs,
        }

    # ------------------------------------------------------------ 持久化
    def to_dict(self) -> Dict:
        return {
            "version": INDEX_VERSION,
            "built_at": self.built_at,
            "params": {
                "k1": self.k1, "b": self.b,
                "heading_boost": self.heading_boost,
                "coverage_weight": self.coverage_weight,
            },
            "chunks": self.chunks,
            "files": self.files,
            "postings": {t: {str(cid): f for cid, f in p.items()}
                         for t, p in self.postings.items()},
            "heading_terms": [sorted(s) for s in self.heading_terms],
            "doc_len": self.doc_len,
            "df": self.df,
            "avgdl": self.avgdl,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BM25Index":
        params = data.get("params", {})
        idx = cls(
            k1=params.get("k1", 1.5),
            b=params.get("b", 0.75),
            heading_boost=params.get("heading_boost", 1.8),
            coverage_weight=params.get("coverage_weight", 0.4),
        )
        idx.chunks = data.get("chunks", [])
        idx.files = data.get("files", {})
        idx.built_at = data.get("built_at", "")
        idx.postings = {
            t: {int(cid): int(f) for cid, f in p.items()}
            for t, p in data.get("postings", {}).items()
        }
        idx.heading_terms = [set(s) for s in data.get("heading_terms", [])]
        idx.doc_len = list(data.get("doc_len", []))
        idx.df = data.get("df", {})
        idx.avgdl = float(data.get("avgdl", 0.0))
        n = len(idx.chunks)
        idx.idf = {
            t: math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for t, df in idx.df.items()
        }
        return idx

    def save(self, index_dir: str) -> str:
        os.makedirs(index_dir, exist_ok=True)
        path = os.path.join(index_dir, INDEX_FILENAME)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, index_dir: str) -> Optional["BM25Index"]:
        path = os.path.join(index_dir, INDEX_FILENAME)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get("version") != INDEX_VERSION:
            return None
        return cls.from_dict(data)
