#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中文友好的 BM25 分词函数（本项目新增，不属于教程原文）。

为什么需要它
------------
`langchain_community.retrievers.BM25Retriever` 的默认分词是

    default_preprocessing_func = lambda text: text.split()

即**按空白字符切分**。中文没有词间空格，于是：

* 文档里一整段中文会变成**一个超长词元**；
* 查询 `'宫保鸡丁'` 只切出一个词元 `'宫保鸡丁'`，而文档中的词元是
  `'宫保鸡丁的做法'`，两者不相等 —— 查询**完全无法命中**。

实测（`benchmark_retrieval.py`）：BM25 分支单独使用时，中文菜名查询的
命中率为 **0%**。

本模块提供 `zh_preprocess`，用**二元字组（bigram）**替代空格切分：

    宫保鸡丁  ->  ['宫保', '保鸡', '鸡丁']

只要文档中出现同一串汉字，就必然产生重叠词元，因此无需分词词典也能稳定召回。
代价是词表规模约为原来的 2 倍，且精度略低于专业分词器（如 jieba）。

用法
----
    from zh_tokenizer import zh_preprocess
    BM25Retriever.from_documents(chunks, k=5, preprocess_func=zh_preprocess)
"""

from __future__ import annotations

import re

# CJK 统一表意文字（含扩展 A、兼容表意文字）
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
# 英文单词 / 数字（含小数）
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?")


def zh_preprocess(text: str) -> list[str]:
    """把文本切成「中文二元字组 + 英文/数字词」的词元列表。

    Args:
        text: 待切分文本

    Returns:
        词元列表。中文部分按相邻两字滑窗切分，英文与数字按整词保留并小写化。
    """
    tokens = [m.group().lower() for m in _LATIN_WORD.finditer(text)]
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens
