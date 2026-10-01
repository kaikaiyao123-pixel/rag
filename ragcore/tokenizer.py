# -*- coding: utf-8 -*-
"""中英文混合分词器（纯标准库实现）。

设计取舍
--------
中文不依赖第三方分词器（jieba / pkuseg 等），改用 **bigram（二元切分）** 方案：
长度为 n 的连续汉字串生成 n-1 个 bigram；n == 1 时退化为 unigram。
这是信息检索领域的经典无词典方案，召回稳定、无需模型文件，代价是索引体积
约为按词切分的 2~3 倍 —— 对小型知识库（数 MB 级）完全可以接受。

英文按字母/数字边界切分并小写化，对高置信度的复数形式做保守归一
（仅 ``-ies`` → ``-y`` 与词尾单个 ``-s``），避免激进词干化引入的语义偏移。

已知局限
--------
- 中文同义词无法归一（"电脑" vs "计算机" 视为不同词项），需靠文档中同时
  出现或人工扩写查询来弥补。
- 单字查询（如 "税"）只能匹配到单字汉字串，命中率低于多字查询。
"""
from __future__ import annotations

import re
from typing import List

# 汉字范围：扩展 A 区 + 基本区 + 兼容区
_CJK_RANGES = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"

# 一个"词元串"：连续汉字 | 英文标识符 | 数字（含小数、版本号）
_RUN_RE = re.compile(
    rf"([{_CJK_RANGES}]+)"
    rf"|([A-Za-z][A-Za-z0-9_+#]*(?:[.\-][A-Za-z0-9_+#]+)*)"
    rf"|([0-9]+(?:[.][0-9]+)*)"
)

# 英文停用词
STOPWORDS_EN = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
without into over under between across is are was were be been being am do does did done
have has had having it its as not no nor so such can could will would shall should
may might must we you he she they i me my your their our his her them us also more most
very much many any all each every some other another which who whom whose what when where
why how there here about above after again against because before below during
""".split())

# 中文单字停用字：仅用于过滤"两字皆为停用字"的 bigram（如 "的了"）
STOPCHARS_ZH = frozenset(
    "的了着过和与及或而但之其是在有为以于对把被给从到就都也还很更最"
    "这那些个中上下里外时年月日一二三四五六七八九十个么呢吧啊呀吗"
)


def _normalize_latin(tok: str) -> str:
    """保守的英文归一：仅处理两类高置信度的复数形式。

    刻意**不**处理 ``-es``（如 classes / databases）：无词典时无法区分
    "词干以 s 结尾"（class→classes）与"词干以 e 结尾"（database→databases），
    强行剥离会产生 ``databas`` 这类错误词干。

    关键点是查询与文档走同一套归一逻辑，因此即使 ``classes`` 被归一到
    ``classe``，查询侧与文档侧仍会一致命中；而错误剥离造成的词干破坏
    则会让不同词意外合并，危害更大。
    """
    t = tok.lower()
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def _cjk_grams(run: str) -> List[str]:
    """汉字串 → bigram 列表。"""
    n = len(run)
    if n == 1:
        return [] if run in STOPCHARS_ZH else [run]
    out: List[str] = []
    for i in range(n - 1):
        bg = run[i:i + 2]
        # 两字皆为停用字时丢弃，其余保留（避免误伤 "数据的" 这类组合）
        if bg[0] in STOPCHARS_ZH and bg[1] in STOPCHARS_ZH:
            continue
        out.append(bg)
    return out


def tokenize(text: str) -> List[str]:
    """把任意文本切分为检索词元列表，**保留重复**（用于词频统计）。"""
    if not text:
        return []
    tokens: List[str] = []
    for m in _RUN_RE.finditer(text):
        cjk, latin, num = m.group(1), m.group(2), m.group(3)
        if cjk:
            tokens.extend(_cjk_grams(cjk))
        elif latin:
            t = _normalize_latin(latin)
            if t and t not in STOPWORDS_EN:
                tokens.append(t)
        elif num:
            tokens.append(num)
    return tokens


def tokenize_unique(text: str) -> List[str]:
    """切分并去重，保持首次出现顺序。"""
    seen = set()
    out: List[str] = []
    for t in tokenize(text):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
