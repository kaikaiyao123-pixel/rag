# -*- coding: utf-8 -*-
"""查询改写：同义词扩展（query expansion）。

为什么需要
----------
本项目用中文二元字组（bigram）做词项，检索是**纯词面匹配**：问题与文档必须
共享至少一个二元组才可能命中。同义异形词因此天然失配 —— 菜谱语料里那道菜
叫「西红柿炒鸡蛋」，用户问「番茄炒鸡蛋怎么做」时，查询词项是
``番茄 / 茄炒 / 炒鸡 / 鸡蛋 / 蛋怎 / 怎么 / 么做``，而文档里只有
``西红 / 红柿 / 柿炒 / 炒鸡 / 鸡蛋 / …``。``番茄`` 与 ``西红`` 没有任何
重叠，最相关的那篇文档根本进不了候选集。

实测（菜谱库 372 篇 / 2083 片段）::

    番茄炒鸡蛋怎么做        top1=菠菜炒鸡蛋(15.36)  idf_cov=0.1653  → 拒答
    番茄炒鸡蛋怎么做 西红柿  top1=西红柿炒鸡蛋(43.18) idf_cov=0.3111  → 正常作答

所以扩展是有效的，但**词表必须是数据而非代码**：不同知识库的别名集完全不同，
硬编码进检索逻辑既不可维护也不可解释。

设计约束
--------
1. **默认关闭**：``synonyms`` 为空字典时 ``expand`` 原样返回，检索行为与
   加此功能前逐字节一致（默认语料 ``docs/`` 的 143 条标注集评测不受影响）。
2. **只追加、不替换**：原查询词项全部保留，别名追加在末尾。这样即使别名
   配错，也只是多召回几个候选，不会把原本正确的检索结果挤掉。
3. **别名由分词器再切分**：这里不关心别名有几个字，追加后交给 ``tokenize``
   处理。``西红柿`` 会被切成 ``西红`` / ``红柿`` 两个二元组，正好对上文档。
4. **非法配置降级**：``synonyms`` 不是字典、值不是列表、元素不是字符串时
   一律忽略，绝不抛异常 —— 配置文件写错不应该让服务起不来。
"""
from __future__ import annotations

from typing import Any, Dict, List


def expand(question: str, synonyms: Any = None) -> str:
    """按同义词表扩展查询串。

    参数
    ----
    question : str
        原始问题。
    synonyms : Any
        ``{原词: [别名, ...]}``。为空或结构非法时原样返回。

    返回
    ----
    str
        扩展后的查询串。无任何别名命中时**返回原对象内容**（不追加空格），
        便于调用方与上游做等值比较。

    示例
    ----
    >>> expand("番茄炒鸡蛋怎么做", {"番茄": ["西红柿"]})
    '番茄炒鸡蛋怎么做 西红柿'
    >>> expand("番茄炒鸡蛋怎么做", {})
    '番茄炒鸡蛋怎么做'
    """
    if not isinstance(synonyms, dict) or not synonyms or not question:
        return question

    extra: List[str] = []
    for term, aliases in synonyms.items():
        if not isinstance(term, str) or not term or term not in question:
            continue
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, (list, tuple)):
            continue
        for alias in aliases:
            if not isinstance(alias, str) or not alias:
                continue
            # 别名已在问题里出现时不必重复；已追加过的也不重复
            if alias in question or alias in extra:
                continue
            extra.append(alias)

    if not extra:
        return question
    return question + " " + " ".join(extra)


def describe(question: str, synonyms: Any = None) -> Dict[str, Any]:
    """返回扩展结果与追加的别名，供诊断输出使用。"""
    expanded = expand(question, synonyms)
    if expanded == question:
        return {"query": question, "added": []}
    added = expanded[len(question):].strip().split()
    return {"query": expanded, "added": added}
