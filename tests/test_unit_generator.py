#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单元测试：生成层（上下文拼装 / 离线抽取式回答 / LLM 客户端规则）。

覆盖需求：REQ-GEN-01..09

``_clean_fragment`` 与 ``_sentences`` 是私有函数，但它们的清洗规则是**对外可见的
行为**（决定答案里出现什么文字），因此直接对它们做单元测试，比隔着
``extractive_answer`` 断言整段文本更精确、失败信息也更可读。
"""
from __future__ import annotations

from ragcore.generator import (LLMClient, LLMError, _clean_fragment,
                               extractive_answer, format_context)

from harness import (assert_eq, assert_false, assert_in, assert_le,
                     assert_not_in, assert_raises, assert_true, case)

REFUSAL = "根据现有资料无法回答该问题。"


def _hit(text: str, doc: str = "a.md", heading: str = "",
         start_line: int = 1, score: float = 1.0) -> dict:
    return {"doc": doc, "heading": heading, "start_line": start_line,
            "score": score, "text": text}


# 上下文拼装格式（REQ-GEN-01）
@case("TC-GEN-001", "REQ-GEN-01", "format_context 输出编号、来源、标题与行号",
      technique="等价类划分", level="单元",
      expect='首行 == "[1] 来源：a.md › H（约第 10 行）"')
def tc_gen_001() -> None:
    out = format_context([_hit("正文", heading="H", start_line=10)])
    assert_eq(out, "[1] 来源：a.md › H（约第 10 行）\n正文")

    two = format_context([_hit("甲"), _hit("乙", doc="b.md")])
    assert_in("[1] 来源：a.md", two)
    assert_in("[2] 来源：b.md", two)


# 超长截断的两个分支（REQ-GEN-02）
@case("TC-GEN-002", "REQ-GEN-02", "超长时截断并标注，余量不足 200 字则直接停止",
      technique="边界值分析", level="单元",
      expect="余量 628 字时输出「已截断」；余量 178 字时静默停止")
def tc_gen_002() -> None:
    marked = format_context([_hit("A" * 350), _hit("B" * 900)], max_chars=1000)
    assert_in("已截断", marked, "余量充足时应带截断标记")
    assert_in("[2]", marked, "第二块应有截断后的残余内容")

    silent = format_context([_hit("A" * 800), _hit("B" * 400)], max_chars=1000)
    assert_not_in("已截断", silent, "余量不足 200 字时应直接停止")
    assert_not_in("[2]", silent, "第二块整体不进入上下文")


# 图片语法必须在切句前移除（REQ-GEN-03，历史缺陷回归点）
@case("TC-GEN-003", "REQ-GEN-03", "图片语法被整体移除，不残留路径与残片",
      technique="回归", level="单元",
      expect="答案含「成品色泽红亮」且不含 jpg / 方括号残片")
def tc_gen_003() -> None:
    hits = [_hit("![成品](./000.jpg)成品色泽红亮，肥而不腻，入口即化。")]
    ans = extractive_answer("成品", hits, {})
    assert_in("成品色泽红亮", ans)
    assert_not_in("jpg", ans)
    assert_not_in("000", ans)
    assert_not_in("![", ans)


# 表格行处理判定表（REQ-GEN-04）
@case("TC-GEN-004", "REQ-GEN-04", "表格四类行的去留（判定表）",
      technique="判定表", level="单元",
      expect="数据行转逗号文本；分隔行丢弃；短表头行丢弃；两列行保留")
def tc_gen_004() -> None:
    assert_eq(_clean_fragment("| 参数 | 说明 |"), "参数，说明")
    assert_eq(_clean_fragment("|---|---|"), None, "分隔行应丢弃")
    assert_eq(_clean_fragment("| 免费版 | 个人版 | 团队版 | 企业版 |"), None,
              "四列短表头应丢弃")
    assert_eq(_clean_fragment("| 免费版 | 3 台设备 |"), "免费版，3 台设备")


# Markdown 清洗（REQ-GEN-05）
@case("TC-GEN-005", "REQ-GEN-05", "链接 / 双链 / 加粗 / 代码 / 列表符号被清洗",
      technique="等价类划分", level="单元",
      expect="清洗后只剩可朗读文本")
def tc_gen_005() -> None:
    assert_eq(
        _clean_fragment("**加粗**和`代码`与[[双向链接]]及[文本](http://x)"),
        "加粗和代码与双向链接及文本")
    assert_eq(_clean_fragment("- 条目内容"), "条目内容")
    assert_eq(_clean_fragment("> 引用内容"), "引用内容")
    assert_eq(_clean_fragment("### 标题文字"), "标题文字")
    assert_eq(_clean_fragment("   "), None, "纯空白应返回 None")


# 三种无答案路径必须给出同一文案（REQ-GEN-06）
@case("TC-GEN-006", "REQ-GEN-06", "无命中 / 无词项 / 无重叠返回统一拒答文案",
      technique="等价类划分", level="单元",
      expect="三者都等于「根据现有资料无法回答该问题。」")
def tc_gen_006() -> None:
    assert_eq(extractive_answer("量子退相干", [], {}), REFUSAL)
    assert_eq(extractive_answer("", [_hit("alpha beta gamma")], {}), REFUSAL)
    assert_eq(extractive_answer("完全不相关的词汇",
                                [_hit("alpha beta gamma")], {}), REFUSAL)


# 抽取结果规模约束（REQ-GEN-07）
@case("TC-GEN-007", "REQ-GEN-07", "max_sentences 与 max_chars 约束抽取规模",
      technique="边界值分析", level="单元",
      expect="max_sentences=3 → 3 行；max_chars=20 → 1 行")
def tc_gen_007() -> None:
    text = "".join(f"alpha 第{i}句的内容比较长一些。" for i in range(6))
    hits = [_hit(text)]

    ans = extractive_answer("alpha", hits, {}, max_sentences=3)
    assert_eq(len(ans.split("\n")), 3, "句子数上限")

    ans2 = extractive_answer("alpha", hits, {}, max_chars=20)
    assert_eq(len(ans2.split("\n")), 1, "总长上限应提前停止")
    assert_le(len(ans2.split("\n")[0]), 40)


# available 规则判定表（REQ-GEN-08）
@case("TC-GEN-008", "REQ-GEN-08", "远程需密钥、本机与内网可免密钥（判定表）",
      technique="判定表", level="单元",
      expect="远程无密钥→False；远程有密钥→True；127.0.0.1 与 192.168.x 免密钥→True")
def tc_gen_008() -> None:
    assert_false(LLMClient("https://api.deepseek.com/v1", "", "m").available,
                 "远程服务缺密钥应不可用")
    assert_true(LLMClient("https://api.deepseek.com/v1", "sk-x", "m").available)
    assert_true(LLMClient("http://127.0.0.1:11434/v1", "", "m").available,
                "回环地址免密钥")
    assert_true(LLMClient("http://192.168.1.5:8000/v1", "", "m").available,
                "内网地址免密钥")
    assert_false(LLMClient("", "sk-x", "m").available, "缺 api_base")
    assert_false(LLMClient("https://api.deepseek.com/v1", "sk-x", "").available,
                 "缺 model")


# endpoint 拼接（REQ-GEN-09 等价类：三种 api_base 形态 + 尾部斜杠）
@case("TC-GEN-009", "REQ-GEN-09", "api_base 三种形态都补全为 chat/completions",
      technique="等价类划分", level="单元",
      expect="/v1 结尾补 /chat/completions；裸域名补 /v1/chat/completions；已完整则不变")
def tc_gen_009() -> None:
    assert_eq(LLMClient("https://api.x.com", "", "m")._endpoint(),
              "https://api.x.com/v1/chat/completions")
    assert_eq(LLMClient("https://api.x.com/v1", "", "m")._endpoint(),
              "https://api.x.com/v1/chat/completions")
    assert_eq(LLMClient("https://api.x.com/v1/chat/completions", "", "m")._endpoint(),
              "https://api.x.com/v1/chat/completions")
    assert_eq(LLMClient("https://api.x.com/v1/", "", "m")._endpoint(),
              "https://api.x.com/v1/chat/completions", "尾部斜杠应被归一")


# 未配置时 chat 必须立刻失败（REQ-GEN-10 的前置分支）
@case("TC-GEN-010", "REQ-GEN-10", "未配置时 chat 抛 LLMError 而非发起请求",
      technique="错误推测", level="单元",
      expect="LLMError 信息包含「未配置」")
def tc_gen_010() -> None:
    client = LLMClient("https://api.x.com/v1", "", "m")
    assert_raises(LLMError, client.chat, [{"role": "user", "content": "hi"}],
                  contains="未配置")
