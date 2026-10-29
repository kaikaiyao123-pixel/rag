#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单元测试：分词器 / 分块器 / 文档加载。

覆盖需求：REQ-TOK-01..07、REQ-CHK-01..09、REQ-LOD-01..04
"""
from __future__ import annotations

import os

from fixtures import tempdir, write_bytes, write_file

from ragcore.chunker import chunk_document, parse_sections, split_long_text
from ragcore.loader import discover_documents, read_text
from ragcore.tokenizer import tokenize, tokenize_unique

from harness import (assert_eq, assert_ge, assert_in, assert_le, assert_not_in,
                     assert_raises, assert_true, case)

# ============================================================== 分词器 TOK
# 中文连续汉字串：n 个汉字产生 n-1 个 bigram（REQ-TOK-01 有效等价类）
@case("TC-TOK-001", "REQ-TOK-01", "四字汉字串切分为 3 个 bigram",
      technique="等价类划分", level="单元",
      expect='tokenize("云雀笔记") == ["云雀", "雀笔", "笔记"]')
def tc_tok_001() -> None:
    assert_eq(tokenize("云雀笔记"), ["云雀", "雀笔", "笔记"])


# 单字汉字：退化为 unigram；但停用字被丢弃（REQ-TOK-01 边界 + REQ-TOK-06 边界）
@case("TC-TOK-002", "REQ-TOK-01", "单字汉字退化为 unigram，停用字除外",
      technique="边界值分析", level="单元",
      expect='tokenize("税") == ["税"]；tokenize("的") == []')
def tc_tok_002() -> None:
    assert_eq(tokenize("税"), ["税"], "非停用单字")
    assert_eq(tokenize("的"), [], "停用单字")


# 英文小写化（REQ-TOK-02 有效等价类）
@case("TC-TOK-003", "REQ-TOK-02", "英文标识符统一小写",
      technique="等价类划分", level="单元",
      expect='tokenize("Markdown") == ["markdown"]')
def tc_tok_003() -> None:
    assert_eq(tokenize("Markdown"), ["markdown"])
    assert_eq(tokenize("API_BASE"), ["api_base"])


# 复数归一的四个边界：-ies 长词、-ies 短词、-ss、长度≤3（REQ-TOK-02 边界）
@case("TC-TOK-004", "REQ-TOK-02", "复数归一覆盖 -ies / -s / -ss / 短词四个边界",
      technique="边界值分析", level="单元",
      expect="databases→database；class 不变；studies→study；ties→tie")
def tc_tok_004() -> None:
    assert_eq(tokenize("databases"), ["database"], "词尾单 -s 剥离")
    assert_eq(tokenize("class"), ["class"], "-ss 不得剥离")
    assert_eq(tokenize("studies"), ["study"], "-ies 归一为 -y")
    # ties 长度恰为 4，不满足 len>4，落到 -s 分支 → tie（这是刻意的保守策略）
    assert_eq(tokenize("ties"), ["tie"], "-ies 短词按 -s 处理")


# 英文停用词过滤（REQ-TOK-03 判定表：停用词 vs 实词）
@case("TC-TOK-005", "REQ-TOK-03", "英文停用词被过滤，同句实词保留",
      technique="判定表", level="单元",
      expect='tokenize("the cat is on the mat") 只含 cat / mat')
def tc_tok_005() -> None:
    toks = tokenize("the cat is on the mat")
    assert_eq(toks, ["cat", "mat"])


# 数字与版本号整体保留（REQ-TOK-04 边界：小数点不得拆词）
@case("TC-TOK-006", "REQ-TOK-04", "小数与多段版本号保留为单一词元",
      technique="边界值分析", level="单元",
      expect='tokenize("3.14") == ["3.14"]；"2024.10.26" 不被拆分')
def tc_tok_006() -> None:
    assert_eq(tokenize("3.14"), ["3.14"])
    assert_eq(tokenize("2024.10.26"), ["2024.10.26"])
    # 字母数字混合标识符走英文分支，同样整体保留
    assert_eq(tokenize("FTS5"), ["fts5"])


# 双停用字 bigram 丢弃、含实字的 bigram 保留（REQ-TOK-05 判定表两行）
@case("TC-TOK-009", "REQ-TOK-05", "双停用字 bigram 丢弃，其余保留",
      technique="判定表", level="单元",
      expect='tokenize("是在") == []；tokenize("数据的") 保留两个 bigram')
def tc_tok_009() -> None:
    assert_eq(tokenize("是在"), [], "两字皆为停用字")
    assert_eq(tokenize("数据的"), ["数据", "据的"], "仅含一个停用字时保留")


# 空输入的三个等价类（REQ-TOK-06）
@case("TC-TOK-007", "REQ-TOK-06", "空串 / 纯空白 / 纯标点均返回空列表",
      technique="等价类划分", level="单元",
      expect="三种输入都返回 []，不抛异常")
def tc_tok_007() -> None:
    assert_eq(tokenize(""), [])
    assert_eq(tokenize("   \n\t  "), [])
    assert_eq(tokenize("。，！？；："), [])


# tokenize_unique 保序去重（REQ-TOK-07）
@case("TC-TOK-008", "REQ-TOK-07", "tokenize_unique 去重且保持首次出现顺序",
      technique="等价类划分", level="单元",
      expect='tokenize_unique("数据 数据 分析") == ["数据", "分析"]')
def tc_tok_008() -> None:
    assert_eq(tokenize_unique("数据 数据 分析"), ["数据", "分析"])


# ============================================================== 分块器 CHK
# 标题栈生成层级路径（REQ-CHK-01）
@case("TC-CHK-001", "REQ-CHK-01", "标题栈生成 A › B 形式的层级路径",
      technique="等价类划分", level="单元",
      expect="两级标题产生 heading='A › B'")
def tc_chk_001() -> None:
    md = "# A\n\n一级正文。\n\n## B\n\n二级正文。\n\n### C\n\n三级正文。\n"
    headings = [h for h, _, _ in parse_sections(md)]
    assert_eq(headings, ["A", "A › B", "A › B › C"])


# 同级标题出栈（REQ-CHK-01 边界：栈必须弹出同级及更深项）
@case("TC-CHK-002", "REQ-CHK-01", "同级标题不叠加，栈正确回退",
      technique="边界值分析", level="单元",
      expect="## B 之后的 ## C 路径为 'A › C'，而非 'A › B › C'")
def tc_chk_002() -> None:
    md = "# A\n\n一。\n\n## B\n\n二。\n\n## C\n\n三。\n"
    headings = [h for h, _, _ in parse_sections(md)]
    assert_eq(headings, ["A", "A › B", "A › C"])


# 围栏代码块内的 # 不得成为标题（REQ-CHK-02 判定表：``` 与 ~~~ 两种围栏）
@case("TC-CHK-003", "REQ-CHK-02", "围栏代码块内的 # 行不被解析为标题",
      technique="判定表", level="单元",
      expect="``` 与 ~~~ 内的注释都不进入 heading")
def tc_chk_003() -> None:
    for fence in ("```", "~~~"):
        md = (f"# 真标题\n\n正文段落。\n\n{fence}python\n"
              f"# 这是注释\nprint(1)\n{fence}\n\n结尾段落。\n")
        headings = [h for h, _, _ in parse_sections(md)]
        assert_eq(headings, ["真标题"], f"围栏 {fence}")
        chunks = chunk_document("t.md", md, 600, 1, 0)
        assert_true(any("结尾段落" in c["text"] for c in chunks),
                    f"围栏 {fence} 之后的正文丢失")


# 未闭合围栏会吞掉后续所有标题（REQ-CHK-02 边界：状态机不闭合）
@case("TC-CHK-004", "REQ-CHK-02", "未闭合围栏吞掉后续标题（状态机不退出）",
      technique="边界值分析", level="单元",
      expect="``` 未闭合时，其后的 # B 不被当作标题")
def tc_chk_004() -> None:
    md = "# A\n\n正文。\n\n```\ncode\n# B\nmore\n"
    headings = [h for h, _, _ in parse_sections(md)]
    assert_eq(headings, ["A"])


# 短文档单块且段落完整（REQ-CHK-03）
@case("TC-CHK-005", "REQ-CHK-03", "短文档产生单块且两段均完整保留",
      technique="等价类划分", level="单元",
      expect="块数 == 1，且同时包含两段正文")
def tc_chk_005() -> None:
    md = "# T\n\n第一段内容。\n\n第二段内容。\n"
    chunks = chunk_document("t.md", md, 600, 1, 0)
    assert_eq(len(chunks), 1)
    assert_in("第一段内容", chunks[0]["text"])
    assert_in("第二段内容", chunks[0]["text"])


# 块长上限：无重叠时 ≤ max_chars；有重叠时 ≤ max_chars + overlap + 2（REQ-CHK-04）
@case("TC-CHK-006", "REQ-CHK-04", "块长不超过 max_chars（含重叠时的额外开销）",
      technique="边界值分析", level="单元",
      expect="overlap=0 时每块 ≤ 200；overlap=100 时每块 ≤ 302")
def tc_chk_006() -> None:
    md = "# T\n\n" + "很长的段落。" * 200
    plain = chunk_document("t.md", md, 200, 1, 0)
    assert_ge(len(plain), 3, "应切出多块")
    for c in plain:
        assert_le(len(c["text"]), 200, "无重叠时块长上限")

    with_ov = chunk_document("t.md", md, 200, 1, 100)
    for c in with_ov:
        assert_le(len(c["text"]), 200 + 100 + 2, "重叠带来的额外开销上限")


# 超长无标点段落必须硬切而不是放弃（REQ-CHK-04 边界）
@case("TC-CHK-007", "REQ-CHK-04", "无标点超长段落被硬切且不丢字符",
      technique="边界值分析", level="单元",
      expect="切分后各片段长度 ≤ max_chars，拼接后字符总数不变")
def tc_chk_007() -> None:
    text = "甲" * 1000
    pieces = split_long_text(text, 120)
    assert_eq("".join(pieces), text, "字符不得丢失")
    for p in pieces:
        assert_le(len(p), 120)


# overlap 被 clamp 到 max_chars // 2（REQ-CHK-05 边界：overlap > max_chars//2）
@case("TC-CHK-008", "REQ-CHK-05", "overlap 超过 max_chars//2 时被 clamp",
      technique="边界值分析", level="单元",
      expect="overlap=1000 与 overlap=max_chars//2 产生完全相同的分块")
def tc_chk_008() -> None:
    md = "# T\n\n" + "".join(f"第{i:03d}句。" for i in range(100))
    clamped = chunk_document("t.md", md, 100, 1, 1000)
    exact = chunk_document("t.md", md, 100, 1, 50)
    assert_eq([c["text"] for c in clamped], [c["text"] for c in exact],
              "clamp 后结果应与显式传 50 完全一致")

    prefix, _ = clamped[1]["text"].split("\n\n", 1)
    assert_eq(len(prefix), 50, "重叠前缀长度应为 max_chars//2")
    assert_true(clamped[0]["text"].endswith(prefix), "前缀应来自上一块尾部")


# 无重叠时不携带前缀（REQ-CHK-05 边界：overlap=0）
@case("TC-CHK-009", "REQ-CHK-05", "overlap=0 时不携带上一块尾部",
      technique="边界值分析", level="单元",
      expect="第二块文本中不含双换行拼接的残留前缀")
def tc_chk_009() -> None:
    md = "# T\n\n" + "".join(f"第{i:03d}句。" for i in range(100))
    chunks = chunk_document("t.md", md, 100, 1, 0)
    assert_not_in("\n\n", chunks[1]["text"], "不应出现前缀拼接")


# max_chars 的非法值与极小值（REQ-CHK-06 边界）
@case("TC-CHK-010", "REQ-CHK-06", "max_chars<=0 抛 ValueError，=1 仍可工作",
      technique="边界值分析", level="单元",
      expect="0 与 -1 抛 ValueError；1 正常返回非空块")
def tc_chk_010() -> None:
    assert_raises(ValueError, chunk_document, "t.md", "# T\n\n正文。", 0, 1, 0,
                  contains="max_chars")
    assert_raises(ValueError, chunk_document, "t.md", "# T\n\n正文。", -1, 1, 0)
    tiny = chunk_document("t.md", "# T\n\n" + "甲" * 20, 1, 1, 0)
    assert_ge(len(tiny), 1)
    for c in tiny:
        assert_true(c["text"].strip() != "", "块内容不得为空")


# 尾部短块合并（REQ-CHK-07）
@case("TC-CHK-011", "REQ-CHK-07", "尾部短块在同标题下并入前一块",
      technique="边界值分析", level="单元",
      expect="两块的文档合并为一块，且内容完整保留")
def tc_chk_011() -> None:
    md = "# T\n\n" + "甲" * 150 + "\n\n" + "乙" * 10
    chunks = chunk_document("t.md", md, 100, 120, 0)
    assert_eq(len(chunks), 1, "短尾块应被合并")
    assert_in("甲", chunks[0]["text"])
    assert_in("乙", chunks[0]["text"])


# start_line 语义与单调性（REQ-CHK-08）
@case("TC-CHK-012", "REQ-CHK-08", "start_line 指向正文起始行且单调不减",
      technique="等价类划分", level="单元",
      expect="首块 start_line == 3（标题下一行的正文行）；逐块非递减")
def tc_chk_012() -> None:
    md = "# T\n\n甲段落内容。\n\n乙段落内容。\n"
    chunks = chunk_document("t.md", md, 600, 1, 0)
    assert_eq(chunks[0]["start_line"], 3, "首个正文段落所在行")

    md2 = "# T\n\n" + "\n\n".join("甲" * 300 for _ in range(3))
    lines = [c["start_line"] for c in chunk_document("t.md", md2, 100, 1, 0)]
    assert_ge(len(lines), 3)
    for a, b in zip(lines, lines[1:]):
        assert_true(b >= a, f"start_line 出现回退：{lines}")


# 空文档（REQ-CHK-09）
@case("TC-CHK-013", "REQ-CHK-09", "空文档与纯空白文档返回空列表",
      technique="边界值分析", level="单元",
      expect="chunk_document 返回 []，不抛异常")
def tc_chk_013() -> None:
    assert_eq(chunk_document("t.md", "", 600, 120, 100), [])
    assert_eq(chunk_document("t.md", "   \n\n \t \n", 600, 120, 100), [])


# ============================================================ 文档加载 LOD
# 扩展名过滤的大小写与点号形式（REQ-LOD-01 等价类）
@case("TC-LOD-001", "REQ-LOD-01", "扩展名过滤大小写不敏感且点号可省略",
      technique="等价类划分", level="单元",
      expect='[".md",".txt"] 与 ["md","txt"] 结果一致，且 B.MD 被识别')
def tc_lod_001() -> None:
    with tempdir("TC-LOD-001") as d:
        write_file(os.path.join(d, "a.md"), "# A\n")
        write_file(os.path.join(d, "B.MD"), "# B\n")
        write_file(os.path.join(d, "c.txt"), "c\n")
        write_file(os.path.join(d, "d.json"), "{}\n")

        with_dot = {os.path.basename(p) for p in
                    discover_documents([d], [".md", ".txt"])}
        no_dot = {os.path.basename(p) for p in
                  discover_documents([d], ["md", "txt"])}
        assert_eq(with_dot, {"a.md", "B.MD", "c.txt"})
        assert_eq(no_dot, with_dot, "带点与不带点应等价")


# exclude 的子串与基名匹配（REQ-LOD-02 判定表）
@case("TC-LOD-002", "REQ-LOD-02", "exclude 支持子串匹配与文件名精确匹配",
      technique="判定表", level="单元",
      expect="exclude=['template'] 排除 template.md，保留 note.md")
def tc_lod_002() -> None:
    with tempdir("TC-LOD-002") as d:
        write_file(os.path.join(d, "template.md"), "模板\n")
        write_file(os.path.join(d, "note.md"), "笔记\n")
        write_file(os.path.join(d, "sub", "template-2.md"), "模板 2\n")

        kept = {os.path.basename(p) for p in
                discover_documents([d], [".md"], exclude=["template"])}
        assert_eq(kept, {"note.md"}, "子串匹配应排除两个 template 文件")

        kept2 = {os.path.basename(p) for p in
                 discover_documents([d], [".md"], exclude=["note.md"])}
        assert_eq(kept2, {"template.md", "template-2.md"}, "基名精确匹配")


# 隐藏目录与 SKIP_DIRS（REQ-LOD-03 判定表）
@case("TC-LOD-003", "REQ-LOD-03", "隐藏目录与依赖目录中的文档不被发现",
      technique="判定表", level="单元",
      expect=".hidden/、node_modules/、.git/ 内文件被跳过，仅 ok.md 被发现")
def tc_lod_003() -> None:
    with tempdir("TC-LOD-003") as d:
        write_file(os.path.join(d, "ok.md"), "正常\n")
        write_file(os.path.join(d, ".hidden", "x.md"), "隐藏\n")
        write_file(os.path.join(d, "node_modules", "y.md"), "依赖\n")
        write_file(os.path.join(d, ".git", "z.md"), "版本库\n")
        kept = {os.path.basename(p) for p in discover_documents([d], [".md"])}
        assert_eq(kept, {"ok.md"})


# 非 UTF-8 编码与 CRLF（REQ-LOD-04 错误推测）
@case("TC-LOD-004", "REQ-LOD-04", "GB18030 文件可解码且 CRLF 被归一为 LF",
      technique="错误推测", level="单元",
      expect="中文内容正确还原，换行全部为 \\n")
def tc_lod_004() -> None:
    with tempdir("TC-LOD-004") as d:
        p = os.path.join(d, "gbk.txt")
        write_bytes(p, "第一行\r\n第二行\r\n".encode("gb18030"))
        text = read_text(p)
        assert_eq(text, "第一行\n第二行\n")
        assert_not_in("\r", text)
