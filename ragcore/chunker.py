# -*- coding: utf-8 -*-
"""Markdown 感知的分块器（纯标准库实现）。

分块策略
--------
1. **按标题切分**：解析 ATX 标题（``#`` ~ ``######``），维护标题栈，
   为每段正文生成形如 ``一级 › 二级 › 三级`` 的标题路径。该路径既是
   检索时的加权字段，也是回答时展示的引用位置。围栏代码块内部的行
   不参与标题解析，避免示例代码里的 ``# 注释`` 被误判为标题。
2. **按空行切段**：段落是语义的最小自然单元，优先保持完整。
3. **贪心装箱**：把段落依次装入不超过 ``max_chars`` 的块中，超长段落
   先按句末标点切分、必要时硬切。
4. **重叠窗口**：新块携带上一块尾部 ``overlap`` 个字符，缓解"答案被
   切断在边界"的问题。

字符数而非 token 数作为度量单位：中文场景下字符数与信息量近似线性，
且无需引入 tokenizer 依赖。
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# 围栏代码块起止（``` 或 ~~~，允许缩进，允许带语言名）
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
# 句末标点（中英文），用于超长段落切分
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*")


def _is_heading(line: str) -> Tuple[int, str] | None:
    m = HEADING_RE.match(line)
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


def parse_sections(text: str) -> List[Tuple[str, int, List[str]]]:
    """解析为 [(标题路径, 起始行号, 正文行列表), ...]。

    围栏代码块（``` / ~~~）内部的行不参与标题解析。否则像 Python 或 Shell
    示例里的 ``# 注释`` 会被误判为一级标题，污染标题路径，进而影响检索时
    的标题加权与引用位置展示。
    """
    lines = text.split("\n")
    sections: List[Tuple[str, int, List[str]]] = []
    stack: List[Tuple[int, str]] = []
    cur_title = ""
    buf: List[str] = []
    buf_start = 1
    fence: str = ""          # 当前处于代码块中时，记录开启它的围栏标记

    for lineno, line in enumerate(lines, start=1):
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0]
            if not fence:
                fence = marker
            elif fence == marker:
                fence = ""
            buf.append(line)
            continue

        h = None if fence else _is_heading(line)
        if h is None:
            buf.append(line)
            continue
        # 遇到新标题：结算上一段
        if any(s.strip() for s in buf):
            sections.append((cur_title, buf_start, buf))
        buf = []
        level, title = h
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        cur_title = " › ".join(t for _, t in stack)
        buf_start = lineno + 1

    if any(s.strip() for s in buf):
        sections.append((cur_title, buf_start, buf))
    return sections


def split_paragraphs(lines: List[str], start_line: int) -> List[Tuple[str, int]]:
    """按空行切段，返回 [(段落文本, 起始行号), ...]。"""
    paras: List[Tuple[str, int]] = []
    buf: List[str] = []
    buf_start = start_line
    for i, line in enumerate(lines):
        if line.strip():
            if not buf:
                buf_start = start_line + i
            buf.append(line)
        elif buf:
            paras.append(("\n".join(buf).strip(), buf_start))
            buf = []
    if buf:
        paras.append(("\n".join(buf).strip(), buf_start))
    return paras


def _hard_split(text: str, max_chars: int) -> List[str]:
    out: List[str] = []
    while len(text) > max_chars:
        out.append(text[:max_chars])
        text = text[max_chars:]
    if text:
        out.append(text)
    return out


def split_long_text(text: str, max_chars: int) -> List[str]:
    """超长文本先按句末标点切分，再合并/硬切成不超过 max_chars 的片段。"""
    if len(text) <= max_chars:
        return [text]
    pieces: List[str] = []
    for sent in _SENT_SPLIT_RE.split(text):
        if not sent:
            continue
        pieces.extend(_hard_split(sent, max_chars) if len(sent) > max_chars else [sent])
    out: List[str] = []
    buf = ""
    for p in pieces:
        if buf and len(buf) + len(p) > max_chars:
            out.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        out.append(buf)
    return out


def chunk_document(doc_rel: str,
                   text: str,
                   max_chars: int = 600,
                   min_chars: int = 120,
                   overlap: int = 100) -> List[Dict]:
    """把单个文档切分为块列表。

    返回的每个 dict 字段：doc / heading / text / start_line。
    ``id`` 由索引层统一分配，保证稳定有序。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须为正整数")
    overlap = max(0, min(overlap, max_chars // 2))

    chunks: List[Dict] = []
    for heading, sec_start, lines in parse_sections(text):
        paras = split_paragraphs(lines, sec_start)
        cur: List[str] = []
        cur_len = 0
        cur_start = sec_start

        def flush() -> None:
            nonlocal cur, cur_len
            if not cur:
                return
            body = "\n\n".join(cur).strip()
            if body:
                chunks.append({
                    "doc": doc_rel,
                    "heading": heading,
                    "text": body,
                    "start_line": cur_start,
                })
            cur, cur_len = [], 0

        for para, pline in paras:
            for piece in split_long_text(para, max_chars):
                if cur and cur_len + len(piece) + 2 > max_chars:
                    body = "\n\n".join(cur).strip()
                    tail = body[-overlap:] if overlap else ""
                    flush()
                    cur_start = pline
                    if tail:
                        cur.append(tail)
                        cur_len = len(tail)
                if not cur:
                    cur_start = pline
                cur.append(piece)
                cur_len += len(piece) + 2
        flush()

    # 合并尾部过短的块，避免产生无信息量的碎片
    if len(chunks) >= 2 and len(chunks[-1]["text"]) < min_chars:
        last = chunks.pop()
        prev = chunks[-1]
        if prev["doc"] == last["doc"] and prev["heading"] == last["heading"]:
            prev["text"] = prev["text"] + "\n\n" + last["text"]
        else:
            chunks.append(last)

    return chunks
