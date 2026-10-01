# -*- coding: utf-8 -*-
"""文档发现与读取（纯标准库实现）。"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable, List, Sequence

DEFAULT_EXTENSIONS = (".md", ".markdown", ".txt")

# 遍历时跳过的目录
SKIP_DIRS = frozenset({
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".rag_index",
    ".venv", "venv", "env", ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".next", "target",
})

# 读取时依次尝试的编码
_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "big5", "latin-1")


def _norm(p: str) -> str:
    """统一为相对当前目录的正斜杠路径，保证索引可移植。"""
    ap = os.path.abspath(p)
    try:
        rel = os.path.relpath(ap, os.getcwd())
    except ValueError:  # 跨盘符（Windows）
        rel = ap
    return rel.replace(os.sep, "/")


def _excluded(rel: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    low = rel.lower()
    base = os.path.basename(rel).lower()
    for pat in patterns:
        p = pat.lower().strip()
        if not p:
            continue
        if p in low or p == base:
            return True
    return False


def discover_documents(roots: Iterable[str],
                       extensions: Sequence[str] = DEFAULT_EXTENSIONS,
                       exclude: Sequence[str] = ()) -> List[str]:
    """展开文件/目录，返回排序去重后的相对路径列表。"""
    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in extensions)
    found: List[str] = []
    for root in roots:
        if not root:
            continue
        if os.path.isfile(root):
            rel = _norm(root)
            if rel.lower().endswith(exts) and not _excluded(rel, exclude):
                found.append(rel)
            continue
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                if not name.lower().endswith(exts):
                    continue
                rel = _norm(os.path.join(dirpath, name))
                if not _excluded(rel, exclude):
                    found.append(rel)
    return sorted(set(found))


def read_text(path: str) -> str:
    """多编码尝试读取文本，统一换行为 \\n。"""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 不会失败
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha1_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def file_stat(path: str) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime": int(st.st_mtime)}
