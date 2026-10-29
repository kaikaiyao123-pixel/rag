#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""共享测试夹具：确定性语料、内存索引、临时目录、本地 Web 服务。

设计约束
--------
1. **临时目录必须落在项目内**（``tests/.tmp/``）。``tempfile.mkdtemp()`` 的父目录
   是本机用户名 ``C:\\Users\\黄凯斌\\...``，含中文；本项目虽为纯标准库、不受影响，
   但为了让"路径含非 ASCII"成为**被显式测试的一项**而不是所有用例的隐含前提，
   夹具统一使用 ASCII 路径，中文路径单独由鲁棒性用例覆盖。
2. **语料必须小且确定**。索引结果要能逐字断言，因此不用 ``docs/`` 的真实语料
   （那 12 篇文档的分块结果会随 chunker 参数漂移）。``docs/`` 的真实语料只在
   集成/性能用例里用。
3. **配置不读环境变量**。夹具直接基于 ``DEFAULT_CONFIG`` 深拷贝，不经
   ``load_config``，避免 CI 上的 ``RAG_API_KEY`` 把用例拖进真实网络请求。
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Dict, Iterator, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TMP_ROOT = os.path.join(HERE, ".tmp")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ragcore.chunker import chunk_document                    # noqa: E402
from ragcore.cli import DEFAULT_CONFIG                       # noqa: E402
from ragcore.index import BM25Index                          # noqa: E402
from ragcore.loader import discover_documents, read_text      # noqa: E402


# --------------------------------------------------------------------- 临时目录
@contextlib.contextmanager
def tempdir(name: str) -> Iterator[str]:
    """在 ``tests/.tmp/<name>`` 下创建临时目录，退出时删除。"""
    os.makedirs(TMP_ROOT, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    path = os.path.join(TMP_ROOT, safe)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as f:
        f.write(content)
    return path


def write_bytes(path: str, data: bytes) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ----------------------------------------------------------------------- 语料
MINI_CORPUS: Dict[str, str] = {
    "mini/a-overview.md": (
        "# 云雀笔记\n\n"
        "## 产品定位\n\n"
        "云雀笔记是一款面向个人的本地优先笔记工具，支持双向链接与全文检索。\n\n"
        "## 导入方式\n\n"
        "支持从 Markdown 文件导入已有笔记，也支持导入 Evernote 导出的 enex 文件。\n"
    ),
    "mini/b-pricing.md": (
        "# 定价\n\n"
        "## 免费版\n\n"
        "免费版可登录 3 台设备，单个笔记上限 100 MB。\n\n"
        "## 团队版\n\n"
        "团队版每人每月 25 元，支持共享空间与变更审计。\n"
    ),
    "mini/c-faq.md": (
        "# 常见问题\n\n"
        "## 导出格式\n\n"
        "导出支持 Markdown、PDF、HTML 三种格式。\n\n"
        "## 数据存储\n\n"
        "数据默认存储在本机，开启云同步后加密上传至对象存储。\n"
    ),
    "mini/d-api.md": (
        "# 接口\n\n"
        "## 检索接口\n\n"
        "检索接口接受 query 与 top_k 两个参数，返回片段列表。\n"
    ),
}


def materialize(corpus: Dict[str, str], root: str) -> str:
    """把 ``{相对路径: 内容}`` 落盘到 root，返回 root。"""
    for rel, text in corpus.items():
        write_file(os.path.join(root, rel), text)
    return root


_MINI_INDEX: Optional[BM25Index] = None


def mini_index() -> BM25Index:
    """4 篇小语料构建的确定性索引（进程内缓存，只读复用）。"""
    global _MINI_INDEX
    if _MINI_INDEX is not None:
        return _MINI_INDEX
    chunks: List[Dict] = []
    for rel, text in MINI_CORPUS.items():
        chunks.extend(chunk_document(rel, text, 600, 120, 100))
    chunks.sort(key=lambda c: (c["doc"], c["start_line"]))
    _MINI_INDEX = BM25Index().build(chunks)
    return _MINI_INDEX


def mini_cfg() -> Dict:
    """基于 DEFAULT_CONFIG 的深拷贝，密钥清空 → 一定走离线抽取式回答。"""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["llm"]["api_key"] = ""
    cfg["retrieval"]["top_k"] = 5
    return cfg


def build_index(chunks_spec: List[Dict], **params) -> BM25Index:
    """按给定的 chunk 字典列表构建索引（用于需要精确控制内容的用例）。"""
    return BM25Index(**params).build(chunks_spec)


def chunk(doc: str, heading: str, text: str, start_line: int = 1) -> Dict:
    return {"doc": doc, "heading": heading, "text": text, "start_line": start_line}


def ingest_mini(root: str, index_dir: str) -> None:
    """把 ``tests/.tmp`` 下的 mini 语料建库到 index_dir（供 Web/CLI 集成用例使用）。"""
    rels = discover_documents([os.path.join(root, "mini")], [".md"])
    chunks: List[Dict] = []
    for rel in rels:
        chunks.extend(chunk_document(rel, read_text(rel), 600, 120, 100))
    chunks.sort(key=lambda c: (c["doc"], c["start_line"]))
    BM25Index().build(chunks).save(index_dir)


# ---------------------------------------------------------------------- HTTP
def http_request(url: str, method: str = "GET", data: Optional[bytes] = None,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: float = 15.0) -> Tuple[int, Dict[str, str], str]:
    """发起请求并返回 (状态码, 响应头, 响应体文本)；4xx/5xx 不抛异常。"""
    req = urllib.request.Request(url, data=data, headers=headers or {},
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8", "replace")


def http_json(url: str, obj: Dict, method: str = "POST",
              timeout: float = 15.0) -> Tuple[int, Dict]:
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    if method.upper() == "GET":
        status, _, body = http_request(url, method="GET", timeout=timeout)
    else:
        status, _, body = http_request(
            url, method=method, data=raw,
            headers={"Content-Type": "application/json"}, timeout=timeout)
    try:
        return status, json.loads(body)
    except ValueError:
        return status, {"_raw": body}


@contextlib.contextmanager
def web_server(index: Optional[BM25Index] = None,
               cfg: Optional[Dict] = None) -> Iterator[Tuple[str, Dict]]:
    """起一个绑定随机端口的真实 HTTP 服务，退出时关闭。"""
    from ragcore.web import make_context, make_handler

    ctx = make_context(index or mini_index(), cfg or mini_cfg())
    ctx["quiet"] = True
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ctx))
    srv.daemon_threads = True
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}", ctx
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


# ----------------------------------------------------------------------- CLI
def run_cli(args: List[str], cwd: str = ROOT,
            env_extra: Optional[Dict[str, str]] = None,
            timeout: float = 180.0) -> subprocess.CompletedProcess:
    """以子进程方式运行 ``rag.py``，剥离可能存在的真实密钥环境变量。"""
    env = dict(os.environ)
    for k in ("RAG_API_KEY", "RAG_API_BASE", "RAG_MODEL",
              "RAG_TIMEOUT", "RAG_TEMPERATURE"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "rag.py"] + args,
        cwd=cwd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout)


def run_cli_json(args: List[str], cwd: str = ROOT,
                 env_extra: Optional[Dict[str, str]] = None) -> Tuple[int, Dict, str]:
    """运行 CLI 并把 stdout 解析为 JSON；解析失败时返回空 dict 与原文。"""
    cp = run_cli(args, cwd=cwd, env_extra=env_extra)
    try:
        return cp.returncode, json.loads(cp.stdout), cp.stderr
    except ValueError:
        return cp.returncode, {}, (cp.stdout + "\n" + cp.stderr)


# ------------------------------------------------------------------- 子进程探针
def run_python(code: str, env_extra: Optional[Dict[str, str]] = None,
               cwd: str = ROOT, timeout: float = 120.0) -> subprocess.CompletedProcess:
    """在独立解释器里执行一段代码（用于验证跨进程可复现性）。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)
