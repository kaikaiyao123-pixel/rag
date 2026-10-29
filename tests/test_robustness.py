#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""鲁棒性测试：异常输入、失效依赖、并发与坏路径。

覆盖需求：REQ-CLI-05..06、REQ-GEN-10、REQ-LOD-01、REQ-LOD-04、REQ-WEB-01、REQ-WEB-07

这一组用例主要用**错误推测法**设计：不是从规格正向推导，而是从"这里最可能怎么坏"
反向构造输入。LLM 相关的失败路径用本机回环上的假服务复现，**不访问外网**。
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator, List, Tuple

from fixtures import (ROOT, http_json, http_request, mini_cfg, mini_index,
                      run_cli, run_python, tempdir, web_server, write_bytes,
                      write_file)

from ragcore.generator import LLMClient, LLMError
from ragcore.loader import read_text

from harness import (assert_eq, assert_ge, assert_in, assert_raises,
                     assert_true, case)


# ------------------------------------------------------------ 假 LLM 服务
@contextlib.contextmanager
def fake_llm(reply: Callable[[], Tuple[int, str]]) -> Iterator[str]:
    """起一个只处理 POST 的假 /chat/completions 服务，返回 api_base。"""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:                       # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            code, body = reply()
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:            # noqa: A003
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


MESSAGES = [{"role": "user", "content": "hi"}]


# ============================================================ CLI 错误路径
@case("TC-ROB-001", "REQ-CLI-05", "未发现任何文档时 ingest 返回退出码 2",
      technique="错误推测", level="系统",
      expect="退出码 2，提示「未在 ... 下发现任何文档」")
def tc_rob_001() -> None:
    with tempdir("TC-ROB-001") as d:
        os.makedirs(os.path.join(d, "empty"))
        cfg_path = os.path.join(d, "config.json")
        write_file(cfg_path, json.dumps({"docs": [os.path.join(d, "empty")],
                                         "index_dir": os.path.join(d, "idx")}))
        cp = run_cli(["--config", cfg_path, "ingest"])
        assert_eq(cp.returncode, 2)
        assert_in("未在", cp.stderr)
        assert_in("发现任何文档", cp.stderr)


@case("TC-ROB-002", "REQ-CLI-05", "索引缺失时 ask / search 返回退出码 2",
      technique="错误推测", level="系统",
      expect="退出码 2，提示「请先执行：python rag.py ingest」")
def tc_rob_002() -> None:
    with tempdir("TC-ROB-002") as d:
        cfg_path = os.path.join(d, "config.json")
        write_file(cfg_path, json.dumps({"index_dir": os.path.join(d, "none")}))
        for sub in (["ask", "导入", "--no-llm"], ["search", "导入"],
                    ["stats"], ["docs"]):
            cp = run_cli(["--config", cfg_path] + sub)
            assert_eq(cp.returncode, 2, f"{sub[0]} 应返回 2")
            assert_in("请先执行", cp.stderr, f"{sub[0]} 缺少操作指引")


@case("TC-ROB-003", "REQ-CLI-06", "配置文件损坏时回退默认配置并给出警告",
      technique="错误推测", level="系统",
      expect="退出码 0，stderr 含「警告」，生效配置为默认值")
def tc_rob_003() -> None:
    with tempdir("TC-ROB-003") as d:
        cfg_path = os.path.join(d, "config.json")
        write_file(cfg_path, "{ 这不是合法 JSON ")
        cp = run_cli(["--config", cfg_path, "config"])
        assert_eq(cp.returncode, 0, "配置损坏不应中断执行")
        assert_in("警告", cp.stderr)
        shown = json.loads(cp.stdout)
        assert_eq(shown["index_dir"], ".rag_index", "应回退默认索引目录")
        assert_eq(shown["chunk"]["max_chars"], 600)


# ==================================================== LLM 客户端失败路径
@case("TC-ROB-004", "REQ-GEN-10", "服务端返回 5xx 时抛 LLMError 且带上状态码",
      technique="错误推测", level="集成",
      expect="LLMError 信息含「HTTP 500」")
def tc_rob_004() -> None:
    with fake_llm(lambda: (500, '{"error":"internal"}')) as base:
        client = LLMClient(base, "sk-x", "m")
        assert_raises(LLMError, client.chat, MESSAGES, contains="HTTP 500")


@case("TC-ROB-005", "REQ-GEN-10", "响应不是 JSON 时抛 LLMError",
      technique="错误推测", level="集成",
      expect="LLMError 信息含「响应解析失败」")
def tc_rob_005() -> None:
    with fake_llm(lambda: (200, "<html>502 Bad Gateway</html>")) as base:
        client = LLMClient(base, "sk-x", "m")
        assert_raises(LLMError, client.chat, MESSAGES, contains="响应解析失败")


@case("TC-ROB-006", "REQ-GEN-10", "JSON 结构不符（choices 为空）时抛 LLMError",
      technique="错误推测", level="集成",
      expect="LLMError 信息含「响应解析失败」")
def tc_rob_006() -> None:
    with fake_llm(lambda: (200, '{"choices": []}')) as base:
        client = LLMClient(base, "sk-x", "m")
        assert_raises(LLMError, client.chat, MESSAGES, contains="响应解析失败")


@case("TC-ROB-007", "REQ-GEN-10", "端口不可达时抛 LLMError 而非阻塞或泄漏原始异常",
      technique="错误推测", level="集成",
      expect="直连不可达端口 → LLMError，信息含「网络错误」")
def tc_rob_007() -> None:
    # 必须放到子进程里跑，原因有二：
    # 1. urllib 的默认 opener 在**首次请求时**构造并缓存 ProxyHandler，之后再改
    #    代理环境变量已无效；
    # 2. 本机若配置了系统级 HTTP 代理且未豁免 localhost，回环请求会被代理转发，
    #    连接拒绝会变成代理返回的 502 —— 那是环境问题，不是被测代码的问题。
    # 子进程里显式设 no_proxy=* ，把「连接被拒」这个被测场景还原出来。
    code = (
        "import sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "from ragcore.generator import LLMClient, LLMError\n"
        "c = LLMClient('http://127.0.0.1:1/v1', 'sk-x', 'm', timeout=3)\n"
        "try:\n"
        "    c.chat([{'role': 'user', 'content': 'hi'}])\n"
        "except LLMError as e:\n"
        "    print('LLMERROR:' + str(e))\n"
        "except Exception as e:\n"
        "    print('RAW:' + type(e).__name__ + ':' + str(e))\n"
        "else:\n"
        "    print('NOEXC')\n"
    )
    cp = run_python(code, env_extra={"no_proxy": "*", "NO_PROXY": "*"})
    out = cp.stdout.strip()
    assert_true(out.startswith("LLMERROR:"),
                f"未得到 LLMError：{out!r}；stderr={cp.stderr[-300:]}")
    assert_in("网络错误", out)


@case("TC-ROB-008", "REQ-GEN-10", "正常响应能取到 content",
      technique="等价类划分", level="集成",
      expect="chat 返回 choices[0].message.content 的去空白结果")
def tc_rob_008() -> None:
    body = json.dumps({"choices": [{"message": {"content": "  答案文本  "}}]})
    with fake_llm(lambda: (200, body)) as base:
        client = LLMClient(base, "sk-x", "m")
        assert_eq(client.chat(MESSAGES), "答案文本")


# ================================================================ 坏输入
@case("TC-ROB-009", "REQ-LOD-04", "二进制文件不会让读取崩溃",
      technique="错误推测", level="单元",
      expect="read_text 返回 str（latin-1 兜底），不抛异常")
def tc_rob_009() -> None:
    with tempdir("TC-ROB-009") as d:
        p = os.path.join(d, "blob.md")
        write_bytes(p, bytes(range(256)) * 4)
        text = read_text(p)
        assert_true(isinstance(text, str))
        assert_ge(len(text), 1)


@case("TC-ROB-010", "REQ-LOD-01", "含空格与中文的路径可正常发现与读取",
      technique="错误推测", level="单元",
      expect="文件名含中文、目录名含空格时仍能发现并读到内容")
def tc_rob_010() -> None:
    from ragcore.loader import discover_documents

    with tempdir("TC-ROB-010") as d:
        sub = os.path.join(d, "含空格 的目录")
        write_file(os.path.join(sub, "笔记 文件.md"), "# 标题\n\n正文内容。\n")
        found = discover_documents([d], [".md"])
        assert_eq(len(found), 1)
        assert_in("笔记 文件.md", found[0])
        assert_in("正文内容", read_text(os.path.join(d, sub, "笔记 文件.md")))


# ============================================================ Web 并发与存活
@case("TC-ROB-011", "REQ-WEB-07", "并发 20 个提问请求全部成功且服务不崩溃",
      technique="错误推测", level="集成",
      expect="20 个并发请求均返回 200，且响应内容非空")
def tc_rob_011() -> None:
    results: List[int] = []
    lock = threading.Lock()

    with web_server() as (base, _):
        def worker() -> None:
            st, body = http_json(base + "/api/ask",
                                 {"question": "导入已有笔记", "no_llm": True})
            with lock:
                results.append(st)
                if st == 200:
                    assert_true(bool(body.get("answer")), "响应缺少 answer")

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert_eq(len(results), 20, "有线程未完成")
        assert_eq(sorted(set(results)), [200], f"出现非 200 响应：{sorted(set(results))}")

        # 并发之后再发一个请求，确认服务仍然健康
        assert_eq(http_json(base + "/api/ask", {"question": "导出格式"})[0], 200)


@case("TC-ROB-012", "REQ-WEB-01", "未实现的 HTTP 方法返回 501 而不是崩溃",
      technique="错误推测", level="集成",
      expect="PUT /api/ask → 501，随后 GET / 仍为 200")
def tc_rob_012() -> None:
    with web_server() as (base, _):
        st, _, _ = http_request(base + "/api/ask", method="PUT",
                                data=b"{}",
                                headers={"Content-Type": "application/json"})
        assert_eq(st, 501)
        assert_eq(http_request(base + "/")[0], 200)


@case("TC-ROB-013", "REQ-WEB-07", "答案生成抛异常时返回 500 且服务存活",
      technique="错误推测", level="集成",
      expect="注入故障后请求返回 500，恢复正常后请求返回 200")
def tc_rob_013() -> None:
    idx = mini_index()
    cfg = mini_cfg()
    with web_server(index=idx, cfg=cfg) as (base, ctx):
        # 注入故障：让索引对象在 search 时抛异常（模拟检索层崩溃）
        class Broken:
            def __getattr__(self, name):
                def boom(*a, **kw):
                    raise RuntimeError("注入故障")
                return boom

        original = ctx["index"]
        ctx["index"] = Broken()
        try:
            st, body = http_json(base + "/api/ask", {"question": "导入"})
            assert_eq(st, 500, "单个请求异常应被兜底为 500")
            assert_in("error", body)
        finally:
            ctx["index"] = original

        assert_eq(http_json(base + "/api/ask", {"question": "导入"})[0], 200,
                  "故障恢复后服务应可用")
