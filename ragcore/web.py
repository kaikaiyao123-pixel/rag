# -*- coding: utf-8 -*-
"""本地 Web 问答界面：用标准库 http.server 把检索问答引擎暴露成一个小型 HTTP 服务。

    python rag_web.py                    # http://127.0.0.1:8765
    python rag_web.py --port 9000 --open
    python rag_web.py --selftest         # 起真实服务跑一遍自检，无需浏览器

设计约束
--------
1. **零第三方依赖**：只用标准库（http.server / json / threading / urllib）。
2. **检索链路保持离线**：服务默认只绑定 ``127.0.0.1``，不发起任何外网请求。
   仅当配置了大模型时，生成环节才按与 CLI **完全相同**的规则访问 ``api_base``。
3. **生成逻辑不复制**：直接复用 ``cli._answer_once``。否则 Web 与 CLI 会各自
   演化出不同的拒答/回退行为，而这类漂移极难被发现。

接口
----
``GET  /``             问答页面（``web/index.html``）
``GET  /api/meta``     索引统计、文档清单、检索与拒答参数、大模型可用性
``POST /api/ask``      ``{"question": str, "top_k": int?, "no_llm": bool?}``
                       → 答案、引用来源、召回片段、拒答特征
``GET  /api/llm``      当前大模型配置（密钥仅回显末 4 位）
``POST /api/llm``      ``{"api_base"?, "model"?, "api_key"?, "clear_key"?}``
                       → 写入**进程内存**，不落盘
``POST /api/llm/test`` 用当前配置发一次最小请求，验证连通性

密钥处理
--------
浏览器里填的密钥只存在于本进程内存，重启即失效；接口响应一律只回显末 4 位，
日志不打印请求体。这样即使误开了 ``--host 0.0.0.0``，抓包/日志也不会泄露密钥。
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .abstain import features as abstain_features
from .cli import _answer_once, build_llm, make_retriever, open_index
from .generator import LLMError

HTML_PATH = Path(__file__).resolve().parent.parent / "web" / "index.html"

MAX_BODY_BYTES = 64 * 1024
MAX_QUESTION_CHARS = 500
MAX_TOP_K = 20
MAX_API_BASE_CHARS = 300
MAX_MODEL_CHARS = 100
MAX_KEY_CHARS = 300
# 「测试连接」用的超时上限：让用户尽快拿到失败结论，而不是干等 60 秒
TEST_TIMEOUT_CAP = 20


# --------------------------------------------------------------------- 上下文
def make_context(index, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """构造 handler 共享的只读上下文。

    索引加载后只读，因此多线程并发检索是安全的（BM25Index 不持有可变状态）。
    唯一会被并发改写的是 ``cfg["llm"]``（前端「模型设置」），故用 ``lock`` 保护，
    并在每次请求开始时快照一份，避免读到写了一半的配置。
    """
    llm = build_llm(cfg)
    html = b""
    if HTML_PATH.is_file():
        html = HTML_PATH.read_bytes()
    return {
        "index": index,
        "cfg": cfg,
        "html": html,
        "html_path": HTML_PATH,
        # 启动时是否已有环境变量密钥：用于区分密钥来源（env / 界面 / 无）
        "env_key_present": bool(os.environ.get("RAG_API_KEY")),
        "runtime_key_set": False,
        "lock": threading.Lock(),
        "quiet": False,
        # 兼容旧字段：启动瞬间的可用性快照
        "llm_available": bool(llm.available),
        "llm_model": cfg["llm"].get("model", ""),
        "llm_base": cfg["llm"].get("api_base", ""),
    }


# ------------------------------------------------------------------ 模型配置
def _has_ctl(s: str) -> bool:
    """是否含 ASCII 控制字符（含换行）——用于阻断请求头注入。"""
    return any(ord(c) < 32 or ord(c) == 127 for c in s)


def mask_key(key: str) -> str:
    """密钥脱敏：只保留末 4 位，长度不足则完全遮蔽。"""
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * 8
    return "•" * 8 + key[-4:]


def llm_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """当前大模型配置的可对外视图（绝不含完整密钥）。"""
    with ctx["lock"]:
        l = dict(ctx["cfg"]["llm"])
    client = build_llm({"llm": l})
    key = l.get("api_key", "") or ""
    if not key:
        source = "none"
    elif ctx.get("runtime_key_set"):
        source = "runtime"
    elif ctx.get("env_key_present"):
        source = "env"
    else:
        source = "none"
    return {
        "api_base": l.get("api_base", ""),
        "model": l.get("model", ""),
        "key_set": bool(key),
        "key_masked": mask_key(key),
        "available": bool(client.available),
        "is_local": client._is_local(l.get("api_base", "")),
        "source": source,
        "temperature": l.get("temperature", 0.2),
        "timeout": l.get("timeout", 60),
    }


# ---------------------------------------------------------------------- 处理
def make_handler(ctx: Dict[str, Any]):
    """把上下文闭包进一个 handler 类，避免使用全局变量。"""

    class Handler(BaseHTTPRequestHandler):
        server_version = "RagWeb/" + __version__

        # ---------------------------------------------------------- 响应工具
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # 浏览器提前关闭连接（刷新/取消），不是错误
                pass

        def _json(self, status: int, obj: Any) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            if ctx.get("quiet"):
                return
            code = args[1] if len(args) > 1 else ""
            sys.stderr.write(f"  {self.command} {self.path} → {code}\n")

        # ------------------------------------------------------------ 路由
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]

            if path in ("/", "/index.html"):
                if not ctx["html"]:
                    self._error(500, f"页面文件缺失：{ctx['html_path']}")
                    return
                self._send(200, ctx["html"], "text/html; charset=utf-8")
                return

            if path == "/api/meta":
                self._json(200, self._meta())
                return

            if path == "/api/llm":
                self._json(200, llm_state(ctx))
                return

            if path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return

            if path in ("/api/ask", "/api/llm/test"):
                self._error(405, "请使用 POST")
                return

            self._error(404, f"未找到：{path}")

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/ask":
                self._handle_ask()
                return
            if path == "/api/llm":
                self._handle_llm_set()
                return
            if path == "/api/llm/test":
                self._handle_llm_test()
                return
            self._error(404, f"未找到：{path}")

        # ------------------------------------------------------------ 业务
        def _meta(self) -> Dict[str, Any]:
            idx = ctx["index"]
            cfg = ctx["cfg"]
            s = idx.stats()
            return {
                "version": __version__,
                "index": {
                    "built_at": s["built_at"],
                    "documents": s["documents"],
                    "chunks": s["chunks"],
                    "terms": s["terms"],
                    "avgdl": s["avgdl"],
                },
                "documents": sorted(s["per_doc"]),
                "llm": llm_state(ctx),
                "retrieval": dict(cfg["retrieval"]),
                "abstain": dict(cfg["abstain"]),
                "chunk": dict(cfg["chunk"]),
                "limits": {"max_question_chars": MAX_QUESTION_CHARS,
                           "max_top_k": MAX_TOP_K},
            }

        def _read_json(self) -> Optional[Dict[str, Any]]:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            if n <= 0 or n > MAX_BODY_BYTES:
                return None
            raw = self.rfile.read(n)
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                return None
            return obj if isinstance(obj, dict) else None

        def _handle_ask(self) -> None:
            payload = self._read_json()
            if payload is None:
                self._error(400, "请求体必须是 JSON 对象")
                return

            question = payload.get("question")
            if not isinstance(question, str) or not question.strip():
                self._error(400, "question 不能为空")
                return
            question = question.strip()
            if len(question) > MAX_QUESTION_CHARS:
                self._error(400, f"question 过长（上限 {MAX_QUESTION_CHARS} 字符）")
                return

            top_k = payload.get("top_k")
            if top_k is None:
                top_k = ctx["cfg"]["retrieval"]["top_k"]
            if not isinstance(top_k, int) or isinstance(top_k, bool) \
                    or not 1 <= top_k <= MAX_TOP_K:
                self._error(400, f"top_k 必须是 1~{MAX_TOP_K} 的整数")
                return

            no_llm = payload.get("no_llm", False)
            if not isinstance(no_llm, bool):
                self._error(400, "no_llm 必须是布尔值")
                return

            idx = ctx["index"]
            # 快照一份配置：模型设置可能被并发改写，避免读到半更新的 llm 块
            with ctx["lock"]:
                cfg_snap = dict(ctx["cfg"])
                cfg_snap["llm"] = dict(ctx["cfg"]["llm"])
            retriever = make_retriever(idx, cfg_snap, top_k=top_k)
            t0 = time.perf_counter()
            try:
                result = _answer_once(question, retriever, cfg_snap, no_llm, top_k)
            except Exception as e:  # 兜底：不让单个请求打挂服务
                self._error(500, f"处理失败：{type(e).__name__}: {e}")
                return
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            hits = result["hits"]
            feats = abstain_features(question, hits, retriever.idf,
                                     idx.postings)
            self._json(200, {
                "question": question,
                "answer": result["answer"],
                "note": result["note"],
                "abstained": result["abstained"],
                "used_llm": result["used_llm"],
                "elapsed_ms": round(elapsed_ms, 2),
                "sources": [
                    {"n": i, "doc": h["doc"], "heading": h["heading"],
                     "line": h["start_line"], "score": round(h["score"], 4)}
                    for i, h in enumerate(hits, 1)
                ],
                "hits": [
                    {"n": i, "doc": h["doc"], "heading": h["heading"],
                     "line": h["start_line"], "score": round(h["score"], 4),
                     "matched_terms": h.get("matched_terms", []),
                     "text": h["text"]}
                    for i, h in enumerate(hits, 1)
                ],
                "debug": {
                    "features": feats,
                    "thresholds": {
                        "vocab_cov": ctx["cfg"]["abstain"]["vocab_cov"],
                        "idf_cov": ctx["cfg"]["abstain"]["idf_cov"],
                        "norm_score": ctx["cfg"]["abstain"]["norm_score"],
                    },
                    "params": {
                        "top_k": top_k,
                        "candidate_k": ctx["cfg"]["retrieval"]["candidate_k"],
                        "max_per_doc": ctx["cfg"]["retrieval"]["max_per_doc"],
                    },
                },
            })

        # ------------------------------------------------------------ 模型配置
        def _handle_llm_set(self) -> None:
            payload = self._read_json()
            if payload is None:
                self._error(400, "请求体必须是 JSON 对象")
                return

            updates: Dict[str, Any] = {}

            if "api_base" in payload:
                v = payload["api_base"]
                if not isinstance(v, str):
                    self._error(400, "api_base 必须是字符串")
                    return
                v = v.strip()
                if len(v) > MAX_API_BASE_CHARS:
                    self._error(400, f"api_base 过长（上限 {MAX_API_BASE_CHARS} 字符）")
                    return
                if _has_ctl(v):
                    self._error(400, "api_base 含控制字符")
                    return
                if v and not (v.startswith("http://") or v.startswith("https://")):
                    self._error(400, "api_base 必须以 http:// 或 https:// 开头")
                    return
                updates["api_base"] = v

            if "model" in payload:
                v = payload["model"]
                if not isinstance(v, str):
                    self._error(400, "model 必须是字符串")
                    return
                v = v.strip()
                if not v:
                    self._error(400, "model 不能为空")
                    return
                if len(v) > MAX_MODEL_CHARS or _has_ctl(v):
                    self._error(400, f"model 过长（上限 {MAX_MODEL_CHARS} 字符）或含控制字符")
                    return
                updates["model"] = v

            if "api_key" in payload:
                v = payload["api_key"]
                if not isinstance(v, str):
                    self._error(400, "api_key 必须是字符串")
                    return
                v = v.strip()
                if len(v) > MAX_KEY_CHARS:
                    self._error(400, f"api_key 过长（上限 {MAX_KEY_CHARS} 字符）")
                    return
                # 换行会被拼进 Authorization 头，必须显式拒绝
                if _has_ctl(v):
                    self._error(400, "api_key 含换行或控制字符")
                    return
                updates["api_key"] = v

            if payload.get("clear_key") is True:
                updates["api_key"] = ""

            if not updates:
                self._error(400, "没有可更新的字段")
                return

            with ctx["lock"]:
                ctx["cfg"]["llm"].update(updates)
                if "api_key" in updates:
                    ctx["runtime_key_set"] = bool(updates["api_key"])

            self._json(200, {"ok": True, "llm": llm_state(ctx)})

        def _handle_llm_test(self) -> None:
            """发一次最小请求验证连通性。失败也返回 200，用 ok 字段区分结果。"""
            payload = self._read_json()
            if payload is None:
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    n = -1
                if n > 0:
                    self._error(400, "请求体必须是 JSON 对象")
                    return

            if not llm_state(ctx)["available"]:
                self._json(200, {"ok": False,
                                 "error": "尚未配置可用的服务地址 / 模型 / 密钥"})
                return

            with ctx["lock"]:
                l = dict(ctx["cfg"]["llm"])
            client = build_llm({"llm": l})
            client.timeout = min(int(l.get("timeout", 60) or 60), TEST_TIMEOUT_CAP)

            t0 = time.perf_counter()
            try:
                reply = client.chat([{"role": "user", "content": "只回复两个字母：ok"}])
            except LLMError as e:
                self._json(200, {"ok": False, "error": str(e),
                                 "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1)})
                return
            except Exception as e:  # 兜底：任何异常都不该打挂服务
                self._json(200, {"ok": False, "error": f"{type(e).__name__}: {e}",
                                 "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1)})
                return

            self._json(200, {
                "ok": True,
                "reply": (reply or "")[:120],
                "model": client.model,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            })

    return Handler


# ---------------------------------------------------------------------- 启动
def serve(cfg: Dict[str, Any], host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = False) -> int:
    index = open_index(cfg["index_dir"])
    if index is None or not index.chunks:
        print(f"索引不存在或为空：{cfg['index_dir']}\n"
              f"请先执行：python rag.py ingest", file=sys.stderr)
        return 2

    if not HTML_PATH.is_file():
        print(f"页面文件缺失：{HTML_PATH}", file=sys.stderr)
        return 2

    ctx = make_context(index, cfg)
    try:
        httpd = ThreadingHTTPServer((host, port), make_handler(ctx))
    except OSError as e:
        print(f"无法监听 {host}:{port} —— {e}\n"
              f"换一个端口：python rag_web.py --port 9000", file=sys.stderr)
        return 2

    real_host, real_port = httpd.server_address[0], httpd.server_address[1]
    s = index.stats()
    st = llm_state(ctx)
    llm_note = (f"{st['model']}（{st['source']}）" if st["available"]
                else "未配置（将使用离线抽取式回答）")
    print("=" * 62)
    print("  RAG 本地问答界面")
    print("=" * 62)
    print(f"  地址    : http://{real_host}:{real_port}")
    print(f"  索引    : {s['documents']} 篇 / {s['chunks']} 片段 / {s['terms']} 词项")
    print(f"  大模型  : {llm_note}")
    print(f"  绑定    : {host}（仅本机可访问，不产生外网请求）")
    print("=" * 62)
    print("  页面右上角「模型设置」可填 API Key（仅存内存，重启失效）")
    print("  Ctrl+C 停止")

    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(
            f"http://{real_host}:{real_port}")).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0


# -------------------------------------------------------------------- 自检
_TAG_LIST = ["html", "head", "body", "header", "main", "section", "div", "h1",
             "h2", "h3", "p", "ul", "li", "button", "textarea", "select",
             "option", "label", "span", "table", "thead", "tbody", "tr", "th",
             "td", "details", "summary", "pre", "code", "strong", "a"]

_EXT_ATTR_RE = re.compile(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", re.I)
_EXT_CSS_RE = re.compile(r"""@import|url\(\s*["']?\s*https?:""", re.I)


def check_html(html: str) -> List[Tuple[str, bool, str]]:
    """页面静态自检：标签配对、外部资源、关键结构。

    这几类缺陷无法靠肉眼可靠发现（漏一个闭合标签在浏览器里往往仍能渲染），
    因此固化成断言。返回值与 check() 的签名保持一致。
    """
    out: List[Tuple[str, bool, str]] = []
    for tag in _TAG_LIST:
        n_open = len(re.findall(r"<%s(?=[\s>])" % tag, html, re.I))
        n_close = len(re.findall(r"</%s\s*>" % tag, html, re.I))
        if n_open or n_close:
            out.append((f"HTML 标签配对 <{tag}>", n_open == n_close,
                        f"开 {n_open} / 闭 {n_close}"))
    ext = _EXT_ATTR_RE.findall(html) + _EXT_CSS_RE.findall(html)
    out.append(("页面无外部资源引用（离线可用）", not ext, str(ext[:3])))
    out.append(("页面声明 UTF-8", 'charset="utf-8"' in html.lower(), ""))
    out.append(("页面含问答表单", 'id="q"' in html and 'id="send"' in html, ""))
    out.append(("页面含调试面板容器", 'id="debug"' in html, ""))
    out.append(("页面含模型设置入口",
                'id="cfgbtn"' in html and 'id="cfgmodal"' in html, ""))
    out.append(("页面含密钥输入框（password 类型）",
                'id="apikey"' in html and 'type="password"' in html.lower(), ""))
    out.append(("页面含保存与测试按钮",
                'id="cfgsave"' in html and 'id="cfgtest"' in html, ""))
    return out


def selftest(cfg: Dict[str, Any]) -> int:
    """起一个真实 HTTP 服务并逐项断言（不依赖浏览器，也不依赖网络）。"""
    index = open_index(cfg["index_dir"])
    if index is None or not index.chunks:
        print(f"索引不存在或为空：{cfg['index_dir']}\n"
              f"请先执行：python rag.py ingest", file=sys.stderr)
        return 2

    ctx = make_context(index, cfg)
    ctx["quiet"] = True
    orig_llm = dict(cfg["llm"])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ctx))
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    passed = failed = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name} {detail}")

    def req(method: str, path: str, payload: Any = None,
            raw: Optional[bytes] = None) -> Tuple[int, str, bytes]:
        import urllib.error
        import urllib.request
        data = raw
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if data is not None:
            headers["Content-Type"] = "application/json"
        r = urllib.request.Request(base + path, data=data, headers=headers,
                                   method=method)
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    try:
        print(f"== 0. 服务启动 ==\n  监听 {base}")

        print("\n== 1. 页面静态自检 ==")
        status, ctype, body = req("GET", "/")
        html = body.decode("utf-8", "replace")
        check("GET / 返回 200", status == 200, str(status))
        check("Content-Type 为 text/html", "text/html" in ctype, ctype)
        check("页面非空且含 DOCTYPE", len(html) > 2000 and "<!doctype html>" in html.lower(),
              f"{len(html)} 字节")
        for name, ok, detail in check_html(html):
            check(name, ok, detail)

        print("\n== 2. 元信息接口 ==")
        status, _, body = req("GET", "/api/meta")
        meta = json.loads(body.decode("utf-8"))
        check("GET /api/meta 返回 200", status == 200, str(status))
        check("文档数 > 0", meta["index"]["documents"] > 0,
              str(meta["index"]))
        check("片段数与索引一致", meta["index"]["chunks"] == len(index.chunks),
              f"{meta['index']['chunks']} vs {len(index.chunks)}")
        check("文档清单非空", len(meta["documents"]) > 0)
        check("meta 不含 api_key（避免泄露密钥）",
              "api_key" not in json.dumps(meta, ensure_ascii=False), "")

        print("\n== 3. 问答接口（领域内）==")
        status, _, body = req("POST", "/api/ask", {"question": "如何导入已有笔记？"})
        r = json.loads(body.decode("utf-8"))
        check("POST /api/ask 返回 200", status == 200, str(status))
        check("答案非空", bool(r.get("answer", "").strip()), str(r.get("answer"))[:60])
        check("返回引用来源", len(r.get("sources", [])) > 0, str(len(r.get("sources", []))))
        check("未配置密钥时走离线抽取", r.get("used_llm") is False, str(r.get("used_llm")))
        check("领域内问题未被误拒", r.get("abstained") is False, str(r.get("abstained")))
        check("引用来源字段完整",
              all({"n", "doc", "heading", "line", "score"} <= set(s)
                  for s in r.get("sources", [])), "")
        check("返回拒答特征（供调试面板）",
              {"vocab_cov", "idf_cov", "norm_score"} <= set(r["debug"]["features"]), "")
        check("召回片段含全文与命中词项",
              all("text" in h and "matched_terms" in h for h in r.get("hits", [])), "")

        print("\n== 4. 问答接口（领域外应拒答）==")
        status, _, body = req("POST", "/api/ask",
                              {"question": "量子纠缠的退相干时间是多少？"})
        r = json.loads(body.decode("utf-8"))
        check("领域外问题被拒答", status == 200 and r.get("abstained") is True,
              f"status={status} abstained={r.get('abstained')}")

        print("\n== 5. 参数与错误处理 ==")
        status, _, body = req("POST", "/api/ask", {"question": "   "})
        check("空问题返回 400", status == 400, str(status))
        status, _, body = req("POST", "/api/ask", raw=b"{not json")
        check("非法 JSON 返回 400", status == 400, str(status))
        status, _, body = req("POST", "/api/ask", {"question": "导入", "top_k": 99})
        check("top_k 越界返回 400", status == 400, str(status))
        status, _, body = req("POST", "/api/ask", {"question": "导入", "top_k": "5"})
        check("top_k 类型错误返回 400", status == 400, str(status))
        status, _, body = req("POST", "/api/ask", {"question": "x" * 501})
        check("问题过长返回 400", status == 400, str(status))
        status, _, body = req("GET", "/api/ask")
        check("GET /api/ask 返回 405", status == 405, str(status))
        status, _, body = req("GET", "/api/nope")
        check("未知路径返回 404", status == 404, str(status))

        print("\n== 6. top_k 生效 ==")
        status, _, body = req("POST", "/api/ask", {"question": "导出支持哪些格式", "top_k": 2})
        r = json.loads(body.decode("utf-8"))
        check("top_k=2 时引用数不超过 2", len(r.get("sources", [])) <= 2,
              str(len(r.get("sources", []))))

        print("\n== 7. 模型配置接口 ==")
        status, _, body = req("GET", "/api/llm")
        st = json.loads(body.decode("utf-8"))
        check("GET /api/llm 返回 200", status == 200, str(status))
        check("初始状态含 key_set 字段", "key_set" in st, str(sorted(st)))

        fake = "sk-selftest-SECRET-0123456789"
        status, _, body = req("POST", "/api/llm",
                              {"api_key": fake, "model": "deepseek-chat"})
        st = json.loads(body.decode("utf-8"))
        check("POST /api/llm 保存成功", status == 200 and st.get("ok") is True, str(status))
        check("保存后 key_set 为真", st["llm"]["key_set"] is True, str(st["llm"]))
        check("密钥只回显末 4 位且完整密钥不出现在响应里",
              st["llm"]["key_masked"].endswith(fake[-4:]) and fake not in body.decode("utf-8"),
              st["llm"]["key_masked"])
        check("保存后大模型可用", st["llm"]["available"] is True, str(st["llm"]))

        status, _, body = req("GET", "/api/meta")
        check("meta 不含完整密钥", fake not in body.decode("utf-8"), "")

        # 指向本机 1 端口：连接立即被拒，可离线验证「测试连接」的失败路径
        req("POST", "/api/llm", {"api_base": "http://127.0.0.1:1"})
        status, _, body = req("POST", "/api/llm/test", {})
        t = json.loads(body.decode("utf-8"))
        check("测试连接失败时返回 ok=false",
              status == 200 and t.get("ok") is False, str(t)[:120])
        check("失败时给出错误原因", bool(t.get("error")), str(t.get("error"))[:80])

        status, _, _ = req("POST", "/api/llm", {"api_base": "ftp://x/y"})
        check("非法 api_base 协议返回 400", status == 400, str(status))
        status, _, _ = req("POST", "/api/llm", {"api_key": 123})
        check("api_key 类型错误返回 400", status == 400, str(status))
        status, _, _ = req("POST", "/api/llm", {"api_key": "abc\ndef"})
        check("api_key 含换行返回 400", status == 400, str(status))
        status, _, _ = req("POST", "/api/llm", {})
        check("空更新返回 400", status == 400, str(status))
        status, _, _ = req("GET", "/api/llm/test")
        check("GET /api/llm/test 返回 405", status == 405, str(status))

        status, _, body = req("POST", "/api/llm", {"clear_key": True})
        st = json.loads(body.decode("utf-8"))
        check("清除密钥后 key_set 为假", st["llm"]["key_set"] is False, str(st["llm"]))
        check("清除后不再回显密钥", st["llm"]["key_masked"] == "", st["llm"]["key_masked"])
    finally:
        with ctx["lock"]:
            ctx["cfg"]["llm"].clear()
            ctx["cfg"]["llm"].update(orig_llm)
            ctx["runtime_key_set"] = False
        httpd.shutdown()
        httpd.server_close()

    print("\n" + "=" * 46)
    print(f"通过 {passed} 项，失败 {failed} 项")
    return 1 if failed else 0
