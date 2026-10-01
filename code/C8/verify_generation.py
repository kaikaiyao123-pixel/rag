#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端验证生成链路：用本地 mock 的 OpenAI 兼容服务替代真实 Moonshot API。

为什么需要它
------------
`selftest.py` 用 `FakeListChatModel` 替换了 `gen.llm`，验证的是 LCEL 链的组装、
提示词渲染与输出解析——但**从未真正发出过一次 HTTP 请求**。
而 `MoonshotChat` 继承自 `ChatOpenAI`，其请求构造依赖 openai SDK 的版本
（本项目实测解析到 openai 1.109.1），是否与 langchain-community 0.3.27
兼容，只有真的发一次请求才能确认。

本脚本在本地起一个最小的 OpenAI 兼容服务（`/v1/chat/completions`，同时支持
流式与非流式），把 `MoonshotChat` 的请求**重定向**过去，于是可以在
**不需要真实 API Key、不访问外网** 的前提下走通完整的生成链路：

    查询路由 → 查询重写 → 三种生成模式 → 流式输出

注意：重定向**不能**靠 `base_url` / `openai_api_base` / `api_base`——这三个参数
都被 `langchain-community==0.3.27` 的 `MoonshotChat` 忽略（详见 README 5.4），
必须注入预构建的 `openai.OpenAI(...)` client。本脚本会把这一点也断言下来。

用法
----
    python verify_generation.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 模块初始化时会校验该变量；这里用占位值，真实请求全部发往本地 mock。
os.environ.setdefault("MOONSHOT_API_KEY", "sk-verify-placeholder")

# 本地回环必须绕过沙箱注入的 http_proxy，否则请求会被转发到代理而失败。
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

from langchain_community.chat_models.moonshot import MoonshotChat  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from rag_modules import GenerationIntegrationModule  # noqa: E402

PASSED = 0
FAILED = 0
REQUESTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


def reply_for(prompt: str) -> str:
    """按提示词特征返回可预期的回答，用于验证路由与生成分流。"""
    if "分类结果" in prompt:
        return "detail"
    if "请输出最终查询" in prompt:
        return "简单易做的家常菜谱"
    if "分步骤" in prompt:
        return "## 🥘 菜品介绍\n模拟分步回答"
    return "模拟基础回答"


class MockHandler(BaseHTTPRequestHandler):
    """最小 OpenAI 兼容端点。"""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # 静默，避免污染测试输出
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
        REQUESTS.append({"path": self.path, "payload": payload})

        prompt = json.dumps(payload, ensure_ascii=False)
        content = reply_for(prompt)
        stream = bool(payload.get("stream"))

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            # 分片吐出，模拟真实流式
            pieces = [content[i:i + 4] for i in range(0, len(content), 4)] or [content]
            for piece in pieces:
                chunk = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "kimi-k2-0711-preview",
                    "choices": [{"index": 0, "delta": {"content": piece},
                                 "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            done = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "kimi-k2-0711-preview",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "kimi-k2-0711-preview",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), MockHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{port}/v1"
    print(f"mock OpenAI 兼容服务已启动: {base_url}\n")

    docs = [
        Document(page_content="## 必备原料和工具\n鸡腿肉 200g、花生米 50g",
                 metadata={"dish_name": "宫保鸡丁", "category": "荤菜", "difficulty": "中等"}),
        Document(page_content="## 操作\n1. 鸡腿肉切丁腌制 15 分钟",
                 metadata={"dish_name": "宫保鸡丁", "category": "荤菜", "difficulty": "中等"}),
    ]

    print("== 1. MoonshotChat 构造与端点重定向 ==")
    import openai

    # 上游缺陷：MoonshotChat 的 base_url / openai_api_base / api_base **三个参数
    # 全部被忽略**，端点固定为 https://api.moonshot.cn/v1（详见 README 5.4）。
    # 先把这一点断言下来，避免误以为 base_url 生效。
    probe = MoonshotChat(
        model="kimi-k2-0711-preview",
        moonshot_api_key=os.environ["MOONSHOT_API_KEY"],
        base_url=base_url,
    )
    check("确认 base_url 参数被上游忽略",
          "127.0.0.1" not in str(probe.base_url), f"base_url={probe.base_url}")

    # 唯一可行的重定向方式：注入预构建的 openai client。
    try:
        llm = MoonshotChat(
            model="kimi-k2-0711-preview",
            temperature=0.1,
            max_tokens=2048,
            moonshot_api_key=os.environ["MOONSHOT_API_KEY"],
            client=openai.OpenAI(
                api_key=os.environ["MOONSHOT_API_KEY"], base_url=base_url
            ).chat.completions,
            async_client=openai.AsyncOpenAI(
                api_key=os.environ["MOONSHOT_API_KEY"], base_url=base_url
            ).chat.completions,
        )
        check("MoonshotChat 构造成功", llm is not None)
        check("端点已重定向到本地 mock",
              "127.0.0.1" in str(llm.client._client.base_url),
              f"got {llm.client._client.base_url}")
    except Exception as exc:
        check("MoonshotChat 构造成功", False, f"{type(exc).__name__}: {exc}")
        server.shutdown()
        return 1

    gen = GenerationIntegrationModule(
        model_name="kimi-k2-0711-preview", temperature=0.1, max_tokens=2048)
    gen.llm = llm

    print("\n== 2. 查询路由（真实请求，非 mock 对象） ==")
    route = gen.query_router("宫保鸡丁怎么做")
    check("query_router 返回合法类型", route in ("list", "detail", "general"), f"got {route!r}")
    check("query_router 请求已送达服务端", len(REQUESTS) >= 1, f"requests={len(REQUESTS)}")

    print("\n== 3. 查询重写 ==")
    rewritten = gen.query_rewrite("做菜")
    check("query_rewrite 返回重写结果", rewritten == "简单易做的家常菜谱", f"got {rewritten!r}")

    print("\n== 4. 三种生成模式 ==")
    ans = gen.generate_step_by_step_answer("宫保鸡丁怎么做", docs)
    check("分步指导链走通 HTTP", "菜品介绍" in ans, f"got {ans[:40]!r}")

    ans2 = gen.generate_basic_answer("什么是川菜", docs)
    check("基础回答链走通 HTTP", ans2 == "模拟基础回答", f"got {ans2[:40]!r}")

    list_ans = gen.generate_list_answer("推荐几个素菜", docs)
    check("列表回答不调用 LLM 且含菜名",
          "宫保鸡丁" in list_ans, f"got {list_ans!r}")

    print("\n== 5. 流式输出（SSE） ==")
    chunks = list(gen.generate_basic_answer_stream("什么是川菜", docs))
    check("流式返回多个分片", len(chunks) > 1, f"got {len(chunks)} 片")
    check("流式拼接结果与预期一致", "".join(chunks) == "模拟基础回答",
          f"got {''.join(chunks)!r}")

    print("\n== 6. 请求内容核对 ==")
    with_model = [r for r in REQUESTS if r["payload"].get("model")]
    check("请求体携带 model 字段", len(with_model) == len(REQUESTS),
          f"{len(with_model)}/{len(REQUESTS)}")
    check("请求体携带 messages 字段",
          all("messages" in r["payload"] for r in REQUESTS))
    check("请求路径为 /v1/chat/completions",
          all(r["path"].endswith("/chat/completions") for r in REQUESTS),
          f"paths={sorted({r['path'] for r in REQUESTS})}")
    # 路由 1 + 重写 1 + 分步 1 + 基础 1 + 流式 1 = 5；
    # 列表回答是纯规则拼接，不应产生任何请求。
    check("服务端恰好收到 5 次请求（列表回答不调用 LLM）",
          len(REQUESTS) == 5, f"got {len(REQUESTS)}")

    server.shutdown()

    print("\n" + "=" * 46)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
