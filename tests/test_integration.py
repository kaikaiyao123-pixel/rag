#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""集成测试：CLI 端到端链路 与 Web 层 HTTP 契约。

覆盖需求：REQ-CLI-03..04、REQ-WEB-01..06、REQ-WEB-08..09

Web 用例在**随机端口**上起真实服务，每个用例一个独立上下文，互不干扰；
CLI 用例以子进程调用真实入口 ``rag.py``，并剥离 ``RAG_*`` 环境变量，
保证不会因为开发机上的真实密钥而访问外网。
"""
from __future__ import annotations

import json
import os
import re

from fixtures import (MINI_CORPUS, http_json, http_request, materialize,
                      mini_cfg, mini_index, run_cli, run_cli_json, tempdir,
                      web_server, write_file)

from harness import (assert_eq, assert_ge, assert_in, assert_not_in,
                     assert_true, case)

REFUSAL = "根据现有资料无法回答该问题。"


# ==================================================== CLI 端到端（系统层级）
@case("TC-INT-001", "REQ-CLI-03", "CLI 全链路：ingest → search → ask → stats",
      technique="等价类划分", level="系统",
      expect="四个子命令退出码均为 0，检索命中正确文档，统计输出可读")
def tc_int_001() -> None:
    with tempdir("TC-INT-001") as d:
        materialize(MINI_CORPUS, d)
        cfg_path = os.path.join(d, "config.json")
        write_file(cfg_path, json.dumps({
            "docs": [os.path.join(d, "mini")],
            "index_dir": os.path.join(d, "idx"),
        }))

        cp = run_cli(["--config", cfg_path, "ingest"])
        assert_eq(cp.returncode, 0, f"ingest 失败：{cp.stderr}")
        assert_in("索引完成：4 篇文档 / 7 个片段", cp.stdout)

        rc, hits, err = run_cli_json(["--config", cfg_path, "search",
                                      "导入已有笔记", "--json"])
        assert_eq(rc, 0, f"search 失败：{err}")
        assert_true(isinstance(hits, list) and hits, "应返回非空命中列表")
        assert_in("a-overview", hits[0]["doc"])

        rc, obj, err = run_cli_json(["--config", cfg_path, "ask",
                                     "如何导入已有笔记", "--json", "--no-llm"])
        assert_eq(rc, 0, f"ask 失败：{err}")
        assert_eq(obj.get("question"), "如何导入已有笔记")
        assert_eq(obj.get("used_llm"), False, "带 --no-llm 时不得调用大模型")
        assert_ge(len(obj.get("sources") or []), 1, "应给出引用来源")
        assert_true(obj.get("answer") != REFUSAL, "域内问题不应被拒答")

        cp4 = run_cli(["--config", cfg_path, "stats"])
        assert_eq(cp4.returncode, 0)
        m = re.search(r"片段数\s*:\s*(\d+)", cp4.stdout)
        assert_true(m is not None, f"stats 输出缺少片段数：{cp4.stdout}")
        assert_eq(int(m.group(1)), 7)


@case("TC-INT-002", "REQ-CLI-04", "增量 ingest 复用缓存并在文档删除后移除片段",
      technique="边界值分析", level="系统",
      expect="第二次运行复用 4 篇；删除 1 篇后提示移除 1 篇、文档数降为 3")
def tc_int_002() -> None:
    with tempdir("TC-INT-002") as d:
        materialize(MINI_CORPUS, d)
        cfg_path = os.path.join(d, "config.json")
        write_file(cfg_path, json.dumps({
            "docs": [os.path.join(d, "mini")],
            "index_dir": os.path.join(d, "idx"),
        }))

        assert_eq(run_cli(["--config", cfg_path, "ingest"]).returncode, 0)
        again = run_cli(["--config", cfg_path, "ingest"])
        assert_eq(again.returncode, 0)
        assert_in("复用缓存：4 篇", again.stdout, "未变更文档应走缓存")

        os.remove(os.path.join(d, "mini", "d-api.md"))
        after = run_cli(["--config", cfg_path, "ingest"])
        assert_eq(after.returncode, 0)
        assert_in("移除：1 篇", after.stdout)
        assert_in("索引完成：3 篇文档", after.stdout)


# ======================================================= Web 层（集成层级）
@case("TC-INT-003", "REQ-WEB-01", "路由与状态码契约",
      technique="判定表", level="集成",
      expect="/ →200；/favicon.ico →204；未知路径 →404；GET /api/ask →405")
def tc_int_003() -> None:
    with web_server() as (base, _):
        st, headers, body = http_request(base + "/")
        assert_eq(st, 200)
        assert_in("text/html", headers.get("Content-Type", ""))
        assert_ge(len(body), 1000, "首页应返回完整 HTML")

        assert_eq(http_request(base + "/favicon.ico")[0], 204)

        st, _, body = http_request(base + "/api/nope")
        assert_eq(st, 404)
        assert_in("未找到", body)

        assert_eq(http_request(base + "/api/ask")[0], 405)
        assert_eq(http_request(base + "/api/llm/test")[0], 405)
        assert_eq(http_request(base + "/api/nope", method="POST")[0], 404)


@case("TC-INT-004", "REQ-WEB-01", "/api/meta 返回索引统计与生效参数",
      technique="等价类划分", level="集成",
      expect="含 index / documents / llm / retrieval / abstain / limits 六组字段")
def tc_int_004() -> None:
    with web_server() as (base, _):
        st, meta = http_json(base + "/api/meta", {}, method="GET")
        assert_eq(st, 200)
        for key in ("version", "index", "documents", "llm",
                    "retrieval", "abstain", "chunk", "limits"):
            assert_in(key, meta)
        assert_eq(meta["index"]["documents"], 4)
        assert_eq(meta["index"]["chunks"], 7)
        assert_eq(meta["limits"]["max_top_k"], 20)


@case("TC-INT-005", "REQ-WEB-02", "/api/ask 请求校验（判定表六行）",
      technique="判定表", level="集成",
      expect="非对象 / 空问题 / 超长问题 / top_k 越界 / 非整数 / 非布尔 一律 400")
def tc_int_005() -> None:
    with web_server() as (base, _):
        url = base + "/api/ask"
        # 行 1：请求体不是 JSON 对象
        st, _, _ = http_request(url, method="POST", data=b"[]",
                                headers={"Content-Type": "application/json"})
        assert_eq(st, 400, "数组不是 JSON 对象")
        # 行 2：question 为空
        assert_eq(http_json(url, {"question": "   "})[0], 400)
        # 行 3：question 超长（上限 500）
        assert_eq(http_json(url, {"question": "问" * 501})[0], 400)
        assert_eq(http_json(url, {"question": "问" * 500})[0], 200,
                  "恰好 500 字应被接受")
        # 行 4：top_k 越界
        assert_eq(http_json(url, {"question": "导入", "top_k": 0})[0], 400)
        assert_eq(http_json(url, {"question": "导入", "top_k": 21})[0], 400)
        assert_eq(http_json(url, {"question": "导入", "top_k": 20})[0], 200,
                  "恰好 20 应被接受")
        # 行 5：top_k 类型错误（布尔是 int 子类，必须单独拦截）
        assert_eq(http_json(url, {"question": "导入", "top_k": True})[0], 400)
        assert_eq(http_json(url, {"question": "导入", "top_k": "5"})[0], 400)
        # 行 6：no_llm 非布尔
        assert_eq(http_json(url, {"question": "导入", "no_llm": "yes"})[0], 400)


@case("TC-INT-006", "REQ-WEB-05", "/api/ask 与 CLI 的回答完全一致（不得分叉）",
      technique="回归", level="集成",
      expect="同一问题、同一 top_k 下两侧 answer 字符串逐字相同")
def tc_int_006() -> None:
    from ragcore.cli import _answer_once, make_retriever

    question = "团队版多少钱"
    idx = mini_index()
    cfg = mini_cfg()
    cli_result = _answer_once(question, make_retriever(idx, cfg, top_k=3),
                              cfg, True, 3)

    with web_server() as (base, _):
        st, body = http_json(base + "/api/ask",
                             {"question": question, "no_llm": True, "top_k": 3})
        assert_eq(st, 200)
        assert_in("debug", body)
        assert_in("features", body["debug"])
        assert_eq(body["answer"], cli_result["answer"],
                  "Web 与 CLI 的答案必须一致，否则说明生成逻辑出现了分叉")
        assert_eq(len(body["hits"]), len(cli_result["hits"]),
                  "召回片段数也必须一致")
        assert_eq([h["doc"] for h in body["hits"]],
                  [h["doc"] for h in cli_result["hits"]],
                  "召回顺序也必须一致")


@case("TC-INT-007", "REQ-WEB-03", "密钥只以掩码形式出现，响应体不含明文",
      technique="错误推测", level="集成",
      expect="POST 后 GET /api/llm 与 /api/meta 的原始响应文本中找不到明文密钥")
def tc_int_007() -> None:
    secret = "sk-test-ABCD1234efgh"
    with web_server() as (base, _):
        st, _, body = http_request(
            base + "/api/llm", method="POST",
            data=json.dumps({"api_key": secret}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        assert_eq(st, 200)
        assert_not_in(secret, body, "更新响应不得回显明文")
        assert_in("efgh", body, "应回显末 4 位")
        assert_in("•", body, "应使用掩码字符")

        for path in ("/api/llm", "/api/meta", "/"):
            _, _, raw = http_request(base + path)
            assert_not_in(secret, raw, f"{path} 泄漏了明文密钥")


@case("TC-INT-008", "REQ-WEB-04", "含控制字符的配置项一律 400（头注入防护）",
      technique="错误推测", level="集成",
      expect="api_key / api_base / model 含换行、制表符时均被拒")
def tc_int_008() -> None:
    with web_server() as (base, _):
        url = base + "/api/llm"
        assert_eq(http_json(url, {"api_key": "sk-abc\nInjected: 1"})[0], 400)
        assert_eq(http_json(url, {"api_base": "http://x\r\nHost: evil"})[0], 400)
        assert_eq(http_json(url, {"model": "m\tx"})[0], 400)
        assert_eq(http_json(url, {"api_base": "ftp://x"})[0], 400,
                  "非 http(s) 协议应被拒")
        assert_eq(http_json(url, {"model": ""})[0], 400, "model 不得为空")
        assert_eq(http_json(url, {})[0], 400, "空更新应被拒")


@case("TC-INT-009", "REQ-WEB-06", "请求体非 JSON 或超过 64KB 时返回 400",
      technique="边界值分析", level="集成",
      expect="垃圾字节与 70KB 请求体都返回 400，且服务不崩溃")
def tc_int_009() -> None:
    with web_server() as (base, _):
        url = base + "/api/ask"
        st, _, _ = http_request(url, method="POST", data=b"\x00\x01garbage",
                                headers={"Content-Type": "application/json"})
        assert_eq(st, 400)

        big = json.dumps({"question": "导入", "pad": "x" * 70000}).encode("utf-8")
        st, _, _ = http_request(url, method="POST", data=big,
                                headers={"Content-Type": "application/json"})
        assert_eq(st, 400, "超过 MAX_BODY_BYTES 应被拒")

        # 服务仍存活
        assert_eq(http_json(url, {"question": "导入已有笔记"})[0], 200)


@case("TC-INT-010", "REQ-WEB-08", "/api/llm 更新后可读回，本地地址判定正确",
      technique="等价类划分", level="集成",
      expect="保存 api_base/model 后 GET 反映新值，is_local=True")
def tc_int_010() -> None:
    with web_server() as (base, _):
        st, body = http_json(base + "/api/llm", {
            "api_base": "http://127.0.0.1:11434/v1", "model": "qwen2.5:7b"})
        assert_eq(st, 200)
        assert_eq(body.get("ok"), True)
        assert_eq(body["llm"]["api_base"], "http://127.0.0.1:11434/v1")
        assert_eq(body["llm"]["model"], "qwen2.5:7b")
        assert_true(body["llm"]["is_local"], "回环地址应判定为本地服务")
        assert_eq(body["llm"]["key_set"], False)

        st, cur = http_json(base + "/api/llm", {}, method="GET")
        assert_eq(st, 200)
        assert_eq(cur["model"], "qwen2.5:7b")


@case("TC-INT-011", "REQ-WEB-09", "前端页面满足自包含约束",
      technique="回归", level="集成",
      expect="check_html 全部检查项通过：无外链、无 innerHTML、关键控件齐备")
def tc_int_011() -> None:
    from ragcore.web import HTML_PATH, check_html

    assert_true(HTML_PATH.is_file(), f"页面文件缺失：{HTML_PATH}")
    html = HTML_PATH.read_text(encoding="utf-8")
    checks = check_html(html)
    assert_ge(len(checks), 10, "检查项数量异常，可能被裁剪")
    failed = [f"{name}：{detail}" for name, ok, detail in checks if not ok]
    assert_eq(failed, [], "自包含检查存在失败项")


@case("TC-INT-012", "REQ-WEB-10", "查询在途时提供进度反馈与取消能力",
      technique="错误推测", level="集成",
      expect="页面含取消按钮与 AbortController；超时上限取自服务端 llm.timeout；每秒刷新等待时长")
def tc_int_012() -> None:
    from ragcore.web import HTML_PATH

    html = HTML_PATH.read_text(encoding="utf-8")
    # 用户实测反馈「查询中卡死」：服务端并没有阻塞（配置快照后即释放锁），
    # 卡住的是单个请求，但页面既没有进度也没有取消入口，几十秒的等待
    # 在观感上就是死机。这三条断言锁住该能力，防止后续改版被删掉。
    assert_in('id="cancel"', html)
    assert_in("AbortController", html)
    assert_in("meta.llm.timeout", html, "客户端超时应取自服务端配置，不能写死")
    assert_in("setInterval", html, "已等待时长需定时刷新")
    assert_not_in("innerHTML", html, "取消/计时逻辑不得引入 innerHTML")
