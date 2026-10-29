#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""元测试：框架自验证、需求覆盖率、既有自检不回退。

覆盖需求：REQ-META-01..03

本模块必须**最后**导入 —— TC-META-002 在执行时要读取完整用例注册表来统计
覆盖率，如果它先于其他模块注册，统计结果会偏低。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from fixtures import ROOT, run_cli, tempdir, write_file

from harness import (assert_eq, assert_ge, assert_true, case,
                     load_requirements, registry, verify_harness)


def _tail(text: str, n: int = 25) -> str:
    lines = (text or "").strip().splitlines()
    return "\n".join(lines[-n:])


# 框架自验证（REQ-META-01）
@case("TC-META-001", "REQ-META-01", "执行器与断言助手通过反向注入验证",
      technique="错误推测", level="单元",
      expect="注入的 PASS/FAIL/ERROR 三种探针都被正确判定，断言助手能识别失效")
def tc_meta_001() -> None:
    ok, msgs = verify_harness()
    assert_true(ok, "框架自验证失败：\n" + "\n".join(msgs))


# 需求覆盖率与引用完整性（REQ-META-02）
@case("TC-META-002", "REQ-META-02", "需求覆盖率 100% 且无悬空需求引用",
      technique="等价类划分", level="单元",
      expect="SPEC.md 中每条需求至少被一个用例覆盖，所有 req_id 均可在 SPEC.md 中找到")
def tc_meta_002() -> None:
    reqs = load_requirements()
    assert_ge(len(reqs), 40, f"SPEC.md 需求表解析结果异常：{len(reqs)} 条")

    declared = {rid for rid, _, _ in reqs}
    cases = registry()
    referenced = {c.req_id for c in cases}

    dangling = sorted(referenced - declared)
    assert_eq(dangling, [], "以下用例引用了 SPEC.md 中不存在的需求编号")

    missing = sorted(declared - referenced)
    assert_eq(missing, [], "以下需求没有任何对应用例")

    # 用例编号也必须唯一（注册时已拦截，这里复核一次）
    ids = [c.tc_id for c in cases]
    assert_eq(len(ids), len(set(ids)), "用例编号存在重复")


# 既有自检不得回退（REQ-META-03）
@case("TC-META-003", "REQ-META-03", "既有 selftest 与 web selftest 仍然全部通过",
      technique="回归", level="系统",
      expect="selftest.py 退出码 0；rag_web.py --selftest 退出码 0")
def tc_meta_003() -> None:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    for k in ("RAG_API_KEY", "RAG_API_BASE", "RAG_MODEL"):
        env.pop(k, None)

    cp = subprocess.run([sys.executable, "selftest.py"], cwd=ROOT, env=env,
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=300)
    assert_eq(cp.returncode, 0,
              "selftest.py 失败：\n" + _tail(cp.stdout) + "\n" + _tail(cp.stderr))

    # Web 自检需要一个索引；在临时目录里现建一个，避免依赖仓库里的 .rag_index
    with tempdir("TC-META-003") as d:
        cfg_path = os.path.join(d, "config.json")
        write_file(cfg_path, json.dumps({
            "docs": ["docs"], "index_dir": os.path.join(d, "idx")}))
        ing = run_cli(["--config", cfg_path, "ingest", "-q"])
        assert_eq(ing.returncode, 0, "临时建库失败：\n" + _tail(ing.stderr))

        cp2 = subprocess.run([sys.executable, "rag_web.py", "--config", cfg_path,
                              "--selftest"], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=300)
        assert_eq(cp2.returncode, 0,
                  "rag_web.py --selftest 失败：\n"
                  + _tail(cp2.stdout) + "\n" + _tail(cp2.stderr))
