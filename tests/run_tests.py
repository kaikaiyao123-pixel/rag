#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试套件入口。

    python tests/run_tests.py                  # 全量执行，写出 test_report.md
    python tests/run_tests.py --junit          # 同时写出 test_report.xml（供 CI）
    python tests/run_tests.py --filter chunker # 只跑编号/标题/需求含该关键字的用例
    python tests/run_tests.py --verify-harness # 只做框架自验证
    python tests/run_tests.py --list           # 只列出用例清单

退出码：0 = 全部通过；1 = 存在失败或错误；2 = 环境/规格问题（无法开始执行）。
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import harness                                                # noqa: E402

# 导入顺序 = 报告中的用例顺序。test_meta 必须最后（它统计覆盖率）。
MODULES = (
    "test_unit_text",
    "test_unit_index",
    "test_unit_abstain",
    "test_unit_generator",
    "test_unit_config",
    "test_integration",
    "test_robustness",
    "test_regression",
    "test_performance",
    "test_meta",
)

REPORT_MD = os.path.join(HERE, "test_report.md")
REPORT_XML = os.path.join(HERE, "test_report.xml")


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")     # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _load_modules() -> None:
    for name in MODULES:
        importlib.import_module(name)


def _print_summary(results) -> None:
    cnt = harness._counts(results)
    total = len(results)
    print()
    print("=" * 62)
    print(f"用例总数 {total}　通过 {cnt[harness.PASS]}　失败 {cnt[harness.FAIL]}　"
          f"错误 {cnt[harness.ERROR]}")
    print(f"通过率 {harness._pct(cnt[harness.PASS], total)}　"
          f"总耗时 {sum(r.elapsed_ms for r in results) / 1000.0:.2f}s")
    print("=" * 62)

    bad = [r for r in results if r.status != harness.PASS]
    if bad:
        print("\n未通过用例：")
        for r in bad:
            print(f"  {r.status:<5} {r.case.tc_id}  {r.case.title}")
            for line in r.detail.splitlines()[:4]:
                print(f"        {line}")


def _coverage_note() -> None:
    """覆盖率始终按**全量**注册表统计，不受 --filter 影响。"""
    reqs = harness.load_requirements()
    covered = {c.req_id for c in harness.registry()}
    n = len([r for r, _, _ in reqs if r in covered])
    print(f"需求覆盖率 {n}/{len(reqs)} = {harness._pct(n, len(reqs))}")


def main(argv=None) -> int:
    _setup_console()
    ap = argparse.ArgumentParser(prog="run_tests.py",
                                 description="RAG 引擎测试套件（零第三方依赖）")
    ap.add_argument("--filter", default="", metavar="关键字",
                    help="只执行编号 / 标题 / 需求编号含该关键字的用例")
    ap.add_argument("--junit", action="store_true", help="同时输出 JUnit XML")
    ap.add_argument("--no-report", action="store_true", help="不写报告文件")
    ap.add_argument("--quiet", action="store_true", help="不逐条打印")
    ap.add_argument("--list", action="store_true", help="列出用例后退出")
    ap.add_argument("--verify-harness", action="store_true",
                    help="只做框架反向验证后退出")
    args = ap.parse_args(argv)

    if args.verify_harness:
        ok, msgs = harness.verify_harness()
        for line in msgs:
            print("  " + line)
        print(f"\n框架自验证：{'通过' if ok else '失败'}")
        return 0 if ok else 1

    _load_modules()
    all_cases = harness.registry()
    if not all_cases:
        print("未注册任何用例，检查 MODULES 与各测试模块的导入是否正常。",
              file=sys.stderr)
        return 2

    if args.list:
        print(f"共 {len(all_cases)} 个用例：\n")
        for c in all_cases:
            print(f"  {c.tc_id}  {c.req_id}  [{c.level}/{c.technique}]  {c.title}")
        return 0

    cases = all_cases
    if args.filter:
        kw = args.filter.lower()
        cases = [c for c in all_cases
                 if kw in c.tc_id.lower() or kw in c.title.lower()
                 or kw in c.req_id.lower()]
        if not cases:
            print(f"没有匹配「{args.filter}」的用例。", file=sys.stderr)
            return 2
        print(f"按「{args.filter}」筛选出 {len(cases)} 个用例\n")

    reqs = harness.load_requirements()
    print(f"RAG 引擎测试套件：{len(cases)} 个用例"
          f"（SPEC.md 声明 {len(reqs)} 条需求）\n")

    results = harness.run(cases, progress=not args.quiet)
    _print_summary(results)
    _coverage_note()

    if not args.no_report:
        harness.write_markdown(results, reqs, REPORT_MD)
        print(f"\n报告：{REPORT_MD}")
        if args.junit:
            harness.write_junit(results, REPORT_XML)
            print(f"JUnit：{REPORT_XML}")

    cnt = harness._counts(results)
    return 0 if (cnt[harness.FAIL] == 0 and cnt[harness.ERROR] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
