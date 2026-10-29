#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试执行框架（零第三方依赖）。

为什么自研而不直接用 ``unittest`` / ``pytest``
--------------------------------------------
1. 项目定位是"零第三方依赖"，``pytest`` 不可引入；
2. 标准库 ``unittest`` 无法承载**需求追溯**元数据（TC-ID ↔ REQ-ID ↔ 设计方法），
   而追溯矩阵恰恰是本测试集区别于"随手写的自检脚本"的核心；
3. 框架自身必须可被反向验证 —— ``verify_harness()`` 注入必然失败与必然异常的
   用例，确认执行器确实把它们分别判为 FAIL 与 ERROR。**一个永远输出 PASS 的
   框架比没有测试更危险**，因为它会给出虚假的质量保证。

元数据约定
----------
每个用例必须声明：

- ``tc_id``    用例编号，形如 ``TC-TOK-001``，全库唯一（重复注册直接报错）
- ``req_id``   对应的需求编号，必须存在于 ``SPEC.md`` 的需求表中
- ``title``    一句话说明测什么
- ``technique`` 设计方法：等价类划分 / 边界值分析 / 判定表 / 错误推测 / 回归 / 性能
- ``level``    测试层级：单元 / 集成 / 系统 / 非功能
- ``expect``   预期结果（写给读报告的人看，不参与判定）
"""
from __future__ import annotations

import datetime
import io
import os
import platform
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPEC_PATH = os.path.join(HERE, "SPEC.md")

PASS, FAIL, ERROR, SKIP = "PASS", "FAIL", "ERROR", "SKIP"

# 测试层级（报告按此顺序分组）
LEVELS = ("单元", "集成", "系统", "非功能")

# 设计方法（报告按此顺序分组）
TECHNIQUES = ("等价类划分", "边界值分析", "判定表", "错误推测", "回归", "性能")


# ====================================================================== 用例模型
@dataclass
class Case:
    tc_id: str
    req_id: str
    title: str
    technique: str
    level: str
    expect: str
    fn: Callable[[], None]
    source: str = ""


@dataclass
class Result:
    case: Case
    status: str
    detail: str = ""
    elapsed_ms: float = 0.0
    module: str = ""


_REGISTRY: List[Case] = []
_IDS: Dict[str, str] = {}


def case(tc_id: str, req_id: str, title: str, *,
         technique: str, level: str, expect: str) -> Callable:
    """把一个测试函数登记为用例。编号重复立即报错，避免追溯矩阵出现歧义。"""
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        if tc_id in _IDS:
            raise RuntimeError(
                f"用例编号重复：{tc_id}（已注册于 {_IDS[tc_id]}，"
                f"本次来自 {fn.__module__}）")
        if technique not in TECHNIQUES:
            raise RuntimeError(f"{tc_id}：未知的设计方法「{technique}」")
        if level not in LEVELS:
            raise RuntimeError(f"{tc_id}：未知的测试层级「{level}」")
        _IDS[tc_id] = fn.__module__
        _REGISTRY.append(Case(tc_id, req_id, title, technique, level,
                              expect, fn, fn.__module__))
        return fn
    return deco


def registry() -> List[Case]:
    return list(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
    _IDS.clear()


# ======================================================================== 断言
def fail(msg: str) -> None:
    raise AssertionError(msg)


def assert_true(cond, msg: str = "") -> None:
    if not cond:
        raise AssertionError(msg or "期望条件为真，实际为假")


def assert_false(cond, msg: str = "") -> None:
    if cond:
        raise AssertionError(msg or "期望条件为假，实际为真")


def assert_eq(actual, expected, msg: str = "") -> None:
    if actual != expected:
        prefix = f"{msg}：" if msg else ""
        raise AssertionError(f"{prefix}期望 {expected!r}，实际 {actual!r}")


def assert_ne(actual, unexpected, msg: str = "") -> None:
    if actual == unexpected:
        prefix = f"{msg}：" if msg else ""
        raise AssertionError(f"{prefix}期望不等于 {unexpected!r}，实际相等")


def assert_in(item, container, msg: str = "") -> None:
    if item not in container:
        prefix = f"{msg}：" if msg else ""
        raise AssertionError(f"{prefix}{item!r} 不在 {container!r} 中")


def assert_not_in(item, container, msg: str = "") -> None:
    if item in container:
        prefix = f"{msg}：" if msg else ""
        raise AssertionError(f"{prefix}{item!r} 意外出现在 {container!r} 中")


def assert_ge(actual, floor, msg: str = "") -> None:
    if not actual >= floor:
        prefix = f"{msg}：" if msg else ""
        raise AssertionError(f"{prefix}期望 >= {floor!r}，实际 {actual!r}")


def assert_le(actual, ceil, msg: str = "") -> None:
    if not actual <= ceil:
        prefix = f"{msg}：" if msg else ""
        raise AssertionError(f"{prefix}期望 <= {ceil!r}，实际 {actual!r}")


def assert_between(actual, low, high, msg: str = "") -> None:
    if not (low <= actual <= high):
        prefix = f"{msg}：" if msg else ""
        raise AssertionError(f"{prefix}期望落在 [{low!r}, {high!r}]，实际 {actual!r}")


def assert_close(actual, expected, tol: float = 1e-9, msg: str = "") -> None:
    if abs(actual - expected) > tol:
        prefix = f"{msg}：" if msg else ""
        raise AssertionError(f"{prefix}期望 {expected!r}±{tol}，实际 {actual!r}")


def assert_raises(exc_type, fn: Callable, *args, contains: str = "", **kwargs):
    """断言 fn 抛出指定异常，可选校验异常信息包含某个子串。"""
    try:
        fn(*args, **kwargs)
    except exc_type as e:                                    # noqa: PERF203
        if contains and contains not in str(e):
            raise AssertionError(
                f"异常信息不含「{contains}」：{type(e).__name__}: {e}") from None
        return e
    except Exception as e:                                   # noqa: BLE001
        raise AssertionError(
            f"期望抛出 {exc_type.__name__}，实际抛出 {type(e).__name__}: {e}") from None
    raise AssertionError(f"期望抛出 {exc_type.__name__}，但函数正常返回")


# ======================================================================== 执行
def run(cases: Sequence[Case], progress: bool = True,
        stream=None) -> List[Result]:
    stream = stream if stream is not None else sys.stdout
    results: List[Result] = []
    for c in cases:
        t0 = time.perf_counter()
        status, detail = PASS, ""
        try:
            c.fn()
        except AssertionError as e:
            status = FAIL
            detail = str(e) or "断言失败（无描述）"
        except Exception as e:                               # noqa: BLE001
            status = ERROR
            tb = traceback.format_exc(limit=6).strip().splitlines()
            detail = f"{type(e).__name__}: {e}"
            if tb:
                detail += "\n" + "\n".join("    " + ln for ln in tb[-6:])
        elapsed = (time.perf_counter() - t0) * 1000.0
        res = Result(c, status, detail, elapsed)
        results.append(res)
        if progress:
            stream.write(f"  [{status:<5}] {c.tc_id}  {c.title}  "
                         f"({elapsed:.1f}ms)\n")
            if status != PASS:
                for line in detail.splitlines():
                    stream.write(f"          {line}\n")
            stream.flush()
    return results


# ==================================================================== 需求解析
_REQ_ROW = re.compile(
    r"^\|\s*(REQ-[A-Z]{2,4}-\d{2})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")


def load_requirements(spec_path: str = SPEC_PATH) -> List[Tuple[str, str, str]]:
    """从 SPEC.md 的需求表解析出 [(REQ-ID, 模块, 需求描述), ...]。

    SPEC.md 是需求的**唯一来源**：测试代码只引用编号，描述文本由文档提供。
    这样避免"文档写一套、代码写一套"的双份维护漂移。
    """
    if not os.path.isfile(spec_path):
        return []
    out: List[Tuple[str, str, str]] = []
    with open(spec_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _REQ_ROW.match(line)
            if m:
                out.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    return out


# ==================================================================== 报告生成
def _pct(a: int, b: int) -> str:
    return f"{(100.0 * a / b):.1f}%" if b else "—"


def _counts(results: Sequence[Result]) -> Dict[str, int]:
    d = {PASS: 0, FAIL: 0, ERROR: 0, SKIP: 0}
    for r in results:
        d[r.status] = d.get(r.status, 0) + 1
    return d


def _env_lines() -> List[str]:
    lines = [
        f"- Python：{platform.python_version()}（{platform.python_implementation()}）",
        f"- 操作系统：{platform.system()} {platform.release()} {platform.machine()}",
        f"- 工作目录：`{ROOT}`",
    ]
    try:
        from ragcore import __version__ as core_version
        lines.insert(0, f"- 被测版本：ragcore {core_version}")
    except Exception:                                        # noqa: BLE001
        pass
    return lines


def write_markdown(results: Sequence[Result],
                   requirements: Sequence[Tuple[str, str, str]],
                   path: str,
                   title: str = "RAG 引擎测试报告") -> str:
    req_map = {rid: (mod, desc) for rid, mod, desc in requirements}
    cnt = _counts(results)
    total = len(results)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_ms = sum(r.elapsed_ms for r in results)

    def module_of(r: Result) -> str:
        return req_map.get(r.case.req_id, ("未登记", ""))[0]

    for r in results:
        r.module = module_of(r)

    L: List[str] = []
    L.append(f"# {title}")
    L.append("")
    L.append(f"> 生成时间：{now}　|　总耗时：{total_ms / 1000.0:.2f} s")
    L.append("")
    L.extend(_env_lines())
    L.append("")
    L.append("## 1. 结果总览")
    L.append("")
    L.append("| 指标 | 数值 |")
    L.append("| --- | --- |")
    L.append(f"| 用例总数 | {total} |")
    L.append(f"| 通过 | {cnt[PASS]} |")
    L.append(f"| 失败 | {cnt[FAIL]} |")
    L.append(f"| 错误（非断言异常） | {cnt[ERROR]} |")
    L.append(f"| 通过率 | {_pct(cnt[PASS], total)} |")
    L.append("")

    # ---- 按模块
    L.append("## 2. 按模块统计")
    L.append("")
    L.append("| 模块 | 用例数 | 通过 | 失败 | 错误 | 通过率 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    mods: Dict[str, List[Result]] = {}
    for r in results:
        mods.setdefault(r.module, []).append(r)
    for mod in sorted(mods):
        rs = mods[mod]
        c = _counts(rs)
        L.append(f"| {mod} | {len(rs)} | {c[PASS]} | {c[FAIL]} | {c[ERROR]} | "
                 f"{_pct(c[PASS], len(rs))} |")
    L.append("")

    # ---- 按层级
    L.append("## 3. 按测试层级")
    L.append("")
    L.append("| 层级 | 用例数 | 通过 | 失败 | 错误 | 通过率 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for lv in LEVELS:
        rs = [r for r in results if r.case.level == lv]
        if not rs:
            continue
        c = _counts(rs)
        L.append(f"| {lv} | {len(rs)} | {c[PASS]} | {c[FAIL]} | {c[ERROR]} | "
                 f"{_pct(c[PASS], len(rs))} |")
    L.append("")

    # ---- 按设计方法
    L.append("## 4. 按测试设计方法")
    L.append("")
    L.append("| 设计方法 | 用例数 | 通过 | 失败 | 错误 |")
    L.append("| --- | --- | --- | --- | --- |")
    for tech in TECHNIQUES:
        rs = [r for r in results if r.case.technique == tech]
        if not rs:
            continue
        c = _counts(rs)
        L.append(f"| {tech} | {len(rs)} | {c[PASS]} | {c[FAIL]} | {c[ERROR]} |")
    L.append("")

    # ---- 追溯矩阵
    L.append("## 5. 需求追溯矩阵")
    L.append("")
    by_req: Dict[str, List[Result]] = {}
    for r in results:
        by_req.setdefault(r.case.req_id, []).append(r)
    L.append("| 需求 ID | 模块 | 需求描述 | 用例 | 结果 |")
    L.append("| --- | --- | --- | --- | --- |")
    for rid, mod, desc in requirements:
        rs = by_req.get(rid, [])
        if rs:
            c = _counts(rs)
            mark = "通过" if c[FAIL] == 0 and c[ERROR] == 0 else "**未通过**"
            ids = "、".join(r.case.tc_id for r in rs)
        else:
            mark = "**无用例**"
            ids = "—"
        L.append(f"| {rid} | {mod} | {desc} | {ids} | {mark} |")
    unknown = sorted(set(by_req) - {rid for rid, _, _ in requirements})
    if unknown:
        L.append("")
        L.append("> 以下用例引用了 SPEC.md 中不存在的需求编号："
                 + "、".join(unknown))
    covered = len([rid for rid, _, _ in requirements if by_req.get(rid)])
    L.append("")
    L.append(f"需求覆盖率：**{covered}/{len(requirements)} = "
             f"{_pct(covered, len(requirements))}**")
    L.append("")

    # ---- 缺陷明细
    L.append("## 6. 失败与错误明细")
    L.append("")
    bad = [r for r in results if r.status != PASS]
    if not bad:
        L.append("无。全部用例通过。")
    else:
        for r in bad:
            L.append(f"### {r.case.tc_id}　{r.case.title}")
            L.append("")
            L.append(f"- 需求：`{r.case.req_id}`　层级：{r.case.level}　"
                     f"方法：{r.case.technique}")
            L.append(f"- 预期：{r.case.expect}")
            L.append(f"- 实际：**{r.status}**")
            L.append("- 详情：")
            L.append("")
            L.append("```")
            L.append(r.detail or "(无)")
            L.append("```")
            L.append("")
    L.append("")

    # ---- 全量清单
    L.append("## 7. 用例清单")
    L.append("")
    L.append("| 用例 ID | 需求 | 标题 | 层级 | 设计方法 | 预期 | 结果 | 耗时(ms) |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        c = r.case
        expect = c.expect.replace("|", "\\|")
        L.append(f"| {c.tc_id} | {c.req_id} | {c.title} | {c.level} | "
                 f"{c.technique} | {expect} | {r.status} | {r.elapsed_ms:.1f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("> 本报告由 `tests/run_tests.py` 自动生成，请勿手工编辑。")
    L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


def write_junit(results: Sequence[Result], path: str,
                suite_name: str = "rag.tests") -> str:
    """输出 JUnit XML，供 CI（Jenkins / GitLab / GitHub Actions）直接消费。"""
    cnt = _counts(results)
    suite = ET.Element("testsuite", {
        "name": suite_name,
        "tests": str(len(results)),
        "failures": str(cnt[FAIL]),
        "errors": str(cnt[ERROR]),
        "skipped": str(cnt[SKIP]),
        "time": f"{sum(r.elapsed_ms for r in results) / 1000.0:.3f}",
    })
    for r in results:
        tc = ET.SubElement(suite, "testcase", {
            "classname": f"{r.module}.{r.case.level}",
            "name": f"{r.case.tc_id} {r.case.title}",
            "time": f"{r.elapsed_ms / 1000.0:.3f}",
        })
        if r.status == FAIL:
            el = ET.SubElement(tc, "failure", {
                "message": r.detail.splitlines()[0] if r.detail else "断言失败",
                "type": "AssertionError"})
            el.text = r.detail
        elif r.status == ERROR:
            el = ET.SubElement(tc, "error", {
                "message": r.detail.splitlines()[0] if r.detail else "异常",
                "type": "Error"})
            el.text = r.detail
        elif r.status == SKIP:
            ET.SubElement(tc, "skipped")
    tree = ET.ElementTree(suite)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


# ================================================================= 框架自验证
def verify_harness() -> Tuple[bool, List[str]]:
    """反向验证框架本身：注入必然失败 / 必然异常的用例，确认判定正确。

    这是本测试集的"元测试"。若执行器有 bug（例如把异常吞掉、把断言错误当成
    通过），这里会立刻暴露，而不是让整份报告变成一张虚假的绿色通行证。
    """
    checks: List[Tuple[bool, str]] = []

    def _ok() -> None:
        pass

    def _bad() -> None:
        assert_eq(1, 2, "自检")

    def _boom() -> None:
        raise RuntimeError("自检异常")

    probes = [
        Case("TC-META-001", "REQ-META-01", "正常用例判为 PASS", "—", "—", "PASS", _ok),
        Case("TC-META-002", "REQ-META-01", "断言失败判为 FAIL", "—", "—", "FAIL", _bad),
        Case("TC-META-003", "REQ-META-01", "未捕获异常判为 ERROR", "—", "—", "ERROR", _boom),
    ]
    got = [r.status for r in run(probes, progress=False, stream=io.StringIO())]
    want = [PASS, FAIL, ERROR]
    checks.append((got == want,
                   f"执行器状态判定：期望 {want}，实际 {got}"))

    # 断言助手必须真的会失败
    try:
        assert_raises(ValueError, lambda: 1 / 1)
    except AssertionError:
        checks.append((True, "assert_raises 能识别「未抛异常」"))
    else:
        checks.append((False, "assert_raises 未识别「未抛异常」，断言助手失效"))

    try:
        assert_raises(KeyError, lambda: {}["x"], contains="不存在的键")
    except AssertionError:
        checks.append((True, "assert_raises 的 contains 校验生效"))
    else:
        checks.append((False, "assert_raises 的 contains 校验失效"))

    try:
        assert_eq([1, 2], [1, 2])
        assert_in("a", "abc")
        assert_close(0.1 + 0.2, 0.3, tol=1e-9)
        checks.append((True, "正常断言不误报"))
    except AssertionError as e:
        checks.append((False, f"正常断言误报：{e}"))

    # 重复编号必须被拦截
    saved = dict(_IDS)
    try:
        _IDS["TC-DUP-001"] = "probe"
        try:
            case("TC-DUP-001", "REQ-META-01", "重复", technique="等价类划分",
                 level="单元", expect="报错")(lambda: None)
        except RuntimeError:
            checks.append((True, "重复用例编号被拦截"))
        else:
            checks.append((False, "重复用例编号未被拦截"))
    finally:
        _IDS.clear()
        _IDS.update(saved)

    msgs = [("OK  " if ok else "FAIL") + "  " + m for ok, m in checks]
    return all(ok for ok, _ in checks), msgs
