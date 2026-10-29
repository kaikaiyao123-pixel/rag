#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""变异测试：向产品代码注入已知缺陷，确认测试套件确实会变红。

这不是交付物的一部分，只用于验证"用例不是摆设"。每条变异都对应一条或多条
应当失败的具体用例；若某条变异下测试仍然全绿，说明覆盖是空的。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

MUTATIONS = [
    ("M1 放宽拒答阈值", "ragcore/abstain.py",
     "MIN_VOCAB_COV = 0.32", "MIN_VOCAB_COV = 0.02",
     ["REQ-ABS-04", "TC-REG-002"]),
    ("M2 取消代码块围栏保护", "ragcore/chunker.py",
     "h = None if fence else _is_heading(line)", "h = _is_heading(line)",
     ["TC-CHK-003", "TC-CHK-004"]),
    ("M3 取消 overlap 上限", "ragcore/chunker.py",
     "overlap = max(0, min(overlap, max_chars // 2))", "overlap = max(0, overlap)",
     ["TC-CHK-008"]),
    ("M4 取消图片语法预清洗", "ragcore/generator.py",
     'text = _IMG_RE.sub(" ", text)', 'pass',
     ["TC-GEN-003"]),
]


def run_filtered(keyword: str) -> int:
    cp = subprocess.run([PY, os.path.join(ROOT, "tests", "run_tests.py"),
                         "--filter", keyword, "--no-report", "--quiet"],
                        cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    return cp.returncode


def main() -> int:
    bad = 0
    for name, rel, old, new, expect in MUTATIONS:
        path = os.path.join(ROOT, rel)
        # 必须按字节读写：用文本模式 open(..., "w") 在 Windows 上会把 \n 变成
        # \r\n，还原后整个文件的行尾都被改写，git diff 会出现几百行假变更。
        with open(path, "rb") as f:
            orig = f.read()
        o, n = old.encode("utf-8"), new.encode("utf-8")
        if o not in orig:
            print(f"[SKIP] {name}：锚点未找到（{rel}）")
            bad += 1
            continue
        try:
            with open(path, "wb") as f:
                f.write(orig.replace(o, n, 1))
            codes = {kw: run_filtered(kw) for kw in expect}
        finally:
            with open(path, "wb") as f:
                f.write(orig)
        detected = [kw for kw, code in codes.items() if code != 0]
        status = "捕获" if detected else "未捕获（覆盖空洞）"
        print(f"[{status}] {name}　期望失败项：{expect}　实际失败：{detected or '无'}")
        if not detected:
            bad += 1
    print(f"\n变异测试：{len(MUTATIONS) - bad}/{len(MUTATIONS)} 条变异被捕获")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
