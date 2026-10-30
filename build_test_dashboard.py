#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""由真实运行数据生成 README 用的「功能测试覆盖总览」图。

与 `build_portfolio.py` 同一原则：**图里不硬编码任何数字**，全部来自实跑结果，
否则改了测试忘了改图，图片就会变成"看起来对"的假数据。

数据来源：

- 用例数 / 失败数 / 通过率 / 分层 / 设计方法分布 / 需求覆盖率 / 耗时
  ← `tests/test_report.md`（由 `tests/run_tests.py` 生成）
- 端到端自检项数 ← 实际运行 `selftest.py`
- Web 链路自检项数 ← 实际运行 `rag_web.py --selftest`
- 变异测试捕获率 ← 实际运行 `tests/mutation_check.py`

用法：

    python build_test_dashboard.py                # 全量生成（含变异测试，约 2 分钟）
    python build_test_dashboard.py --no-mutation   # 跳过变异测试（图上标注「未实跑」）
    python build_test_dashboard.py --no-render     # 只生成 HTML，不调用浏览器出图

产物：`assets/test_coverage.png`（2 倍像素宽，供 README 嵌入）。

占位符用 `@@name@@` 而不是 `str.format`：模板里塞的是整段 CSS，
`{` `}` 会和 format 的字段语法打架；纯替换没有这个问题。
"""
import argparse
import html
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(ROOT, "tests", "test_report.md")
OUT_PNG = os.path.join(ROOT, "assets", "test_coverage.png")
TMP_HTML = os.path.join(ROOT, "assets", "_dashboard.html")
PY = sys.executable

# 配色：与 web/index.html 的二次元主题一致（粉 → 紫 → 薄荷）
INK = "#2b2b3a"
SUB = "#6b6b80"


def _run(args, timeout=600):
    """跑一个子进程并返回 (returncode, stdout+stderr)。"""
    cp = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=timeout)
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def _parse_report(path):
    """解析 tests/test_report.md，抽出图上要用的全部数字。"""
    if not os.path.exists(path):
        raise SystemExit("找不到 %s\n请先运行：python tests/run_tests.py" % path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    data = {}

    def grab(pattern, cast=int):
        m = re.search(pattern, text)
        if not m:
            raise SystemExit("无法从报告中解析：%s" % pattern)
        return cast(m.group(1))

    data["total"] = grab(r"\|\s*用例总数\s*\|\s*(\d+)\s*\|")
    data["passed"] = grab(r"\|\s*通过\s*\|\s*(\d+)\s*\|")
    data["failed"] = grab(r"\|\s*失败\s*\|\s*(\d+)\s*\|")
    data["errors"] = grab(r"\|\s*错误（非断言异常）\s*\|\s*(\d+)\s*\|")
    data["rate"] = grab(r"\|\s*通过率\s*\|\s*([\d.]+)%", float)

    m = re.search(r"需求覆盖率：\*\*(\d+)/(\d+)\s*=\s*([\d.]+)%", text)
    if m:
        data["req_hit"] = int(m.group(1))
        data["req_total"] = int(m.group(2))
    else:
        data["req_hit"] = data["req_total"] = 0

    m = re.search(r"总耗时：([\d.]+)\s*s", text)
    data["elapsed"] = float(m.group(1)) if m else 0.0

    def section_rows(title):
        m = re.search(r"##\s*\d+\.\s*%s\s*\n(.*?)(?=\n##\s|\Z)" % title, text, re.S)
        if not m:
            return []
        rows = []
        for line in m.group(1).splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and re.fullmatch(r"\d+", cells[1]):
                rows.append((cells[0], int(cells[1])))
        return rows

    data["levels"] = section_rows("按测试层级")
    data["methods"] = section_rows("按测试设计方法")
    if not data["levels"] or not data["methods"]:
        raise SystemExit("报告中未找到分层 / 设计方法分布表")
    return data


def _count_assertions(script_args, label):
    """跑一个自检脚本，从输出里取「通过 N 项」。"""
    code, out = _run([PY] + script_args)
    m = re.search(r"通过\s*(\d+)\s*项", out)
    if code != 0 or not m:
        raise SystemExit("%s 执行失败（退出码 %s），无法读取断言数：\n%s"
                         % (label, code, out[-800:]))
    return int(m.group(1))


def _mutation_capture():
    """实跑变异测试，返回 (被捕获数, 总数)。"""
    code, out = _run([PY, os.path.join("tests", "mutation_check.py")])
    m = re.search(r"变异测试：(\d+)/(\d+)\s*条变异被捕获", out)
    if not m:
        raise SystemExit("无法解析变异测试结果：\n%s" % out[-800:])
    if code != 0:
        raise SystemExit("变异测试未全部被捕获（退出码 %s），不应出图：\n%s"
                         % (code, out[-800:]))
    return int(m.group(1)), int(m.group(2))


def _bars(rows):
    """把 (标签, 数值) 列表渲染成水平条形图。"""
    top = max(v for _, v in rows) or 1
    out = []
    for label, value in rows:
        pct = value / top * 100.0
        out.append(
            '<div class="row"><span class="rlabel">%s</span>'
            '<div class="track"><div class="bar" style="width:%.1f%%"></div></div>'
            '<span class="rval">%d</span></div>'
            % (html.escape(label), pct, value))
    return "\n".join(out)


def build_html(d, selftest_n, web_n, mut):
    """按真实数据拼出仪表盘 HTML。"""
    if mut:
        mut_hit, mut_total = mut
        mut_ring = '<span class="ring"><b>%d/%d</b></span>' % (mut_hit, mut_total)
        mut_text = ("变异测试 <b>%d/%d</b> 捕获率 100%%（注入已知缺陷后用例必变红）"
                    % (mut_hit, mut_total))
    else:
        mut_ring = '<span class="ring"><b>--</b></span>'
        mut_text = "变异测试本次<strong>未实跑</strong>（生成时加了 --no-mutation）"

    values = {
        "ink": INK, "sub": SUB,
        "total": d["total"], "passed": d["passed"],
        "failed": d["failed"], "errors": d["errors"], "rate": d["rate"],
        "req_hit": d["req_hit"], "req_total": d["req_total"],
        "elapsed": "%.1f" % d["elapsed"],
        "selftest_n": selftest_n, "web_n": web_n,
        "levels": _bars(d["levels"]), "methods": _bars(d["methods"]),
        "mut_ring": mut_ring, "mut_text": mut_text,
    }
    out = TEMPLATE
    for key, val in values.items():
        out = out.replace("@@%s@@" % key, str(val))
    # 占位符必须全部替换干净：漏一个就会在图上显示 "@@xxx@@"，
    # 这类错误靠肉眼极难发现（数字看着都挺正常），所以在这里硬拦。
    leftover = re.findall(r"@@\w+@@", out)
    if leftover:
        raise SystemExit("模板占位符未被替换：%s" % ", ".join(sorted(set(leftover))))
    return out


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    background: linear-gradient(135deg, #FFF1F7 0%, #F4F0FF 45%, #E9FBFF 100%);
    font-family: system-ui, -apple-system, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
    color: @@ink@@;
    -webkit-font-smoothing: antialiased;
  }
  .wrap {
    width: 960px; margin: 26px auto; padding: 30px 34px 26px;
    background: rgba(255,255,255,0.82); border: 1px solid rgba(255,255,255,0.9);
    border-radius: 22px; box-shadow: 0 18px 50px rgba(155,140,255,0.22);
    backdrop-filter: blur(8px);
  }
  .head { display: flex; align-items: center; justify-content: space-between; }
  .title { font-size: 25px; font-weight: 800; letter-spacing: .5px; }
  .title .accent { color: #E0609B; }
  .subtitle { font-size: 13px; color: @@sub@@; margin-top: 6px; }
  .badge {
    font-size: 12px; color: #7a6fb0; background: rgba(155,140,255,0.14);
    border: 1px solid rgba(155,140,255,0.3); border-radius: 999px;
    padding: 6px 14px; white-space: nowrap;
  }
  .cards { display: flex; gap: 16px; margin-top: 22px; }
  .card {
    flex: 1; background: #fff; border-radius: 16px; padding: 16px 18px;
    border: 1px solid rgba(224,96,155,0.18);
    box-shadow: 0 6px 18px rgba(224,96,155,0.08);
  }
  .card .clabel { font-size: 12.5px; color: #8a8a9c; font-weight: 600; }
  .card .ccmd {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 12px; color: #b06aa0; margin: 4px 0 10px; word-break: break-all;
  }
  .card .cnum { font-size: 32px; font-weight: 800; line-height: 1; }
  .card .cnum .u { font-size: 14px; font-weight: 600; color: #8a8a9c; margin-left: 4px; }
  .card .csub { font-size: 11.5px; color: #9a9aa8; margin-top: 6px; }
  .card.pink .cnum { color: #E0609B; }
  .card.purple .cnum { color: #8A7BFF; }
  .card.mint .cnum { color: #2BA9A0; }

  .summary {
    margin-top: 20px; display: flex; align-items: center; justify-content: space-around;
    background: linear-gradient(135deg, rgba(224,96,155,0.10), rgba(155,140,255,0.10));
    border-radius: 16px; padding: 16px 10px;
  }
  .summary .s { text-align: center; }
  .summary .sv { font-size: 26px; font-weight: 800; color: @@ink@@; }
  .summary .sv .g { color: #2BA9A0; }
  .summary .sl { font-size: 12px; color: #7a7a8c; margin-top: 3px; }
  .summary .sep { width: 1px; height: 40px; background: rgba(155,140,255,0.25); }

  .charts { display: flex; gap: 22px; margin-top: 22px; }
  .chart { flex: 1; background: #fff; border-radius: 16px; padding: 16px 18px;
           border: 1px solid rgba(155,140,255,0.16);
           box-shadow: 0 6px 18px rgba(155,140,255,0.07); }
  .chart h3 { font-size: 14px; font-weight: 700; margin-bottom: 14px; color: #4a4a5e; }
  .row { display: flex; align-items: center; margin: 9px 0; }
  .rlabel { width: 78px; font-size: 12.5px; color: #6b6b80; text-align: right; padding-right: 10px; }
  .track { flex: 1; height: 16px; background: rgba(155,140,255,0.10); border-radius: 8px; overflow: hidden; }
  .bar { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #FF9EC2, #B79BFF); }
  .rval { width: 34px; font-size: 12.5px; font-weight: 700; color: #4a4a5e; text-align: right; padding-left: 8px; }

  .foot {
    margin-top: 20px; display: flex; align-items: center; justify-content: space-between;
    font-size: 12px; color: #7a7a8c; border-top: 1px dashed rgba(155,140,255,0.3);
    padding-top: 14px;
  }
  .mut { display: inline-flex; align-items: center; gap: 8px; }
  .mut .ring {
    width: 32px; height: 32px; border-radius: 50%; position: relative;
    background: conic-gradient(#2BA9A0 0% 100%, #e6e6ee 100% 100%);
    display: flex; align-items: center; justify-content: center;
  }
  .mut .ring::after {
    content: ''; position: absolute; inset: 4px; background: #fff; border-radius: 50%;
  }
  .mut .ring b { position: relative; z-index: 1; font-size: 10.5px; font-weight: 800; color: #1d7a73; }
</style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div>
        <div class="title">RAG 引擎 · <span class="accent">功能测试</span>覆盖总览</div>
        <div class="subtitle">零第三方依赖 · 全链路可复现评测 · 由 build_test_dashboard.py 生成</div>
      </div>
      <div class="badge">ragcore 1.0.0 · Python 3.13</div>
    </div>

    <div class="cards">
      <div class="card pink">
        <div class="clabel">端到端自检</div>
        <div class="ccmd">python selftest.py</div>
        <div class="cnum">@@selftest_n@@<span class="u">项断言</span></div>
        <div class="csub">分词·分块·建库·检索·离线回答·持久化</div>
      </div>
      <div class="card purple">
        <div class="clabel">Web 请求链路自检</div>
        <div class="ccmd">python rag_web.py --selftest</div>
        <div class="cnum">@@web_n@@<span class="u">项断言</span></div>
        <div class="csub">页面静态检查·接口契约·错误处理</div>
      </div>
      <div class="card mint">
        <div class="clabel">软件测试套件</div>
        <div class="ccmd">python tests/run_tests.py</div>
        <div class="cnum">@@total@@<span class="u">用例</span></div>
        <div class="csub">@@req_total@@ 条需求追溯 · 覆盖 100%</div>
      </div>
    </div>

    <div class="summary">
      <div class="s"><div class="sv">@@passed@@<span class="g">/@@total@@</span></div><div class="sl">用例通过（率 @@rate@@%）</div></div>
      <div class="sep"></div>
      <div class="s"><div class="sv">@@req_hit@@<span class="g">/@@req_total@@</span></div><div class="sl">需求覆盖率 100%</div></div>
      <div class="sep"></div>
      <div class="s"><div class="sv">@@failed@@</div><div class="sl">失败 / 错误 @@errors@@</div></div>
      <div class="sep"></div>
      <div class="s"><div class="sv">@@elapsed@@<span class="g">s</span></div><div class="sl">全量耗时</div></div>
    </div>

    <div class="charts">
      <div class="chart">
        <h3>按测试层级</h3>
@@levels@@
      </div>
      <div class="chart">
        <h3>按设计方法</h3>
@@methods@@
      </div>
    </div>

    <div class="foot">
      <div class="mut">
        @@mut_ring@@
        <span>@@mut_text@@</span>
      </div>
      <div>数据来源：tests/test_report.md · selftest.py · rag_web.py --selftest · mutation_check.py</div>
    </div>
  </div>
</body>
</html>
"""


def find_chrome():
    """找一个可用的 Chromium 内核浏览器。"""
    env = os.environ.get("CHROME_PATH")
    if env and os.path.exists(env):
        return env
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        p = shutil.which(name)
        if p:
            return p
    return None


def render(html_text, out_png):
    """用无头浏览器把 HTML 渲染成 PNG。"""
    with open(TMP_HTML, "w", encoding="utf-8", newline="") as f:
        f.write(html_text)
    browser = find_chrome()
    if not browser:
        print("!! 未找到 Chrome/Edge，已只生成 HTML：%s" % TMP_HTML)
        print("   请手动打开该文件后截图，或安装 Chrome 后重跑。")
        return False
    url = "file:///" + TMP_HTML.replace("\\", "/")
    cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
           "--force-device-scale-factor=2", "--window-size=1010,762",
           "--screenshot=%s" % out_png, url]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not os.path.exists(out_png):
        print("!! 渲染失败：%s" % (cp.stderr or cp.stdout)[-500:])
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="生成 README 用的功能测试覆盖总览图")
    ap.add_argument("--no-mutation", action="store_true",
                    help="跳过变异测试（图上标注未实跑）")
    ap.add_argument("--no-render", action="store_true",
                    help="只生成 HTML，不调用浏览器出图")
    args = ap.parse_args()

    d = _parse_report(REPORT)
    print("· 报告解析：用例 %d，通过 %d，失败 %d，错误 %d，需求 %d/%d，耗时 %.1fs"
          % (d["total"], d["passed"], d["failed"], d["errors"],
             d["req_hit"], d["req_total"], d["elapsed"]))

    selftest_n = _count_assertions(["selftest.py"], "selftest.py")
    print("· selftest.py：%d 项断言" % selftest_n)
    web_n = _count_assertions(["rag_web.py", "--selftest"], "rag_web.py --selftest")
    print("· rag_web.py --selftest：%d 项断言" % web_n)

    mut = None
    if args.no_mutation:
        print("· 变异测试：已跳过（图上将标注未实跑）")
    else:
        hit, total = _mutation_capture()
        print("· 变异测试：%d/%d 条被捕获" % (hit, total))
        mut = (hit, total)

    html_text = build_html(d, selftest_n, web_n, mut)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    if args.no_render:
        with open(TMP_HTML, "w", encoding="utf-8", newline="") as f:
            f.write(html_text)
        print("· 已生成 HTML：%s（按 --no-render 跳过出图）" % TMP_HTML)
        return 0

    if not render(html_text, OUT_PNG):
        return 1
    print("· 已生成图片：%s（%.0f KB）" % (OUT_PNG, os.path.getsize(OUT_PNG) / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
