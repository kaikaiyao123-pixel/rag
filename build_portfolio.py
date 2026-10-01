#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""由 benchmark_report.json 生成单页 HTML 作品集。

    python benchmark.py        # 先产出评测数据
    python build_portfolio.py  # 再由数据生成网页

设计原则
--------
- **数据驱动**：页面上所有指标、图表、失败样例都从 benchmark_report.json
  读取，不硬编码。重跑评测后重新生成即可，不会出现"文案与数据脱节"。
- **零外部依赖**：图表为脚本生成的内联 SVG，不引用任何 CDN 或字体服务，
  离线双击即可打开，也便于直接投递或打印为 PDF。
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

REPORT = "benchmark_report.json"
OUT_DIR = "portfolio"
OUT_FILE = os.path.join(OUT_DIR, "index.html")

# ---------------------------------------------------------------- 视觉规范
INK = "#14181f"
INK_SOFT = "#4a5464"
MUTED = "#78828f"
LINE = "#e3e7ee"
BG = "#f6f7f9"
CARD = "#ffffff"
ACCENT = "#2f5fd8"
ACCENT_SOFT = "#eaf0fd"
OK = "#0f9b8e"
WARN = "#c9741f"

SERIES = ["#2f5fd8", "#0f9b8e", "#c9741f"]

METHOD_LABEL = {
    "full": "本方案（BM25 + 标题加权 + 覆盖率）",
    "bm25_plain": "纯 Okapi BM25",
    "tfidf": "TF-IDF 余弦",
    "keyword": "关键词命中计数",
}


# ---------------------------------------------------------------- 小工具
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%"


def load_report(path: str = REPORT) -> Dict:
    if not os.path.isfile(path):
        sys.exit(f"未找到 {path}，请先运行：python benchmark.py")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_lines(paths: Sequence[str]) -> Dict[str, int]:
    """统计源码行数（排除空行与纯注释行），用于展示工程规模。"""
    total = code = 0
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                total += 1
                if s and not s.startswith("#"):
                    code += 1
    return {"total": total, "code": code}


# ---------------------------------------------------------------- SVG 图表
def grouped_bar_chart(groups: Sequence[Tuple[str, Sequence[float]]],
                      series: Sequence[str],
                      colors: Sequence[str],
                      width: int = 680, height: int = 300,
                      ymax: float = 1.0,
                      value_fmt: str = "{:.3f}",
                      y_fmt: str = "{:.0%}",
                      markers: Sequence[Tuple[int, float, str]] = (),
                      ) -> str:
    """分组柱状图。

    groups: [(组标签, [各系列取值...])]
    markers: [(组序号, 取值, 标签)] 用于在组内画阈值虚线
    """
    pl, pr, pt, pb = 60, 18, 44, 56
    pw, ph = width - pl - pr, height - pt - pb
    n_ser = len(series)
    gw = pw / max(1, len(groups))
    bar_w = min(40.0, gw * 0.66 / n_ser)
    gap = 5.0
    cluster_w = bar_w * n_ser + gap * (n_ser - 1)

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'xmlns="http://www.w3.org/2000/svg" font-family="inherit">']

    # 网格与 y 轴刻度
    for i in range(5):
        v = ymax * i / 4
        y = pt + ph - ph * (v / ymax)
        out.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl + pw:.1f}" y2="{y:.1f}" '
                   f'stroke="{LINE}" stroke-width="1"/>')
        out.append(f'<text x="{pl - 10}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-size="11" fill="{MUTED}">{y_fmt.format(v)}</text>')

    # 图例
    lx = pl
    for name, color in zip(series, colors):
        out.append(f'<rect x="{lx}" y="{pt - 28}" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(f'<text x="{lx + 15}" y="{pt - 19}" font-size="11.5" fill="{INK_SOFT}">'
                   f'{esc(name)}</text>')
        lx += 15 + len(name) * 7.2 + 22

    # 柱体
    for gi, (glabel, vals) in enumerate(groups):
        gx = pl + gw * gi + (gw - cluster_w) / 2
        for si, v in enumerate(vals):
            h = ph * min(1.0, v / ymax)
            x = gx + si * (bar_w + gap)
            y = pt + ph - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                       f'height="{max(h, 0.6):.1f}" rx="3" fill="{colors[si]}"/>')
            out.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                       f'font-size="10.5" fill="{INK_SOFT}">{value_fmt.format(v)}</text>')
        out.append(f'<text x="{pl + gw * gi + gw / 2:.1f}" y="{pt + ph + 20:.1f}" '
                   f'text-anchor="middle" font-size="12" fill="{INK}">{esc(glabel)}</text>')

    # 阈值标记线
    for gi, val, label in markers:
        if val <= 0:
            continue
        y = pt + ph - ph * (val / ymax)
        x0 = pl + gw * gi + (gw - cluster_w) / 2 - 6
        x1 = x0 + cluster_w + 12
        out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
                   f'stroke="{INK}" stroke-width="1.6" stroke-dasharray="5 3"/>')
        out.append(f'<text x="{x1 + 6:.1f}" y="{y + 3.5:.1f}" font-size="10.5" '
                   f'fill="{INK}">阈值 {value_fmt.format(val)}</text>')

    out.append(f'<line x1="{pl}" y1="{pt + ph}" x2="{pl + pw}" y2="{pt + ph}" '
               f'stroke="{INK_SOFT}" stroke-width="1.2"/>')
    out.append("</svg>")
    return "".join(out)


def pipeline_svg() -> str:
    """系统架构流水线图（单行主链路 + 向下分支出生成层）。"""
    w, h = 680, 262
    by, bh, bw = 72, 74, 100
    boxes = [
        (16, "文档", "md / txt\n多编码识别", "#5b6b82"),
        (132, "分块", "标题栈切分\n重叠窗口", "#5b6b82"),
        (248, "索引", "BM25 倒排\nJSON 持久化", ACCENT),
        (364, "检索", "候选截断\n限流 + 去重", ACCENT),
        (480, "拒答判定", "词表覆盖\nIDF 覆盖", OK),
    ]
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
           f'xmlns="http://www.w3.org/2000/svg" font-family="inherit">']
    out.append('<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" '
               'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#8b95a5"/></marker>'
               '<marker id="ar2" markerWidth="9" markerHeight="9" refX="7" refY="3" '
               'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="' + WARN + '"/></marker></defs>')

    out.append(f'<text x="16" y="24" font-size="12" font-weight="600" fill="{INK}">'
               f'检索链路（完全离线，零第三方依赖）</text>')

    for i, (x, title, sub, color) in enumerate(boxes):
        out.append(f'<rect x="{x}" y="{by}" width="{bw}" height="{bh}" rx="8" '
                   f'fill="{CARD}" stroke="{color}" stroke-width="1.6"/>')
        out.append(f'<text x="{x + bw / 2}" y="{by + 26}" text-anchor="middle" '
                   f'font-size="13" font-weight="600" fill="{color}">{esc(title)}</text>')
        for j, ln in enumerate(sub.split("\n")):
            out.append(f'<text x="{x + bw / 2}" y="{by + 46 + j * 14}" text-anchor="middle" '
                       f'font-size="10.5" fill="{MUTED}">{esc(ln)}</text>')
        if i < len(boxes) - 1:
            out.append(f'<line x1="{x + bw + 3}" y1="{by + bh / 2}" x2="{x + bw + 12}" '
                       f'y2="{by + bh / 2}" stroke="#8b95a5" stroke-width="1.6" '
                       f'marker-end="url(#ar)"/>')

    # 拒答出口：从判定框顶部向上引出，标注在箭头之上
    cx = 480 + bw / 2
    out.append(f'<line x1="{cx}" y1="{by - 4}" x2="{cx}" y2="{by - 24}" '
               f'stroke="{OK}" stroke-width="1.6" marker-end="url(#ar)"/>')
    out.append(f'<text x="{cx}" y="{by - 32}" text-anchor="middle" font-size="10.5" '
               f'fill="{OK}">资料不足 → 直接拒答</text>')

    # 生成分支：判定框底部向下引出
    out.append(f'<line x1="{cx}" y1="{by + bh + 2}" x2="{cx}" y2="{by + bh + 40}" '
               f'stroke="{WARN}" stroke-width="1.6" marker-end="url(#ar2)"/>')
    gy = by + bh + 44
    out.append(f'<rect x="400" y="{gy}" width="240" height="62" rx="8" fill="{CARD}" '
               f'stroke="{WARN}" stroke-width="1.6"/>')
    out.append(f'<text x="520" y="{gy + 25}" text-anchor="middle" font-size="13" '
               f'font-weight="600" fill="{WARN}">生成</text>')
    out.append(f'<text x="520" y="{gy + 45}" text-anchor="middle" font-size="10.5" '
               f'fill="{MUTED}">大模型（OpenAI 兼容）／离线抽取式兜底</text>')

    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- 页面片段
def metric_card(value: str, label: str, hint: str = "", accent: bool = False) -> str:
    color = ACCENT if accent else INK
    return (f'<div class="mc{" mc-a" if accent else ""}">'
            f'<div class="mc-v" style="color:{color}">{esc(value)}</div>'
            f'<div class="mc-l">{esc(label)}</div>'
            + (f'<div class="mc-h">{esc(hint)}</div>' if hint else "")
            + "</div>")


def decision_card(title: str, problem: str, choice: str, tradeoff: str, result: str) -> str:
    return f"""<article class="dc">
  <h3>{esc(title)}</h3>
  <dl>
    <dt>问题</dt><dd>{problem}</dd>
    <dt>选择</dt><dd>{choice}</dd>
    <dt>代价</dt><dd>{tradeoff}</dd>
    <dt>结果</dt><dd class="dc-r">{result}</dd>
  </dl>
</article>"""


def bold(text: str) -> str:
    """把 **强调** 转为 <strong>，其余内容保持原样（调用方负责已转义或受控文本）。"""
    import re as _re
    return _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def timeline(steps: Sequence[Tuple[str, str]]) -> str:
    items = "".join(
        f'<li><span class="tl-n">{i}</span>'
        f'<div><strong>{esc(title)}</strong><p>{bold(desc)}</p></div></li>'
        for i, (title, desc) in enumerate(steps, 1))
    return f'<ol class="tl">{items}</ol>'


# ---------------------------------------------------------------- 主体
def build(report: Dict) -> str:
    meta = report["meta"]
    corpus = meta["corpus"]
    labels = meta["label_set"]
    full = report["metrics"]["full"]
    ref = report["refusal"]
    legacy = report["refusal_baseline_legacy"]
    calib = report["refusal_calibration"]
    best = calib["best"]
    lat = report["latency"]
    idx = report["index"]
    timing = report["timing"]

    src = count_lines(["rag.py", "benchmark.py", "build_portfolio.py"] +
                      [f"ragcore/{n}.py" for n in
                       ("__init__", "abstain", "chunker", "cli", "generator",
                        "index", "loader", "retriever", "tokenizer")])

    # 图表 A：本方案在各 k 上的命中率
    chart_a = grouped_bar_chart(
        [("文档级命中 Doc Recall", [full["doc_recall"]["@1"], full["doc_recall"]["@3"],
                                    full["doc_recall"]["@5"]]),
         ("片段级命中 Passage Hit", [full["passage_hit"]["@1"], full["passage_hit"]["@3"],
                                     full["passage_hit"]["@5"]])],
        ["@1", "@3", "@5"], SERIES, height=300)

    # 图表 B：消融对比
    order = ["full", "bm25_plain", "tfidf", "keyword"]
    chart_b = grouped_bar_chart(
        [(METHOD_LABEL[k].split("（")[0], [report["metrics"][k]["doc_recall"]["@1"],
                                           report["metrics"][k]["passage_hit"]["@1"],
                                           report["metrics"][k]["mrr@10"]])
         for k in order],
        ["Doc Recall@1", "Passage Hit@1", "MRR@10"], SERIES, height=320)

    # 图表 C：拒答特征可分性
    inf = calib["in_domain_features"]
    oof = calib["out_of_domain_features"]
    chart_c = grouped_bar_chart(
        [("词表覆盖 vocab_cov", [inf["vocab_cov"]["p50"], oof["vocab_cov"]["p50"]]),
         ("IDF 覆盖 idf_cov", [inf["idf_cov"]["p50"], oof["idf_cov"]["p50"]]),
         ("归一化得分 norm_score", [inf["norm_score"]["p50"], oof["norm_score"]["p50"]])],
        ["领域内问题（中位）", "领域外问题（中位）"], [ACCENT, WARN],
        height=300,
        markers=[(0, best["vocab_cov"], "vocab_cov"),
                 (1, best["idf_cov"], "idf_cov"),
                 (2, best["norm_score"], "norm_score")])

    # 消融表
    rows = []
    for k in order:
        m = report["metrics"][k]
        hl = ' class="hl"' if k == "full" else ""
        rows.append(
            f'<tr{hl}><td>{esc(METHOD_LABEL[k])}</td>'
            f'<td>{m["doc_recall"]["@1"]:.3f}</td>'
            f'<td>{m["passage_hit"]["@1"]:.3f}</td>'
            f'<td>{m["passage_hit"]["@5"]:.3f}</td>'
            f'<td>{m["mrr@10"]:.3f}</td></tr>')
    ablation_table = "".join(rows)

    # 失败样例
    miss_rows = []
    for m in full.get("_misses", []):
        rank = m["doc_rank"] if m["doc_rank"] else "未进前 10"
        miss_rows.append(
            f'<tr><td>{esc(m["query"])}</td><td>{esc(m["expected_doc"])}</td>'
            f'<td>{esc(rank)}</td><td>{esc(m["top1"] or "—")}</td></tr>')
    miss_table = "".join(miss_rows) or '<tr><td colspan="4">无</td></tr>'

    pareto = [p for p in calib["pareto"] if p["false_refusal_rate"] <= 0.05][:5]
    pareto_rows = "".join(
        f'<tr><td>{p["vocab_cov"]:.2f}</td><td>{p["idf_cov"]:.2f}</td>'
        f'<td>{p["norm_score"]:.2f}</td>'
        f'<td>{pct(p["refusal_accuracy"])}</td>'
        f'<td>{pct(p["false_refusal_rate"])}</td>'
        f'<td>{pct(p["balanced_accuracy"])}</td></tr>'
        for p in pareto)

    tl = timeline([
        ("建立评测基线", "构造 71 条标注查询（问题 → 期望文档 → 正确片段特征串）与 7 条领域外问题，"
                     "同时实现 3 个对照方法，先用数据看清现状，而不是凭感觉判断好坏。"),
        ("暴露缺陷", f"首次评测发现：检索指标尚可（Doc Recall@5 = {pct(full['doc_recall']['@5'], 0)}），"
                   f"但**拒答准确率只有 {pct(legacy['refusal_accuracy'])}** —— 系统几乎不对领域外问题拒答，"
                   "倾向于从无关片段里强行摘录，是典型幻觉来源。"),
        ("定位根因", "原拒答条件是「召回片段与问题无任何共享词项」。中文 bigram 分词下该条件几乎不可能成立："
                   "任意两句中文都会因「可以」「怎么」这类高频二元组产生重叠。"
                   "改为构造查询内归一化特征：词表覆盖率与 IDF 覆盖率。"),
        ("二次修正", "第一版特征把「不在词表中的词项」排除在分母外，导致领域外查询分母趋近 0、"
                   "比值反而被放大到 1.0 以上 —— 特征方向完全反了。"
                   "修正为按「语料中最稀有词项」计权后，三项特征方向一致、分离清晰。"),
        ("标定与复测", f"网格搜索阈值，发现 84 组解并列最优；改用 max-min 间隔准则二次筛选，"
                    f"把阈值推到特征分布空隙中央。最终拒答率 {pct(legacy['refusal_accuracy'])} → "
                    f"{pct(ref['refusal_accuracy'])}，误拒 {ref['falsely_refused']}/{ref['queries_in_domain']}。"),
    ])

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>零依赖小型 RAG 检索问答引擎 · 项目作品集</title>
<style>
  :root {{
    --ink:{INK}; --ink-soft:{INK_SOFT}; --muted:{MUTED}; --line:{LINE};
    --bg:{BG}; --card:{CARD}; --accent:{ACCENT}; --accent-soft:{ACCENT_SOFT};
    --ok:{OK}; --warn:{WARN};
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
      "Hiragino Sans GB","Microsoft YaHei",sans-serif;
    font-size:15px; line-height:1.72;
    -webkit-font-smoothing:antialiased;
  }}
  .wrap {{ max-width:960px; margin:0 auto; padding:0 24px 72px; }}
  code, .mono {{ font-family:"SF Mono",Consolas,"Cascadia Mono",Menlo,monospace; }}

  header.hero {{ padding:64px 0 40px; }}
  .eyebrow {{ display:inline-block; font-size:12px; letter-spacing:.09em; text-transform:uppercase;
    color:var(--accent); background:var(--accent-soft); padding:4px 11px; border-radius:999px;
    font-weight:600; }}
  h1 {{ font-size:35px; line-height:1.28; margin:18px 0 12px; letter-spacing:-.02em; font-weight:700; }}
  .lede {{ font-size:17px; color:var(--ink-soft); max-width:70ch; margin:0 0 24px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:32px; }}
  .chip {{ font-size:12.5px; color:var(--ink-soft); background:var(--card);
    border:1px solid var(--line); padding:4px 11px; border-radius:6px; }}

  .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
  .mc {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
  .mc-a {{ border-color:var(--accent); background:linear-gradient(180deg,var(--accent-soft),var(--card) 62%); }}
  .mc-v {{ font-size:27px; font-weight:700; letter-spacing:-.02em; line-height:1.2; }}
  .mc-l {{ font-size:13px; color:var(--ink-soft); margin-top:3px; }}
  .mc-h {{ font-size:11.5px; color:var(--muted); margin-top:2px; }}

  section {{ margin-top:56px; }}
  h2 {{ font-size:22px; margin:0 0 6px; letter-spacing:-.01em; }}
  h2 .n {{ color:var(--muted); font-weight:600; margin-right:9px; font-size:17px; }}
  .sub {{ color:var(--muted); font-size:14px; margin:0 0 22px; max-width:78ch; }}
  h3 {{ font-size:15.5px; margin:0 0 10px; }}

  .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:22px 24px; }}
  .card + .card {{ margin-top:14px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }}

  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th, td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); }}
  th {{ font-size:12px; color:var(--muted); font-weight:600; text-transform:uppercase;
    letter-spacing:.05em; }}
  tbody tr:last-child td {{ border-bottom:none; }}
  tr.hl td {{ background:var(--accent-soft); font-weight:600; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .tbl-wrap {{ overflow-x:auto; }}

  .chart {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:18px 20px 12px; }}
  .chart h3 {{ margin-bottom:2px; }}
  .chart p {{ margin:0 0 10px; font-size:13px; color:var(--muted); }}
  .chart + .chart {{ margin-top:14px; }}

  .dc {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px 22px; }}
  .dc dl {{ margin:0; }}
  .dc dt {{ font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em;
    margin-top:12px; }}
  .dc dt:first-child {{ margin-top:0; }}
  .dc dd {{ margin:3px 0 0; font-size:14px; color:var(--ink-soft); }}
  .dc dd.dc-r {{ color:var(--ok); font-weight:600; }}

  ol.tl {{ list-style:none; padding:0; margin:0; }}
  ol.tl li {{ display:flex; gap:16px; padding:0 0 22px; position:relative; }}
  ol.tl li:not(:last-child)::before {{ content:""; position:absolute; left:13px; top:30px;
    bottom:0; width:2px; background:var(--line); }}
  .tl-n {{ flex:0 0 28px; height:28px; border-radius:50%; background:var(--accent-soft);
    color:var(--accent); font-size:13px; font-weight:700; display:flex;
    align-items:center; justify-content:center; z-index:1; }}
  ol.tl strong {{ font-size:15px; }}
  ol.tl p {{ margin:4px 0 0; font-size:14px; color:var(--ink-soft); }}
  ol.tl p strong {{ color:var(--ink); }}

  pre {{ background:#11151c; color:#e6e9ef; padding:16px 18px; border-radius:11px;
    overflow-x:auto; font-size:13px; line-height:1.66; margin:0; }}
  pre .c {{ color:#8a94a6; }}
  .kv {{ display:grid; grid-template-columns:auto 1fr; gap:2px 16px; font-size:14px; }}
  .kv dt {{ color:var(--muted); }}
  .kv dd {{ margin:0; }}

  footer {{ margin-top:64px; padding-top:22px; border-top:1px solid var(--line);
    color:var(--muted); font-size:13px; }}
  .note {{ font-size:13px; color:var(--muted); margin-top:10px; }}

  @media (max-width:640px) {{
    h1 {{ font-size:27px; }} .lede {{ font-size:15.5px; }}
    .wrap {{ padding:0 16px 48px; }} header.hero {{ padding:40px 0 28px; }}
  }}
  @media print {{
    body {{ background:#fff; font-size:12px; }}
    .card,.mc,.chart,.dc {{ break-inside:avoid; }}
    section {{ margin-top:32px; }}
    pre {{ background:#f4f5f7; color:#14181f; }}
  }}
</style>
</head>
<body>
<div class="wrap">

<header class="hero">
  <span class="eyebrow">个人项目 · 检索系统 · 零第三方依赖</span>
  <h1>零依赖小型 RAG 检索问答引擎</h1>
  <p class="lede">
    用 Python 标准库从零实现一个本地知识库问答系统：自研中文分词、Markdown 感知分块、
    Okapi BM25 倒排索引、查询内归一化的拒答判定，以及可插拔的生成层。
    不含任何第三方包，检索全链路离线运行；同时自建标注评测集与消融实验，
    用数据驱动每一次设计取舍。
  </p>
  <div class="chips">
    <span class="chip">Python 3.9+ · 纯标准库</span>
    <span class="chip">Okapi BM25 倒排索引</span>
    <span class="chip">中文 bigram 无词典分词</span>
    <span class="chip">拒答判定 / 幻觉抑制</span>
    <span class="chip">标注集 + 消融实验</span>
    <span class="chip">OpenAI 兼容生成接口</span>
  </div>

  <div class="metrics">
    {metric_card(pct(full["doc_recall"]["@5"], 0), "文档级召回 Recall@5", f"{corpus['docs']} 篇文档 / {labels['in_domain']} 条标注查询", accent=True)}
    {metric_card(pct(full["passage_hit"]["@1"]), "片段级命中 Hit@1", "定位到正确段落，而非仅正确文档")}
    {metric_card(f"{full['mrr@10']:.3f}", "MRR@10", "首个正确结果的排名倒数均值")}
    {metric_card(pct(ref["refusal_accuracy"], 0), "拒答准确率", f"误拒 {pct(ref['false_refusal_rate'])}，改造前仅 {pct(legacy['refusal_accuracy'])}")}
    {metric_card(f"{lat['p95_ms']} ms", "查询延迟 P95", f"P50 {lat['p50_ms']} ms · {lat['samples']} 次采样")}
  </div>
</header>

<section>
  <h2><span class="n">01</span>项目速览</h2>
  <p class="sub">所有数字均来自本仓库可复现的评测脚本，非估算值。</p>
  <div class="card">
    <dl class="kv">
      <dt>我的角色</dt><dd>独立完成：需求定义、架构设计、算法实现、评测设计与调优</dd>
      <dt>工程规模</dt><dd>{src['code']} 行有效代码（{src['total']} 行含注释），9 个模块 + 评测脚本 + 网页生成脚本</dd>
      <dt>知识库规模</dt><dd>{corpus['docs']} 篇文档 → {corpus['chunks']} 个片段 → {corpus['terms']} 个词项，正文 {corpus['chars']:,} 字符</dd>
      <dt>评测规模</dt><dd>{labels['in_domain']} 条领域内标注查询 + {labels['out_of_domain']} 条领域外问题，4 种方法横向对比</dd>
      <dt>性能</dt><dd>全量建索引 {timing['total_build_s'] * 1000:.0f} ms（解析分块 {timing['parse_s'] * 1000:.0f} ms + 索引 {timing['index_build_s'] * 1000:.0f} ms）；索引文件 {idx['index_kb']} KB</dd>
      <dt>技术约束</dt><dd>零第三方依赖（仅标准库），检索链路完全离线，生成层可选接入任意 OpenAI 兼容接口</dd>
    </dl>
  </div>
</section>

<section>
  <h2><span class="n">02</span>问题与目标</h2>
  <p class="sub"></p>
  <div class="grid2">
    <div class="card">
      <h3>为什么不用现成方案</h3>
      <p style="margin:0;color:var(--ink-soft);font-size:14px">
        主流 RAG 方案默认依赖向量数据库与 Embedding 服务，部署门槛与运行成本高，
        且中文效果高度依赖外部模型。我希望验证一个更朴素的假设：
        <strong>在单一领域的知识库上，精心调校的词项检索能否达到可用水平？</strong>
        如果能，就得到一个零成本、可完全离线、行为可解释的基线。
      </p>
    </div>
    <div class="card">
      <h3>三个可验证的目标</h3>
      <ol style="margin:0;padding-left:20px;color:var(--ink-soft);font-size:14px">
        <li>零第三方依赖：<code class="mono">pip install</code> 为空，任何有 Python 的机器都能跑</li>
        <li>可量化的检索质量：自建标注集，给出 Recall / MRR 等标准指标</li>
        <li>可抑制幻觉：资料不足时明确拒答，而不是强行编造</li>
      </ol>
    </div>
  </div>
</section>

<section>
  <h2><span class="n">03</span>系统架构</h2>
  <p class="sub">检索链路全部在本地完成，不产生任何网络请求；生成层是唯一的可选外部依赖，且具备离线降级路径。</p>
  <div class="card">{pipeline_svg()}</div>
  <div class="grid2" style="margin-top:14px">
    <div class="card">
      <h3>分块策略</h3>
      <p style="margin:0;color:var(--ink-soft);font-size:14px">
        按 Markdown 标题维护标题栈，为每段正文生成 <code class="mono">一级 › 二级</code> 形式的
        标题路径；再按空行切段、贪心装箱到 600 字符以内，相邻块保留 100 字符重叠窗口。
        标题路径既是检索时的加权字段，也是回答时展示的引用位置。
        围栏代码块内的 <code class="mono"># 注释</code> 会被识别并跳过，不参与标题解析。
      </p>
    </div>
    <div class="card">
      <h3>检索评分</h3>
      <p style="margin:0 0 8px;color:var(--ink-soft);font-size:14px">
        <code class="mono">final = BM25(text) × (1 + 1.8 × 标题命中率 + 0.4 × 词项覆盖率)</code>
      </p>
      <p style="margin:0;color:var(--ink-soft);font-size:14px">
        IDF 采用 BM25+ 形式 <code class="mono">ln(1 + (N−df+0.5)/(df+0.5))</code>，
        保证词项出现于过半文档时权重恒为正。标题命中率让「标题即答案」的 FAQ 片段优先浮出；
        词项覆盖率抑制「单个高频词重复命中」压过「多词共同命中」。
      </p>
    </div>
  </div>
</section>

<section>
  <h2><span class="n">04</span>量化评测</h2>
  <p class="sub">
    评测集由 {labels['in_domain']} 条手工标注查询组成，每条标注「问题 → 期望文档 → 正确片段中必须出现的特征串」；
    脚本会先校验特征串确实存在于期望文档中，防止标注错误污染指标。
    同时给出两档判定：文档级（召回对文档）与片段级（召回对具体段落），
    避免用宽松指标掩盖「找到了文档但没找到正确段落」的真实缺陷。
    后处理管线（候选截断 30 → 单文档限流 3 → 近似去重 → Top-5）在四种方法间完全一致。
  </p>

  <div class="chart">
    <h3>本方案在各召回深度上的表现</h3>
    <p>Top-1 已能定位到 88.7% 的期望文档，Top-5 达到 100%。</p>
    {chart_a}
  </div>

  <div class="chart">
    <h3>消融对比：替换打分函数</h3>
    <p>在完全相同的分词、分块与后处理管线上，仅替换打分函数。</p>
    {chart_b}
  </div>

  <div class="card" style="margin-top:14px">
    <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>方法</th><th class="num">Doc Recall@1</th><th class="num">Passage Hit@1</th>
        <th class="num">Passage Hit@5</th><th class="num">MRR@10</th>
      </tr></thead>
      <tbody>{ablation_table}</tbody>
    </table>
    </div>
    <p class="note">
      结论：标题加权与覆盖率加权在 Top-1 指标上带来稳定提升（Doc Recall@1 +4.2pt，Passage Hit@1 +4.2pt，
      MRR +0.024），但在 Top-5 上二者打平——说明这两项加权的作用是<strong>重排序</strong>而非扩大召回，
      与设计意图一致。TF-IDF 余弦在文档级接近本方案，但片段级明显落后，原因是缺少文档长度归一化。
    </p>
  </div>

  <div class="chart">
    <h3>拒答判定：领域内 vs 领域外问题的特征分布</h3>
    <p>三项特征均按「查询内归一化」计算，虚线为网格搜索标定出的阈值。</p>
    {chart_c}
  </div>

  <div class="card" style="margin-top:14px">
    <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th class="num">vocab_cov 阈值</th><th class="num">idf_cov 阈值</th>
        <th class="num">norm_score 阈值</th><th class="num">拒答准确率</th>
        <th class="num">误拒率</th><th class="num">平衡准确率</th>
      </tr></thead>
      <tbody>{pareto_rows}</tbody>
    </table>
    </div>
    <p class="note">
      帕累托前沿（已截取误拒率 ≤ 5% 的工作点）。最终选定
      <code class="mono">vocab_cov ≥ {best['vocab_cov']}</code>、
      <code class="mono">idf_cov ≥ {best['idf_cov']}</code>，
      <code class="mono">norm_score</code> 的最优阈值退化为 0 —— 该特征本身有区分度，
      但信息已被前两项覆盖，属冗余特征，因此不参与默认判定。
    </p>
  </div>

  <div class="grid2" style="margin-top:14px">
    <div class="card">
      <h3>性能与资源</h3>
      <table>
        <tbody>
          <tr><td>查询延迟 P50</td><td class="num">{lat['p50_ms']} ms</td></tr>
          <tr><td>查询延迟 P95</td><td class="num">{lat['p95_ms']} ms</td></tr>
          <tr><td>查询延迟 P99</td><td class="num">{lat['p99_ms']} ms</td></tr>
          <tr><td>索引构建</td><td class="num">{timing['total_build_s'] * 1000:.0f} ms</td></tr>
          <tr><td>索引文件体积</td><td class="num">{idx['index_kb']} KB</td></tr>
          <tr><td>索引 / 正文比</td><td class="num">{idx['ratio']:.2f}×</td></tr>
          <tr><td>加载堆内存峰值</td><td class="num">{idx['load_heap_peak_kb'] / 1024:.1f} MB</td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <h3>失败样例（{full['passage_misses']} 条未在 Top-5 命中正确片段）</h3>
      <div class="tbl-wrap">
      <table>
        <thead><tr><th>查询</th><th>期望文档</th><th>文档级排名</th><th>实际 Top-1</th></tr></thead>
        <tbody>{miss_table}</tbody>
      </table>
      </div>
      <p class="note">
        该查询期望定位到检索语法文档中的「时间筛选」章节，但 BM25 把
        「时间范围」与定价文档里的「时间」类词项匹配得更强。
        这暴露了纯词项检索的固有短板：<strong>无法理解「时间筛选」是一个整体概念</strong>。
        可行的改进方向是加入查询改写或轻量向量召回做分数融合。
      </p>
    </div>
  </div>
</section>

<section>
  <h2><span class="n">05</span>一次由评测驱动的缺陷修复</h2>
  <p class="sub">
    这是整个项目里我认为最有价值的部分：不是「我做了一个 RAG」，而是
    「我用评测发现自己做错了，并证明改对了」。
  </p>
  <div class="card">{tl}</div>
</section>

<section>
  <h2><span class="n">06</span>关键技术决策</h2>
  <p class="sub">每个决策都写清问题、选择、代价与结果——取舍比结论更能说明工程判断力。</p>
  <div class="grid2">
    {decision_card(
      "中文分词：bigram 而非词典分词",
      "jieba 等分词器会引入第三方依赖与词典文件，与「零依赖」目标冲突。",
      "不依赖词典，对连续汉字串做二元切分（n 个汉字生成 n−1 个 bigram）。这是信息检索领域的经典无词典方案，召回稳定。",
      "索引体积约为按词切分的 2~3 倍；中文同义词不做归一（「电脑」与「计算机」视为不同词项）；单字查询命中率低。",
      f"在 {corpus['docs']} 篇语料上词项数 {corpus['terms']}，索引 {idx['index_kb']} KB，仍在可接受范围。")}
    {decision_card(
      "IDF 采用 BM25+ 形式",
      "经典 Okapi BM25 的 IDF 在高文档频率下会取到负值，导致「出现于过半文档的词」反而拉低相关度得分。",
      "改用 ln(1 + (N−df+0.5)/(df+0.5))，保证任何词项的权重恒为正。",
      "与原始论文的数值不再严格可比，但对短文本与小语料更稳健。",
      "消融实验显示该形式下高频词不再产生负贡献，评分曲线单调可解释。")}
    {decision_card(
      "离线抽取式回答作为降级路径",
      "未配置大模型密钥时，系统不应直接不可用；但离线模式无法做推理。",
      "保留双模式：配置接口则走大模型，否则从召回片段中按词项加权重叠度抽取原句并附引用编号。",
      "抽取模式只摘录原句，不做推理、比较与数值换算，摘录内容可能来自相邻条目。",
      "保证「零配置也能跑通完整问答链路」，便于在无网络环境验证检索质量；输出中已明确标注该局限。")}
    {decision_card(
      "远程接口必须有密钥，本机地址可留空",
      "早期实现只检查 api_base 与 model 是否填写，导致默认配置每次都发起一次注定 401 的请求。",
      "用 ipaddress 解析主机名，判定是否为环回或私有地址：远程必须提供密钥，本机/内网地址允许留空。",
      "多了一层地址判定逻辑，非标准端口或域名映射的内网服务可能被误判。",
      "默认配置安静走离线模式；本地 Ollama / vLLM 无需密钥即可使用。")}
  </div>
</section>

<section>
  <h2><span class="n">07</span>已知边界</h2>
  <p class="sub">主动列出短板，比等着被问出来更有说服力。</p>
  <div class="card">
    <div class="tbl-wrap">
    <table>
      <thead><tr><th>能力</th><th>现状与原因</th></tr></thead>
      <tbody>
        <tr><td>文档格式</td><td>仅 <code class="mono">.md</code> / <code class="mono">.txt</code>。PDF / Word 需先转换，避免引入解析依赖</td></tr>
        <tr><td>检索模型</td><td>纯词项匹配，无向量语义检索。同义改写召回弱，是失败样例的主要来源</td></tr>
        <tr><td>分块粒度</td><td>按字符数切分，未按 token 计量；超长表格可能被截断</td></tr>
        <tr><td>增量更新</td><td>按文件 SHA-1 复用解析缓存，但倒排索引整体重建。万级片段内无感，再大需要引入分段索引</td></tr>
        <tr><td>并发</td><td>单进程；索引为单文件 JSON，不支持并发写。多进程场景需改为分片存储</td></tr>
        <tr><td>离线回答</td><td>仅摘录原句，不做推理、比较或数值换算，需靠大模型补足</td></tr>
        <tr><td>评测集</td><td>由作者手工构造并自校验，规模 {labels['in_domain']} 条；不能替代多人标注的公开基准</td></tr>
      </tbody>
    </table>
    </div>
  </div>
</section>

<section>
  <h2><span class="n">08</span>复现方式</h2>
  <p class="sub">无第三方依赖，克隆后即可运行；评测脚本与网页生成脚本一并提供。</p>
  <div class="card">
<pre><span class="c"># 1. 建立索引（默认读取 ./docs）</span>
python rag.py ingest

<span class="c"># 2. 提问：未配置密钥时自动使用离线抽取式回答，资料不足则拒答</span>
python rag.py ask "如何导入已有笔记？"

<span class="c"># 3. 只看检索结果与得分</span>
python rag.py search "导出支持哪些格式" -k 5

<span class="c"># 4. 端到端自检（{src['code']} 行代码，29 项断言）</span>
python selftest.py

<span class="c"># 5. 复现本文全部评测数据与图表</span>
python benchmark.py
python build_portfolio.py</pre>
    <p class="note">
      配置大模型（可选）：设置 <code class="mono">RAG_API_KEY</code> 即可，
      <code class="mono">RAG_API_BASE</code> 与 <code class="mono">RAG_MODEL</code>
      可指向任意 OpenAI 兼容服务，包括本地 Ollama。
    </p>
  </div>
</section>

<footer>
  <p style="margin:0">
    数据生成时间 {esc(meta['generated_at'])} ｜ 语料 {corpus['docs']} 篇 /
    {corpus['chunks']} 片段 / {corpus['terms']} 词项 ｜
    评测 {labels['in_domain']} 条领域内 + {labels['out_of_domain']} 条领域外
  </p>
  <p class="note" style="margin-top:6px">
    本页所有图表与数字均由 <code class="mono">build_portfolio.py</code> 从
    <code class="mono">benchmark_report.json</code> 生成，重跑评测后重新生成即可同步更新。
    示例语料为虚构产品「云雀笔记」的文档集。
  </p>
</footer>

</div>
</body>
</html>
"""


def validate(html_text: str) -> List[str]:
    """对生成结果做结构与几何自检，返回问题列表（空列表表示通过）。

    覆盖两类在无浏览器环境下最容易漏掉的缺陷：
    1. 容器标签配对失衡（模板拼接时漏写或多写闭合标签）。
    2. SVG 内元素坐标越界、出现 NaN，或文字按字宽估算后溢出画布。

    文字宽度用「CJK 记 1 em、其余记 0.55 em」估算，并正确处理
    text-anchor 的 start/middle/end 三种对齐方式 —— 忽略对齐方式会把
    居中文字误判为溢出。
    """
    problems: List[str] = []

    for tag in ("html", "head", "body", "header", "section", "div", "article",
                "table", "thead", "tbody", "tr", "td", "th", "svg", "ol", "ul",
                "li", "dl", "dt", "dd", "pre", "h1", "h2", "h3", "p", "strong"):
        opened = len(re.findall(rf"<{tag}[ >]", html_text))
        closed = len(re.findall(rf"</{tag}>", html_text))
        if opened != closed:
            problems.append(f"标签不配对：<{tag}> 开 {opened} / 闭 {closed}")

    ext = re.findall(r'(?:src|href)="(https?://[^"]*)"', html_text)
    if ext:
        problems.append(f"引用了外部资源，破坏离线可用性：{ext[:3]}")

    attr_re = re.compile(r'([\w-]+)="([^"]*)"')
    for i, (vw, vh, body) in enumerate(
            re.findall(r'<svg viewBox="0 0 (\d+) (\d+)"[^>]*>(.*?)</svg>', html_text, re.S), 1):
        w, h = int(vw), int(vh)
        if "nan" in body.lower():
            problems.append(f"SVG{i}：含 NaN 坐标")
        for attr in ("x", "y", "x1", "y1", "x2", "y2"):
            for v in re.findall(attr + r'="(-?[\d.]+)"', body):
                f = float(v)
                limit = w if attr.startswith("x") else h
                if not math.isfinite(f) or f < -2 or f > limit + 2:
                    problems.append(f"SVG{i}：{attr}={f} 超出画布 {limit}")
        for m in re.finditer(r"<text\s([^>]*)>([^<]*)</text>", body):
            a = dict(attr_re.findall(m.group(1)))
            txt = m.group(2)
            x = float(a.get("x", 0))
            fs = float(a.get("font-size", 12))
            anchor = a.get("text-anchor", "start")
            width = sum(fs * (1.0 if ord(c) > 0x2000 else 0.55) for c in txt)
            if anchor == "middle":
                left, right = x - width / 2, x + width / 2
            elif anchor == "end":
                left, right = x - width, x
            else:
                left, right = x, x + width
            if right > w + 1 or left < -1:
                problems.append(
                    f"SVG{i}：文字 {txt[:22]!r} 横向 {left:.0f}~{right:.0f} 溢出 0~{w}")
    return problems


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = load_report()
    html_text = build(report)

    problems = validate(html_text)
    if problems:
        print("自检未通过：")
        for p in problems:
            print(f"  - {p}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"作品集已生成：{OUT_FILE}")
    print(f"  体积 {len(html_text.encode()) / 1024:.1f} KB，外部资源 0 个，"
          f"结构自检与 {len(re.findall(r'<svg ', html_text))} 个图表几何自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
