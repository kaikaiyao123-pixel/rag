#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端自检：验证建库、检索、离线回答全链路。

    python selftest.py

不依赖任何网络与第三方库。退出码 0 表示全部通过。
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragcore.chunker import chunk_document          # noqa: E402
from ragcore.generator import extractive_answer     # noqa: E402
from ragcore.index import BM25Index                 # noqa: E402
from ragcore.loader import discover_documents, read_text, sha1_of  # noqa: E402
from ragcore.retriever import Retriever             # noqa: E402
from ragcore.tokenizer import tokenize              # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def main():
    print("== 1. 分词器 ==")
    t = tokenize("云雀笔记支持 Markdown 导入，databases 会被归一化")
    check("中文产生 bigram", "云雀" in t and "雀笔" in t, str(t[:6]))
    check("英文小写化", "markdown" in t)
    check("复数归一", "database" in t and "databases" not in t)
    check("空串安全", tokenize("") == [])

    print("\n== 2. 分块器 ==")
    md = ("# 标题A\n\n第一段内容，讲的是导入方式。\n\n第二段内容，讲的是导出格式。\n\n"
          "## 子标题B\n\n" + "很长的段落。" * 120)
    chunks = chunk_document("t.md", md, max_chars=300, min_chars=50, overlap=50)
    check("产生多个块", len(chunks) >= 3, f"got {len(chunks)}")
    check("标题路径已记录", any("标题A" in c["heading"] for c in chunks))
    check("层级路径用 › 连接", any("标题A › 子标题B" == c["heading"] for c in chunks))
    check("块长受限", all(len(c["text"]) <= 300 * 1.5 for c in chunks))

    # 回归：围栏代码块内的 "# 注释" 不得被解析为标题
    fenced = ("# 真标题\n\n正文段落，描述检索语法。\n\n"
              "```python\n# 这是注释，不是标题\nprint(1)\n```\n\n"
              "结尾段落，仍然属于真标题。\n")
    fc = chunk_document("t2.md", fenced, max_chars=600, min_chars=20, overlap=0)
    check("代码块内注释未被当作标题",
          all("这是注释" not in c["heading"] for c in fc),
          f"headings={[c['heading'] for c in fc]}")
    check("代码块后正文仍归属原标题",
          any(c["heading"] == "真标题" and "结尾段落" in c["text"] for c in fc))

    print("\n== 3. 建库 ==")
    docs = discover_documents(["docs"], [".md", ".txt"])
    check("发现示例文档", len(docs) >= 3, f"got {docs}")

    all_chunks = []
    for rel in docs:
        all_chunks.extend(chunk_document(rel, read_text(rel), 600, 120, 100))
    all_chunks.sort(key=lambda c: (c["doc"], c["start_line"]))
    idx = BM25Index().build(all_chunks)
    check("片段数合理", len(idx.chunks) > 5, f"got {len(idx.chunks)}")
    check("词项表非空", len(idx.df) > 100, f"got {len(idx.df)}")
    check("avgdl 为正", idx.avgdl > 0)

    print("\n== 4. 检索 ==")
    retriever = Retriever(idx, top_k=3, candidate_k=20, max_per_doc=2)

    cases = [
        ("如何导入已有笔记", "faq"),
        ("导出支持哪些格式", "faq"),
        ("版本历史保留多久", None),
        ("团队版多少钱", "pricing"),
        ("数据存储在哪里", None),
        ("双向链接怎么用", "product-overview"),
    ]
    for q, expect in cases:
        hits = retriever.retrieve(q)
        ok = bool(hits)
        detail = "无命中"
        if ok:
            top = hits[0]
            detail = f"top={top['doc']} › {top['heading']} ({top['score']:.3f})"
            if expect:
                ok = expect in top["doc"]
        check(f"查询「{q}」", ok, detail)

    # 检索质量下限：正确答案应在 Top-3 内出现
    hits = retriever.retrieve("免费版可以登录几台设备", top_k=3)
    check("表格类问题可定位到 pricing",
          any("pricing" in h["doc"] for h in hits),
          str([h["doc"] for h in hits]))

    print("\n== 5. 单文档限流与去重 ==")
    hits = retriever.retrieve("笔记 导入 导出 同步 检索", top_k=6)
    from collections import Counter
    cnt = Counter(h["doc"] for h in hits)
    check("单文档不超过 max_per_doc", all(v <= 2 for v in cnt.values()), str(cnt))
    check("片段文本无重复", len({h["text"][:60] for h in hits}) == len(hits))

    print("\n== 6. 离线抽取式回答 ==")
    hits = retriever.retrieve("如何导入已有笔记")
    ans = extractive_answer("如何导入已有笔记", hits, idx.idf)
    check("回答非空且带引用", len(ans) > 20 and "[" in ans, ans[:80])
    check("无关问题被拒答",
          extractive_answer("量子纠缠的退相干时间是多少", retriever.retrieve("量子纠缠"),
                            idx.idf).startswith("根据现有资料"),
          "")

    print("\n== 7. 持久化与增量 ==")
    tmp = tempfile.mkdtemp(prefix="ragtest_")
    try:
        path = idx.save(tmp)
        check("索引文件已写出", os.path.isfile(path))
        idx2 = BM25Index.load(tmp)
        check("可重新加载", idx2 is not None and len(idx2.chunks) == len(idx.chunks))
        check("IDF 一致",
              all(abs(idx2.idf[k] - v) < 1e-9 for k, v in list(idx.idf.items())[:50]))
        h1 = idx.search("如何导入笔记", top_k=3)
        h2 = idx2.search("如何导入笔记", top_k=3)
        check("检索结果一致",
              [h["chunk_id"] for h in h1] == [h["chunk_id"] for h in h2])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n== 8. 文档与评测数据一致性 ==")
    # README 的指标表必须与 benchmark_report.json 逐项一致。这条约定此前只靠人工
    # 遵守，实测已漂移过：keyword 行曾长期停留在一次更早运行的残值上。这里做成
    # 断言，改评测后忘了同步 README 会直接让自检失败。
    root = os.path.dirname(os.path.abspath(__file__))
    rep_path = os.path.join(root, "benchmark_report.json")
    readme_path = os.path.join(root, "README.md")
    if not (os.path.isfile(rep_path) and os.path.isfile(readme_path)):
        check("benchmark_report.json 与 README.md 均存在", False, "缺少文件")
    else:
        with open(rep_path, encoding="utf-8") as f:
            metrics = json.load(f)["metrics"]
        with open(readme_path, encoding="utf-8") as f:
            readme = f.read()
        # README 表格行标签（前缀匹配）→ 报告中的方法键
        label2key = [("本方案", "full"), ("纯 Okapi BM25", "bm25_plain"),
                     ("TF-IDF 余弦", "tfidf"), ("关键词命中计数", "keyword")]
        found = set()
        for line in readme.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip().replace("*", "") for c in line.strip("|").split("|")]
            if len(cells) != 5:
                continue
            hit = next((lb for lb, _ in label2key if cells[0].startswith(lb)), None)
            if hit is None or hit in found:
                continue
            found.add(hit)
            m = metrics[dict(label2key)[hit]]
            want = [m["doc_recall"]["@1"], m["passage_hit"]["@1"],
                    m["passage_hit"]["@5"], m["mrr@10"]]
            try:
                got = [float(c) for c in cells[1:]]
            except ValueError:
                check(f"README 指标行「{hit}」四个数字可解析", False, str(cells[1:]))
                continue
            check(f"README 指标行「{hit}」与 benchmark_report.json 一致",
                  all(abs(round(w, 3) - g) < 5e-4 for w, g in zip(want, got)),
                  f"README={got} 报告={[round(w, 3) for w in want]}")
        check("README 指标表含全部四个方法行", len(found) == 4, f"找到 {sorted(found)}")

    print(f"\n{'=' * 46}\n通过 {PASS} 项，失败 {FAIL} 项")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
