#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索质量与性能基准测试。

    python benchmark.py                # 运行全部评测并写出 benchmark_report.json
    python benchmark.py --check-labels # 仅校验标注集是否与语料一致

评测设计
--------
1. **标注查询集**：手工构造的"问题 → 期望文档 → 期望关键片段"三元组。
   每条标注的 ``must`` 字段是一段必须出现在正确片段中的特征字符串，
   脚本会先校验该字符串确实存在于期望文档的某个片段里，防止标注错误
   导致指标失真。

2. **两档命中判定**：
   - ``Doc Recall@k``：前 k 条结果中包含期望文档（文档级，宽松）
   - ``Passage Hit@k``：前 k 条结果中同时满足"来自期望文档"且"包含 must
     特征串"（片段级，严格）
   两档指标并列呈现，避免用宽松指标掩盖"定位到了文档但没找到正确段落"
   这类真实缺陷。

3. **消融对比**：在完全相同的分词、分块与后处理管线上，替换打分函数：
   - ``full``        BM25 + 标题加权 + 词项覆盖率加权（本方案默认）
   - ``bm25_plain``  纯 Okapi BM25，关闭两项启发式加权
   - ``tfidf``       TF-IDF 向量余弦相似度
   - ``keyword``     去重查询词命中计数（最朴素基线）

4. **拒答能力**：另设一组领域外问题，统计系统是否正确输出拒答话术，
   并对比领域内/领域外的最高分分布，说明两者的可分性。

5. **性能**：索引构建耗时、热查询延迟分位数、索引文件体积、
   索引加载的 Python 堆内存增量（tracemalloc）。

指标说明：本评测每条查询只有一个正确片段，因此不计算 nDCG —— 在单相关
文档设定下 nDCG 与 MRR 的信息量高度重叠，同时列出只会造成冗余。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time
import tracemalloc
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragcore.abstain import features as abstain_features      # noqa: E402
from ragcore.chunker import chunk_document                    # noqa: E402
from ragcore.cli import DEFAULT_CONFIG                        # noqa: E402
from ragcore.generator import extractive_answer                # noqa: E402
from ragcore.index import BM25Index                            # noqa: E402
from ragcore.loader import discover_documents, read_text, sha1_of  # noqa: E402
from ragcore.retriever import dedupe                           # noqa: E402
from ragcore.tokenizer import tokenize                         # noqa: E402

REPORT_PATH = "benchmark_report.json"
KS = (1, 3, 5)
MRR_K = 10
CANDIDATE_K = 30
MAX_PER_DOC = 3

# --------------------------------------------------------------------- 标注集
# (查询, 期望文档, 正确片段中必须出现的特征串)
#
# 按难度分四层。整体指标会被难例稀释，分层统计才能看出方法在哪类问法上失效：
#   basic       问法与文档用词高度重合，考察"能不能检索到"
#   paraphrase  问法刻意避开原文用词，考察词项扩展能力（零依赖二元切分的主要短板）
#   detail      答案埋在具体数值 / 路径 / 协议名里，考察片段级定位精度
#   confusable  多个文档出现相近数字或概念，考察能否区分到正确文档
LABELED_BASIC: List[Tuple[str, str, str]] = [
    # product-overview.md
    ('产品主要面向哪些用户群体？', 'docs/product-overview.md', '需要长期积累资料的研发人员'),
    ('双向链接是怎么建立和查看的？', 'docs/product-overview.md', '反向链接'),
    ('网页剪藏提供了哪几种模式？', 'docs/product-overview.md', '三种剪藏模式'),
    ('版本历史默认保留多少个版本？', 'docs/product-overview.md', '200 个版本'),
    ('数据以什么格式存放在本地？', 'docs/product-overview.md', '纯 Markdown 文件'),
    # faq.md
    ('如何导入 Obsidian 库？', 'docs/faq.md', 'Obsidian 仓库根目录'),
    ('支持导入印象笔记的导出包吗？', 'docs/faq.md', '.enex'),
    ('导出支持哪些格式？', 'docs/faq.md', 'HTML 单文件'),
    ('搜索不到确定存在的内容该怎么排查？', 'docs/faq.md', '重建索引'),
    ('知识库默认存放在哪个目录？', 'docs/faq.md', '文档/云雀笔记/'),
    ('移动端的功能范围有什么限制？', 'docs/faq.md', '功能限于阅读、编辑和检索'),
    ('导入时遇到同名笔记会怎么处理？', 'docs/faq.md', '保留两者'),
    # pricing.md
    ('免费版可以登录几台设备？', 'docs/pricing.md', '1 台'),
    ('个人版一年多少钱？', 'docs/pricing.md', '128 元'),
    ('团队版比个人版多了哪些协作能力？', 'docs/pricing.md', '共享知识库'),
    ('学生可以免费使用吗？', 'docs/pricing.md', '在校学生与教师'),
    ('退款政策是怎样的？', 'docs/pricing.md', '14 天内'),
    ('可以开增值税专用发票吗？', 'docs/pricing.md', '增值税专用'),
    ('团队版按什么口径计费？', 'docs/pricing.md', '实际活跃成员计费'),
    # sync-guide.md
    ('一次同步大概产生多少流量？', 'docs/sync-guide.md', '1~3 KB'),
    ('Git 自动提交的默认间隔是多少？', 'docs/sync-guide.md', '5 分钟'),
    ('网盘同步为什么容易产生冲突？', 'docs/sync-guide.md', '文件级同步'),
    ('局域网同步使用什么加密协议？', 'docs/sync-guide.md', 'TLS 1.3'),
    ('冲突副本的文件名是什么格式？', 'docs/sync-guide.md', '冲突副本'),
    ('同步一直卡在正在比对怎么处理？', 'docs/sync-guide.md', '正在比对'),
    ('同步时为什么要排除 .lark 目录？', 'docs/sync-guide.md', '忽略列表'),
    # search-syntax.md
    ('怎么在检索中排除某个关键词？', 'docs/search-syntax.md', '排除'),
    ('支持正则表达式检索吗？', 'docs/search-syntax.md', '正则'),
    ('怎么按时间范围筛选笔记？', 'docs/search-syntax.md', 'updated:'),
    ('智能笔记本最多能嵌套几层？', 'docs/search-syntax.md', '3 层'),
    ('一万篇笔记的索引占多少内存？', 'docs/search-syntax.md', '180 MB'),
    ('短语精确匹配怎么写？', 'docs/search-syntax.md', '短语精确匹配'),
    # markdown-guide.md
    ('支持 Mermaid 图表吗？', 'docs/markdown-guide.md', 'Mermaid'),
    ('数学公式用哪个渲染引擎？', 'docs/markdown-guide.md', 'KaTeX'),
    ('front matter 里的 aliases 字段有什么用？', 'docs/markdown-guide.md', 'aliases'),
    ('表格支持合并单元格吗？', 'docs/markdown-guide.md', '合并单元格'),
    ('任务列表的勾选状态导出后保存在哪里？', 'docs/markdown-guide.md', 'tasks'),
    ('笔记遵循哪个 Markdown 规范？', 'docs/markdown-guide.md', 'CommonMark'),
    # plugin-api.md
    ('插件运行在什么环境里？', 'docs/plugin-api.md', '独立'),
    ('插件钩子超时会被怎么处理？', 'docs/plugin-api.md', '超时'),
    ('插件能直接写系统文件吗？', 'docs/plugin-api.md', '越权'),
    ('插件市场对开源有什么要求？', 'docs/plugin-api.md', '开源'),
    ('插件的权限怎么声明？', 'docs/plugin-api.md', 'permissions'),
    ('插件怎么调试？', 'docs/plugin-api.md', '开发者模式'),
    # security-privacy.md
    ('软件会收集我的使用数据吗？', 'docs/security-privacy.md', '遥测'),
    ('索引加密使用什么算法？', 'docs/security-privacy.md', 'AES-256-GCM'),
    ('忘记主密码还能找回索引吗？', 'docs/security-privacy.md', '忘记密码'),
    ('浏览器扩展会读取浏览历史吗？', 'docs/security-privacy.md', '浏览历史'),
    ('导出诊断包会包含笔记正文吗？', 'docs/security-privacy.md', '不包含笔记正文'),
    # troubleshooting.md
    ('启动后白屏该怎么排查？', 'docs/troubleshooting.md', '白屏'),
    ('输入中文时出现重复字符怎么办？', 'docs/troubleshooting.md', 'IME'),
    ('误删的笔记怎么恢复？', 'docs/troubleshooting.md', '回收站'),
    ('日志文件保存在哪里？', 'docs/troubleshooting.md', 'logs'),
    ('索引重建速度大概是多少？', 'docs/troubleshooting.md', '3000~5000'),
    ('输入法候选词定位偏移怎么解决？', 'docs/troubleshooting.md', '候选词'),
    # release-notes.md
    ('2.4.0 版本新增了什么检索功能？', 'docs/release-notes.md', '正则表达式'),
    ('索引格式什么时候升级到 v3？', 'docs/release-notes.md', 'v3'),
    ('2.0.0 为什么被标记为不兼容升级？', 'docs/release-notes.md', '不兼容'),
    ('1.x 系列还在维护吗？', 'docs/release-notes.md', '停止维护'),
    ('哪个版本开始支持局域网同步？', 'docs/release-notes.md', '局域网直连同步'),
    # mobile-app.md
    ('移动端支持哪些系统版本？', 'docs/mobile-app.md', 'iOS 15.0'),
    ('移动端怎么和桌面端同步数据？', 'docs/mobile-app.md', '局域网同步'),
    ('移动端的生物识别锁能加密数据吗？', 'docs/mobile-app.md', '界面层锁定'),
    ('移动端同步卡在正在接收索引是什么原因？', 'docs/mobile-app.md', '存储空间不足'),
    ('移动端为什么显示不出复杂公式？', 'docs/mobile-app.md', '精简版渲染引擎'),
    # team-collaboration.md
    ('共享知识库的权限分几档？', 'docs/team-collaboration.md', '只读'),
    ('团队版怎么处理成员离职？', 'docs/team-collaboration.md', '停用'),
    ('私有化部署需要什么服务器配置？', 'docs/team-collaboration.md', '4 核 8 GB'),
    ('审计日志会保存笔记全文吗？', 'docs/team-collaboration.md', '不存储笔记全文'),
    ('评论可以标记状态吗？', 'docs/team-collaboration.md', '待处理'),
    ('共享知识库的编辑冲突怎么解决？', 'docs/team-collaboration.md', '乐观并发控制'),
]

LABELED_PARAPHRASE: List[Tuple[str, str, str]] = [
    # 问法刻意避开原文用词；没有同义词表，这一层是主要短板所在
    # product-overview.md
    ('这个软件适合哪些人使用？', 'docs/product-overview.md', '咨询顾问'),
    ('笔记能用别的编辑器打开吗？', 'docs/product-overview.md', '其他编辑器打开'),
    ('剪藏之后还能查到来处吗？', 'docs/product-overview.md', '原始 URL'),
    # faq.md
    ('换电脑了怎么把旧笔记带过来？', 'docs/faq.md', 'Markdown 文件夹'),
    ('导出的文件能直接发给别人看吗？', 'docs/faq.md', '便于分享'),
    # pricing.md
    ('团队版和个人版的差别主要在哪？', 'docs/pricing.md', '设备数量'),
    # sync-guide.md
    ('同步会不会很费流量？', 'docs/sync-guide.md', '与笔记长度基本无关'),
    ('两台电脑同时改了同一篇会怎样？', 'docs/sync-guide.md', '保留双方版本'),
    # search-syntax.md
    ('只想在标题里找某个词，怎么搜？', 'docs/search-syntax.md', 'title:'),
    ('能按创建时间筛选笔记吗？', 'docs/search-syntax.md', 'created:'),
    # markdown-guide.md
    ('数学公式能正常显示成符号吗？', 'docs/markdown-guide.md', '独立公式用'),
    ('能画流程图和时序图吗？', 'docs/markdown-guide.md', '流程图'),
    # plugin-api.md
    ('自己写的小工具怎么装进去？', 'docs/plugin-api.md', 'manifest.json'),
    ('插件能访问哪些网站由谁决定？', 'docs/plugin-api.md', 'allowedDomains'),
    # security-privacy.md
    ('我的笔记会被传到服务器上吗？', 'docs/security-privacy.md', '不存在任何形式的数据上传'),
    ('程序崩溃会偷偷上报吗？', 'docs/security-privacy.md', '没有崩溃上报'),
    # troubleshooting.md
    ('打开就一片空白怎么排查？', 'docs/troubleshooting.md', '安全模式启动'),
    ('打字时字符重复出现是什么原因？', 'docs/troubleshooting.md', '兼容输入法模式'),
    ('删掉的笔记多久内还能找回？', 'docs/troubleshooting.md', '保留 30 天'),
    # release-notes.md
    ('最新版加了哪些功能？', 'docs/release-notes.md', '2.4.0'),
    # mobile-app.md
    ('手机端最低支持什么系统？', 'docs/mobile-app.md', 'Android 10'),
    ('手机和电脑怎么互传笔记？', 'docs/mobile-app.md', 'mDNS 发现后配对'),
    # team-collaboration.md
    ('团队里谁能删掉整个知识库？', 'docs/team-collaboration.md', '可管理'),
    ('成员离职后他建的笔记会被删吗？', 'docs/team-collaboration.md', '不会删除该成员创建的笔记'),
]

LABELED_DETAIL: List[Tuple[str, str, str]] = [
    # 答案埋在具体数值 / 路径 / 协议名里，考察片段级定位精度
    # product-overview.md
    ('版本快照除数量外还按什么时间范围保留？', 'docs/product-overview.md', '90 天'),
    ('关闭更新检查的入口在哪里？', 'docs/product-overview.md', '设置 → 通用 → 隐私'),
    # faq.md
    ('导入 Markdown 文件夹会保留原有目录结构吗？', 'docs/faq.md', '按原有的子目录结构建立笔记本'),
    ('纯文本导入能识别哪些编码？', 'docs/faq.md', 'Big5'),
    ('导出的 PDF 能附带反向链接吗？', 'docs/faq.md', '反向链接附录'),
    # pricing.md
    ('免费版的版本历史保留多久？', 'docs/pricing.md', '最近 30 天'),
    ('个人版超期的版本归档到哪个目录？', 'docs/pricing.md', '.lark/archive'),
    ('工单响应时间承诺多久？', 'docs/pricing.md', '1 个工作日'),
    # sync-guide.md
    ('自动提交的间隔能调整到多长？', 'docs/sync-guide.md', '1~60 分钟'),
    ('拉取远端更新前会先执行什么 Git 操作？', 'docs/sync-guide.md', 'git stash'),
    ('局域网同步配对时会交换什么？', 'docs/sync-guide.md', '交换公钥'),
    # search-syntax.md
    ('同一类限定符写两次会怎样？', 'docs/search-syntax.md', '只保留最后一个'),
    ('正则检索单次最多执行多久？', 'docs/search-syntax.md', '2 秒'),
    # markdown-guide.md
    ('代码块支持多少种语言高亮？', 'docs/markdown-guide.md', '180'),
    ('引用最多能嵌套几层？', 'docs/markdown-guide.md', '5 层'),
    ('表格过宽导出 PDF 时最小缩到多少？', 'docs/markdown-guide.md', '60%'),
    # plugin-api.md
    ('插件调用了未声明的能力会抛什么错误？', 'docs/plugin-api.md', 'PermissionDeniedError'),
    ('插件钩子连续超时几次会被禁用？', 'docs/plugin-api.md', '连续三次'),
    # security-privacy.md
    ('索引加密的密钥用什么算法派生？', 'docs/security-privacy.md', 'Argon2id'),
    ('Linux 下知识库目录的权限位是多少？', 'docs/security-privacy.md', '0700'),
    # troubleshooting.md
    ('粘贴多长的文本会开始卡顿？', 'docs/troubleshooting.md', '5 万字符'),
    ('撤销栈最多保留多少步？', 'docs/troubleshooting.md', '200 步'),
    ('局域网同步要放行哪个 TCP 端口？', 'docs/troubleshooting.md', '47820'),
    # release-notes.md
    ('2.1.0 改用二元切分后索引体积变成多少倍？', 'docs/release-notes.md', '2.4 倍'),
]

LABELED_CONFUSABLE: List[Tuple[str, str, str]] = [
    # 问法只给主题、不给唯一标识词，答案散落在多篇同主题文档中；
    # 考察能否靠剩余词项把范围收敛到正确的那一篇。
    # team-collaboration.md
    ('服务端存储怎么加密？', 'docs/team-collaboration.md', 'AES-256 静态加密'),
    # pricing.md
    ('团队版的版本历史能保留多久？', 'docs/pricing.md', '永久保留'),
    ('个人版可以同时在几台设备上用？', 'docs/pricing.md', '3 台'),
    # troubleshooting.md
    ('十万片段重建索引大约要多久？', 'docs/troubleshooting.md', '20~35 秒'),
    ('局域网同步找不到设备要检查哪个 UDP 端口？', 'docs/troubleshooting.md', 'UDP 5353'),
    # plugin-api.md
    ('插件要求的最低主程序版本过低会怎样？', 'docs/plugin-api.md', '可能不兼容'),
    # faq.md
    ('导出能带上全部元数据吗？', 'docs/faq.md', '包含全部元数据'),
    # pricing.md
    ('免费版有广告吗？', 'docs/pricing.md', '没有广告'),
    # troubleshooting.md
    ('索引体积正常应该在什么范围？', 'docs/troubleshooting.md', '25%~40%'),
    # mobile-app.md
    ('移动端的索引占正文多少？', 'docs/mobile-app.md', '约为正文的 30%'),
    # team-collaboration.md
    ('审计日志能按哪些条件筛选？', 'docs/team-collaboration.md', '变更类型'),
    # troubleshooting.md
    ('知识库目录被占用要删哪个文件？', 'docs/troubleshooting.md', '.lark/lock'),
    # security-privacy.md
    ('版本快照存在哪个目录？', 'docs/security-privacy.md', '.lark/history/'),
    ('开启索引加密后每次启动要做什么？', 'docs/security-privacy.md', '每次启动需输入主密码'),
    # sync-guide.md
    ('冲突副本合并完成后可以怎么处理？', 'docs/sync-guide.md', '批量清理'),
    # team-collaboration.md
    ('共享知识库的成员权限按什么粒度授予？', 'docs/team-collaboration.md', '成员粒度'),
    # plugin-api.md
    ('插件的网络访问默认允许哪些协议？', 'docs/plugin-api.md', '默认只允许 HTTPS'),
    # troubleshooting.md
    ('撤销和版本历史有什么区别？', 'docs/troubleshooting.md', '撤销栈上限'),
    ('笔记超过多大就只索引一部分？', 'docs/troubleshooting.md', '1 MB'),
    # search-syntax.md
    ('热查询的延迟中位数是多少？', 'docs/search-syntax.md', '20 ms'),
    # markdown-guide.md
    ('表格太宽时导出会怎么处理？', 'docs/markdown-guide.md', '自动缩放'),
    # mobile-app.md
    ('移动端首次同步大知识库要多久？', 'docs/mobile-app.md', '10~30 分钟'),
    # team-collaboration.md
    ('私有化部署用什么数据库？', 'docs/team-collaboration.md', 'PostgreSQL 14'),
    # troubleshooting.md
    ('日志默认保留多久？', 'docs/troubleshooting.md', '7 天'),
]

TIERS: List[Tuple[str, List[Tuple[str, str, str]]]] = [
    ("basic", LABELED_BASIC),
    ("paraphrase", LABELED_PARAPHRASE),
    ("detail", LABELED_DETAIL),
    ("confusable", LABELED_CONFUSABLE),
]

LABELED: List[Tuple[str, str, str]] = [item for _, group in TIERS for item in group]


# 领域外问题：语料中不存在答案，期望系统拒答
OUT_OF_DOMAIN: List[str] = [
    "请解释量子纠缠中的退相干机制",
    "如何用 Python 实现一个稳定的快速排序",
    "2026 年世界杯的冠军是哪支球队",
    "请推荐几家附近的川菜馆",
    "急性肠胃炎应该怎么治疗",
    "光伏逆变器的 MPPT 算法原理是什么",
    "请帮我写一首关于秋天的七言绝句",
]

REFUSAL = "根据现有资料无法回答该问题。"


def passage_text(chunk: Dict) -> str:
    """检索单元的完整可见文本。

    生成阶段送给大模型的参考资料包含「标题路径 + 正文」，因此判定片段命中时
    也把标题路径计入，否则像"启动后白屏"这种答案写在标题里的片段会被误判为
    未命中。
    """
    return f"{chunk.get('heading', '')}\n{chunk.get('text', '')}"


# --------------------------------------------------------------------- 语料
def build_corpus(docs_dir: str = "docs") -> Tuple[List[Dict], Dict[str, Dict], Dict]:
    """解析并分块，返回 (chunks, files_meta, timing)。"""
    files = discover_documents([docs_dir])
    chunks: List[Dict] = []
    meta: Dict[str, Dict] = {}
    t0 = time.perf_counter()
    for rel in files:
        text = read_text(rel)
        part = chunk_document(rel, text, max_chars=600, min_chars=120, overlap=100)
        chunks.extend(part)
        meta[rel] = {"sha1": sha1_of(text), "chars": len(text), "chunks": len(part)}
    parse_s = time.perf_counter() - t0
    return chunks, meta, {"parse_s": parse_s, "docs": len(files)}


# --------------------------------------------------------------------- 基线
def make_rankers(index: BM25Index) -> Dict[str, Callable[[str], List[Tuple[int, float]]]]:
    """构造 4 个打分器，全部返回 (chunk_id, score) 降序列表。"""
    full = index

    plain = copy.copy(index)
    plain.heading_boost = 0.0
    plain.coverage_weight = 0.0

    # TF-IDF 向量余弦：与 BM25 共用同一套分词与词频统计
    doc_norms = [0.0] * len(index.chunks)
    for t, posting in index.postings.items():
        idf = index.idf[t]
        for cid, f in posting.items():
            w = (f / (index.doc_len[cid] or 1)) * idf
            doc_norms[cid] += w * w
    doc_norms = [math.sqrt(x) or 1.0 for x in doc_norms]

    def rank_full(query: str) -> List[Tuple[int, float]]:
        return [(r["chunk_id"], r["score"])
                for r in full.search(query, top_k=len(index.chunks),
                                     candidate_k=len(index.chunks),
                                     max_per_doc=0)]

    def rank_plain(query: str) -> List[Tuple[int, float]]:
        return [(r["chunk_id"], r["score"])
                for r in plain.search(query, top_k=len(index.chunks),
                                      candidate_k=len(index.chunks),
                                      max_per_doc=0)]

    # 两个基线都必须显式打破并列（二级键 chunk_id），且查询词项按字典序展开。
    # 原因：keyword 是整数命中计数，并列极多；而 sorted/sort 属稳定排序，并列项
    # 会继承 acc 的插入顺序，插入顺序又取决于 set(tokenize(query)) 的迭代顺序 ——
    # 后者受 PYTHONHASHSEED 随机化影响。若不打破并列，同一份代码、同一份语料，
    # 换一个进程就会跑出不同指标（实测 keyword MRR@10 在 0.8380~0.8514 间漂移）。
    # 生产路径 index.py 的 results.sort 早已使用 (-score, chunk_id) 二级键，
    # 这里补齐同一约定，使评测结果可复现。
    def rank_tfidf(query: str) -> List[Tuple[int, float]]:
        qv = {t: index.idf[t] for t in sorted(set(tokenize(query))) if t in index.idf}
        if not qv:
            return []
        qnorm = math.sqrt(sum(w * w for w in qv.values())) or 1.0
        acc: Dict[int, float] = {}
        for t, w in qv.items():
            for cid, f in index.postings.get(t, {}).items():
                acc[cid] = acc.get(cid, 0.0) + (f / (index.doc_len[cid] or 1)) * w * w
        scored = [(cid, s / (doc_norms[cid] * qnorm)) for cid, s in acc.items()]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def rank_keyword(query: str) -> List[Tuple[int, float]]:
        acc: Dict[int, float] = {}
        for t in sorted(set(tokenize(query))):
            for cid in index.postings.get(t, {}):
                acc[cid] = acc.get(cid, 0.0) + 1.0
        scored = sorted(acc.items(), key=lambda x: (-x[1], x[0]))
        return scored

    return {"full": rank_full, "bm25_plain": rank_plain,
            "tfidf": rank_tfidf, "keyword": rank_keyword}


def postprocess(ranked: Sequence[Tuple[int, float]], index: BM25Index,
                top_k: int = max(KS)) -> List[Dict]:
    """与线上 Retriever 一致的后处理：候选截断 → 单文档限流 → 去重。"""
    rows = []
    for cid, score in list(ranked)[:CANDIDATE_K]:
        ch = index.chunks[cid]
        rows.append({"chunk_id": cid, "score": score, "doc": ch.get("doc", ""),
                     "heading": ch.get("heading", ""), "text": ch.get("text", "")})
    kept, per_doc, overflow = [], {}, []
    for r in rows:
        d = r["doc"]
        if per_doc.get(d, 0) < MAX_PER_DOC:
            per_doc[d] = per_doc.get(d, 0) + 1
            kept.append(r)
        else:
            overflow.append(r)
    if len(kept) < top_k:
        kept.extend(overflow[: top_k - len(kept)])
    return dedupe(kept)[:top_k]


# --------------------------------------------------------------------- 评测
def evaluate(rank_fn, index: BM25Index, labeled) -> Dict:
    n = len(labeled)
    doc_hits = {k: 0 for k in KS}
    pass_hits = {k: 0 for k in KS}
    rr_sum = 0.0
    misses: List[Dict] = []
    top1_scores: List[float] = []

    for query, exp_doc, must in labeled:
        results = postprocess(rank_fn(query), index)
        if results:
            top1_scores.append(results[0]["score"])

        doc_rank = None
        pass_rank = None
        for i, r in enumerate(results, 1):
            if r["doc"] == exp_doc:
                if doc_rank is None:
                    doc_rank = i
                if must in passage_text(r) and pass_rank is None:
                    pass_rank = i

        for k in KS:
            if doc_rank and doc_rank <= k:
                doc_hits[k] += 1
            if pass_rank and pass_rank <= k:
                pass_hits[k] += 1

        if doc_rank and doc_rank <= MRR_K:
            rr_sum += 1.0 / doc_rank

        if pass_rank is None:
            misses.append({
                "query": query, "expected_doc": exp_doc, "must": must,
                "doc_rank": doc_rank,
                "top1": results[0]["doc"] if results else None,
                "top1_head": (results[0]["heading"][:60] if results else None),
            })

    return {
        "queries": n,
        "doc_recall": {f"@{k}": round(doc_hits[k] / n, 4) for k in KS},
        "passage_hit": {f"@{k}": round(pass_hits[k] / n, 4) for k in KS},
        "mrr@10": round(rr_sum / n, 4),
        "passage_misses": len(misses),
        "mean_top1_score": round(statistics.mean(top1_scores), 3) if top1_scores else 0.0,
        "_misses": misses,
    }


def _trips(f: Dict[str, float], v: float, t: float, n: float) -> bool:
    """给定阈值组合，判断某条查询是否会被判定为拒答。"""
    return f["vocab_cov"] < v or f["idf_cov"] < t or f["norm_score"] < n


def _summarize(feats: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """输出各特征的均值与分位数，用于观察领域内/外的可分性。"""
    out = {}
    for key in ("vocab_cov", "idf_cov", "norm_score"):
        vals = sorted(f[key] for f in feats)
        if not vals:
            out[key] = {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
            continue

        def pct(p):
            i = min(len(vals) - 1, int(round(p / 100.0 * (len(vals) - 1))))
            return round(vals[i], 4)

        out[key] = {"mean": round(statistics.mean(vals), 4),
                    "p10": pct(10), "p50": pct(50), "p90": pct(90)}
    return out


def calibrate_abstention(index: BM25Index, rank_fn, labeled, ood) -> Dict:
    """在标注集上网格搜索拒答阈值。

    目标函数取**平衡准确率**：领域外拒答率与领域内不误拒率的算术平均。
    单纯最大化拒答率会把阈值推高到"什么都拒答"的退化解，因此必须同时
    惩罚误拒。

    由于平衡准确率存在大量并列最优解（本例中有数十组阈值同为 100%），
    仅按准确率取第一个解会选中 ``(0.0, 0.2, 0.0)`` 这类**退化解**——
    两项阈值被压到 0 相当于禁用，系统只靠单一特征判定，换一批数据极易失效。
    因此在并列解中改用 **max-min 间隔**准则二次筛选：取每个阈值到最近样本
    值的距离（按特征量程归一），再取三项中的最小值，选该值最大者。这等价于
    把所有阈值都推到特征分布的空隙中央，对分布漂移最不敏感。

    搜索结束后一并返回帕累托前沿，便于按业务偏好另选工作点。
    """
    in_feats = [abstain_features(q, postprocess(rank_fn(q), index), index.idf, index.postings)
                for q, _, _ in labeled]
    ood_feats = [abstain_features(q, postprocess(rank_fn(q), index), index.idf, index.postings)
                 for q in ood]

    grid = {
        "vocab_cov": [0.0, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45],
        "idf_cov": [0.0, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30],
        "norm_score": [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    }

    # 每个特征的量程，用于把间隔归一化到可比较的尺度
    spans = {}
    for key in grid:
        vals = [f[key] for f in in_feats] + [f[key] for f in ood_feats]
        spans[key] = (max(vals) - min(vals)) or 1.0

    def margin(key: str, thr: float) -> float:
        vals = [f[key] for f in in_feats] + [f[key] for f in ood_feats]
        return min(abs(thr - v) for v in vals) / spans[key]

    candidates = []
    frontier: Dict[Tuple[float, float], Dict] = {}
    for v in grid["vocab_cov"]:
        for t in grid["idf_cov"]:
            for n in grid["norm_score"]:
                fa = sum(1 for f in ood_feats if _trips(f, v, t, n)) / len(ood_feats)
                fr = sum(1 for f in in_feats if _trips(f, v, t, n)) / len(in_feats)
                bal = 0.5 * (fa + (1.0 - fr))
                robustness = min(margin("vocab_cov", v), margin("idf_cov", t),
                                 margin("norm_score", n))
                cand = {"vocab_cov": v, "idf_cov": t, "norm_score": n,
                        "refusal_accuracy": round(fa, 4),
                        "false_refusal_rate": round(fr, 4),
                        "balanced_accuracy": round(bal, 4),
                        "robustness_margin": round(robustness, 4)}
                candidates.append(cand)
                frontier[(round(fr, 3), round(fa, 3))] = cand

    best_bal = max(c["balanced_accuracy"] for c in candidates)
    tied = [c for c in candidates if c["balanced_accuracy"] >= best_bal - 1e-9]
    best = max(tied, key=lambda c: (c["robustness_margin"],
                                    -c["false_refusal_rate"],
                                    c["vocab_cov"], c["idf_cov"], c["norm_score"]))
    best = dict(best)
    best["tied_optima"] = len(tied)

    pareto = sorted(frontier.values(),
                    key=lambda c: (c["false_refusal_rate"], -c["refusal_accuracy"]))
    return {
        "best": best,
        "pareto": pareto[:12],
        "in_domain_features": _summarize(in_feats),
        "out_of_domain_features": _summarize(ood_feats),
    }


def evaluate_abstention(index: BM25Index, rank_fn, labeled, ood, thresholds: Dict) -> Dict:
    """用给定阈值评估拒答表现（含误拒率）。"""
    in_feats = [abstain_features(q, postprocess(rank_fn(q), index), index.idf, index.postings)
                for q, _, _ in labeled]
    ood_feats = [abstain_features(q, postprocess(rank_fn(q), index), index.idf, index.postings)
                 for q in ood]
    v = thresholds["vocab_cov"]
    t = thresholds["idf_cov"]
    n = thresholds["norm_score"]
    fa = sum(1 for f in ood_feats if _trips(f, v, t, n))
    fr = sum(1 for f in in_feats if _trips(f, v, t, n))
    return {
        "queries_out_of_domain": len(ood),
        "refused": fa,
        "refusal_accuracy": round(fa / len(ood), 4),
        "queries_in_domain": len(labeled),
        "falsely_refused": fr,
        "false_refusal_rate": round(fr / len(labeled), 4),
        "balanced_accuracy": round(0.5 * (fa / len(ood) + 1 - fr / len(labeled)), 4),
    }


def evaluate_refusal_legacy(index: BM25Index, ood: List[str]) -> Dict:
    """改造前的行为：仅当召回片段与问题完全没有共享词项时才拒答。"""
    refused, top_scores = 0, []
    for q in ood:
        hits = postprocess(make_rankers(index)["full"](q), index)
        if hits:
            top_scores.append(hits[0]["score"])
        ans = extractive_answer(q, hits, index.idf)
        if ans.strip() == REFUSAL:
            refused += 1
    return {
        "queries": len(ood),
        "refused": refused,
        "refusal_accuracy": round(refused / len(ood), 4),
        "mean_top1_score": round(statistics.mean(top_scores), 3) if top_scores else 0.0,
    }


def measure_latency(rank_fn, queries: List[str], repeats: int = 5) -> Dict:
    """热查询延迟：先跑一轮预热，再统计分位数。"""
    for q in queries:
        rank_fn(q)
    samples: List[float] = []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            rank_fn(q)
            samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()

    def pct(p: float) -> float:
        if not samples:
            return 0.0
        idx = min(len(samples) - 1, int(round(p / 100.0 * (len(samples) - 1))))
        return round(samples[idx], 3)

    return {
        "samples": len(samples),
        "mean_ms": round(statistics.mean(samples), 3),
        "p50_ms": pct(50), "p95_ms": pct(95), "p99_ms": pct(99),
        "max_ms": round(samples[-1], 3),
    }


# --------------------------------------------------------------------- 主流程
def check_labels(index: BM25Index) -> List[str]:
    """校验每条标注的 must 特征串确实存在于期望文档的某个片段中。"""
    problems = []
    by_doc: Dict[str, List[str]] = {}
    for ch in index.chunks:
        by_doc.setdefault(ch.get("doc", ""), []).append(passage_text(ch))
    for query, exp_doc, must in LABELED:
        texts = by_doc.get(exp_doc)
        if texts is None:
            problems.append(f"[文档不存在] {exp_doc}（查询：{query}）")
            continue
        if not any(must in t for t in texts):
            problems.append(f"[特征串未命中] {must!r} 不在 {exp_doc}（查询：{query}）")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="检索质量与性能基准测试")
    ap.add_argument("--docs", default="docs", help="语料目录")
    ap.add_argument("--repeats", type=int, default=5, help="延迟测试重复轮数")
    ap.add_argument("--check-labels", action="store_true", help="仅校验标注集")
    ap.add_argument("--out", default=REPORT_PATH, help="报告输出路径")
    args = ap.parse_args(argv)

    # --- 语料 ---
    chunks, files_meta, parse_info = build_corpus(args.docs)
    t0 = time.perf_counter()
    index = BM25Index().build(chunks, files_meta)
    build_s = time.perf_counter() - t0

    print(f"语料：{parse_info['docs']} 篇 / {len(chunks)} 片段 / {len(index.df)} 词项")
    print(f"构建：解析分块 {parse_info['parse_s']*1000:.0f} ms + 索引 {build_s*1000:.0f} ms")

    problems = check_labels(index)
    if problems:
        print("\n标注集校验未通过：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"标注集校验通过：{len(LABELED)} 条领域内 + {len(OUT_OF_DOMAIN)} 条领域外\n")
    if args.check_labels:
        return 0

    # --- 消融 ---
    rankers = make_rankers(index)
    metrics: Dict[str, Dict] = {}
    for name in ("full", "bm25_plain", "tfidf", "keyword"):
        m = evaluate(rankers[name], index, LABELED)
        metrics[name] = m
        print(f"[{name:11s}] Doc Recall@1={m['doc_recall']['@1']:.3f} "
              f"@5={m['doc_recall']['@5']:.3f} | "
              f"Passage Hit@1={m['passage_hit']['@1']:.3f} "
              f"@5={m['passage_hit']['@5']:.3f} | MRR@10={m['mrr@10']:.3f}")

    # --- 分层指标 ---
    # 143 条里混着四类难度，整体数字会被难例稀释。分层统计回答的是
    # "方法在哪一类问法上失效"，比单一总分更有诊断价值。
    tier_metrics: Dict[str, Dict] = {}
    print("\n分层表现（本方案）：")
    for tname, tset in TIERS:
        tm = evaluate(rankers["full"], index, tset)
        # 误拒率按**生产阈值**算，而不是本层标定值：用户实际感受到的是生产行为
        ta = evaluate_abstention(index, rankers["full"], tset, OUT_OF_DOMAIN,
                                 DEFAULT_CONFIG["abstain"])
        tier_metrics[tname] = {
            "queries": len(tset),
            "doc_recall@1": tm["doc_recall"]["@1"],
            "passage_hit@1": tm["passage_hit"]["@1"],
            "passage_hit@5": tm["passage_hit"]["@5"],
            "mrr@10": tm["mrr@10"],
            "false_refusal_rate": ta["false_refusal_rate"],
            # 每个对照方法在该层的表现：让"加权收益集中在哪一层"这类结论
            # 可以直接从报告复算，而不是靠人工记忆引用某个数字
            "by_method": {k: evaluate(rankers[k], index, tset)["passage_hit"]["@1"]
                          for k in ("full", "bm25_plain", "tfidf", "keyword")},
        }
        print(f"  {tname:11s} n={len(tset):3d}  Doc Recall@1={tm['doc_recall']['@1']:.3f}  "
              f"Passage Hit@1={tm['passage_hit']['@1']:.3f}  "
              f"@5={tm['passage_hit']['@5']:.3f}  "
              f"误拒={ta['false_refusal_rate']:.1%}")

    # --- 拒答阈值标定 ---
    legacy = evaluate_refusal_legacy(index, OUT_OF_DOMAIN)
    calib = calibrate_abstention(index, rankers["full"], LABELED, OUT_OF_DOMAIN)
    refusal = evaluate_abstention(index, rankers["full"], LABELED, OUT_OF_DOMAIN, calib["best"])
    # 生产配置实际使用的阈值：与网格最优对照，决定要不要动线上参数。
    # 差距在噪声范围内就不改——改阈值会改变线上行为，收益必须大到值得冒险。
    prod = evaluate_abstention(index, rankers["full"], LABELED, OUT_OF_DOMAIN,
                               DEFAULT_CONFIG["abstain"])

    print("\n拒答阈值标定（领域内 vs 领域外特征可分性）：")
    for key in ("vocab_cov", "idf_cov", "norm_score"):
        a = calib["in_domain_features"][key]
        b = calib["out_of_domain_features"][key]
        print(f"  {key:14s} 领域内 中位 {a['p50']:.3f} / p10 {a['p10']:.3f} ｜ "
              f"领域外 中位 {b['p50']:.3f} / p90 {b['p90']:.3f}")
    print(f"  选定阈值：vocab_cov≥{calib['best']['vocab_cov']} "
          f"idf_cov≥{calib['best']['idf_cov']} "
          f"norm_score≥{calib['best']['norm_score']}")
    print(f"  （{calib['best']['tied_optima']} 组阈值并列最优，"
          f"已按 max-min 间隔准则取最稳健者，间隔 {calib['best']['robustness_margin']:.3f}）")
    print(f"  改造前拒答率 {legacy['refusal_accuracy']:.1%} → "
          f"改造后 {refusal['refusal_accuracy']:.1%}"
          f"（误拒 {refusal['falsely_refused']}/{refusal['queries_in_domain']} = "
          f"{refusal['false_refusal_rate']:.1%}，"
          f"平衡准确率 {refusal['balanced_accuracy']:.1%}）")
    prod_th = DEFAULT_CONFIG["abstain"]
    print(f"  生产阈值 vocab_cov≥{prod_th['vocab_cov']} idf_cov≥{prod_th['idf_cov']} "
          f"norm_score≥{prod_th['norm_score']}："
          f"拒答 {prod['refusal_accuracy']:.1%}，"
          f"误拒 {prod['falsely_refused']}/{prod['queries_in_domain']} = "
          f"{prod['false_refusal_rate']:.1%}，平衡 {prod['balanced_accuracy']:.1%}")

    # --- 延迟 ---
    queries = [q for q, _, _ in LABELED] + OUT_OF_DOMAIN
    latency = measure_latency(rankers["full"], queries, args.repeats)
    print(f"延迟：p50={latency['p50_ms']} ms  p95={latency['p95_ms']} ms  "
          f"p99={latency['p99_ms']} ms（{latency['samples']} 次采样）")

    # --- 索引体积与加载内存 ---
    os.makedirs(".rag_index", exist_ok=True)
    path = index.save(".rag_index")
    index_bytes = os.path.getsize(path)
    total_chars = sum(m["chars"] for m in files_meta.values())

    tracemalloc.start()
    _loaded = BM25Index.load(".rag_index")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    size_info = {
        "index_bytes": index_bytes,
        "index_kb": round(index_bytes / 1024, 1),
        "corpus_chars": total_chars,
        "ratio": round(index_bytes / total_chars, 3) if total_chars else 0.0,
        "load_heap_peak_kb": round(peak / 1024, 1),
    }
    print(f"索引：{size_info['index_kb']} KB（正文 {total_chars} 字符，"
          f"比率 {size_info['ratio']:.2f}）｜加载堆内存峰值 {size_info['load_heap_peak_kb']} KB")

    # --- 报告 ---
    report = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus": {"docs": parse_info["docs"], "chunks": len(chunks),
                       "terms": len(index.df), "chars": total_chars,
                       "avgdl": round(index.avgdl, 1)},
            "label_set": {"in_domain": len(LABELED), "out_of_domain": len(OUT_OF_DOMAIN),
                          "by_tier": {n: len(s) for n, s in TIERS}},
            "protocol": {"candidate_k": CANDIDATE_K, "max_per_doc": MAX_PER_DOC,
                         "top_k": max(KS), "mrr_k": MRR_K},
        },
        "metrics": metrics,
        "tiers": tier_metrics,
        "refusal": refusal,
        "refusal_production": prod,
        "refusal_calibration": calib,
        "refusal_baseline_legacy": legacy,
        "latency": latency,
        "index": size_info,
        "timing": {"parse_s": round(parse_info["parse_s"], 4),
                   "index_build_s": round(build_s, 4),
                   "total_build_s": round(parse_info["parse_s"] + build_s, 4)},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写出：{args.out}")

    # --- 失败样例（最有价值的分析材料） ---
    misses = metrics["full"]["_misses"]
    if misses:
        print(f"\n本方案未在 Top-5 命中正确片段的样例（{len(misses)} 条）：")
        for m in misses[:8]:
            print(f"  · {m['query']}")
            print(f"    期望 {m['expected_doc']}，文档级排名 {m['doc_rank']}，"
                  f"Top-1 实为 {m['top1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
