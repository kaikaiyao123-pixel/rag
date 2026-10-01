# -*- coding: utf-8 -*-
"""ragcore —— 零第三方依赖的小型 RAG 引擎。

模块划分
--------
tokenizer : 中英文混合分词（中文 bigram 方案）
loader    : 文档发现与多编码读取
chunker   : Markdown 感知的分块
index     : Okapi BM25 倒排索引 + JSON 持久化
retriever : 检索编排（候选集 → 去重 → 单文档限流）
generator : LLM 生成（OpenAI 兼容）与离线抽取式回答
cli       : 命令行入口
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
