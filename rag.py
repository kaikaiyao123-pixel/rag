#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""小型 RAG 程序入口。

    python rag.py --help
    python rag.py ingest
    python rag.py ask "如何导入笔记？"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragcore.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
