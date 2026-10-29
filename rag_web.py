#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地 Web 问答界面入口。

    python rag_web.py                  # 默认 http://127.0.0.1:8765
    python rag_web.py --port 9000 --open
    python rag_web.py --selftest       # 起真实服务跑一遍自检，无需浏览器

需要先建立索引：``python rag.py ingest``
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragcore.cli import load_config          # noqa: E402
from ragcore.web import selftest, serve      # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        prog="rag_web.py",
        description="本地 Web 问答界面（零第三方依赖，仅绑定本机）")
    p.add_argument("--host", default="127.0.0.1",
                   help="监听地址（默认 127.0.0.1；改成 0.0.0.0 会让同网段可访问）")
    p.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    p.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    p.add_argument("--config", default=None,
                   help="配置文件路径（默认 ./config.json）")
    p.add_argument("--selftest", action="store_true",
                   help="起真实服务并逐项自检，退出码 0 表示全部通过")
    args = p.parse_args()

    cfg = load_config(args.config or ("config.json"
                                      if os.path.isfile("config.json") else None))

    if args.selftest:
        return selftest(cfg)
    return serve(cfg, host=args.host, port=args.port,
                 open_browser=args.open)


if __name__ == "__main__":
    sys.exit(main())
