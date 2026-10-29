#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单元测试：配置合并与环境变量覆盖。

覆盖需求：REQ-CLI-01..02

配置层是"隐性输入"——它不显式出现在任何函数调用里，却决定检索参数、拒答阈值
和模型地址。这里的用例刻意断言**类型**（int / float），因为 `"30"` 与 `30` 在
JSON 里长得像，进到 BM25 的 `k1` 或 `timeout` 里就会炸。
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Iterator

from fixtures import tempdir, write_file

from ragcore.cli import DEFAULT_CONFIG, deep_merge, load_config

from harness import assert_eq, assert_true, case

_ENV_KEYS = ("RAG_API_BASE", "RAG_API_KEY", "RAG_MODEL",
             "RAG_TIMEOUT", "RAG_TEMPERATURE")


@contextlib.contextmanager
def clean_env(**overrides: str) -> Iterator[None]:
    """清空 RAG_* 环境变量，按需注入，退出时恢复原状。"""
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# 深合并（REQ-CLI-01）
@case("TC-CLI-001", "REQ-CLI-01", "deep_merge 递归合并嵌套字典，标量覆盖",
      technique="等价类划分", level="单元",
      expect="嵌套键保留未覆盖项，同名标量被覆盖，新增键被加入")
def tc_cli_001() -> None:
    base = {"a": {"b": 1, "c": 2}, "x": 1}
    got = deep_merge(base, {"a": {"c": 3, "d": 4}, "x": 9, "y": 5})
    assert_eq(got, {"a": {"b": 1, "c": 3, "d": 4}, "x": 9, "y": 5})


# 入参不可变（REQ-CLI-01 边界）
@case("TC-CLI-002", "REQ-CLI-01", "deep_merge 不修改入参",
      technique="边界值分析", level="单元",
      expect="合并后 base 与 override 保持原值")
def tc_cli_002() -> None:
    base = {"a": {"b": 1}}
    override = {"a": {"b": 2}}
    deep_merge(base, override)
    assert_eq(base, {"a": {"b": 1}}, "base 被就地修改")
    assert_eq(override, {"a": {"b": 2}}, "override 被就地修改")


# 配置文件合并到默认值（REQ-CLI-01）
@case("TC-CLI-003", "REQ-CLI-01", "配置文件只覆盖指定键，其余取默认值",
      technique="等价类划分", level="单元",
      expect="仅 chunk.max_chars 被改，retrieval 与 abstain 保持默认")
def tc_cli_003() -> None:
    with tempdir("TC-CLI-003") as d:
        p = write_file(os.path.join(d, "config.json"),
                       json.dumps({"chunk": {"max_chars": 800}}))
        cfg = load_config(p)
        assert_eq(cfg["chunk"]["max_chars"], 800)
        assert_eq(cfg["chunk"]["min_chars"], DEFAULT_CONFIG["chunk"]["min_chars"])
        assert_eq(cfg["retrieval"], DEFAULT_CONFIG["retrieval"])
        assert_eq(cfg["abstain"], DEFAULT_CONFIG["abstain"])


# 环境变量覆盖且类型正确（REQ-CLI-02）
@case("TC-CLI-004", "REQ-CLI-02", "环境变量覆盖 llm 配置并完成类型转换",
      technique="等价类划分", level="单元",
      expect="RAG_TIMEOUT='30' → int 30；RAG_TEMPERATURE='0.7' → float 0.7")
def tc_cli_004() -> None:
    with clean_env(RAG_MODEL="qwen2.5:7b", RAG_TIMEOUT="30",
                   RAG_TEMPERATURE="0.7", RAG_API_BASE="http://127.0.0.1:8000/v1"):
        cfg = load_config(None)
        assert_eq(cfg["llm"]["model"], "qwen2.5:7b")
        assert_eq(cfg["llm"]["timeout"], 30)
        assert_true(isinstance(cfg["llm"]["timeout"], int), "timeout 必须是 int")
        assert_eq(cfg["llm"]["temperature"], 0.7)
        assert_true(isinstance(cfg["llm"]["temperature"], float),
                    "temperature 必须是 float")
        assert_eq(cfg["llm"]["api_base"], "http://127.0.0.1:8000/v1")


# 非法数值与空值被跳过（REQ-CLI-02 边界）
@case("TC-CLI-005", "REQ-CLI-02", "非法数值与空值的环境变量被忽略",
      technique="边界值分析", level="单元",
      expect="RAG_TIMEOUT='abc' 与空字符串都回退默认值，不抛异常")
def tc_cli_005() -> None:
    with clean_env(RAG_TIMEOUT="abc", RAG_TEMPERATURE="xyz", RAG_MODEL=""):
        cfg = load_config(None)
        assert_eq(cfg["llm"]["timeout"], DEFAULT_CONFIG["llm"]["timeout"])
        assert_eq(cfg["llm"]["temperature"], DEFAULT_CONFIG["llm"]["temperature"])
        assert_eq(cfg["llm"]["model"], DEFAULT_CONFIG["llm"]["model"])
