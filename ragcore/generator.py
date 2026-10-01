# -*- coding: utf-8 -*-
"""答案生成层。

两种模式
--------
1. **LLM 模式**：调用任意 OpenAI 兼容的 ``/chat/completions`` 接口
   （DeepSeek、Moonshot、通义、本地 vLLM/Ollama 均可），仅用标准库
   ``urllib`` 实现，无需 openai SDK。
2. **离线抽取模式**：不依赖任何外部服务，从召回片段中挑选与问题词项
   重叠度最高的句子并按引用编号拼装。回答质量低于 LLM，但保证"零配置
   也能跑通完整问答链路"，同时便于在没有网络的环境里验证检索质量。

两种模式都强制输出 ``[n]`` 形式的引用标记，n 对应召回片段的序号。
"""
from __future__ import annotations

import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from .tokenizer import tokenize

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。请严格遵守以下规则：\n"
    "1. 只依据【参考资料】作答，不得使用资料之外的知识，不得推测或编造。\n"
    "2. 每个事实性结论后面必须标注来源编号，格式为 [1]、[2]，可标注多个。\n"
    "3. 若参考资料不足以回答问题，直接回答：根据现有资料无法回答该问题。\n"
    "4. 回答使用简体中文，结构清晰，先给结论再给依据，避免复述无关内容。"
)

_SENT_RE = re.compile(r"[^。！？；\n.!?;]+[。！？；.!?;]?")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WIKILINK_RE = re.compile(r"\[\[([^\]]*)\]\]")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.findall(text)]


# --------------------------------------------------------------------- 上下文
def format_context(hits: List[Dict], max_chars: int = 6000) -> str:
    """把召回片段拼装为带编号的参考资料，超长时按序截断。"""
    parts: List[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        head = f"[{i}] 来源：{h.get('doc', '')}"
        if h.get("heading"):
            head += f" › {h['heading']}"
        if h.get("start_line"):
            head += f"（约第 {h['start_line']} 行）"
        body = h.get("text", "").strip()
        block = f"{head}\n{body}"
        if used + len(block) > max_chars and parts:
            remain = max_chars - used
            if remain > 200:
                parts.append(block[:remain] + " …（已截断）")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def build_messages(question: str, hits: List[Dict], max_context_chars: int = 6000) -> List[Dict]:
    context = format_context(hits, max_context_chars)
    user = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{question}\n\n"
        "请依据上述参考资料作答，并标注引用编号。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------- LLM
class LLMError(RuntimeError):
    pass


class LLMClient:
    """OpenAI 兼容 Chat Completions 客户端（标准库实现）。"""

    # 本机/内网地址：Ollama、vLLM、LM Studio 等本地推理服务通常不校验密钥
    @staticmethod
    def _is_local(base: str) -> bool:
        import ipaddress
        import urllib.parse
        try:
            host = urllib.parse.urlparse(base).hostname or ""
        except ValueError:
            return False
        if host == "localhost":
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        return ip.is_loopback or ip.is_private

    def __init__(self,
                 api_base: str = "",
                 api_key: str = "",
                 model: str = "deepseek-chat",
                 temperature: float = 0.2,
                 timeout: int = 60):
        self.api_base = (api_base or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.temperature = float(temperature)
        self.timeout = int(timeout)

    @property
    def available(self) -> bool:
        """是否具备调用条件。

        远程服务必须提供 api_key；本机/内网地址允许留空，以适配本地推理
        服务。这样默认配置（已填 api_base 但未填密钥）会安静地走离线抽取
        模式，而不是每次都发起一次注定失败的请求。
        """
        if not (self.api_base and self.model):
            return False
        return bool(self.api_key) or self._is_local(self.api_base)

    def _endpoint(self) -> str:
        base = self.api_base
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def chat(self, messages: List[Dict]) -> str:
        if not self.available:
            raise LLMError("未配置 api_base 或 model")
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(self._endpoint(), data=payload,
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"HTTP {e.code}：{detail}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"网络错误：{e.reason}") from e
        except TimeoutError as e:
            raise LLMError(f"请求超时（{self.timeout}s）") from e

        try:
            data = json.loads(raw)
            return data["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError) as e:
            raise LLMError(f"响应解析失败：{raw[:300]}") from e


# --------------------------------------------------------------- 离线抽取式
def _clean_fragment(text: str) -> Optional[str]:
    """把 Markdown 片段清洗为可直接朗读的句子；返回 None 表示该片段无信息量。"""
    s = text.strip()
    if not s:
        return None

    # 表格行：转成逗号分隔的可读文本；分隔行（|---|）与表头行直接丢弃
    if s.count("|") >= 2:
        cells = [c.strip() for c in s.strip("|").split("|")]
        cells = [c for c in cells if c]
        if not cells:
            return None
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            return None
        # 所有单元格都很短且数量较多 → 判定为表头行，无回答价值
        if len(cells) >= 4 and all(len(c) <= 5 for c in cells):
            return None
        s = "，".join(cells)

    s = re.sub(r"^[\-\*\+>]\s+", "", s)          # 列表符号 / 引用符
    s = re.sub(r"^#{1,6}\s+", "", s)             # 标题符号
    s = _WIKILINK_RE.sub(r"\1", s)               # [[双向链接]]
    s = _MD_LINK_RE.sub(r"\1", s)                # [文本](url)
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def extractive_answer(question: str,
                      hits: List[Dict],
                      idf: Optional[Dict[str, float]] = None,
                      max_sentences: int = 4,
                      max_chars: int = 700) -> str:
    """无 LLM 时的兜底回答：按词项加权重叠度抽取原句并附带引用编号。

    该模式不做任何推理，只是"从最相关的片段里挑出与问题重叠度最高的句子"。
    它能保证答案有据可查（不会幻觉），但无法综合多句、无法换算或比较数值，
    因此只作为未配置大模型时的降级方案。
    """
    if not hits:
        return "根据现有资料无法回答该问题。"

    idf = idf or {}
    q_terms = set(tokenize(question))
    if not q_terms:
        return "根据现有资料无法回答该问题。"

    best = hits[0]["score"] or 1.0
    scored = []
    for rank, h in enumerate(hits, 1):
        rel = h.get("score", 0.0) / best
        for raw in _sentences(h.get("text", "")):
            sent = _clean_fragment(raw)
            if not sent or len(sent) < 8:
                continue
            toks = tokenize(sent)
            if not toks:
                continue
            matched = sum(idf.get(t, 1.0) for t in set(toks) & q_terms)
            if matched <= 0:
                continue
            # 长度归一化：抑制长句靠词数堆叠得分
            score = matched / (len(toks) ** 0.5) * (0.6 + 0.4 * rel)
            scored.append((score, rank, sent))

    if not scored:
        return "根据现有资料无法回答该问题。"

    scored.sort(key=lambda x: -x[0])
    picked: List[str] = []
    seen: set = set()
    total = 0
    for _, rank, sent in scored:
        key = sent[:40]
        if key in seen:
            continue
        seen.add(key)
        if total + len(sent) > max_chars and picked:
            break
        picked.append(f"{sent} [{rank}]")
        total += len(sent)
        if len(picked) >= max_sentences:
            break

    if not picked:
        return "根据现有资料无法回答该问题。"
    return "\n".join(picked)
