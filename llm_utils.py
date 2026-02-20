"""
Shared LLM helpers with thinking enabled (streaming required by Aval/Dashscope).
Returns (content, thinking); thinking may be empty if the API does not expose it.
"""
import re
from typing import List, Dict, Any, Optional, Tuple

# When True, use stream=True and enable_thinking=True; collect content and optional thinking.
ENABLE_THINKING = True


def _split_think_tags(full: str) -> Tuple[str, str]:
    """If content contains think-tags (think.../think), return (thinking, content). Else return ('', full)."""
    if not full or "</think>" not in full:
        return ("", full)
    m = re.search(r"<think>(.*?)</think>", full, re.DOTALL)
    if not m:
        return ("", full)
    thinking = m.group(1).strip()
    content = (full[: m.start()] + full[m.end() :]).strip()
    return (thinking, content)


def chat_completion_with_thinking(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Call chat completions. When ENABLE_THINKING is True, uses stream=True and
    enable_thinking=True, then collects content and reasoning/thinking from the stream.
    Returns (content, thinking). thinking may come from delta.reasoning_content,
    delta.thinking, or from think-tags in the main content.
    """
    body = dict(extra_body or {})
    if ENABLE_THINKING:
        body["enable_thinking"] = True

    if not ENABLE_THINKING:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            extra_body=body if body else None,
        )
        content = (resp.choices[0].message.content or "").strip()
        return (content, "")

    # Streaming with thinking: required by some APIs when enable_thinking=True
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        extra_body=body,
    )
    content_parts: List[str] = []
    thinking_parts: List[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        if getattr(delta, "reasoning_content", None):
            thinking_parts.append(delta.reasoning_content)
        if getattr(delta, "thinking", None):
            thinking_parts.append(delta.thinking)
    full_content = "".join(content_parts).strip()
    explicit_thinking = "".join(thinking_parts).strip()
    if explicit_thinking:
        return (full_content, explicit_thinking)
    thinking_from_tags, content_only = _split_think_tags(full_content)
    if thinking_from_tags:
        return (content_only, thinking_from_tags)
    return (full_content, "")
