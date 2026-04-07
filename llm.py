"""
LLMOrchestrator — unified interface for Claude, OpenAI, OpenRouter, and local models.

All backends expose the same call signature:
    orchestrator.chat(messages, system_prompt) -> str
    orchestrator.chat_with_tools(messages, tools, system_prompt) -> (str, list[tool_call])
"""

import json
import logging
from typing import Any
from dataclasses import dataclass, field

import config

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Normalized tool call returned by any backend."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


class LLMOrchestrator:
    """
    Unified LLM interface. Instantiate once and call .chat() or .chat_with_tools().

    Usage:
        llm = LLMOrchestrator()                    # uses config.LLM_BACKEND
        llm = LLMOrchestrator(backend="openai")    # override
    """

    def __init__(self, backend: str | None = None):
        self.backend = (backend or config.LLM_BACKEND).lower()
        self.model   = config.MODELS[self.backend]
        self._client = self._build_client()
        logger.info(f"LLMOrchestrator initialized: backend={self.backend}, model={self.model}")

    # ──────────────────────────────────────────────────────────────────────────
    # Client construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_client(self):
        if self.backend == "claude":
            import anthropic
            return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        elif self.backend in ("openai", "openrouter", "local"):
            from openai import OpenAI
            if self.backend == "openai":
                return OpenAI(api_key=config.OPENAI_API_KEY)
            elif self.backend == "openrouter":
                return OpenAI(
                    api_key=config.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                )
            else:  # local
                return OpenAI(
                    api_key=config.LOCAL_API_KEY,
                    base_url=config.LOCAL_BASE_URL,
                )
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}. "
                             "Choose from: claude, openai, openrouter, local")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        """Simple chat — returns text string."""
        response = self._dispatch(messages, system_prompt, tools=None)
        return response.content

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str = "",
    ) -> LLMResponse:
        """
        Chat with tool definitions. Returns LLMResponse with .content and .tool_calls.
        Tool definitions follow Anthropic format internally; we convert for OpenAI backends.
        """
        return self._dispatch(messages, system_prompt, tools=tools)

    # ──────────────────────────────────────────────────────────────────────────
    # Backend dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _dispatch(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None,
    ) -> LLMResponse:
        if self.backend == "claude":
            return self._call_claude(messages, system_prompt, tools)
        else:
            return self._call_openai_compat(messages, system_prompt, tools)

    # ── Claude ────────────────────────────────────────────────────────────────

    def _call_claude(self, messages, system_prompt, tools) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=4096,
            messages=messages,
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools  # already in Anthropic format

        response = self._client.messages.create(**kwargs)

        text_content = ""
        tool_calls   = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
        )

    # ── OpenAI-compatible (OpenAI / OpenRouter / Local) ───────────────────────

    def _call_openai_compat(self, messages, system_prompt, tools) -> LLMResponse:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=full_messages,
            max_tokens=4096,
        )
        if tools:
            kwargs["tools"] = self._anthropic_tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        text_content = msg.content or ""
        tool_calls   = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason,
        )

    # ── Format conversion ─────────────────────────────────────────────────────

    @staticmethod
    def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
        """Convert Anthropic tool schema to OpenAI function-calling schema."""
        converted = []
        for t in tools:
            converted.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })
        return converted

    def tool_result_message(self, tool_call: ToolCall, result: str) -> dict:
        """
        Build the correct tool-result message format for the active backend.
        This goes back into the messages list for the next turn.
        """
        if self.backend == "claude":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                }],
            }
        else:
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }

    def assistant_tool_use_message(self, response: LLMResponse) -> dict:
        """
        Build the assistant message that contains tool_use blocks.
        Required by Claude to maintain valid conversation history.
        """
        if self.backend == "claude":
            content = []
            if response.content:
                content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            return {"role": "assistant", "content": content}
        else:
            # OpenAI format
            return {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
