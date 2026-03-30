"""
Provider-agnostic HTTP adapter for Market Context generation.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import aiohttp

from backend.config import Config

log = logging.getLogger(__name__)


class MarketContextLLMError(RuntimeError):
    pass


def market_context_provider_ready() -> bool:
    return Config.market_context_llm_ready()


class MarketContextLLMAdapter:
    def __init__(self) -> None:
        self.provider = Config.MARKET_CONTEXT_LLM_PROVIDER
        self.model = Config.MARKET_CONTEXT_LLM_MODEL
        self.base_url = Config.MARKET_CONTEXT_LLM_BASE_URL.strip()
        self.api_key = (
            Config.MARKET_CONTEXT_LLM_API_KEY
            or {
                "anthropic": Config.MARKET_CONTEXT_LLM_API_KEY or os.environ.get("ANTHROPIC_API_KEY", ""),
                "openai": Config.MARKET_CONTEXT_LLM_API_KEY or os.environ.get("OPENAI_API_KEY", ""),
            }.get(self.provider, Config.MARKET_CONTEXT_LLM_API_KEY)
        )

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        if not market_context_provider_ready():
            raise MarketContextLLMError("Market Context LLM provider is not configured")

        if self.provider == "anthropic":
            text = await self._call_anthropic(system_prompt=system_prompt, user_prompt=user_prompt)
        elif self.provider == "openai":
            text = await self._call_openai(system_prompt=system_prompt, user_prompt=user_prompt)
        else:
            raise MarketContextLLMError(f"Unsupported Market Context provider: {self.provider or 'unset'}")

        try:
            return _extract_json_payload(text)
        except ValueError as exc:
            raise MarketContextLLMError(str(exc)) from exc

    async def _call_anthropic(self, *, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url or "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": 1400,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        body = await _post_json(url, payload, headers=headers)
        content = body.get("content") or []
        parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        text = "\n".join(part for part in parts if part).strip()
        if not text:
            raise MarketContextLLMError("Anthropic response did not contain text content")
        return text

    async def _call_openai(self, *, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url or "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        body = await _post_json(url, payload, headers=headers)
        choices = body.get("choices") or []
        if not choices:
            raise MarketContextLLMError("OpenAI response did not contain choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            text = "\n".join(parts).strip()
            if text:
                return text
        raise MarketContextLLMError("OpenAI response did not contain usable text content")


async def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=Config.MARKET_CONTEXT_LLM_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.post(url, json=payload) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise MarketContextLLMError(
                    f"LLM provider HTTP {resp.status}: {text[:500]}"
                )
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                log.warning("market_context: provider returned non-JSON body: %s", text[:200])
                raise MarketContextLLMError("LLM provider returned invalid JSON response") from exc


def _extract_json_payload(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM returned an empty response")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object")
        return parsed
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON object")
        parsed = json.loads(raw[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object")
        return parsed
