"""LLM clients (บทที่ 6.3)

ค่าเริ่มต้นของงานวิจัยคือโมเดล open-weight ที่ติดตั้งภายในองค์กร ผ่าน OpenAI-compatible endpoint (vLLM, Ollama ฯลฯ)
Claude API ใช้ได้เฉพาะข้อมูลที่ปิดบังแล้วและได้รับอนุมัติตามนโยบายข้อมูลขององค์กร
ตรวจชื่อโมเดลและพารามิเตอร์ล่าสุดที่ https://docs.claude.com ก่อนใช้งานจริง และล็อกเวอร์ชันโมเดลตลอดการทดลอง
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 1200,
                 json_schema: dict | None = None) -> str: ...


class AnthropicClient:
    def __init__(self, model: str | None = None):
        import anthropic                                   # pip install anthropic

        self.client = anthropic.Anthropic()                # อ่าน ANTHROPIC_API_KEY จาก environment
        self.model = model or os.getenv("BLP_ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, prompt: str, max_tokens: int = 1200, json_schema: dict | None = None) -> str:
        kwargs = {"model": self.model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": prompt}]}
        if json_schema is not None:                        # structured outputs
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        resp = self.client.messages.create(**kwargs)
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")


class OpenAICompatibleClient:
    """โมเดลภายใน เช่น Thai LLM ที่ serve ด้วย vLLM: BLP_LLM_BASE_URL=http://gpu-server:8000/v1"""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 timeout: int = 120):
        self.base_url = (base_url or os.getenv("BLP_LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.getenv("BLP_LLM_MODEL", "local-thai-llm")
        self.api_key = api_key or os.getenv("BLP_LLM_API_KEY", "EMPTY")
        self.timeout = timeout
        self.name = f"local:{self.model}"

    def complete(self, system: str, prompt: str, max_tokens: int = 1200, json_schema: dict | None = None) -> str:
        import requests

        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "blp_output", "schema": json_schema}}
        url, headers = f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.api_key}"}
        r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
        if r.status_code == 400 and "response_format" in body:   # เซิร์ฟเวอร์บางตัวไม่รองรับ json_schema
            body.pop("response_format")
            r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def make_llm(kind: str | None, model: str | None = None, base_url: str | None = None) -> LLMClient | None:
    if kind in (None, "", "none"):
        return None
    if kind == "anthropic":
        return AnthropicClient(model=model)
    if kind in ("local", "openai"):
        return OpenAICompatibleClient(base_url=base_url, model=model)
    raise ValueError(f"unknown LLM kind: {kind}")


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in LLM output")
    return json.loads(text[start:end + 1])
