# pyplanner/base.py
"""
Shared base class, data structures, and LLM backend abstraction for all PyPlanner methods.

Supported backends:
  - "ollama"    : local Ollama server (default)
  - "openai"    : OpenAI API (GPT-4o, GPT-4o-mini, o1, ...)
  - "anthropic" : Anthropic API (claude-opus-4-6, claude-sonnet-4-6, ...)
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_HOST     = "http://localhost:11434"
DEFAULT_MODEL    = "llama3.2"
DEFAULT_BACKEND  = "ollama"

OPENAI_API_URL    = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# ── Model presets per provider ─────────────────────────────────────────
PROVIDER_MODELS: dict[str, list[str]] = {
    "ollama": [
        "llama3.2", "llama3.1", "llama3",
        "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b",
        "mistral", "mistral-nemo",
        "gemma2:2b", "gemma2:9b",
        "phi3.5", "deepseek-r1:7b",
    ],
    "openai": [
        "gpt-4o", "gpt-4o-mini",
        "gpt-4-turbo", "gpt-4",
        "o1", "o1-mini", "o3-mini",
    ],
    "anthropic": [
        "claude-sonnet-4-6", "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
}

ROBOT_ACTIONS = [
    "Navigate", "Find", "Grab", "Place", "PutIn",
    "Open", "Close", "TurnOn", "TurnOff",
    "Wash", "Sit", "LieOn", "Serve", "Wait",
]

ACTIONS_STR = "\n".join(f"  {a}" for a in ROBOT_ACTIONS)

STEP_SCHEMA = """\
Each step must be a JSON object with these exact fields:
  "action"  : one action from the list above (exact spelling)
  "object"  : target object in snake_case, e.g. apple, coffee_machine, mug, stove
  "target"  : destination receptacle/surface (only for Place or PutIn, else use "")
  "reason"  : one short sentence explaining why this step is needed

Rules:
- Always Navigate or Find an object BEFORE interacting with it
- Use object names from the visible_objects list when possible
- If the task needs multiple objects, handle them one at a time
- Maximum 15 steps"""

JSON_EXAMPLE = """\
{"steps": [
  {"action": "Navigate", "object": "apple", "target": "", "reason": "Move to the apple"},
  {"action": "Grab",     "object": "apple", "target": "", "reason": "Pick up the apple"}
]}"""


# ══════════════════════════════════════════════════════════════════════
# PlanMetrics
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PlanMetrics:
    """Metrics collected for every generate_plan / replan call."""
    method:        str   = ""
    model:         str   = ""
    backend:       str   = ""
    latency_s:     float = 0.0
    llm_calls:     int   = 0
    input_tokens:  int   = 0
    output_tokens: int   = 0
    num_steps:     int   = 0
    parse_ok:      bool  = True
    notes:         str   = ""
    extra:         dict  = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_step(self) -> float:
        return self.total_tokens / self.num_steps if self.num_steps else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method":          self.method,
            "model":           self.model,
            "backend":         self.backend,
            "latency_s":       round(self.latency_s, 3),
            "llm_calls":       self.llm_calls,
            "input_tokens":    self.input_tokens,
            "output_tokens":   self.output_tokens,
            "total_tokens":    self.total_tokens,
            "num_steps":       self.num_steps,
            "tokens_per_step": round(self.tokens_per_step, 1),
            "parse_ok":        self.parse_ok,
            "notes":           self.notes,
        }


# ══════════════════════════════════════════════════════════════════════
# LLMBackend  — unified _chat() for Ollama / OpenAI / Anthropic
# ══════════════════════════════════════════════════════════════════════
class LLMBackend:
    """
    Thin wrapper that exposes a single chat() method regardless of provider.

    Args:
        provider:    "ollama" | "openai" | "anthropic"
        model:       model string for the chosen provider
        host:        Ollama host URL (ignored for openai/anthropic)
        api_key:     API key for openai/anthropic (falls back to env vars)
        temperature: default sampling temperature
    """

    def __init__(
        self,
        provider:    str   = DEFAULT_BACKEND,
        model:       str   = DEFAULT_MODEL,
        host:        str   = DEFAULT_HOST,
        api_key:     str   = "",
        temperature: float = 0.2,
    ):
        self.provider    = provider.lower()
        self.model       = model
        self.host        = host
        self.temperature = temperature

        # Resolve API key: arg > env var
        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        else:
            self.api_key = ""  # Ollama needs no key

        # Lazy-init Ollama client only when needed
        self._ollama_client = None

    def _get_ollama_client(self):
        if self._ollama_client is None:
            import ollama as _ollama
            self._ollama_client = _ollama.Client(host=self.host)
        return self._ollama_client

    def chat(
        self,
        messages:    list[dict],
        temperature: float | None = None,
    ) -> tuple[str, int, int]:
        """
        Send messages and return (content, input_tokens, output_tokens).
        temperature=None uses the instance default.
        """
        temp = temperature if temperature is not None else self.temperature

        if self.provider == "ollama":
            return self._chat_ollama(messages, temp)
        elif self.provider == "openai":
            return self._chat_openai(messages, temp)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages, temp)
        else:
            raise ValueError(f"Unknown provider '{self.provider}'. Use: ollama, openai, anthropic")

    # ── Ollama ──
    def _chat_ollama(self, messages, temperature):
        client  = self._get_ollama_client()
        resp    = client.chat(model=self.model, messages=messages, options={"temperature": temperature})
        content = resp["message"]["content"]
        in_tok  = resp.get("prompt_eval_count") or _approx_tokens(" ".join(m["content"] for m in messages))
        out_tok = resp.get("eval_count")         or _approx_tokens(content)
        return content, in_tok, out_tok

    # ── OpenAI ──
    def _chat_openai(self, messages, temperature):
        import requests as _req
        if not self.api_key:
            raise ValueError("OpenAI API key not set. Pass api_key= or set OPENAI_API_KEY env var.")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body    = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": 2048}
        resp    = _req.post(OPENAI_API_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage   = data.get("usage", {})
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    # ── Anthropic ──
    def _chat_anthropic(self, messages, temperature):
        import requests as _req
        if not self.api_key:
            raise ValueError("Anthropic API key not set. Pass api_key= or set ANTHROPIC_API_KEY env var.")
        # Extract system message if present
        system   = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)
        headers = {
            "x-api-key":         self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type":      "application/json",
        }
        body = {
            "model":       self.model,
            "max_tokens":  2048,
            "temperature": temperature,
            "messages":    filtered,
        }
        if system:
            body["system"] = system
        resp  = _req.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data  = resp.json()
        content = data["content"][0]["text"]
        usage   = data.get("usage", {})
        return content, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def parse_steps(raw: str) -> list[dict]:
    """Robustly parse LLM output into a list of step dicts."""
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get("steps", [])
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    steps = []
    for m in re.finditer(r"\{[^{}]+\}", raw):
        try:
            step = json.loads(m.group())
            if "action" in step:
                steps.append(step)
        except json.JSONDecodeError:
            pass
    return steps


# ══════════════════════════════════════════════════════════════════════
# BasePlanner
# ══════════════════════════════════════════════════════════════════════
class BasePlanner(ABC):
    """
    Abstract base for all PyPlanner methods.

    Accepts any supported backend (ollama / openai / anthropic) via LLMBackend.
    All subclasses use self._chat() — backend-agnostic.
    """

    name: str        = "base"
    description: str = ""

    def __init__(
        self,
        host:     str = DEFAULT_HOST,
        model:    str = DEFAULT_MODEL,
        provider: str = DEFAULT_BACKEND,
        api_key:  str = "",
        **kwargs,
    ):
        self.host     = host
        self.model    = model
        self.provider = provider
        self._backend = LLMBackend(provider=provider, model=model, host=host, api_key=api_key)

    def _chat(self, messages: list[dict], temperature: float = 0.2) -> tuple[str, int, int]:
        """Call the configured LLM backend. Returns (content, in_tokens, out_tokens)."""
        return self._backend.chat(messages, temperature=temperature)

    def _make_metrics(self, **kwargs) -> PlanMetrics:
        """Helper to create a PlanMetrics pre-filled with method/model/backend."""
        return PlanMetrics(
            method  = self.name,
            model   = self.model,
            backend = self.provider,
            **kwargs,
        )

    @abstractmethod
    def generate_plan(
        self,
        task: str,
        obs: str,
        visible_objects: list[str],
    ) -> tuple[list[dict], PlanMetrics]:
        ...

    @abstractmethod
    def replan(
        self,
        task: str,
        completed: list[dict],
        failed_step: dict,
        failure_reason: str,
        obs: str,
        visible_objects: list[str],
    ) -> tuple[list[dict], PlanMetrics]:
        ...

    # ── Shared prompt helpers ──
    def _context_str(self, task: str, obs: str, visible_objects: list[str]) -> str:
        obj_str = ", ".join(visible_objects[:30]) if visible_objects else "none visible yet"
        return (
            f"User request: {task}\n\n"
            f"Current robot observation:\n{obs}\n\n"
            f"Objects currently visible:\n{obj_str}"
        )

    def _replan_context(self, task: str, completed: list[dict], failed_step: dict,
                        failure_reason: str, obs: str, visible_objects: list[str]) -> str:
        completed_str = "\n".join(
            f"  {i+1}. {s.get('action','')} {s.get('object','')}"
            + (f" → {s['target']}" if s.get("target") else "")
            for i, s in enumerate(completed)
        ) or "  (none yet)"
        obj_str = ", ".join(visible_objects[:30]) if visible_objects else "none visible"
        return (
            f"User request: {task}\n\nSteps already completed:\n{completed_str}\n\n"
            f"Failed step:\n  action : {failed_step.get('action','')}\n"
            f"  object : {failed_step.get('object','')}\n  target : {failed_step.get('target','')}\n"
            f"  reason : {failure_reason}\n\nCurrent observation:\n{obs}\n\nVisible objects:\n{obj_str}\n\n"
            "Generate ONLY the remaining steps. Do NOT repeat completed steps. Fix the root cause."
        )
