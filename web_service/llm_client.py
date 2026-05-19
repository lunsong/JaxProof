"""LLM API client with support for OpenAI-compatible APIs and streaming."""

import os
import re
from pathlib import Path
from typing import Optional, Iterator, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _load_keys_file() -> dict:
    """Parse the local keys file (export VAR=value format) if it exists."""
    keys_path = Path(__file__).with_name("keys")
    result = {}
    if keys_path.exists():
        with open(keys_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r'export\s+(\w+)=["\']?(.*?)["\']?\s*$', line)
                if m:
                    result[m.group(1)] = m.group(2)
    return result


class LLMClient:
    """Generic LLM client wrapping OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # Environment variables take highest priority, then explicit args,
        # then the local keys file.
        file_keys = _load_keys_file()

        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or file_keys.get("OPENAI_API_KEY", "")
        )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL", "")
            or file_keys.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = (
            model
            or os.environ.get("OPENAI_MODEL", "")
            or file_keys.get("OPENAI_MODEL", "gpt-4o")
        )

        if not self.api_key:
            raise RuntimeError(
                "No API key provided. Set OPENAI_API_KEY environment variable, pass api_key, "
                "or create a web_service/keys file with export OPENAI_API_KEY=..."
            )

        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")

        # Use a generous timeout because proof-generation prompts are large
        # and some models can take 60-120s to respond.
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=180,
        )

    def chat_stream(
        self, system_prompt: str, user_prompt: str
    ) -> Iterator[Tuple[str, str]]:
        """
        Stream a chat completion request.
        Yields ("reasoning", chunk) or ("content", chunk) tuples.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield ("reasoning", delta.reasoning_content)
            if delta.content:
                yield ("content", delta.content)

    def formalize_stream(self, spec: str) -> Iterator[Tuple[str, str]]:
        """Stream formalization of a natural language specification."""
        from prompts import FORMALIZATION_CONTEXT, FORMALIZE_PROMPT
        prompt = FORMALIZE_PROMPT.replace("{spec}", spec)
        yield from self.chat_stream(FORMALIZATION_CONTEXT, prompt)

    def implement_stream(self, theorem_text: str) -> Iterator[Tuple[str, str]]:
        """Stream SSA implementation and proof generation."""
        from prompts import FRAMEWORK_CONTEXT, IMPLEMENT_PROMPT
        prompt = IMPLEMENT_PROMPT.replace("{theorem_text}", theorem_text)
        yield from self.chat_stream(FRAMEWORK_CONTEXT, prompt)

    def fix_stream(self, lean_code: str, error: str) -> Iterator[Tuple[str, str]]:
        """Stream compilation error fixes."""
        from prompts import FRAMEWORK_CONTEXT, FIX_PROMPT
        prompt = FIX_PROMPT.replace("{lean_code}", lean_code).replace("{error}", error)
        yield from self.chat_stream(FRAMEWORK_CONTEXT, prompt)

    # Non-streaming wrappers (used by lean_runner and legacy endpoints)

    def formalize(self, spec: str) -> str:
        """Formalize a natural language specification into a Lean theorem."""
        content_parts = []
        for role, chunk in self.formalize_stream(spec):
            if role == "content":
                content_parts.append(chunk)
        return "".join(content_parts)

    def implement(self, theorem_text: str) -> str:
        """Generate SSA implementation and proof from a theorem statement."""
        content_parts = []
        for role, chunk in self.implement_stream(theorem_text):
            if role == "content":
                content_parts.append(chunk)
        return "".join(content_parts)

    def fix(self, lean_code: str, error: str) -> str:
        """Fix compilation errors in Lean code."""
        content_parts = []
        for role, chunk in self.fix_stream(lean_code, error):
            if role == "content":
                content_parts.append(chunk)
        return "".join(content_parts)
