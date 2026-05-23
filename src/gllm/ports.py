"""Provider port.

Each concrete adapter implements `generate(request) -> Response` synchronously.
Sync deliberately: gllm is a one-shot CLI; async would force every caller to
juggle an event loop for no benefit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .domain import Request, Response


class LLMProvider(ABC):
    name: str  # "anthropic" | "openai" | "gemini"

    @abstractmethod
    def generate(self, request: Request) -> Response:
        ...
