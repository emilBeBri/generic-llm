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

    def list_models(self) -> list[str]:
        """Live model ids the provider's API currently serves, filtered to
        text-generation models. Optional capability: adapters that can't
        enumerate (e.g. Azure deployments) leave this raising. Powers
        `gllm --models`, which probes the live API rather than trusting a
        hand-maintained catalog that drifts out of sync."""
        raise NotImplementedError(
            f"{self.name} adapter cannot list models (not catalog-listable)"
        )
